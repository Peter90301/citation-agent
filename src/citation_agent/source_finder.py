import json
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from citation_agent.models import CitationNeed, CitationSuggestion, SourceCandidate


SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
CROSSREF_WORKS_URL = "https://api.crossref.org/works"
DEFAULT_TIMEOUT_SECONDS = 12

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}


class SourceLookupError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceFinderConfig:
    per_provider_limit: int = 3
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    semantic_scholar_api_key: str | None = None
    crossref_mailto: str | None = None

    @classmethod
    def from_env(cls, per_provider_limit: int = 3) -> "SourceFinderConfig":
        return cls(
            per_provider_limit=per_provider_limit,
            semantic_scholar_api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
            crossref_mailto=os.getenv("CROSSREF_MAILTO"),
        )


class SourceFinder:
    def __init__(self, config: SourceFinderConfig | None = None) -> None:
        self.config = config or SourceFinderConfig.from_env()

    def find_for_need(
        self,
        need: CitationNeed,
        query_override: str | None = None,
    ) -> CitationSuggestion:
        query = query_override or build_query(need.sentence)
        sources: list[SourceCandidate] = []
        warnings: list[str] = []

        for provider_name, search in (
            ("Semantic Scholar", self.search_semantic_scholar),
            ("Crossref", self.search_crossref),
        ):
            try:
                sources.extend(search(query))
            except SourceLookupError as exc:
                warnings.append(f"{provider_name}: {exc}")

        return CitationSuggestion(
            citation_need=need,
            query=query,
            sources=tuple(rank_sources(sources)),
            warnings=tuple(warnings),
        )

    def search_semantic_scholar(self, query: str) -> list[SourceCandidate]:
        params = {
            "query": query,
            "limit": str(self.config.per_provider_limit),
            "fields": "title,authors,year,venue,url,abstract,citationCount,externalIds",
        }
        headers = {}
        if self.config.semantic_scholar_api_key:
            headers["x-api-key"] = self.config.semantic_scholar_api_key

        payload = fetch_json(
            SEMANTIC_SCHOLAR_SEARCH_URL,
            params=params,
            headers=headers,
            timeout_seconds=self.config.timeout_seconds,
        )
        return [semantic_scholar_to_candidate(item) for item in payload.get("data", [])]

    def search_crossref(self, query: str) -> list[SourceCandidate]:
        params = {
            "query.bibliographic": query,
            "rows": str(self.config.per_provider_limit),
            "select": "title,author,published-print,published-online,issued,container-title,DOI,URL,is-referenced-by-count,abstract,score",
        }
        if self.config.crossref_mailto:
            params["mailto"] = self.config.crossref_mailto

        payload = fetch_json(
            CROSSREF_WORKS_URL,
            params=params,
            headers={"User-Agent": user_agent(self.config.crossref_mailto)},
            timeout_seconds=self.config.timeout_seconds,
        )
        items = payload.get("message", {}).get("items", [])
        return [crossref_to_candidate(item) for item in items]


def build_query(sentence: str, max_terms: int = 12) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z-]{2,}|\d{4}", sentence)
    terms: list[str] = []

    for word in words:
        normalized = word.lower()
        if normalized in STOPWORDS:
            continue
        if normalized not in terms:
            terms.append(normalized)
        if len(terms) >= max_terms:
            break

    return " ".join(terms) or sentence[:160]


def rank_sources(sources: list[SourceCandidate]) -> list[SourceCandidate]:
    return sorted(
        sources,
        key=lambda source: (
            source.relevance_score or 0,
            source.citation_count or 0,
            source.year or 0,
        ),
        reverse=True,
    )


def fetch_json(
    url: str,
    *,
    params: dict[str, str],
    headers: dict[str, str] | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    request_url = f"{url}?{urlencode(params)}"
    request = Request(request_url, headers=headers or {})

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise SourceLookupError(f"{url} returned HTTP {exc.code}") from exc
    except URLError as exc:
        raise SourceLookupError(f"Could not reach {url}: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise SourceLookupError(f"{url} returned invalid JSON") from exc


def semantic_scholar_to_candidate(item: dict[str, Any]) -> SourceCandidate:
    external_ids = item.get("externalIds") or {}
    authors = tuple(author.get("name", "") for author in item.get("authors", []) if author.get("name"))
    return SourceCandidate(
        provider="Semantic Scholar",
        title=item.get("title") or "Untitled",
        authors=authors,
        year=item.get("year"),
        venue=item.get("venue") or None,
        doi=external_ids.get("DOI"),
        url=item.get("url"),
        abstract=item.get("abstract"),
        citation_count=item.get("citationCount"),
        relevance_score=None,
    )


def crossref_to_candidate(item: dict[str, Any]) -> SourceCandidate:
    return SourceCandidate(
        provider="Crossref",
        title=first(item.get("title")) or "Untitled",
        authors=tuple(format_crossref_author(author) for author in item.get("author", [])),
        year=published_year(item),
        venue=first(item.get("container-title")),
        doi=item.get("DOI"),
        url=item.get("URL"),
        abstract=strip_tags(item.get("abstract")),
        citation_count=item.get("is-referenced-by-count"),
        relevance_score=item.get("score"),
    )


def first(value: list[Any] | None) -> Any:
    if not value:
        return None
    return value[0]


def published_year(item: dict[str, Any]) -> int | None:
    for key in ("published-print", "published-online", "issued"):
        date_parts = item.get(key, {}).get("date-parts", [])
        if date_parts and date_parts[0]:
            return date_parts[0][0]
    return None


def format_crossref_author(author: dict[str, Any]) -> str:
    given = author.get("given", "")
    family = author.get("family", "")
    return " ".join(part for part in (given, family) if part).strip() or author.get("name", "")


def strip_tags(value: str | None) -> str | None:
    if not value:
        return None
    return re.sub(r"<[^>]+>", "", value).strip()


def user_agent(mailto: str | None) -> str:
    base = "citation-agent/0.1.0 (https://example.local/citation-agent)"
    if mailto:
        return f"{base}; mailto:{mailto}"
    return base
