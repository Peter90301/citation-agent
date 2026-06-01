import json
import os
from dataclasses import dataclass
from typing import Any

from citation_agent.llm_reviewer import (
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    LLMReviewError,
    find_first_text,
    post_openai_json,
)
from citation_agent.models import (
    CitationSuggestion,
    MatchedCitationSuggestion,
    SourceCandidate,
    SourceMatch,
)


@dataclass(frozen=True)
class LLMMatcherConfig:
    api_key: str
    model: str = DEFAULT_MODEL
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS

    @classmethod
    def from_env(cls) -> "LLMMatcherConfig":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMReviewError("OPENAI_API_KEY is required for --match-sources")
        return cls(
            api_key=api_key,
            model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        )


class OpenAILLMSourceMatcher:
    def __init__(self, config: LLMMatcherConfig | None = None) -> None:
        self.config = config or LLMMatcherConfig.from_env()

    def match_many(
        self,
        suggestions: list[CitationSuggestion],
    ) -> list[MatchedCitationSuggestion]:
        matched: list[MatchedCitationSuggestion] = []
        for suggestion in suggestions:
            if not suggestion.sources:
                matched.append(MatchedCitationSuggestion(suggestion, ()))
                continue
            matched.append(self.match_one(suggestion))
        return matched

    def match_one(self, suggestion: CitationSuggestion) -> MatchedCitationSuggestion:
        payload = build_match_payload(self.config.model, suggestion)
        response = post_openai_json(
            payload,
            api_key=self.config.api_key,
            timeout_seconds=self.config.timeout_seconds,
        )
        match_items = parse_match_output(response)
        matches_by_index = {
            item["source_index"]: source_match_from_item(
                suggestion.sources[item["source_index"] - 1],
                item,
            )
            for item in match_items
            if valid_source_index(item, suggestion.sources)
        }

        matches = tuple(
            matches_by_index.get(
                index,
                SourceMatch(
                    source=source,
                    supports_claim=False,
                    support_score=0.0,
                    rationale="LLM did not return a match judgment for this source.",
                    limitations="No source-match review was available.",
                ),
            )
            for index, source in enumerate(suggestion.sources, start=1)
        )
        return MatchedCitationSuggestion(suggestion, rank_matches(matches))


def build_match_payload(model: str, suggestion: CitationSuggestion) -> dict[str, Any]:
    sources = [
        {
            "source_index": index,
            "provider": source.provider,
            "title": source.title,
            "authors": list(source.authors),
            "year": source.year,
            "venue": source.venue,
            "doi": source.doi,
            "url": source.url,
            "abstract": source.abstract,
            "citation_count": source.citation_count,
        }
        for index, source in enumerate(suggestion.sources, start=1)
    ]

    return {
        "model": model,
        "max_output_tokens": 1200,
        "input": [
            {
                "role": "system",
                "content": (
                    "You judge whether candidate academic sources support a specific "
                    "article sentence. Return JSON only. Be conservative: a source "
                    "supports the sentence only when its title, abstract, or metadata "
                    "directly addresses the same factual claim."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "sentence": suggestion.citation_need.sentence,
                        "citation_need_confidence": suggestion.citation_need.confidence,
                        "search_query": suggestion.query,
                        "candidate_sources": sources,
                        "task": (
                            "For each source, decide whether it supports the sentence. "
                            "Set support_score from 0 to 1. Explain the rationale and "
                            "any limitations in one concise sentence each."
                        ),
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "source_matches",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "matches": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "source_index": {"type": "integer"},
                                    "supports_claim": {"type": "boolean"},
                                    "support_score": {"type": "number"},
                                    "rationale": {"type": "string"},
                                    "limitations": {"type": "string"},
                                },
                                "required": [
                                    "source_index",
                                    "supports_claim",
                                    "support_score",
                                    "rationale",
                                    "limitations",
                                ],
                            },
                        }
                    },
                    "required": ["matches"],
                },
            }
        },
    }


def parse_match_output(response: dict[str, Any]) -> list[dict[str, Any]]:
    output_text = response.get("output_text") or find_first_text(response.get("output", []))
    if not output_text:
        raise LLMReviewError("OpenAI response did not include source match output text")

    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise LLMReviewError("Source match output was not valid JSON") from exc

    matches = parsed.get("matches")
    if not isinstance(matches, list):
        raise LLMReviewError("Source match output did not include a matches array")
    return matches


def source_match_from_item(source: SourceCandidate, item: dict[str, Any]) -> SourceMatch:
    return SourceMatch(
        source=source,
        supports_claim=bool(item["supports_claim"]),
        support_score=max(0.0, min(float(item["support_score"]), 1.0)),
        rationale=str(item.get("rationale", "")).strip(),
        limitations=str(item.get("limitations", "")).strip(),
    )


def valid_source_index(
    item: dict[str, Any],
    sources: tuple[SourceCandidate, ...],
) -> bool:
    index = item.get("source_index")
    return isinstance(index, int) and 1 <= index <= len(sources)


def rank_matches(matches: tuple[SourceMatch, ...]) -> tuple[SourceMatch, ...]:
    return tuple(
        sorted(
            matches,
            key=lambda match: (
                match.supports_claim,
                match.support_score,
                match.source.citation_count or 0,
            ),
            reverse=True,
        )
    )
