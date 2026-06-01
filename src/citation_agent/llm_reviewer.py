import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from citation_agent.models import CitationNeed, LLMReview, ReviewedCitationNeed


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = "gpt-5.2"
DEFAULT_TIMEOUT_SECONDS = 30


class LLMReviewError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMReviewerConfig:
    api_key: str
    model: str = DEFAULT_MODEL
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS

    @classmethod
    def from_env(cls) -> "LLMReviewerConfig":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMReviewError("OPENAI_API_KEY is required for --llm-review")
        return cls(
            api_key=api_key,
            model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        )


class OpenAILLMReviewer:
    def __init__(self, config: LLMReviewerConfig | None = None) -> None:
        self.config = config or LLMReviewerConfig.from_env()

    def review_many(
        self,
        article_text: str,
        citation_needs: list[CitationNeed],
    ) -> list[ReviewedCitationNeed]:
        if not citation_needs:
            return []

        payload = build_review_payload(self.config.model, article_text, citation_needs)
        response = post_openai_json(
            payload,
            api_key=self.config.api_key,
            timeout_seconds=self.config.timeout_seconds,
        )
        review_items = parse_structured_output(response)
        reviews_by_index = {
            item["index"]: review_from_item(item)
            for item in review_items
            if isinstance(item.get("index"), int)
        }

        reviewed: list[ReviewedCitationNeed] = []
        for index, need in enumerate(citation_needs, start=1):
            review = reviews_by_index.get(index)
            if not review:
                review = LLMReview(
                    needs_citation=True,
                    confidence=need.confidence,
                    reasons=tuple(need.reason_labels),
                    rationale="LLM did not return a review for this sentence; using rule-based confidence.",
                )
            reviewed.append(ReviewedCitationNeed(citation_need=need, review=review))

        return reviewed


def build_review_payload(
    model: str,
    article_text: str,
    citation_needs: list[CitationNeed],
) -> dict[str, Any]:
    candidates = [
        {
            "index": index,
            "sentence": need.sentence,
            "rule_reasons": need.reason_labels,
            "rule_confidence": need.confidence,
        }
        for index, need in enumerate(citation_needs, start=1)
    ]

    return {
        "model": model,
        "max_output_tokens": 1000,
        "input": [
            {
                "role": "system",
                "content": (
                    "You review article sentences for citation needs. "
                    "Return JSON only. A sentence needs a citation when it makes "
                    "a factual, empirical, statistical, causal, historical, comparative, "
                    "medical, legal, financial, or policy claim that a reader may reasonably "
                    "ask to verify. Opinions, transitions, and common writing glue usually "
                    "do not need citations."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "article": article_text,
                        "candidate_sentences": candidates,
                        "task": (
                            "For each candidate, decide whether it needs a citation. "
                            "Set confidence from 0 to 1 for your citation-need judgment. "
                            "Provide concise reasons, a one-sentence rationale, and a "
                            "scholarly search query suitable for Semantic Scholar or Crossref."
                        ),
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "citation_reviews",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "reviews": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "index": {"type": "integer"},
                                    "needs_citation": {"type": "boolean"},
                                    "confidence": {
                                        "type": "number",
                                    },
                                    "reasons": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "rationale": {"type": "string"},
                                    "suggested_query": {"type": "string"},
                                },
                                "required": [
                                    "index",
                                    "needs_citation",
                                    "confidence",
                                    "reasons",
                                    "rationale",
                                    "suggested_query",
                                ],
                            },
                        }
                    },
                    "required": ["reviews"],
                },
            }
        },
    }


def post_openai_json(
    payload: dict[str, Any],
    *,
    api_key: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    request = Request(
        OPENAI_RESPONSES_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise LLMReviewError(f"OpenAI API returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise LLMReviewError(f"Could not reach OpenAI API: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise LLMReviewError("OpenAI API returned invalid JSON") from exc


def parse_structured_output(response: dict[str, Any]) -> list[dict[str, Any]]:
    output_text = response.get("output_text") or find_first_text(response.get("output", []))
    if not output_text:
        raise LLMReviewError("OpenAI response did not include output text")

    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise LLMReviewError("LLM review output was not valid JSON") from exc

    reviews = parsed.get("reviews")
    if not isinstance(reviews, list):
        raise LLMReviewError("LLM review output did not include a reviews array")
    return reviews


def find_first_text(value: Any) -> str | None:
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        for child in value.values():
            found = find_first_text(child)
            if found:
                return found
    if isinstance(value, list):
        for child in value:
            found = find_first_text(child)
            if found:
                return found
    return None


def review_from_item(item: dict[str, Any]) -> LLMReview:
    return LLMReview(
        needs_citation=bool(item["needs_citation"]),
        confidence=max(0.0, min(float(item["confidence"]), 1.0)),
        reasons=tuple(str(reason) for reason in item.get("reasons", [])),
        rationale=str(item.get("rationale", "")).strip(),
        suggested_query=str(item.get("suggested_query", "")).strip() or None,
    )
