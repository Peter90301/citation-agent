from dataclasses import dataclass
from enum import Enum


class CitationReason(str, Enum):
    STATISTIC = "statistic_or_number"
    RESEARCH_CUE = "research_or_report_claim"
    COMPARISON = "comparison_or_ranking"
    CAUSAL = "causal_claim"
    HIGH_STAKES = "high_stakes_claim"
    HISTORICAL = "historical_or_time_specific_claim"


@dataclass(frozen=True)
class CitationNeed:
    sentence: str
    start: int
    end: int
    reasons: tuple[CitationReason, ...]
    confidence: float

    @property
    def reason_labels(self) -> list[str]:
        return [reason.value for reason in self.reasons]


@dataclass(frozen=True)
class SourceCandidate:
    provider: str
    title: str
    authors: tuple[str, ...]
    year: int | None
    venue: str | None
    doi: str | None
    url: str | None
    abstract: str | None
    citation_count: int | None
    relevance_score: float | None

    @property
    def author_text(self) -> str:
        if not self.authors:
            return "Unknown authors"
        if len(self.authors) <= 3:
            return ", ".join(self.authors)
        return f"{', '.join(self.authors[:3])}, et al."


@dataclass(frozen=True)
class CitationSuggestion:
    citation_need: CitationNeed
    query: str
    sources: tuple[SourceCandidate, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class LLMReview:
    needs_citation: bool
    confidence: float
    reasons: tuple[str, ...]
    rationale: str
    suggested_query: str | None = None


@dataclass(frozen=True)
class ReviewedCitationNeed:
    citation_need: CitationNeed
    review: LLMReview
