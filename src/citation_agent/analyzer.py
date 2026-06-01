import re
from collections.abc import Iterable

from citation_agent.models import CitationNeed, CitationReason


SENTENCE_END_RE = re.compile(r"(?<=[.!?。！？])\s+")
NUMBER_RE = re.compile(
    r"(\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b|\b\d+(?:\.\d+)?%|\$\d+(?:,\d{3})*(?:\.\d+)?)"
)
YEAR_RE = re.compile(r"\b(18|19|20)\d{2}\b")

RESEARCH_CUES = (
    "study",
    "studies",
    "research",
    "report",
    "survey",
    "poll",
    "data shows",
    "evidence suggests",
    "according to",
    "analysis found",
    "研究",
    "報告",
    "調查",
    "數據顯示",
    "資料顯示",
    "根據",
)

COMPARISON_CUES = (
    "more than",
    "less than",
    "higher than",
    "lower than",
    "better than",
    "worse than",
    "most",
    "least",
    "largest",
    "smallest",
    "best",
    "worst",
    "faster",
    "slower",
    "更高",
    "更低",
    "更多",
    "最",
)

CAUSAL_CUES = (
    "causes",
    "caused by",
    "leads to",
    "led to",
    "results in",
    "resulted in",
    "because of",
    "due to",
    "drives",
    "increases the risk",
    "reduces the risk",
    "導致",
    "造成",
    "因為",
    "提高風險",
    "降低風險",
)

HIGH_STAKES_CUES = (
    "medical",
    "health",
    "disease",
    "treatment",
    "legal",
    "law",
    "policy",
    "financial",
    "investment",
    "tax",
    "medicine",
    "clinical",
    "醫療",
    "健康",
    "疾病",
    "治療",
    "法律",
    "政策",
    "金融",
    "投資",
    "稅務",
)


class CitationAnalyzer:
    """Rule-based first pass for detecting claims that likely need citations."""

    def analyze(self, text: str) -> list[CitationNeed]:
        findings: list[CitationNeed] = []

        for sentence, start, end in split_sentences(text):
            reasons = tuple(self._reasons_for(sentence))
            if not reasons:
                continue

            findings.append(
                CitationNeed(
                    sentence=sentence.strip(),
                    start=start,
                    end=end,
                    reasons=reasons,
                    confidence=self._confidence(reasons),
                )
            )

        return findings

    def _reasons_for(self, sentence: str) -> Iterable[CitationReason]:
        normalized = sentence.lower()

        if NUMBER_RE.search(sentence):
            yield CitationReason.STATISTIC
        if any(cue in normalized for cue in RESEARCH_CUES):
            yield CitationReason.RESEARCH_CUE
        if any(cue in normalized for cue in COMPARISON_CUES):
            yield CitationReason.COMPARISON
        if any(cue in normalized for cue in CAUSAL_CUES):
            yield CitationReason.CAUSAL
        if any(cue in normalized for cue in HIGH_STAKES_CUES):
            yield CitationReason.HIGH_STAKES
        if YEAR_RE.search(sentence):
            yield CitationReason.HISTORICAL

    def _confidence(self, reasons: tuple[CitationReason, ...]) -> float:
        base = 0.48
        score = base + (0.13 * len(reasons))
        if CitationReason.RESEARCH_CUE in reasons:
            score += 0.12
        if CitationReason.HIGH_STAKES in reasons:
            score += 0.1
        return min(score, 0.95)


def split_sentences(text: str) -> list[tuple[str, int, int]]:
    chunks = SENTENCE_END_RE.split(text.strip())
    sentences: list[tuple[str, int, int]] = []
    cursor = 0

    for chunk in chunks:
        sentence = chunk.strip()
        if not sentence:
            continue
        start = text.find(sentence, cursor)
        end = start + len(sentence)
        sentences.append((sentence, start, end))
        cursor = end

    return sentences
