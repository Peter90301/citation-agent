import json

from citation_agent.llm_matcher import (
    parse_match_output,
    rank_matches,
    source_match_from_item,
)
from citation_agent.models import SourceCandidate, SourceMatch


def source(title: str, citation_count: int = 0) -> SourceCandidate:
    return SourceCandidate(
        provider="Crossref",
        title=title,
        authors=("Ada Lovelace",),
        year=2024,
        venue="Journal",
        doi=None,
        url=None,
        abstract="Abstract.",
        citation_count=citation_count,
        relevance_score=None,
    )


def test_parse_match_output_from_output_text():
    response = {
        "output_text": json.dumps(
            {
                "matches": [
                    {
                        "source_index": 1,
                        "supports_claim": True,
                        "support_score": 0.86,
                        "rationale": "The abstract directly discusses the claim.",
                        "limitations": "The sample is domain-specific.",
                    }
                ]
            }
        )
    }

    matches = parse_match_output(response)

    assert matches[0]["source_index"] == 1
    assert matches[0]["support_score"] == 0.86


def test_source_match_from_item_clamps_support_score():
    match = source_match_from_item(
        source("Useful Paper"),
        {
            "supports_claim": True,
            "support_score": 1.3,
            "rationale": "Directly relevant.",
            "limitations": "Limited metadata.",
        },
    )

    assert match.support_score == 1.0
    assert match.supports_claim is True


def test_rank_matches_prefers_support_and_score():
    weak = SourceMatch(
        source=source("Weak", citation_count=100),
        supports_claim=False,
        support_score=0.95,
        rationale="Not about the same claim.",
        limitations="Mismatch.",
    )
    strong = SourceMatch(
        source=source("Strong", citation_count=1),
        supports_claim=True,
        support_score=0.8,
        rationale="Same claim.",
        limitations="None.",
    )

    ranked = rank_matches((weak, strong))

    assert ranked[0].source.title == "Strong"
