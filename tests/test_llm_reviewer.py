import json

from citation_agent.llm_reviewer import parse_structured_output, review_from_item


def test_parse_structured_output_from_output_text():
    response = {
        "output_text": json.dumps(
            {
                "reviews": [
                    {
                        "index": 1,
                        "needs_citation": True,
                        "confidence": 0.91,
                        "reasons": ["statistical claim"],
                        "rationale": "The sentence makes a verifiable claim.",
                        "suggested_query": "hybrid work employee preference survey",
                    }
                ]
            }
        )
    }

    reviews = parse_structured_output(response)

    assert reviews[0]["index"] == 1
    assert reviews[0]["confidence"] == 0.91


def test_parse_structured_output_from_nested_response_content():
    response = {
        "output": [
            {
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps({"reviews": []}),
                    }
                ]
            }
        ]
    }

    assert parse_structured_output(response) == []


def test_review_from_item_clamps_confidence():
    review = review_from_item(
        {
            "needs_citation": True,
            "confidence": 1.4,
            "reasons": ["medical claim"],
            "rationale": "Needs verification.",
            "suggested_query": "delayed diagnosis complications",
        }
    )

    assert review.confidence == 1.0
    assert review.reasons == ("medical claim",)
