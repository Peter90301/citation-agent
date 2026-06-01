from citation_agent.source_finder import (
    build_query,
    crossref_to_candidate,
    semantic_scholar_to_candidate,
)


def test_build_query_removes_common_words():
    query = build_query(
        "In healthcare, delayed diagnosis can increase the risk of complications."
    )

    assert query == "healthcare delayed diagnosis increase risk complications"


def test_semantic_scholar_to_candidate():
    candidate = semantic_scholar_to_candidate(
        {
            "title": "A Useful Paper",
            "authors": [{"name": "Ada Lovelace"}, {"name": "Grace Hopper"}],
            "year": 2024,
            "venue": "Journal of Useful Things",
            "url": "https://example.com/paper",
            "abstract": "Important findings.",
            "citationCount": 12,
            "externalIds": {"DOI": "10.123/example"},
        }
    )

    assert candidate.provider == "Semantic Scholar"
    assert candidate.title == "A Useful Paper"
    assert candidate.authors == ("Ada Lovelace", "Grace Hopper")
    assert candidate.doi == "10.123/example"
    assert candidate.citation_count == 12


def test_crossref_to_candidate():
    candidate = crossref_to_candidate(
        {
            "title": ["A Crossref Paper"],
            "author": [{"given": "Alan", "family": "Turing"}],
            "issued": {"date-parts": [[1950]]},
            "container-title": ["Mind"],
            "DOI": "10.456/example",
            "URL": "https://doi.org/10.456/example",
            "is-referenced-by-count": 99,
            "abstract": "<jats:p>Classic article.</jats:p>",
            "score": 88.5,
        }
    )

    assert candidate.provider == "Crossref"
    assert candidate.title == "A Crossref Paper"
    assert candidate.authors == ("Alan Turing",)
    assert candidate.year == 1950
    assert candidate.abstract == "Classic article."
    assert candidate.relevance_score == 88.5
