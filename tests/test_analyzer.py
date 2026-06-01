from citation_agent import CitationAnalyzer


def test_detects_statistic_and_research_claims():
    text = "A 2024 report found that 62% of teams use hybrid work. This is interesting."

    findings = CitationAnalyzer().analyze(text)

    assert len(findings) == 1
    assert "2024 report found" in findings[0].sentence
    assert "statistic_or_number" in findings[0].reason_labels
    assert "research_or_report_claim" in findings[0].reason_labels


def test_ignores_plain_opinion():
    text = "The product feels polished. The interface is pleasant."

    findings = CitationAnalyzer().analyze(text)

    assert findings == []
