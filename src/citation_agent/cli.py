import argparse
import json
from pathlib import Path

from citation_agent.analyzer import CitationAnalyzer
from citation_agent.llm_matcher import OpenAILLMSourceMatcher
from citation_agent.llm_reviewer import LLMReviewError, OpenAILLMReviewer
from citation_agent.models import CitationNeed
from citation_agent.source_finder import SourceFinder, SourceFinderConfig


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect article sentences that likely need citations."
    )
    parser.add_argument("article", type=Path, help="Path to a plain text article.")
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format.",
    )
    parser.add_argument(
        "--with-sources",
        action="store_true",
        help="Search Semantic Scholar and Crossref for candidate sources.",
    )
    parser.add_argument(
        "--llm-review",
        action="store_true",
        help="Use an LLM reviewer to recalculate citation-need confidence.",
    )
    parser.add_argument(
        "--source-limit",
        type=int,
        default=3,
        help="Candidate sources to request from each provider for each sentence.",
    )
    parser.add_argument(
        "--match-sources",
        action="store_true",
        help="Use an LLM to score whether candidate sources support each sentence.",
    )
    args = parser.parse_args()
    if args.match_sources and not args.with_sources:
        parser.error("--match-sources requires --with-sources")

    text = args.article.read_text(encoding="utf-8")
    findings = CitationAnalyzer().analyze(text)
    reviewed_findings = []
    suggestions = []
    matched_suggestions = []
    source_matcher = None

    if args.llm_review:
        try:
            reviewed_findings = OpenAILLMReviewer().review_many(text, findings)
        except LLMReviewError as exc:
            parser.error(str(exc))

    if args.match_sources:
        try:
            source_matcher = OpenAILLMSourceMatcher()
        except LLMReviewError as exc:
            parser.error(str(exc))

    if args.with_sources:
        finder = SourceFinder(SourceFinderConfig.from_env(per_provider_limit=args.source_limit))
        if args.llm_review:
            for reviewed in reviewed_findings:
                if not reviewed.review.needs_citation:
                    continue
                need = apply_review_confidence(reviewed.citation_need, reviewed.review.confidence)
                suggestions.append(
                    finder.find_for_need(
                        need,
                        query_override=reviewed.review.suggested_query,
                    )
                )
        else:
            for finding in findings:
                suggestions.append(finder.find_for_need(finding))

    if args.match_sources:
        matched_suggestions = source_matcher.match_many(suggestions)

    if args.format == "json":
        print(
            json.dumps(
                serialize_matched_suggestions(matched_suggestions)
                if args.match_sources
                else serialize_suggestions(suggestions)
                if args.with_sources
                else serialize_reviewed_findings(reviewed_findings)
                if args.llm_review
                else [serialize_finding(item) for item in findings],
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        if args.match_sources:
            print_matched_suggestions_markdown(matched_suggestions)
        elif args.with_sources:
            print_suggestions_markdown(suggestions)
        elif args.llm_review:
            print_reviewed_markdown(reviewed_findings)
        else:
            print_markdown(findings)

    return 0


def print_markdown(findings) -> None:
    if not findings:
        print("No likely citation needs found.")
        return

    print(f"Found {len(findings)} likely citation need(s).\n")
    for index, item in enumerate(findings, start=1):
        reasons = ", ".join(item.reason_labels)
        print(f"## {index}. Confidence {item.confidence:.2f}")
        print(item.sentence)
        print(f"\nReasons: {reasons}\n")


def print_reviewed_markdown(reviewed_findings) -> None:
    if not reviewed_findings:
        print("No likely citation needs found.")
        return

    print(f"Reviewed {len(reviewed_findings)} candidate citation need(s).\n")
    for index, reviewed in enumerate(reviewed_findings, start=1):
        item = reviewed.citation_need
        review = reviewed.review
        status = "needs citation" if review.needs_citation else "does not need citation"
        rule_reasons = ", ".join(item.reason_labels)
        llm_reasons = ", ".join(review.reasons) or "none"
        print(f"## {index}. LLM confidence {review.confidence:.2f} - {status}")
        print(item.sentence)
        print(f"\nRule confidence: {item.confidence:.2f}")
        print(f"Rule reasons: {rule_reasons}")
        print(f"LLM reasons: {llm_reasons}")
        print(f"Rationale: {review.rationale}")
        if review.suggested_query:
            print(f"Suggested query: {review.suggested_query}")
        print()


def print_suggestions_markdown(suggestions) -> None:
    if not suggestions:
        print("No citation suggestions found.")
        return

    print(f"Found {len(suggestions)} citation need(s) with source suggestions.\n")
    for index, suggestion in enumerate(suggestions, start=1):
        item = suggestion.citation_need
        reasons = ", ".join(item.reason_labels)
        print(f"## {index}. Confidence {item.confidence:.2f}")
        print(item.sentence)
        print(f"\nReasons: {reasons}")
        print(f"Search query: {suggestion.query}\n")
        for warning in suggestion.warnings:
            print(f"Warning: {warning}")
        if suggestion.warnings:
            print()

        if not suggestion.sources:
            print("No candidate sources found.\n")
            continue

        for source_index, source in enumerate(suggestion.sources, start=1):
            year = source.year or "n.d."
            venue = f" - {source.venue}" if source.venue else ""
            doi = f" DOI: {source.doi}" if source.doi else ""
            url = f" {source.url}" if source.url else ""
            citations = (
                f" Citations: {source.citation_count}."
                if source.citation_count is not None
                else ""
            )
            print(f"{source_index}. [{source.provider}] {source.title}")
            print(f"   {source.author_text} ({year}){venue}.{doi}{citations}{url}")
        print()


def print_matched_suggestions_markdown(matched_suggestions) -> None:
    if not matched_suggestions:
        print("No matched citation suggestions found.")
        return

    print(
        f"Found {len(matched_suggestions)} citation need(s) with source match scores.\n"
    )
    for index, matched in enumerate(matched_suggestions, start=1):
        suggestion = matched.citation_suggestion
        item = suggestion.citation_need
        reasons = ", ".join(item.reason_labels)
        print(f"## {index}. Citation need confidence {item.confidence:.2f}")
        print(item.sentence)
        print(f"\nReasons: {reasons}")
        print(f"Search query: {suggestion.query}\n")
        for warning in suggestion.warnings:
            print(f"Warning: {warning}")
        if suggestion.warnings:
            print()

        if not matched.matches:
            print("No candidate sources found.\n")
            continue

        for source_index, match in enumerate(matched.matches, start=1):
            source = match.source
            year = source.year or "n.d."
            venue = f" - {source.venue}" if source.venue else ""
            doi = f" DOI: {source.doi}" if source.doi else ""
            url = f" {source.url}" if source.url else ""
            support = "supports" if match.supports_claim else "weak/no support"
            print(
                f"{source_index}. [{source.provider}] {source.title} "
                f"- source match confidence {match.support_score:.2f} ({support})"
            )
            print(f"   {source.author_text} ({year}){venue}.{doi}{url}")
            print(f"   Rationale: {match.rationale}")
            print(f"   Limitations: {match.limitations}")
        print()


def serialize_finding(item) -> dict:
    return {
        "sentence": item.sentence,
        "start": item.start,
        "end": item.end,
        "reasons": item.reason_labels,
        "confidence": item.confidence,
        "citation_need_confidence": item.confidence,
    }


def serialize_reviewed_findings(reviewed_findings) -> list[dict]:
    return [
        {
            **serialize_finding(reviewed.citation_need),
            "llm_review": {
                "needs_citation": reviewed.review.needs_citation,
                "confidence": reviewed.review.confidence,
                "reasons": list(reviewed.review.reasons),
                "rationale": reviewed.review.rationale,
                "suggested_query": reviewed.review.suggested_query,
            },
        }
        for reviewed in reviewed_findings
    ]


def serialize_suggestions(suggestions) -> list[dict]:
    return [
        {
            **serialize_finding(suggestion.citation_need),
            "query": suggestion.query,
            "warnings": list(suggestion.warnings),
            "sources": [
                {
                    "provider": source.provider,
                    "title": source.title,
                    "authors": list(source.authors),
                    "year": source.year,
                    "venue": source.venue,
                    "doi": source.doi,
                    "url": source.url,
                    "abstract": source.abstract,
                    "citation_count": source.citation_count,
                    "relevance_score": source.relevance_score,
                }
                for source in suggestion.sources
            ],
        }
        for suggestion in suggestions
    ]


def serialize_matched_suggestions(matched_suggestions) -> list[dict]:
    return [
        {
            **serialize_finding(matched.citation_suggestion.citation_need),
            "query": matched.citation_suggestion.query,
            "warnings": list(matched.citation_suggestion.warnings),
            "matches": [
                {
                    "source": {
                        "provider": match.source.provider,
                        "title": match.source.title,
                        "authors": list(match.source.authors),
                        "year": match.source.year,
                        "venue": match.source.venue,
                        "doi": match.source.doi,
                        "url": match.source.url,
                        "abstract": match.source.abstract,
                        "citation_count": match.source.citation_count,
                        "relevance_score": match.source.relevance_score,
                    },
                    "supports_claim": match.supports_claim,
                    "source_match_confidence": match.support_score,
                    "rationale": match.rationale,
                    "limitations": match.limitations,
                }
                for match in matched.matches
            ],
        }
        for matched in matched_suggestions
    ]


def apply_review_confidence(need: CitationNeed, confidence: float) -> CitationNeed:
    return CitationNeed(
        sentence=need.sentence,
        start=need.start,
        end=need.end,
        reasons=need.reasons,
        confidence=confidence,
    )
