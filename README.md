# Citation Agent

A minimal agentic citation detector. Give it an article, and it identifies sentences that likely need citations, explains why, and can optionally search for candidate academic sources.

## Getting Started

```powershell
cd "C:\Users\peter\Desktop\side project\citation"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m citation_agent examples\sample_article.txt
```

Analyze your own article:

```powershell
python -m citation_agent path\to\your_article.txt --format markdown
```

Search for candidate academic sources:

```powershell
python -m citation_agent examples\sample_article.txt --with-sources
```

Use an LLM reviewer to recalculate citation confidence:

```powershell
$env:OPENAI_API_KEY="your OpenAI API key"
python -m citation_agent examples\sample_article.txt --llm-review
```

Use the LLM reviewer to generate stronger search queries, then search for sources:

```powershell
python -m citation_agent examples\sample_article.txt --llm-review --with-sources
```

Use an LLM matcher to score whether each candidate source actually supports the sentence:

```powershell
python -m citation_agent examples\sample_article.txt --llm-review --with-sources --match-sources
```

The matcher separates two confidence scores:

- `citation_need_confidence`: how likely the sentence needs a citation
- `source_match_confidence`: how well a candidate source supports that sentence

Optional environment variables:

```powershell
$env:SEMANTIC_SCHOLAR_API_KEY="your Semantic Scholar API key"
$env:CROSSREF_MAILTO="your email"
$env:OPENAI_MODEL="gpt-5.2"
```

`SEMANTIC_SCHOLAR_API_KEY` is optional, but using one usually improves reliability. `CROSSREF_MAILTO` is also optional, but Crossref recommends that tools provide a contact email.

If you see `Semantic Scholar ... HTTP 429`, Semantic Scholar is temporarily rate-limiting the request. The tool will still continue with Crossref results. Setting `SEMANTIC_SCHOLAR_API_KEY` usually helps.

## What It Detects

- Statistics, percentages, money amounts, and years
- Comparative or ranking claims, such as more effective, largest, or best
- Evidence cues, such as study, report, survey, data shows, or according to
- High-stakes claims in domains such as medicine, law, finance, and policy
- Causal claims, such as caused by, leads to, or results in

## Project Structure

```text
src/citation_agent/
  analyzer.py      # Core rule-based analysis
  cli.py           # Command-line interface
  llm_matcher.py   # Optional OpenAI-based source matcher
  llm_reviewer.py  # Optional OpenAI-based reviewer
  source_finder.py # Semantic Scholar and Crossref lookup
  models.py        # Data models
tests/
examples/
```
