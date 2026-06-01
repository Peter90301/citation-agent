# Citation Agent

一個最小可用的 agentic citation detector：輸入一篇文章，工具會自動標出「可能需要 citation」的句子，並說明原因。

## 怎麼開始

```powershell
cd "C:\Users\peter\Desktop\side project\citation"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m citation_agent examples\sample_article.txt
```

你也可以分析自己的文章：

```powershell
python -m citation_agent path\to\your_article.txt --format markdown
```

如果你想自動找候選文獻：

```powershell
python -m citation_agent examples\sample_article.txt --with-sources
```

如果你想用 LLM 重新計算 citation confidence：

```powershell
$env:OPENAI_API_KEY="你的 OpenAI API key"
python -m citation_agent examples\sample_article.txt --llm-review
```

也可以同時讓 LLM 產生更好的搜尋 query，再去找來源：

```powershell
python -m citation_agent examples\sample_article.txt --llm-review --with-sources
```

可選環境變數：

```powershell
$env:SEMANTIC_SCHOLAR_API_KEY="你的 Semantic Scholar API key"
$env:CROSSREF_MAILTO="你的 email"
$env:OPENAI_MODEL="gpt-5.2"
```

`SEMANTIC_SCHOLAR_API_KEY` 不是必填，但有 key 通常比較穩。`CROSSREF_MAILTO` 也不是必填，不過 Crossref 建議工具提供聯絡 email。

如果看到 `Semantic Scholar ... HTTP 429`，代表 Semantic Scholar 暫時限流；工具仍會繼續用 Crossref 找候選來源。設定 `SEMANTIC_SCHOLAR_API_KEY` 後通常會改善。

## 目前能抓什麼

- 統計數字、百分比、金額、年份
- 比較級或最高級主張，例如 more effective、largest、best
- 研究、報告、調查、資料顯示等 evidence cue
- 醫療、法律、金融、政策等高風險主張
- 明確的 causal claim，例如 caused by、leads to、results in

## 專案結構

```text
src/citation_agent/
  analyzer.py   # 核心分析流程
  cli.py        # 命令列介面
  models.py     # 資料結構
tests/
examples/
```

## 下一步

1. 用你的真實文章測試這版規則。
2. 把 false positive / false negative 收集起來。
3. 加上 LLM reviewer：讓模型判斷每句是否真的需要 citation。
4. 加上 citation matcher：讓模型判斷哪篇候選文獻最能支持原句。
5. 加上 citation inserter：把來源插回文章，產出 markdown 或 Word-ready 版本。
