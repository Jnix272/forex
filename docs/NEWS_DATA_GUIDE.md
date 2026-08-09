# Forex ML Pipeline - News Data Generation & Acquisition Guide

This document outlines the various methods supported by the pipeline to acquire, generate, and augment historical financial news data for training the NLP components.

## 0. Current State of `historical_news_combined.parquet` (2026-08-07)

The canonical news file `data/raw/news/historical_news_combined.parquet` contains **28,492,015 rows** spanning 2008 → 2026 Q3. Schema: `timestamp_utc, event_type, currency, impact, headline, actual, forecast, source, url, sentiment_score, event_category`.

**Sentiment score coverage**:
- **20,081,059** rows carry a real FinBERT/VADER `sentiment_score` in `[-1.000, +1.000]` (mean ≈ −0.047). These are rows whose `headline` field contained actual headline text.
- **8,410,956** rows have `sentiment_score IS NULL`. Of these, the entire 2026 batch (≈ rows where `headline` was a URL like `http…` rather than real text, because the 2026 source only provided URLs) was explicitly nullified on 2026-08-07 — see step 2 below. URL-fallback rows are never usable for sentiment training.

**Important caveats for downstream code**:
- Backing up the file before any score-touching script is mandatory; the pipeline keeps `historical_news_combined.parquet.pre_step2.bak` for one cycle as a rollback.
- The 2026 source feed (`historical_news_2026.csv`) supplies URLs in the `url` column but left `headline` empty. The ingestion step coalesces `headline ← url` so the column is non-null, but the value is a URL and should be filtered from any NLP feature extraction. Use:
  ```sql
  WHERE sentiment_score IS NOT NULL
    AND lower(substring(trim(headline),1,5)) NOT IN ('http:','https')
  ```
- Pre-2026 unscored rows (no headline / null headline) are also kept null; the FinBERT scoring queue at `scripts/score_historical_news_sentiment.py` skips URL-only and empty headlines by construction.

**How we got here (recent ops log)**:
1. **2026-08-06** — memor-safe merge of `historical_news_2026.csv` into `combined.parquet` via DuckDB (3GB cap, durable `.2026_append.parquet` sidecar, atomic replace). Size grew 1.4G → 1.7G.
2. **2026-08-06 → 2026-08-07** — stream-scored ~19M unique unscored 2008-2025 headlines with `scripts/score_historical_news_sentiment.py`. DuckDB-built queue (`sentiment_queue_unscored.parquet`) + PyArrow batch iteration + checkpoint shards in `data/raw/news/sentiment_map/part_*.parquet`, then joined back into `combined.parquet` with `LEFT JOIN … ON trim(CAST(n.headline AS VARCHAR)) = m.headline` and `coalesce(m.sentiment_score, n.sentiment_score)`.
3. **2026-08-07** — step 2 data-quality fix: nullify sentiment scores on URL-fallback headlines:
   ```sql
   UPDATE sentiment_score = NULL
   WHERE lower(substring(trim(headline),1,5)) IN ('http:','https')
   ```
   applied via an atomic stream-rewrite (`CASE WHEN … THEN NULL::DOUBLE ELSE sentiment_score END`). Verified 0 URL rows still carry a score. Backup retained at `historical_news_combined.parquet.pre_step2.bak`.

If you add a new year's news (e.g., 2027), re-run: (a) merge into `combined.parquet`, (b) `python scripts/score_historical_news_sentiment.py --in data/raw/news/historical_news_combined.parquet` to score the new unscored text headlines, (c) re-verify the score distribution and re-apply the URL-nullify step.

## 1. The Main Multi-Source Downloader (`download_historical_news.py`)
This is the primary ingestion script. It pulls from three distinct feeds and seamlessly merges them into `data/raw/news/historical_news_combined.parquet`.

* **Usage**: `python scripts/download_historical_news.py --start 2008-01-01 --end 2026-01-01 --source free`
* **Sources**:
  * **GDELT (Free)**: Searches the global internet for articles mentioning specific currency pairs (e.g., "EURUSD", "European Central Bank"). Prone to API rate limits (`429 Too Many Requests`).
  * **Official Central Bank Feeds (Free)**: Pulls standard RSS/Atom feeds from the Fed, ECB, BOE, BOJ for exact policy announcements.
  * **EODHD API (Paid)**: Uses an `EODHD_API_KEY` to download high-quality, curated forex news.

## 2. Bulk GDELT Masterfiles (`download_gdelt2_bulk.py`)
Downloads massive 15-minute raw CSV zip files directly from GDELT's servers in bulk and filters them locally using Polars.
* **Pros**: Significantly faster for downloading 18+ years of data compared to the REST API.
* **Cons**: Requires massive amounts of local disk space and RAM to parse the global datasets.

## 3. ForexLive Scraper (`scrape_forexlive.py`)
A custom web scraper that directly pulls headlines and timestamps from ForexLive.com.
* **Pros**: High-quality, low-latency retail forex news.
* **Cons**: Brittle; will break if the website layout or HTML DOM changes.

## 4. HuggingFace Datasets (`download_hf_news.py`)
Connects to the HuggingFace AI Hub to download pre-packaged NLP datasets (e.g., `financial_phrasebank`, `twitter-financial-news-sentiment`).
* **Purpose**: Used primarily for fine-tuning sentiment analysis models (like FinBERT or local Ollama models) on pre-labeled financial data, rather than live trading.

## 5. ForexFactory Economic Calendar (`scrape_forexfactory.py`)
Downloads structured Economic Calendar events (NFP, CPI releases, rate decisions) rather than unstructured news articles.
* **Output**: `data/raw/eco_calendar/events.csv`
* **Purpose**: Allows the ML model to anticipate high-volatility scheduled macroeconomic events.

---

## 6. Synthetic Data Generation (Local LLM / Gemma)
If you have a local LLM running (like `gemma4:e2b` via Ollama), you can synthetically generate or augment data to fill gaps or multiply your dataset size.

### A. 2008 Financial Crisis Generation (`download_2008_news.py`)
* **Usage**: `python scripts/download_2008_news.py`
* **Action**: Connects to `http://localhost:11434` and prompts the LLM to hallucinate realistic headlines from the 2008 financial crisis (Lehman Brothers, AIG bailouts). 
* **Purpose**: Teaches the model how to trade during extreme "black swan" volatility without needing 18-year-old web archives.

### B. News Gap Filling (`generate_synthetic_news_fill.py`)
* **Usage**: `python scripts/generate_synthetic_news_fill.py`
* **Action**: If a specific pair (e.g., EURGBP) is missing news for a specific month, it looks at the real news for related pairs (EURUSD, GBPUSD) and uses templates to generate synthetic cross-pair news.

### C. Data Augmentation (`augment_news.py`)
* **Usage**: `python scripts/augment_news.py --limit 1000`
* **Action**: Takes *real* historical headlines from your `.parquet` file and asks the LLM to rewrite each one in 3 different ways (forced via `format: json`).
* **Purpose**: A classic NLP augmentation technique to multiply your dataset size and prevent overfitting to specific phrasing. Outputs to `historical_news_augmented.csv`.
