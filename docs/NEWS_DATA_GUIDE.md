# Forex ML Pipeline - News Data Generation & Acquisition Guide

This document outlines the various methods supported by the pipeline to acquire, generate, and augment historical financial news data for training the NLP components.

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
