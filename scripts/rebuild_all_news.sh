#!/usr/bin/env bash
# Rebuild all news + eco sources into historical_news_combined.parquet
#
# Sources:
#   1) GDELT bulk masterfiles   (download_gdelt2_bulk.py) — reuse 2021-2025, extend 2026+
#   2) ForexLive scraper        (scrape_forexlive.py)
#   3) HuggingFace datasets     (download_hf_news.py) — skip if Hugging_Face.csv already present
#   4) ForexFactory calendar    (scrape_forexfactory.py) → eco_calendar/events.csv
#   5) Merge + FX filter        (merge_massive_datasets.py)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv/bin/python"
TODAY="$(date -u +%Y-%m-%d)"
HF_OUT="data/raw/news/Hugging_Face.csv"

echo "=== 1/5 HuggingFace financial news ==="
if [[ -f "$HF_OUT" ]] && [[ $(stat -c%s "$HF_OUT") -gt 100000000 ]]; then
  echo "Skipping HF download — $HF_OUT already exists ($(du -h "$HF_OUT" | cut -f1))"
else
  "$PY" scripts/download_hf_news.py || echo "WARN: HF download failed"
fi

echo "=== 2/5 ForexLive scrape ==="
if ! "$PY" scripts/scrape_forexlive.py --out data/raw/news/historical_news_forexlive.csv; then
  echo "WARN: ForexLive scrape failed — continuing with other sources"
fi

echo "=== 3/5 GDELT bulk (extend 2026 → ${TODAY}) ==="
if ! "$PY" scripts/download_gdelt2_bulk.py \
  --start 2026-01-01 \
  --end "$TODAY" \
  --workers 8 \
  --out data/raw/news/historical_news_2026.csv; then
  echo "WARN: GDELT 2026 extend failed — merge will use existing 2021-2025 only"
fi

echo "=== 4/5 ForexFactory economic calendar ==="
if ! "$PY" scripts/scrape_forexfactory.py \
  --start 2008-01-01 \
  --end "$TODAY" \
  --out data/raw/eco_calendar/events.csv; then
  echo "WARN: ForexFactory scrape failed — keeping existing events.csv if present"
fi

echo "=== 5/5 Merge → historical_news_combined.parquet ==="
rm -f data/raw/news/historical_news_master.csv
INPUTS=(
  data/raw/news/Hugging_Face.csv
  data/raw/news/historical_news_2021_2025.csv
  data/raw/news/historical_news_fnspid_full.csv
)
[[ -f data/raw/news/historical_news_2026.csv ]] && INPUTS+=(data/raw/news/historical_news_2026.csv)
[[ -f data/raw/news/historical_news_forexlive.csv ]] && INPUTS+=(data/raw/news/historical_news_forexlive.csv)

"$PY" scripts/merge_massive_datasets.py \
  --input "${INPUTS[@]}" \
  --output data/raw/news/historical_news_combined.parquet \
  --pairs EURUSD USDJPY GBPUSD \
  --start-year 2008 \
  --end-year 2026

echo "=== News rebuild complete ==="
ls -lh \
  data/raw/news/historical_news_combined.parquet \
  data/raw/eco_calendar/events.csv \
  data/raw/news/historical_news_forexlive.csv \
  data/raw/news/Hugging_Face.csv \
  data/raw/news/historical_news_2026.csv 2>/dev/null || true
