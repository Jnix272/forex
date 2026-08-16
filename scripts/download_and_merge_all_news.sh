#!/bin/bash
set -e

# echo "=== 1. Starting FNSPID Download (Sequential) ==="
# .venv/bin/python3.12 scripts/download_fnspid.py

echo "=== 2. Starting GDELT 2021-2025 Download (Sequential) ==="
.venv/bin/python3.12 scripts/download_gdelt2_bulk.py --start 2021-01-01 --end 2025-12-31 --out data/raw/news/historical_news_2021_2025.csv --workers 16

echo "=== 3. Merging Datasets ==="
.venv/bin/python3.12 scripts/merge_datasets.py

echo "=== 4. Deduplicating and Compiling Final Parquet ==="
.venv/bin/python3.12 scripts/merge_massive_datasets.py

echo "=== Pipeline Complete! ==="
