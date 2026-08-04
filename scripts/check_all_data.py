import polars as pl
from pathlib import Path
import os

print("\n" + "="*50)
print("  DATASET AUDIT ")
print("="*50)

# 1. Historical News
news_file = Path("data/raw/news/historical_news_combined.parquet")
if news_file.exists():
    df = pl.read_parquet(news_file)
    print(f"\n[NEWS] {news_file.name}")
    print(f"  Rows: {len(df):,}")
    if "timestamp_utc" in df.columns:
        print(f"  Range: {df['timestamp_utc'].min()} -> {df['timestamp_utc'].max()}")
else:
    print("\n[NEWS] Not found")

# 2. Economic Calendar
eco_file = Path("data/raw/eco_calendar/events.csv")
if eco_file.exists():
    df = pl.read_csv(eco_file, ignore_errors=True)
    print(f"\n[ECO CALENDAR] {eco_file.name}")
    print(f"  Rows: {len(df):,}")
    if "timestamp_utc" in df.columns:
        print(f"  Range: {df['timestamp_utc'].min()} -> {df['timestamp_utc'].max()}")
else:
    print("\n[ECO CALENDAR] Not found")

# 3. COT Data
cot_file = Path("data/raw/cot/cot_financials_cleaned.parquet")
if cot_file.exists():
    df = pl.read_parquet(cot_file)
    print(f"\n[COT] {cot_file.name}")
    print(f"  Rows: {len(df):,}")
    if "date" in df.columns:
        print(f"  Range: {df['date'].min()} -> {df['date'].max()}")
else:
    print("\n[COT] Not found")

# 4. Cross Asset
ca_dir = Path("data/raw/cross_asset")
if ca_dir.exists():
    ca_files = list(ca_dir.glob("*.parquet"))
    print(f"\n[CROSS ASSET] {len(ca_files)} symbols")
    for f in sorted(ca_files)[:3]: # just sample a few
        df = pl.read_parquet(f)
        print(f"  {f.stem}: {len(df):,} rows ({df['timestamp_utc'].min()} -> {df['timestamp_utc'].max()})")
    if len(ca_files) > 3:
        print("  ...")
else:
    print("\n[CROSS ASSET] Not found")

# 5. OANDA
oanda = Path("data/raw/oanda_sentiment.csv")
if oanda.exists():
    print(f"\n[OANDA] {oanda.name} exists")
else:
    print("\n[OANDA] Not found")

# 6. Dukascopy (just count files per year)
print("\n[DUKASCOPY TICKS]")
duk_dir = Path("data/raw/dukascopy")
if duk_dir.exists():
    for pair in sorted([d.name for d in duk_dir.iterdir() if d.is_dir()]):
        years = sorted([d.name for d in (duk_dir / pair).iterdir() if d.is_dir()])
        print(f"  {pair}: {len(years)} years ({years[0] if years else ''} -> {years[-1] if years else ''})")
else:
    print("  Not found")

print("\n" + "="*50)
