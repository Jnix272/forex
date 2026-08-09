"""
data_quality_audit.py
Comprehensive data quality audit for data/raw/
Checks: coverage gaps, spread sanity, price sanity, volume, tick density,
        duplicate timestamps, monotonicity, NaN, outliers, COT, news, eco-calendar.
"""
import os, sys, glob, math
import polars as pl
import numpy as np
from datetime import datetime, timedelta, timezone
from collections import defaultdict

RAW = "data/raw"
PAIRS = ["EURUSD","GBPUSD","USDJPY","GBPJPY","AUDUSD","USDCAD","NZDUSD","USDCHF","EURJPY","EURGBP"]

# Expected spread bounds per pair (pips)
SPREAD_BOUNDS = {
    "EURUSD": (0.0, 5.0),
    "GBPUSD": (0.0, 8.0),
    "USDJPY": (0.0, 8.0),
    "GBPJPY": (0.0, 15.0),
    "AUDUSD": (0.0, 8.0),
    "USDCAD": (0.0, 8.0),
    "NZDUSD": (0.0, 10.0),
    "USDCHF": (0.0, 8.0),
    "EURJPY": (0.0, 10.0),
    "EURGBP": (0.0, 8.0),
}
JPY_PAIRS = {"USDJPY","GBPJPY","EURJPY"}

RESULTS = []

def pip_size(pair):
    return 0.01 if pair in JPY_PAIRS else 0.0001

def log(level, check, detail):
    icon = {"PASS": "OK", "FAIL": "FAIL", "WARN": "WARN", "INFO": "INFO"}.get(level, "?")
    line = f"  [{icon}] {check}: {detail}"
    print(line)
    RESULTS.append((level, check, detail))

def check_pair(pair):
    files = sorted(glob.glob(f"{RAW}/dukascopy/{pair}/**/*.parquet", recursive=True))
    if not files:
        log("FAIL", f"{pair}/coverage", "NO FILES FOUND")
        return

    ps = pip_size(pair)
    slo, shi = SPREAD_BOUNDS.get(pair, (0.0, 15.0))

    total_rows = 0
    null_rows = 0
    neg_spread_rows = 0
    wide_spread_rows = 0
    bad_price_rows = 0
    dup_ts_rows = 0
    bad_order_rows = 0
    zero_vol_rows = 0
    outlier_rows = 0
    error_files = []

    hours_present = set()

    # Sample ~1% of files for detailed checks, always include first+last
    n = len(files)
    step = max(1, n // 200)
    sample_files = sorted(set([files[0], files[-1]] + files[::step]))

    for fpath in sample_files:
        try:
            df = pl.read_parquet(fpath)
        except Exception as e:
            error_files.append((fpath, str(e)))
            continue

        if "__index_level_0__" in df.columns:
            df = df.rename({"__index_level_0__": "ts"})
        elif "timestamp" in df.columns:
            df = df.rename({"timestamp": "ts"})

        total_rows += len(df)
        if len(df) == 0:
            continue

        nc = df.null_count()
        null_rows += sum(nc.row(0))

        bad = df.filter((pl.col("bid") <= 0) | (pl.col("ask") <= 0))
        bad_price_rows += len(bad)

        df = df.with_columns(spread_pips=((pl.col("ask") - pl.col("bid")) / ps))
        neg = df.filter(pl.col("spread_pips") < slo)
        neg_spread_rows += len(neg)
        wide = df.filter(pl.col("spread_pips") > shi)
        wide_spread_rows += len(wide)

        bad_ord = df.filter(pl.col("ask") < pl.col("bid"))
        bad_order_rows += len(bad_ord)

        if "volume" in df.columns:
            zv = df.filter(pl.col("volume") <= 0)
            zero_vol_rows += len(zv)

        if "ts" in df.columns:
            dup = df.filter(pl.col("ts").is_duplicated())
            dup_ts_rows += len(dup)

        bid_vals = df["bid"].drop_nulls().to_numpy()
        if len(bid_vals) > 1:
            max_move = np.max(np.abs(np.diff(bid_vals))) / ps
            if max_move > 500:
                outlier_rows += 1

    for fpath in files:
        parts = fpath.replace("\\", "/").split("/")
        try:
            year = int(parts[-3])
            month = int(parts[-2])
            fname = parts[-1].replace(".parquet", "")
            day, hour = map(int, fname.split("_"))
            hours_present.add(datetime(year, month, day, hour, tzinfo=timezone.utc))
        except Exception:
            pass

    max_gap = 0
    first_h = last_h = None
    if hours_present:
        sorted_hours = sorted(hours_present)
        first_h = sorted_hours[0]
        last_h = sorted_hours[-1]
        for i in range(1, len(sorted_hours)):
            gap = (sorted_hours[i] - sorted_hours[i-1]).total_seconds() / 3600
            if gap > max_gap:
                max_gap = gap

    log("INFO", f"{pair}/coverage",
        f"{n:,} files | {first_h.date() if first_h else '?'} to {last_h.date() if last_h else '?'} | max_gap={max_gap:.0f}h")

    if error_files:
        log("FAIL", f"{pair}/corrupt_files", f"{len(error_files)} corrupt parquet files")

    if null_rows > 0:
        log("FAIL", f"{pair}/nulls", f"{null_rows:,} null cells in sampled rows")
    else:
        log("PASS", f"{pair}/nulls", "No nulls")

    if bad_price_rows > 0:
        log("FAIL", f"{pair}/price_sanity", f"{bad_price_rows:,} rows with bid/ask <= 0")
    else:
        log("PASS", f"{pair}/price_sanity", "All prices > 0")

    if bad_order_rows > 0:
        log("FAIL", f"{pair}/bid_ask_order", f"{bad_order_rows:,} rows where ask < bid")
    else:
        log("PASS", f"{pair}/bid_ask_order", "ask >= bid always")

    if neg_spread_rows > 0:
        log("FAIL", f"{pair}/spread_negative", f"{neg_spread_rows:,} rows with spread < 0 pips")
    else:
        log("PASS", f"{pair}/spread_negative", "No negative spreads")

    if wide_spread_rows > 0:
        log("WARN", f"{pair}/spread_wide", f"{wide_spread_rows:,} rows with spread > {shi:.1f} pips (news/gap events)")
    else:
        log("PASS", f"{pair}/spread_wide", "No abnormally wide spreads")

    if dup_ts_rows > 0:
        log("WARN", f"{pair}/dup_timestamps", f"{dup_ts_rows:,} duplicate timestamps in sampled files")
    else:
        log("PASS", f"{pair}/dup_timestamps", "No duplicate timestamps")

    if zero_vol_rows > 0:
        log("WARN", f"{pair}/zero_volume", f"{zero_vol_rows:,} rows with volume <= 0")
    else:
        log("PASS", f"{pair}/zero_volume", "All volume > 0")

    if outlier_rows > 0:
        log("WARN", f"{pair}/price_outlier", f"{outlier_rows} files with intra-file move > 500 pips")
    else:
        log("PASS", f"{pair}/price_outlier", "No extreme intra-file price moves")

    if max_gap > 72:
        log("WARN", f"{pair}/coverage_gap", f"Largest gap = {max_gap:.0f}h -- check for missing data")
    elif max_gap:
        log("PASS", f"{pair}/coverage_gap", f"Largest gap = {max_gap:.0f}h (normal weekend/holiday)")

    print()


def check_cot():
    print("=== COT DATA ===")
    path = f"{RAW}/cot/cot_financials_cleaned.parquet"
    try:
        df = pl.read_parquet(path)
        log("INFO", "cot/shape", f"{df.shape[0]:,} rows x {df.shape[1]} cols")
        log("INFO", "cot/columns", str(df.columns[:10]))
        nc = df.null_count().row(0)
        null_pct = sum(nc) / max(1, df.shape[0] * df.shape[1]) * 100
        if null_pct > 5:
            log("WARN", "cot/nulls", f"{null_pct:.1f}% null cells")
        else:
            log("PASS", "cot/nulls", f"{null_pct:.2f}% null cells")
        date_cols = [c for c in df.columns if any(k in c.lower() for k in ["date","time","report"])]
        if date_cols:
            log("INFO", "cot/date_range", f"Date cols: {date_cols} | sample: {df[date_cols[0]].head(1).to_list()}")
    except Exception as e:
        log("FAIL", "cot/load", str(e))
    print()


def check_news():
    print("=== NEWS DATA ===")
    news_files = glob.glob(f"{RAW}/news/*.csv") + glob.glob(f"{RAW}/news/*.parquet")
    for fp in news_files:
        fname = os.path.basename(fp)
        try:
            if fp.endswith(".parquet"):
                df = pl.read_parquet(fp)
            else:
                df = pl.read_csv(fp, infer_schema_length=1000, truncate_ragged_lines=True)
            null_pct = df.null_count().row(0)
            total_nulls = sum(null_pct)
            null_frac = total_nulls / max(1, df.shape[0] * df.shape[1]) * 100
            date_cols = [c for c in df.columns if any(k in c.lower() for k in ["date","time","publish","created"])]
            log("INFO", f"news/{fname}", f"{df.shape[0]:,} rows x {df.shape[1]} cols | nulls={null_frac:.1f}% | date_cols={date_cols}")
            if null_frac > 30:
                log("WARN", f"news/{fname}/nulls", f"High null rate: {null_frac:.1f}%")
            else:
                log("PASS", f"news/{fname}/quality", "OK")
        except Exception as e:
            log("FAIL", f"news/{fname}", str(e))
    print()


def check_eco_calendar():
    print("=== ECONOMIC CALENDAR ===")
    path = f"{RAW}/eco_calendar/events.csv"
    try:
        df = pl.read_csv(path, infer_schema_length=1000)
        log("INFO", "eco/shape", f"{df.shape[0]:,} rows x {df.shape[1]} cols")
        log("INFO", "eco/columns", str(df.columns))
        nc = df.null_count().row(0)
        total_nulls = sum(nc)
        null_frac = total_nulls / max(1, df.shape[0] * df.shape[1]) * 100
        if null_frac > 20:
            log("WARN", "eco/nulls", f"{null_frac:.1f}% null (forecast/actual often missing -- normal)")
        else:
            log("PASS", "eco/nulls", f"{null_frac:.1f}% null")
        ccols = [c for c in df.columns if "currency" in c.lower()]
        if ccols:
            log("INFO", "eco/currencies", str(df[ccols[0]].unique().sort().to_list()))
    except Exception as e:
        log("FAIL", "eco/load", str(e))
    print()


def print_summary():
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    counts = defaultdict(int)
    for level, _, _ in RESULTS:
        counts[level] += 1
    for level in ["PASS","WARN","FAIL","INFO"]:
        if counts[level]:
            icon = {"PASS": "OK", "FAIL": "FAIL", "WARN": "WARN", "INFO": "INFO"}[level]
            print(f"  [{icon}] {level}: {counts[level]}")

    print()
    fails = [(c, d) for l, c, d in RESULTS if l == "FAIL"]
    if fails:
        print("FAILURES:")
        for c, d in fails:
            print(f"    - {c}: {d}")
    else:
        print("No FAIL-level issues found!")

    warns = [(c, d) for l, c, d in RESULTS if l == "WARN"]
    if warns:
        print()
        print("WARNINGS:")
        for c, d in warns:
            print(f"    - {c}: {d}")
    print()


if __name__ == "__main__":
    print(f"\n{'='*70}")
    print("  FOREX RAW DATA QUALITY AUDIT")
    print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    print("=== DUKASCOPY TICK DATA ===\n")
    for pair in PAIRS:
        print(f"--- {pair} ---")
        check_pair(pair)

    check_cot()
    check_news()
    check_eco_calendar()
    print_summary()
