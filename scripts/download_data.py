"""
scripts/download_data.py
========================
Standalone bulk downloader / ingester for:
  1) Dukascopy tick data     (data/raw/dukascopy/<PAIR>/...)
  2) Cross-asset panel       (data/processed/cross_asset/...)
  3) Myfxbook daily OHLCV    (data/raw/myfxbook/<PAIR>/...)
  4) EODHD daily OHLCV       (data/raw/eodhd/<PAIR>/...)
  5) EODHD cross-asset       (data/raw/eodhd/cross_asset/...)

Defaults mirror config/run.yaml (EURUSD/GBPUSD/USDJPY/AUDUSD, 2018-2025).

Usage examples
--------------
  # Full run -- all 4 pairs + cross-asset, session hours only (07-17 UTC)
  python scripts/download_data.py

  # All 24 hours per day (2x more data, ~2x longer)
  python scripts/download_data.py --full-day

  # Custom pair / date window (use the specific --check-missing-months command to auto-repair gaps!)
  python scripts/download_data.py --pairs EURUSD USDCAD --start 2022-01-01 --end 2023-12-31 --check-missing-months

  # Skip cross-asset (tick data only)
  python scripts/download_data.py --no-cross-asset

  # Cross-asset only (no tick re-download)
  python scripts/download_data.py --no-ticks

  # Ingest a Myfxbook CSV export (copies it into data/raw/myfxbook/<PAIR>/)
  python scripts/download_data.py --ingest-myfxbook C:/Users/you/Downloads/EURGBP_historical_data.csv --myfxbook-pair EURGBP

  # Ingest + verify (shows parsed row count and date range)
  python scripts/download_data.py --ingest-myfxbook path/to/file.csv --myfxbook-pair EURGBP --verify-myfxbook

  # Download EODHD daily forex for all pairs + cross-asset (set EODHD_API_KEY first)
  python scripts/download_data.py --eodhd --no-ticks --no-cross-asset

  # EODHD for specific pairs only
  python scripts/download_data.py --eodhd --eodhd-pairs EURUSD GBPUSD EURGBP --no-ticks --no-cross-asset

  # EODHD cross-asset panel only
  python scripts/download_data.py --eodhd-cross-asset --no-ticks --no-cross-asset

  # Tune politeness vs speed (defaults are conservative for Dukascopy)
  python scripts/download_data.py --concurrency 20 --max-pair-parallelism 3
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import yaml

# Suppress Python 3.14 DeprecationWarnings from aiohttp/asyncio
warnings.filterwarnings("ignore", category=DeprecationWarning, module="aiohttp")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="asyncio")

try:
    import wandb
except ImportError:
    wandb = None

# ── make sure project root is on sys.path ─────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from data.cross_asset import load_cross_asset_panel
from data.eodhd import DEFAULT_EODHD_CACHE_DIR, EODHD_FOREX_PAIRS, EODHDLoader
from data.myfxbook import DEFAULT_MYFXBOOK_DATA_DIR, MyfxbookLoader
from data.sources import (
    DEFAULT_DUKASCOPY_CACHE_DIR,
    DEFAULT_DUKASCOPY_COMPACT_DIR,
    DukascopyLoader,
    ForexDataManager,
)

# ── Load run config for defaults ───────────────────────────────────────────────
_yaml_config = {}
for _config_name in ("run.yaml", "run_ubuntu.yaml"):
    _config_path = _ROOT / "config" / _config_name
    if not _config_path.exists():
        continue
    try:
        with open(_config_path, encoding="utf-8") as _f:
            _yaml_config = yaml.safe_load(_f) or {}
        break
    except Exception:
        pass

_d_cfg = _yaml_config.get("download", {})
_data_cfg = _yaml_config.get("data", {})
_training_cfg = _yaml_config.get("training", {})

DEF_PAIRS = _data_cfg.get("pairs", ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "EURGBP", "NZDUSD", "EURJPY", "GBPJPY"])
DEF_START = str(_data_cfg.get("start", "2018-01-01"))
DEF_END   = str(_data_cfg.get("end", "2025-12-31"))
DEF_FULL_DAY = bool(_data_cfg.get("full_day_data", False))

DEFAULT_PAIRS       = DEF_PAIRS
DEFAULT_START       = DEF_START
DEFAULT_END         = DEF_END
SESSION_HOURS       = list(range(7, 18))   # 07-17 UTC  (London + NY open)
FULL_DAY_HOURS      = list(range(0, 24))   # 00-23 UTC
CROSS_ASSET_CACHE   = str(_ROOT / "data" / "processed" / "cross_asset")
CROSS_ASSET_SOURCE  = (
    os.getenv("CROSS_ASSET_SOURCE", "").strip()
    or str(_training_cfg.get("cross_asset_provider") or _yaml_config.get("cross_asset_source") or "auto").strip()
).lower()

# Download module flags from config
DEF_EODHD = _d_cfg.get("eodhd", False)
DEF_EODHD_CA = _d_cfg.get("eodhd_cross_asset", False)
DEF_TICKS = _d_cfg.get("ticks", True)
DEF_CA = _d_cfg.get("cross_asset", True)
DEF_YEARLY = _d_cfg.get("yearly", False)
DEF_KEEP_GOING = _d_cfg.get("keep_going", True)
DEF_VERIFY = _d_cfg.get("verify", False)
DEF_VERIFY_FIX = _d_cfg.get("verify_fix", False)
DEF_CHECK_MISSING_MONTHS = _d_cfg.get("check_missing_months", False)
DEF_CHECK_MISSING_SCOPE = _d_cfg.get("check_missing_scope", "both")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_rows(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def _bar_size_mb(cache_dir: str, pair: str) -> float:
    """Total MB of cached Parquet files for a pair."""
    p = Path(cache_dir) / pair
    if not p.exists():
        return 0.0
    return sum(f.stat().st_size for f in p.rglob("*.parquet")) / 1e6


def _cross_asset_cache_mb(cache_dir: str) -> float:
    p = Path(cache_dir)
    if not p.exists():
        return 0.0
    return sum(f.stat().st_size for f in p.rglob("*.csv")) / 1e6


# ─────────────────────────────────────────────────────────────────────────────
# Tick downloader
# ─────────────────────────────────────────────────────────────────────────────

def download_ticks(
    pairs:                list[str],
    start:                str,
    end:                  str,
    hours:                list[int],
    concurrency:        int,
    cache_dir:            str,
    max_parallel_pairs: int = 1,
    request_delay:      float = 0.05,
) -> None:
    print("\n" + "=" * 66)
    print("  TICK DATA  --  Dukascopy")
    print(f"  Pairs      : {', '.join(pairs)}")
    print(f"  Range      : {start}  ->  {end}")
    print(f"  Hours/day  : {len(hours)}  ({hours[0]:02d}:00 - {hours[-1]:02d}:59 UTC)")
    print(f"  Concurrency (per pair): {concurrency}")
    print(f"  Pair parallelism      : {max_parallel_pairs}  (1 = one symbol at a time)")
    print(f"  Request delay (s)     : {request_delay}")
    print(f"  Cache      : {cache_dir}")
    print("=" * 66)

    loader = DukascopyLoader(
        cache_dir            = cache_dir,
        concurrency          = concurrency,
        max_parallel_pairs   = max_parallel_pairs,
        request_delay        = request_delay,
        verbose              = True,
    )

    t0 = time.perf_counter()
    results = loader.load_multiple(pairs, start=start, end=end, hours=hours)
    elapsed = time.perf_counter() - t0

    print("\n" + "-" * 66)
    print(f"  {'Pair':<10} {'Rows':>10}  {'Cache (MB)':>12}")
    print("  " + "-" * 36)
    total_rows = 0
    for pair in pairs:
        df    = results.get(pair)
        rows  = len(df) if df is not None else 0
        mb    = _bar_size_mb(cache_dir, pair)
        total_rows += rows
        print(f"  {pair:<10} {_fmt_rows(rows):>10}  {mb:>10.1f} MB")
    print("  " + "-" * 36)
    total_mb = sum(_bar_size_mb(cache_dir, p) for p in pairs)
    print(f"  {'TOTAL':<10} {_fmt_rows(total_rows):>10}  {total_mb:>10.1f} MB")
    print(f"\n  Done in {elapsed:.1f}s  ({total_rows/max(elapsed,1):.0f} ticks/s)")
    print("-" * 66)


def download_ticks_yearly(
    pairs: list[str],
    start: str,
    end: str,
    full_day: bool,
    concurrency: int,
    cache_dir: str,
    compact_dir: str,
    request_delay: float,
    redownload_passes: int,
    keep_going: bool,
    auto_compact: bool,
    compact_granularity: str,
    auto_duckdb: bool,
) -> None:
    print("\n" + "=" * 66)
    print("  TICK DATA  --  Dukascopy Year-by-Year")
    print(f"  Pairs      : {', '.join(pairs)}")
    print(f"  Range      : {start}  ->  {end}")
    print(f"  Hours/day  : {'24 (00:00 - 23:59 UTC)' if full_day else '11 (07:00 - 17:59 UTC)'}")
    print(f"  Concurrency: {concurrency}")
    print(f"  Delay      : {request_delay}")
    print(f"  Retries    : {redownload_passes}")
    print(f"  Keep going : {keep_going}")
    print(f"  Cache      : {cache_dir}")
    print(f"  Compact    : {compact_dir}")
    print(f"  Auto build : compact={auto_compact} duckdb={auto_duckdb} ({compact_granularity})")
    print("=" * 66)

    start_year = int(start[:4])
    end_year = int(end[:4])

    manager = ForexDataManager(
        dukascopy_dir=cache_dir,
        dukascopy_compact_dir=compact_dir,
        verbose=True,
    )
    manager.duka.concurrency = max(1, concurrency)
    manager.duka.delay = max(0.0, request_delay)

    summary = manager.download_dukascopy_year_by_year(
        pairs=pairs,
        start_year=start_year,
        end_year=end_year,
        session_only=not full_day,
        max_redownload_passes=max(0, redownload_passes),
        fail_on_missing=not keep_going,
    )

    print("\n" + "-" * 66)
    print(f"  {'Pair':<10} {'Year':<6} {'Ticks':>12}  {'Coverage':>12}  {'Missing':>8}")
    print("  " + "-" * 52)
    for pair in pairs:
        for year in sorted(summary.get(pair, {})):
            item = summary[pair][year]
            coverage = item["coverage"]
            cov = f"{coverage['present_hours_count']}/{coverage['requested_hours_count']}"
            print(f"  {pair:<10} {year:<6} {item['ticks']:>12,}  {cov:>12}  {coverage['missing_hours_count']:>8}")
    print("-" * 66)

    if auto_compact:
        print("\n" + "-" * 66)
        print("  Building compacted Parquet store")
        compact_summary = manager.compact_dukascopy_cache(pairs, granularity=compact_granularity)
        for pair in pairs:
            item = compact_summary.get(pair, {})
            print(
                f"  {pair:<10} partitions={item.get('partitions_written', 0):>4} "
                f"ticks={item.get('ticks_written', 0):>12,}"
            )
        if auto_duckdb:
            db_path = manager.build_dukascopy_duckdb(granularity=compact_granularity)
            print(f"\n  DuckDB view ready at: {db_path}")
        print("-" * 66)


# ---------------------------------------------------------------------------
# Cross-asset downloader
# ---------------------------------------------------------------------------

def download_cross_asset(
    start:     str,
    end:       str,
    cache_dir: str,
    source:    str,
) -> None:
    print("\n" + "=" * 66)
    print("  CROSS-ASSET PANEL")
    print(f"  Range  : {start}  ->  {end}")
    print(f"  Source : {source}  (auto = Stooq -> Yahoo -> FRED)")
    print(f"  Cache  : {cache_dir}")
    print("=" * 66)

    t0    = time.perf_counter()
    panel = load_cross_asset_panel(
        start     = start,
        end       = end,
        cache_dir = cache_dir,
        source    = source,
    )
    elapsed = time.perf_counter() - t0

    print("\n" + "-" * 66)
    print(f"  {'Asset':<22} {'Points':>8}  {'From':>12}  {'To':>12}")
    print("  " + "-" * 58)
    for asset, ser in sorted(panel.items()):
        if ser is None or ser.empty:
            print(f"  {asset:<22} {'--':>8}  {'--':>12}  {'--':>12}")
        else:
            d0 = str(ser.index.min().date())
            d1 = str(ser.index.max().date())
            print(f"  {asset:<22} {len(ser):>8,}  {d0:>12}  {d1:>12}")
    print("  " + "-" * 58)
    cache_mb = _cross_asset_cache_mb(cache_dir)
    print(f"  {len(panel)} assets  |  {cache_mb:.1f} MB on disk  |  {elapsed:.1f}s")
    print("-" * 66)

    if not panel:
        print("\n  [WARN] No cross-asset data was loaded.")
        print("         Check your internet connection or set FRED_API_KEY for yields.")


# ─────────────────────────────────────────────────────────────────────────────
# EODHD downloaders
# ─────────────────────────────────────────────────────────────────────────────

def download_eodhd_forex(
    pairs:     list[str],
    start:     str,
    end:       str,
    cache_dir: str,
    api_key:   str = "",
) -> None:
    print("\n" + "=" * 66)
    print("  EODHD FOREX  --  Daily OHLCV")
    print(f"  Pairs  : {', '.join(pairs)}")
    print(f"  Range  : {start}  ->  {end}")
    print(f"  Cache  : {cache_dir}")
    print("=" * 66)

    loader = EODHDLoader(api_key=api_key, cache_dir=cache_dir, verbose=False)
    if not loader.api_key:
        print("\n  [ERROR] EODHD_API_KEY not set. Export it before running:")
        print("          Windows: set EODHD_API_KEY=your_key")
        print("          Linux:   export EODHD_API_KEY=your_key")
        return

    t0 = time.perf_counter()
    print("\n" + "-" * 66)
    print(f"  {'Pair':<10} {'Rows':>8}  {'From':>12}  {'To':>12}  {'MB':>6}")
    print("  " + "-" * 52)
    total_rows = 0
    for pair in pairs:
        df  = loader.load(pair, start=start, end=end, use_cache=False)
        rows = len(df)
        total_rows += rows
        mb   = _bar_size_mb(cache_dir, pair)
        if rows:
            d0, d1 = str(df.index.min().date()), str(df.index.max().date())
        else:
            d0, d1 = "--", "--"
        print(f"  {pair:<10} {_fmt_rows(rows):>8}  {d0:>12}  {d1:>12}  {mb:>5.1f}")
    elapsed = time.perf_counter() - t0
    print("  " + "-" * 52)
    total_mb = sum(_bar_size_mb(cache_dir, p) for p in pairs)
    print(f"  {'TOTAL':<10} {_fmt_rows(total_rows):>8}  {'':>12}  {'':>12}  {total_mb:>5.1f}")
    print(f"\n  Done in {elapsed:.1f}s")
    print("-" * 66)


def download_eodhd_cross_asset(
    start:     str,
    end:       str,
    cache_dir: str,
    api_key:   str = "",
) -> None:
    print("\n" + "=" * 66)
    print("  EODHD CROSS-ASSET PANEL")
    print(f"  Range  : {start}  ->  {end}")
    print(f"  Cache  : {cache_dir}")
    print("=" * 66)

    loader = EODHDLoader(api_key=api_key, cache_dir=cache_dir, verbose=False)
    if not loader.api_key:
        print("\n  [ERROR] EODHD_API_KEY not set.")
        return

    t0    = time.perf_counter()
    panel = loader.load_cross_asset(start=start, end=end, use_cache=False)
    elapsed = time.perf_counter() - t0

    print("\n" + "-" * 66)
    print(f"  {'Asset':<22} {'Points':>8}  {'From':>12}  {'To':>12}")
    print("  " + "-" * 58)
    for asset, ser in sorted(panel.items()):
        if ser is None or ser.empty:
            print(f"  {asset:<22} {'--':>8}  {'--':>12}  {'--':>12}")
        else:
            d0 = str(ser.index.min().date())
            d1 = str(ser.index.max().date())
            print(f"  {asset:<22} {len(ser):>8,}  {d0:>12}  {d1:>12}")
    print("  " + "-" * 58)
    print(f"  {len(panel)} assets loaded  |  {elapsed:.1f}s")
    print("-" * 66)

    if not panel:
        print("\n  [WARN] No EODHD cross-asset data loaded. Check your API key and plan.")


# ─────────────────────────────────────────────────────────────────────────────
# Myfxbook ingester
# ─────────────────────────────────────────────────────────────────────────────

def ingest_myfxbook(
    filepath: str,
    pair:     str,
    verify:   bool,
    data_dir: str,
) -> None:
    print("\n" + "=" * 66)
    print("  MYFXBOOK INGEST")
    print(f"  File : {filepath}")
    print(f"  Pair : {pair}")
    print(f"  Dest : {data_dir}")
    print("=" * 66)

    loader = MyfxbookLoader(data_dir=data_dir, verbose=True)
    dest   = loader.ingest_file(filepath, pair)

    if verify:
        print("\n  Verifying ingested file ...")
        df = loader.load_file(str(dest), pair)
        if df.empty:
            print("  [WARN] Parsed 0 rows - check CSV format.")
        else:
            print(f"  Rows      : {len(df):,}")
            print(f"  Date range: {df.index[0].date()}  ->  {df.index[-1].date()}")
            print(f"  Columns   : {list(df.columns)}")
            print(f"  Bid range : {df['bid'].min():.5f}  ->  {df['bid'].max():.5f}")

    print("-" * 66)


# ─────────────────────────────────────────────────────────────────────────────
# Auto Check and Repair Missing Data Grouped by Month
# ─────────────────────────────────────────────────────────────────────────────

def _group_missing_datetimes(
    suspicious: list,
    scope: str,
) -> dict[str, int]:
    if scope == "pairs":
        return {"ALL": len(suspicious)}
    if scope == "years":
        grouped = {}
        for dt in suspicious:
            key = dt.strftime("%Y")
            grouped[key] = grouped.get(key, 0) + 1
        return grouped
    if scope == "both":
        grouped = {}
        for dt in suspicious:
            key = dt.strftime("%Y")
            grouped[key] = grouped.get(key, 0) + 1
        return grouped

    grouped = {}
    for dt in suspicious:
        key = dt.strftime("%Y-%m")
        grouped[key] = grouped.get(key, 0) + 1
    return grouped


def auto_redownload_missing_data(
    pairs: list[str],
    start: str,
    end: str,
    hours: list[int],
    cache_dir: str,
    concurrency: int,
    request_delay: float,
    scope: str = "both",
) -> None:
    from pathlib import Path

    from scripts.verify_data import _expected_hours, _filter_suspicious_missing, redownload_hours, scan_missing

    scope = (scope or "both").lower()
    if scope not in {"months", "years", "pairs", "both"}:
        raise ValueError("scope must be one of: months, years, pairs, both")

    cache = Path(cache_dir)
    expected = _expected_hours(start, end, hours)

    print("\n" + "=" * 66)
    print(f"  AUTO-CHECK: Missing Data by {scope.capitalize()}")
    print("=" * 66)

    total_missing_pairs = 0
    all_suspicious_to_fix = {}

    for pair in pairs:
        miss_stats = scan_missing(cache, pair, expected, min_ticks=0)
        n_missing = len(miss_stats["missing"])

        if n_missing > 0:
            suspicious = _filter_suspicious_missing(cache, pair, miss_stats["missing"])
            if suspicious:
                all_suspicious_to_fix[pair] = suspicious
                total_missing_pairs += 1

                grouped = _group_missing_datetimes(suspicious, scope)
                if scope == "pairs":
                    print(f"\n  [{pair}] Missing {len(suspicious)} suspicious hour(s)")
                elif scope == "years":
                    print(f"\n  [{pair}] Missing {len(suspicious)} hours across {len(grouped)} year(s):")
                    print("  " + "-" * 28)
                    print(f"  {'Year':<10} | {'Missing Hours':<13}")
                    print("  " + "-" * 28)
                    for key in sorted(grouped):
                        print(f"  {key:<10} | {grouped[key]:<13}")
                    print("  " + "-" * 28)
                elif scope == "both":
                    print(f"\n  [{pair}] Missing {len(suspicious)} hours across {len(grouped)} year bucket(s):")
                    print("  " + "-" * 28)
                    print(f"  {'Year':<10} | {'Missing Hours':<13}")
                    print("  " + "-" * 28)
                    for key in sorted(grouped):
                        print(f"  {key:<10} | {grouped[key]:<13}")
                    print("  " + "-" * 28)
                else:
                    print(f"\n  [{pair}] Missing {len(suspicious)} hours across {len(grouped)} month(s):")
                    print("  " + "-" * 32)
                    print(f"  {'Month':<12} | {'Missing Hours':<15}")
                    print("  " + "-" * 32)
                    for key in sorted(grouped):
                        print(f"  {key:<12} | {grouped[key]:<15}")
                    print("  " + "-" * 32)

    if total_missing_pairs == 0:
        print("\n  All pairs fully downloaded. No missing data detected.")
        return

    print("\n  Auto-redownloading missing data for affected pairs...")
    for pair, suspicious in all_suspicious_to_fix.items():
        rd_stats = redownload_hours(
            cache, pair, suspicious,
            concurrency=concurrency,
            request_delay=request_delay,
        )
        print(f"  [{pair}] Repaired | Re-fetched: {rd_stats['refetched']} | Still missing: {rd_stats['still_missing']}")
        if wandb and wandb.run:
            wandb.log({
                f"dukascopy/{pair}/missing_hours_repaired": rd_stats['refetched'],
                f"dukascopy/{pair}/missing_hours_unfixable": rd_stats['still_missing'],
            })

    print("-" * 66)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download Dukascopy tick data and cross-asset panel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pairs",     nargs="+", default=DEFAULT_PAIRS,
                   metavar="PAIR", help="FX pairs to download")
    p.add_argument("--start",     default=DEFAULT_START,  help="Start date YYYY-MM-DD")
    p.add_argument("--end",       default=DEFAULT_END,    help="End date   YYYY-MM-DD")
    p.add_argument("--full-day",  action="store_true", default=DEF_FULL_DAY,
                   help="Download all 24 hours/day (default: session only 07-17 UTC)")
    p.add_argument("--yearly", action="store_true", default=DEF_YEARLY,
                   help="Download Dukascopy ticks pair-by-pair and year-by-year with verification gates")
    p.add_argument("--concurrency", type=int, default=12,
                   help="Max concurrent hour-downloads per pair (lower = kinder to the feed)")
    p.add_argument("--max-pair-parallelism", type=int, default=1,
                   help="How many FX pairs to download at once (1 = safest; raise if stable)")
    p.add_argument("--request-delay", type=float, default=0.05,
                   help="Extra pause (seconds) before/after each HTTP get (0 = fastest)")
    p.add_argument("--redownload-passes", type=int, default=2,
                   help="Automatic missing-hour redownload passes for yearly mode")
    p.add_argument("--keep-going", action="store_true", default=DEF_KEEP_GOING,
                   help="In yearly mode, continue to later years and pairs after a failed year")
    p.add_argument("--tick-cache",  default=DEFAULT_DUKASCOPY_CACHE_DIR,
                   help="Root directory for raw Parquet tick cache")
    p.add_argument("--compact-cache", default=DEFAULT_DUKASCOPY_COMPACT_DIR,
                   help="Root directory for compacted Dukascopy Parquet partitions")
    p.add_argument("--compact-granularity", choices=["daily", "monthly"], default="daily",
                   help="Partition size for automatic compaction in yearly mode")
    p.add_argument("--no-auto-compact", action="store_true",
                   help="In yearly mode, skip automatic compaction after download")
    p.add_argument("--no-auto-duckdb", action="store_true",
                   help="In yearly mode, skip automatic DuckDB view build after compaction")
    p.add_argument("--cross-asset-cache", default=CROSS_ASSET_CACHE,
                   help="Directory for cross-asset CSV cache")
    p.add_argument("--cross-asset-source", default=CROSS_ASSET_SOURCE,
                   choices=["auto", "stooq", "yahoo", "fred", "eodhd"],
                   help="Cross-asset data provider")
    p.add_argument("--no-ticks",        action="store_true", default=not DEF_TICKS, help="Skip tick download")
    p.add_argument("--no-cross-asset",  action="store_true", default=not DEF_CA, help="Skip cross-asset download")
    # Myfxbook ingestion
    p.add_argument("--ingest-myfxbook", metavar="FILE",
                   help="Path to a Myfxbook CSV export to ingest")
    p.add_argument("--myfxbook-pair",   metavar="PAIR",
                   help="FX pair symbol for the Myfxbook file (e.g. EURGBP)")
    p.add_argument("--myfxbook-dir",    default=DEFAULT_MYFXBOOK_DATA_DIR,
                   help="Root directory for Myfxbook data")
    p.add_argument("--verify-myfxbook", action="store_true",
                   help="After ingesting, parse and print row count / date range")
    # EODHD
    _all_eodhd_pairs = sorted(EODHD_FOREX_PAIRS.keys())
    p.add_argument("--eodhd", action="store_true", default=DEF_EODHD,
                   help="Download EODHD daily forex for --eodhd-pairs (requires EODHD_API_KEY)")
    p.add_argument("--eodhd-pairs", nargs="+", default=_all_eodhd_pairs,
                   metavar="PAIR",
                   help=f"Pairs to download via EODHD (default: all {len(_all_eodhd_pairs)} supported pairs)")
    p.add_argument("--eodhd-cross-asset", action="store_true", default=DEF_EODHD_CA,
                   help="Download EODHD cross-asset panel (requires EODHD_API_KEY)")
    p.add_argument("--eodhd-cache", default=DEFAULT_EODHD_CACHE_DIR,
                   help="Root directory for EODHD cache")
    p.add_argument("--eodhd-api-key", default="",
                   help="EODHD API key (overrides EODHD_API_KEY env var)")
    p.add_argument("--no-eodhd", action="store_true",
                   help="Skip EODHD forex download (overrides config default)")
    p.add_argument("--no-eodhd-cross-asset", action="store_true",
                   help="Skip EODHD cross-asset download (overrides config default)")
    # Data verification
    p.add_argument("--verify", action="store_true", default=DEF_VERIFY,
                   help="After download, run data quality verification (duplicates, gaps, missing)")
    p.add_argument("--verify-fix", action="store_true", default=DEF_VERIFY_FIX,
                   help="Like --verify but also auto-repair (remove dupes, redownload missing)")
    p.add_argument("--check-missing-months", action="store_true", default=DEF_CHECK_MISSING_MONTHS,
                   help="After download, automatically check and redownload missing data grouped by month")
    p.add_argument("--check-missing-scope", choices=["months", "years", "pairs", "both"],
                   default=DEF_CHECK_MISSING_SCOPE,
                   help="How to summarize missing-data checks before automatic redownload")
    p.add_argument("--verify-min-ticks", type=int, default=0,
                   help="Flag files with fewer than N ticks as suspicious (0=off)")
    return p.parse_args()


def main() -> None:
    args  = _parse_args()
    if args.no_eodhd:
        args.eodhd = False
    if args.no_eodhd_cross_asset:
        args.eodhd_cross_asset = False
    hours = FULL_DAY_HOURS if args.full_day else SESSION_HOURS

    print("\n" + "=" * 66)
    print("  Forex Scaling Model -- Bulk Data Downloader")
    print("=" * 66)

    if wandb and os.getenv("WANDB_RUN_GROUP"):
        try:
            wandb.init(
                project=os.getenv("WANDB_PROJECT", "forex-scaling-model"),
                group=os.environ["WANDB_RUN_GROUP"],
                name="price_downloader",
                job_type="data-download-child",
            )
            print(f"[W&B] Data downloader attached to group {os.environ['WANDB_RUN_GROUP']}", flush=True)
        except Exception as e:
            print(f"[W&B] Failed to attach to run group: {e}", flush=True)

    # EODHD forex
    if args.eodhd:
        download_eodhd_forex(
            pairs     = [p.upper().replace("/", "") for p in args.eodhd_pairs],
            start     = args.start,
            end       = args.end,
            cache_dir = args.eodhd_cache,
            api_key   = args.eodhd_api_key,
        )

    # EODHD cross-asset
    if args.eodhd_cross_asset:
        download_eodhd_cross_asset(
            start     = args.start,
            end       = args.end,
            cache_dir = args.eodhd_cache,
            api_key   = args.eodhd_api_key,
        )

    # Myfxbook ingest (standalone - can run without tick/cross-asset)
    if args.ingest_myfxbook:
        if not args.myfxbook_pair:
            print("\n  [ERROR] --myfxbook-pair is required with --ingest-myfxbook")
            sys.exit(1)
        ingest_myfxbook(
            filepath = args.ingest_myfxbook,
            pair     = args.myfxbook_pair,
            verify   = args.verify_myfxbook,
            data_dir = args.myfxbook_dir,
        )
        # If ONLY ingesting (no ticks/cross-asset flags flipped), exit early
        if args.no_ticks and args.no_cross_asset:
            print("\n  All done.\n")
            return

    if not args.no_ticks:
        tick_pairs = [p.upper().replace("/", "") for p in args.pairs]
        if args.yearly:
            download_ticks_yearly(
                pairs              = tick_pairs,
                start              = args.start,
                end                = args.end,
                full_day           = args.full_day,
                concurrency        = args.concurrency,
                cache_dir          = args.tick_cache,
                compact_dir        = args.compact_cache,
                request_delay      = args.request_delay,
                redownload_passes  = args.redownload_passes,
                keep_going         = args.keep_going,
                auto_compact       = not args.no_auto_compact,
                compact_granularity= args.compact_granularity,
                auto_duckdb        = not args.no_auto_duckdb,
            )
        else:
            download_ticks(
                pairs                = tick_pairs,
                start                = args.start,
                end                  = args.end,
                hours                = hours,
                concurrency          = args.concurrency,
                cache_dir            = args.tick_cache,
                max_parallel_pairs   = args.max_pair_parallelism,
                request_delay        = args.request_delay,
            )

    if not args.no_cross_asset:
        download_cross_asset(
            start     = args.start,
            end       = args.end,
            cache_dir = args.cross_asset_cache,
            source    = args.cross_asset_source,
        )

    # Post-download verification
    if args.verify or args.verify_fix:
        from scripts.verify_data import run_verification
        run_verification(
            pairs         = [p.upper().replace("/", "") for p in args.pairs],
            start         = args.start,
            end           = args.end,
            hours         = hours,
            cache_dir     = args.tick_cache,
            fix           = args.verify_fix,
            min_ticks     = args.verify_min_ticks,
            concurrency   = args.concurrency,
            request_delay = args.request_delay,
        )
    elif args.check_missing_months:
        auto_redownload_missing_data(
            pairs         = [p.upper().replace("/", "") for p in args.pairs],
            start         = args.start,
            end           = args.end,
            hours         = hours,
            cache_dir     = args.tick_cache,
            concurrency   = args.concurrency,
            request_delay = args.request_delay,
            scope         = args.check_missing_scope,
        )

    print("\n  All done.\n")


if __name__ == "__main__":
    main()
