"""
scripts/download_yearly.py
==========================
CLI for pair-by-pair, year-by-year Dukascopy downloads with verification.

Each (pair, year) slice is downloaded, checked for missing requested hours, and
optionally retried before the script moves on to the next year or pair.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import yaml

from data.sources import (
    DEFAULT_DUKASCOPY_CACHE_DIR,
    DEFAULT_DUKASCOPY_COMPACT_DIR,
    ForexDataManager,
)

# ── Load config/run_ubuntu.yaml for defaults ──────────────────────────────────
_yaml_config = {}
_config_path = _ROOT / "config" / "run_ubuntu.yaml"
if _config_path.exists():
    try:
        with open(_config_path, encoding="utf-8") as _f:
            _yaml_config = yaml.safe_load(_f) or {}
    except Exception:
        pass

_d_cfg = _yaml_config.get("download", {})
_data_cfg = _yaml_config.get("data", {})

DEF_PAIRS = _data_cfg.get(
    "pairs", ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "EURGBP", "NZDUSD", "EURJPY", "GBPJPY"]
)
DEF_START = int(str(_data_cfg.get("start", "2018-01-01"))[:4])
DEF_END = int(str(_data_cfg.get("end", "2025-12-31"))[:4])
DEF_FULL_DAY = bool(_data_cfg.get("full_day_data", False))
DEF_KEEP_GOING = _d_cfg.get("keep_going", True)

DEFAULT_PAIRS = DEF_PAIRS


def _auto_finalize_storage(
    manager: ForexDataManager,
    pairs: list[str],
    *,
    granularity: str,
    build_duckdb: bool,
    as_table: bool = False,
) -> None:
    print("\n" + "=" * 72)
    print("  Post-Download Storage Build")
    print(f"  Compaction granularity : {granularity}")
    print(f"  Build DuckDB view      : {build_duckdb}")
    print(f"  Compact root           : {manager.duka_compact_dir}")
    print("=" * 72)

    compact_summary = manager.compact_dukascopy_cache(pairs, granularity=granularity)
    for pair in pairs:
        item = compact_summary.get(pair, {})
        print(
            f"  {pair:<10} partitions={item.get('partitions_written', 0):>4} ticks={item.get('ticks_written', 0):>12,}"
        )

    if build_duckdb:
        db_path = manager.build_dukascopy_duckdb(granularity=granularity, as_table=as_table)
        print(f"\n  DuckDB view ready at: {db_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Dukascopy data pair-by-pair and year-by-year with verification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS, metavar="PAIR", help="FX pairs to download")
    parser.add_argument("--start", type=int, default=DEF_START, metavar="YEAR", help="First year to download")
    parser.add_argument("--end", type=int, default=DEF_END, metavar="YEAR", help="Last year to download, inclusive")
    parser.add_argument(
        "--full-day",
        action="store_true",
        default=DEF_FULL_DAY,
        help="Download all 24 hours instead of session hours (07-17 UTC)",
    )
    parser.add_argument("--concurrency", type=int, default=12, help="Concurrent hour downloads per pair")
    parser.add_argument("--request-delay", type=float, default=0.05, help="Pause around each HTTP GET")
    parser.add_argument("--tick-cache", default=DEFAULT_DUKASCOPY_CACHE_DIR, help="Parquet cache root")
    parser.add_argument("--compact-cache", default=DEFAULT_DUKASCOPY_COMPACT_DIR, help="Compacted Parquet root")
    parser.add_argument(
        "--redownload-passes", type=int, default=2, help="How many automatic missing-hour redownload passes to allow"
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        default=DEF_KEEP_GOING,
        help="Continue to later years/pairs even if one year still has missing hours",
    )
    parser.add_argument(
        "--compact-granularity",
        choices=["daily", "monthly"],
        default="daily",
        help="Partition size for automatic compaction after download",
    )
    parser.add_argument("--no-auto-compact", action="store_true", help="Skip automatic compaction after download")
    parser.add_argument(
        "--no-auto-duckdb", action="store_true", help="Skip automatic DuckDB view build after compaction"
    )
    parser.add_argument("--as-table", action="store_true", help="Build DuckDB as a physical TABLE instead of a VIEW")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    pairs = [pair.upper().replace("/", "") for pair in args.pairs]

    print("\n" + "=" * 72)
    print("  Dukascopy Year-by-Year Downloader")
    print(f"  Pairs            : {', '.join(pairs)}")
    print(f"  Years            : {args.start} -> {args.end}")
    print(f"  Hours            : {'00-23 UTC' if args.full_day else '07-17 UTC'}")
    print(f"  Cache            : {args.tick_cache}")
    print(f"  Compact cache    : {args.compact_cache}")
    print(f"  Concurrency      : {args.concurrency}")
    print(f"  Request delay    : {args.request_delay}")
    print(f"  Redownload passes: {args.redownload_passes}")
    print(f"  Fail on missing  : {not args.keep_going}")
    print(f"  Auto compact     : {not args.no_auto_compact} ({args.compact_granularity})")
    print(f"  Auto DuckDB      : {not args.no_auto_duckdb}")
    print("=" * 72)

    manager = ForexDataManager(
        dukascopy_dir=args.tick_cache,
        dukascopy_compact_dir=args.compact_cache,
        verbose=True,
    )
    manager.duka.concurrency = max(1, args.concurrency)
    manager.duka.delay = max(0.0, args.request_delay)

    summary = manager.download_dukascopy_year_by_year(
        pairs=pairs,
        start_year=args.start,
        end_year=args.end,
        session_only=not args.full_day,
        max_redownload_passes=max(0, args.redownload_passes),
        fail_on_missing=not args.keep_going,
    )

    print("\n" + "=" * 72)
    print("  Summary")
    print("=" * 72)
    print(f"  {'Pair':<10} {'Year':<6} {'Ticks':>12} {'Coverage':>14} {'Missing':>8}")
    print("  " + "-" * 60)
    for pair in pairs:
        pair_summary = summary.get(pair, {})
        for year in sorted(pair_summary):
            item = pair_summary[year]
            coverage = item["coverage"]
            cov_txt = f"{coverage['present_hours_count']}/{coverage['requested_hours_count']}"
            print(f"  {pair:<10} {year:<6} {item['ticks']:>12,} {cov_txt:>14} {coverage['missing_hours_count']:>8}")
    print("=" * 72)

    if not args.no_auto_compact:
        _auto_finalize_storage(
            manager,
            pairs,
            granularity=args.compact_granularity,
            build_duckdb=not args.no_auto_duckdb,
            as_table=args.as_table,
        )


if __name__ == "__main__":
    main()
