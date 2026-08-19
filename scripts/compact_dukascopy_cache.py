#!/usr/bin/env python3
"""
Compact hourly Dukascopy parquet files into daily/monthly partitions.

This dramatically speeds up training I/O by reducing the number of files
from thousands (hourly) to hundreds (daily) or dozens (monthly).

Usage:
    python scripts/compact_dukascopy_cache.py --pairs EURUSD GBPUSD USDJPY --granularity daily
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.sources import ForexDataManager


def main():
    parser = argparse.ArgumentParser(description="Compact Dukascopy cache")
    parser.add_argument("--pairs", nargs="+", default=["EURUSD", "GBPUSD", "USDJPY"], help="Pairs to compact")
    parser.add_argument(
        "--granularity", choices=["daily", "monthly"], default="daily", help="Partition granularity (default: daily)"
    )
    parser.add_argument("--start", default="2008-01-01", help="Start date")
    parser.add_argument("--end", default="2024-12-31", help="End date")
    parser.add_argument("--overwrite", action="store_true", default=True, help="Overwrite existing partitions")
    parser.add_argument("--no-overwrite", action="store_false", dest="overwrite", help="Skip existing partitions")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress")

    args = parser.parse_args()

    print(f"{'=' * 60}")
    print("COMPACT DUKASCOPY CACHE")
    print(f"{'=' * 60}")
    print(f"Pairs:        {', '.join(args.pairs)}")
    print(f"Granularity:  {args.granularity}")
    print(f"Date range:   {args.start} to {args.end}")
    print(f"Overwrite:    {args.overwrite}")
    print(f"{'=' * 60}\n")

    mgr = ForexDataManager(verbose=not args.quiet)

    try:
        summary = mgr.compact_dukascopy_cache(
            pairs=args.pairs,
            granularity=args.granularity,
            start=args.start,
            end=args.end,
            overwrite=args.overwrite,
        )

        print(f"\n{'=' * 60}")
        print("COMPACTION SUMMARY")
        print(f"{'=' * 60}")

        total_partitions = 0
        total_ticks = 0

        for pair, stats in summary.items():
            partitions = stats.get("partitions_written", 0)
            ticks = stats.get("ticks_written", 0)
            hour_files = stats.get("hour_files", 0)

            total_partitions += partitions
            total_ticks += ticks

            print(f"\n{pair}:")
            print(f"  Hourly files:   {hour_files}")
            print(f"  {args.granularity.capitalize()} partitions: {partitions}")
            print(f"  Ticks written:  {ticks:,}")

            if not args.quiet and stats.get("outputs"):
                print("  Output paths:")
                for out in stats["outputs"][:5]:
                    print(f"    {out}")
                if len(stats["outputs"]) > 5:
                    print(f"    ... and {len(stats['outputs']) - 5} more")

        print(f"\n{'=' * 60}")
        print(f"TOTAL: {total_partitions} {args.granularity} partitions | {total_ticks:,} ticks")
        print(f"{'=' * 60}")

        # Show storage savings
        print("\nStorage savings omitted (Windows does not support 'du')")
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
