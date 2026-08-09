#!/usr/bin/env python3
"""
Download missing historical data for 7 pairs from Dukascopy (2008-2023).

Pairs: AUDUSD, EURGBP, EURJPY, GBPJPY, NZDUSD, USDCAD, USDCHF
These currently only have 2024 data; this script fetches 2008-2023.

Usage:
    python scripts/download_missing_pairs.py [--start-year 2008] [--end-year 2023] [--concurrency 24] [--resume]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.sources import DukascopyLoader


MISSING_PAIRS = ["AUDUSD", "EURGBP", "EURJPY", "GBPJPY", "NZDUSD", "USDCAD", "USDCHF"]


def main():
    parser = argparse.ArgumentParser(description="Download missing pair data from Dukascopy")
    parser.add_argument("--pairs", nargs="+", default=MISSING_PAIRS,
                        help="Pairs to download (default: all 7 missing pairs)")
    parser.add_argument("--start-year", type=int, default=2008,
                        help="Start year (default: 2008)")
    parser.add_argument("--end-year", type=int, default=2023,
                        help="End year (default: 2023, 2024 already exists)")
    parser.add_argument("--concurrency", type=int, default=24,
                        help="Concurrent downloads (default: 24)")
    parser.add_argument("--max-retries", type=int, default=8,
                        help="Max retries per hour (default: 8)")
    parser.add_argument("--request-delay", type=float, default=0.05,
                        help="Delay between requests in seconds (default: 0.05)")
    parser.add_argument("--session-only", action="store_true", default=True,
                        help="Only download London+NY session (07-17 UTC)")
    parser.add_argument("--full-day", action="store_true",
                        help="Download full 24h (overrides --session-only)")
    parser.add_argument("--fail-on-missing", action="store_true", default=False,
                        help="Fail if any hours missing after retries")
    parser.add_argument("--max-redownload-passes", type=int, default=3,
                        help="Redownload passes for missing hours (default: 3)")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from cached data (default: True)")
    parser.add_argument("--no-resume", action="store_false", dest="resume",
                        help="Force re-download all")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose output")
    
    args = parser.parse_args()
    
    if args.full_day:
        args.session_only = False
    
    print(f"{'='*60}")
    print(f"Dukascopy Historical Data Download")
    print(f"{'='*60}")
    print(f"Pairs:          {', '.join(args.pairs)}")
    print(f"Date range:     {args.start_year}-01-01 to {args.end_year}-12-31")
    print(f"Session only:   {args.session_only} ({'07-17 UTC' if args.session_only else '00-23 UTC'})")
    print(f"Concurrency:    {args.concurrency}")
    print(f"Max retries:    {args.max_retries}")
    print(f"Redownload passes: {args.max_redownload_passes}")
    print(f"Fail on missing: {args.fail_on_missing}")
    print(f"Resume:         {args.resume}")
    print(f"{'='*60}\n")
    
    loader = DukascopyLoader(
        concurrency=args.concurrency,
        max_retries=args.max_retries,
        request_delay=args.request_delay,
        verbose=not args.quiet,
    )
    
    try:
        results = loader.download_dukascopy_year_by_year(
            pairs=args.pairs,
            start_year=args.start_year,
            end_year=args.end_year,
            session_only=args.session_only,
            max_redownload_passes=args.max_redownload_passes,
            fail_on_missing=args.fail_on_missing,
        )
        
        print(f"\n{'='*60}")
        print(f"DOWNLOAD SUMMARY")
        print(f"{'='*60}")
        
        total_ticks = 0
        total_missing = 0
        
        for pair, years in results.items():
            pair_ticks = 0
            pair_missing = 0
            print(f"\n{pair}:")
            for year, data in years.items():
                ticks = data.get('ticks', 0)
                coverage = data.get('coverage', {})
                missing = coverage.get('missing_hours_count', 0)
                requested = coverage.get('requested_hours_count', 0)
                present = coverage.get('present_hours_count', 0)
                
                pair_ticks += ticks
                pair_missing += missing
                
                status = "✅" if missing == 0 else f"⚠️  {missing}/{requested} missing"
                print(f"  {year}: {ticks:>12,} ticks | {present}/{requested} hours {status}")
            
            total_ticks += pair_ticks
            total_missing += pair_missing
            print(f"  TOTAL: {pair_ticks:>12,} ticks | {pair_missing} missing hours")
        
        print(f"\n{'='*60}")
        print(f"GRAND TOTAL: {total_ticks:>12,} ticks | {total_missing} missing hours")
        print(f"{'='*60}")
        
        if total_missing > 0 and args.fail_on_missing:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Progress saved in cache.")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()