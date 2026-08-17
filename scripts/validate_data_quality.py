#!/usr/bin/env python3
"""
Validate data quality for all pairs.

Runs the ForexDataManager quality_report on each pair and flags issues.

Usage:
    python scripts/validate_data_quality.py [--pairs EURUSD GBPUSD] [--min-score 95] [--start 2008-01-01] [--end 2024-12-31]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.sources import ForexDataManager


def main():
    parser = argparse.ArgumentParser(description="Validate data quality")
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "EURGBP", "EURJPY", "GBPJPY", "NZDUSD", "USDCAD", "USDCHF"],
        help="Pairs to validate",
    )
    parser.add_argument(
        "--source",
        default="dukascopy",
        choices=["dukascopy", "tds", "lmax_historical", "auto"],
        help="Data source to validate",
    )
    parser.add_argument("--start", default="2008-01-01", help="Start date")
    parser.add_argument("--end", default="2024-12-31", help="End date")
    parser.add_argument("--min-score", type=float, default=95.0, help="Minimum quality score (0-100)")
    parser.add_argument("--session-only", action="store_true", default=True, help="Session hours only")
    parser.add_argument("--fail-fast", action="store_true", help="Exit on first failure")
    parser.add_argument("--json", action="store_true", help="Output JSON summary")
    parser.add_argument("--quiet", action="store_true", help="Only show failures")

    args = parser.parse_args()

    mgr = ForexDataManager(verbose=not args.quiet)

    print(f"{'=' * 70}")
    print("DATA QUALITY VALIDATION")
    print(f"{'=' * 70}")
    print(f"Source:     {args.source}")
    print(f"Date range: {args.start} to {args.end}")
    print(f"Min score:  {args.min_score}%")
    print(f"Pairs:      {', '.join(args.pairs)}")
    print(f"{'=' * 70}\n")

    results = []
    failed = []

    for pair in args.pairs:
        try:
            if not args.quiet:
                print(f"Loading {pair}...", end=" ", flush=True)

            df = mgr.load(
                pair,
                source=args.source,
                start=args.start,
                end=args.end,
                session_only=args.session_only,
            )

            if df.empty:
                result = {"pair": pair, "status": "NO_DATA", "quality_score": 0, "error": "No data loaded"}
                failed.append(result)
                if not args.quiet:
                    print("❌ NO DATA")
                continue

            report = mgr.quality_report(df, pair)
            report["pair"] = pair
            report["status"] = "PASS" if report["quality_score"] >= args.min_score else "FAIL"
            results.append(report)

            if report["quality_score"] < args.min_score:
                failed.append(report)

            if not args.quiet:
                status_icon = "✅" if report["status"] == "PASS" else "❌"
                print(
                    f"{status_icon} {report['quality_score']:.2f}% | "
                    f"{report['n_ticks']:,} ticks | "
                    f"{report['avg_spread_pips']:.2f} pips avg | "
                    f"{report['n_gaps_over_1min']} gaps>1min"
                )

            if args.fail_fast and report["status"] == "FAIL":
                break

        except Exception as e:
            result = {"pair": pair, "status": "ERROR", "quality_score": 0, "error": str(e)}
            failed.append(result)
            if not args.quiet:
                print(f"❌ ERROR: {e}")
            if args.fail_fast:
                break

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    passed = [r for r in results if r["status"] == "PASS"]
    failed_count = len(failed)

    print(f"Total checked: {len(args.pairs)}")
    print(f"Passed:        {len(passed)}")
    print(f"Failed:        {failed_count}")
    print(f"Errors:        {len([r for r in results if r.get('status') == 'ERROR'])}")

    if failed:
        print(f"\n{'FAILURES / ERRORS':-^70}")
        for r in failed:
            if r["status"] == "NO_DATA":
                print(f"  {r['pair']}: NO DATA LOADED")
            elif r["status"] == "ERROR":
                print(f"  {r['pair']}: ERROR - {r.get('error', 'unknown')}")
            else:
                print(
                    f"  {r['pair']}: {r['quality_score']:.2f}% "
                    f"(spread={r['avg_spread_pips']:.1f}pips, "
                    f"gaps={r['n_gaps_over_1min']}, "
                    f"anomalies={r['n_spread_anomalies']}, "
                    f"inversions={r['n_bid_ask_inversions']})"
                )

    if args.json:
        import json

        output = {
            "summary": {
                "total": len(args.pairs),
                "passed": len(passed),
                "failed": failed_count,
                "min_score": args.min_score,
            },
            "results": results + failed,
        }
        print(json.dumps(output, indent=2, default=str))

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
