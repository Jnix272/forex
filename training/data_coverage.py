"""
Data Coverage Validation.

Checks raw data directories for coverage gaps before training.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def validate_pair_coverage(
    pairs: list[str],
    data_source: str = "dukascopy",
    raw_dir: str = "data/compact",
    min_years: int = 2,
    expected_years: int = 18,
) -> tuple[list[str], list[dict]]:
    """
    Validate data coverage for each pair.

    Returns:
        (valid_pairs, coverage_report) - pairs with >= min_years of data, and full report.
    """
    # Check both raw and compact directories
    raw_path = Path(raw_dir) / data_source
    if not raw_path.exists() or not any(d.is_dir() for d in raw_path.iterdir()):
        # Fallback to raw directory for backwards compatibility - also when the
        # preferred dir exists but is an empty stub (e.g. data/compact/dukascopy).
        raw_path = Path("data/raw") / data_source
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data directory not found: {raw_path}")

    valid_pairs = []
    reports = []

    for pair in pairs:
        pair_dir = raw_path / pair.upper().replace("/", "")
        hive_pair_dir = raw_path / "granularity=daily" / f"pair={pair}"

        if hive_pair_dir.exists():
            # New Hive partitioned structure
            years = sorted(
                [
                    int(d.name.replace("year=", ""))
                    for d in hive_pair_dir.iterdir()
                    if d.is_dir() and d.name.startswith("year=")
                ]
            )
            n_files = len(list(hive_pair_dir.rglob("*.parquet")))
        elif pair_dir.exists():
            # Check for granularity=daily subdirectory (compact structure)
            granularity_dir = pair_dir / "granularity=daily"
            if granularity_dir.exists():
                years = sorted([int(d.name) for d in granularity_dir.iterdir() if d.is_dir() and d.name.isdigit()])
                n_files = len(list(granularity_dir.rglob("*.parquet")))
            else:
                # Fallback to old structure (year directories directly under pair)
                years = sorted([int(d.name) for d in pair_dir.iterdir() if d.is_dir() and d.name.isdigit()])
                n_files = len(list(pair_dir.rglob("*.parquet")))
        else:
            reports.append(
                {
                    "pair": pair,
                    "status": "MISSING",
                    "years": 0,
                    "message": f"Pair directory not found: {pair_dir} or {hive_pair_dir}",
                }
            )
            continue

        if not years:
            reports.append(
                {
                    "pair": pair,
                    "status": "EMPTY",
                    "years": 0,
                    "files": n_files,
                    "message": f"No year directories found in {pair_dir}",
                }
            )
            continue

        n_years = len(years)
        if n_years >= min_years:
            valid_pairs.append(pair)
            status = "OK" if n_years >= expected_years else "LOW"
        else:
            status = "SKIPPED"

        reports.append(
            {
                "pair": pair,
                "status": status,
                "years": n_years,
                "year_range": f"{years[0]}-{years[-1]}" if years else "N/A",
                "files": n_files,
                "min_years": min_years,
                "expected_years": expected_years,
                "message": (
                    f"{status}: {n_years} year(s) of data"
                    + (f" (need >= {min_years} to train)" if n_years < min_years else "")
                    + (f" - shorter than expected {expected_years} years" if status == "LOW" else "")
                ),
            }
        )

    return valid_pairs, reports


def validate_news_data(
    news_dir: str = "data/raw/news",
) -> dict[str, Any]:
    """Validate news data integrity."""
    path = Path(news_dir)
    if not path.exists():
        return {"status": "MISSING", "message": f"News directory not found: {news_dir}"}

    report = {"status": "OK", "issues": [], "stats": {}}

    # Check for empty/small files
    for f in sorted(path.glob("*")):
        if f.is_dir():
            continue
        size = f.stat().st_size
        if size == 0:
            report["issues"].append(f"EMPTY: {f.name} is 0 bytes")
        elif size < 100:
            report["issues"].append(f"SMALL: {f.name} is only {size} bytes")

    # Check for redundant CSV files
    csv_files = list(path.glob("*.csv"))
    parquet_globs = list(path.glob("*.parquet"))
    parq_globs = list(path.glob("*.parq"))
    parquet_files = parquet_globs + parq_globs
    if len(csv_files) > 1 and len(parquet_files) > 0:
        csv_total = sum(f.stat().st_size for f in csv_files) / 1e9
        pq_total = sum(f.stat().st_size for f in parquet_files) / 1e9
        report["issues"].append(
            f"REDUNDANT: {len(csv_files)} CSV files ({csv_total:.1f}GB) likely duplicate "
            f"{len(parquet_files)} parquet files ({pq_total:.1f}GB)"
        )

    # Check combined parquet if exists
    combined = path / "historical_news_combined.parquet"
    if combined.exists():
        try:
            import pandas as pd
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(combined)
            report["stats"]["rows"] = pf.metadata.num_rows
            report["stats"]["columns"] = pf.schema_arrow.names
            report["stats"]["row_groups"] = pf.metadata.num_row_groups

            # Check timestamps in first and last row group
            table = pf.read_row_group(0, columns=["timestamp_utc"])
            df = table.to_pandas()
            ts = pd.to_datetime(df["timestamp_utc"], errors="coerce")
            bad_ts = int(ts.isna().sum())
            if bad_ts > 0:
                report["issues"].append(
                    f"BAD_TIMESTAMPS: {bad_ts}/{len(ts)} ({bad_ts / len(ts) * 100:.1f}%) "
                    f"unparseable timestamps in first row group"
                )
                report["stats"]["bad_timestamps"] = bad_ts

            if "sentiment_score" in pf.schema_arrow.names:
                table2 = pf.read_row_group(0, columns=["sentiment_score"])
                df2 = table2.to_pandas()
                scores = df2["sentiment_score"].dropna()
                report["stats"]["sentiment_min"] = float(scores.min()) if len(scores) else None
                report["stats"]["sentiment_max"] = float(scores.max()) if len(scores) else None
                report["stats"]["sentiment_mean"] = float(scores.mean()) if len(scores) else None
        except Exception as e:
            report["issues"].append(f"PARSE_ERROR: {e}")

    if not report["issues"]:
        report["status"] = "OK"
    else:
        report["status"] = "WARN"

    return report


def validate_source_directories(
    raw_dir: str = "data/raw",
    expected_sources: list[str] | None = None,
) -> dict[str, Any]:
    """Validate that expected source directories exist and are populated."""
    if expected_sources is None:
        expected_sources = ["dukascopy", "news", "cot", "eco_calendar", "eodhd", "lmax"]

    path = Path(raw_dir)
    report = {"status": "OK", "issues": [], "empty": [], "missing": [], "populated": []}

    for src in expected_sources:
        src_path = path / src
        if not src_path.exists():
            report["missing"].append(src)
        elif not list(src_path.iterdir()):
            report["empty"].append(src)
        else:
            report["populated"].append(src)

    if report["empty"]:
        report["issues"].append(
            f"EMPTY: {len(report['empty'])} source directories have no data: " + ", ".join(report["empty"])
        )
    if report["missing"]:
        report["issues"].append(
            f"MISSING: {len(report['missing'])} source directories not found: " + ", ".join(report["missing"])
        )

    if report["issues"]:
        report["status"] = "WARN"

    return report


def run_data_coverage_check(
    args: Any,  # argparse.Namespace
    min_years: int = 2,
    expected_years: int = 18,
) -> dict[str, Any]:
    """Run full data coverage check and return report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "sections": {},
        "issues": [],
        "recommendations": [],
    }

    pairs = list(getattr(args, "pairs", ["EURUSD", "GBPUSD", "USDJPY"]))

    # 1. Pair coverage
    valid_pairs, coverage = validate_pair_coverage(
        pairs=pairs,
        data_source=getattr(args, "data_source", "dukascopy"),
        min_years=min_years,
        expected_years=expected_years,
    )
    report["sections"]["pair_coverage"] = coverage

    # Check for insufficient pairs
    if len(valid_pairs) < len(pairs):
        skipped = [p for p in pairs if p not in valid_pairs]
        report["issues"].append(
            f"INSUFFICIENT_DATA: {len(skipped)}/{len(pairs)} pairs have < {min_years} years "
            f"of data: {', '.join(skipped)}. Training only on {len(valid_pairs)} pairs."
        )
        if len(valid_pairs) < 2:
            report["issues"].append(
                f"CRITICAL: Only {len(valid_pairs)} pair(s) available - "
                f"multi-pair training requires at least 2. Run download_data.py first."
            )
    elif any(r["status"] == "LOW" for r in coverage):
        low_pairs = [r["pair"] for r in coverage if r["status"] == "LOW"]
        report["issues"].append(
            f"LOW_COVERAGE: {len(low_pairs)} pairs have < {expected_years} years: {', '.join(low_pairs)}. "
            f"Models may not generalize across full regimes."
        )

    report["valid_pairs"] = valid_pairs

    # 2. News
    news = validate_news_data()
    report["sections"]["news"] = news
    if news.get("issues"):
        report["issues"].extend(news["issues"])

    # 3. Source directories
    sources = validate_source_directories()
    report["sections"]["sources"] = sources
    if sources.get("issues"):
        report["issues"].extend(sources["issues"])

    # 4. Recommendations
    if len(valid_pairs) < len(pairs):
        report["recommendations"].append(
            "Run: python scripts/download_data.py --pairs " + " ".join(p for p in pairs if p not in valid_pairs)
        )
    if any(r["status"] == "LOW" for r in coverage):
        report["recommendations"].append(
            "Run: python scripts/download_data.py to fill missing years for low-coverage pairs"
        )

    return report


def save_coverage_report(report: dict, output_dir: str = "logs"):
    """Save coverage report to JSON."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    report_path = path / "data_coverage_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[Coverage] Report saved to {report_path}")
    return str(report_path)


def print_coverage_summary(report: dict):
    """Print human-readable coverage summary."""
    print(f"\n{'=' * 60}")
    print("DATA COVERAGE CHECK")
    print(f"{'=' * 60}")

    # Pair coverage
    print("\n📊 Pair Coverage:")
    for r in report["sections"].get("pair_coverage", []):
        icon = {"OK": "✅", "LOW": "⚠️", "SKIPPED": "❌", "MISSING": "❌", "EMPTY": "❌"}.get(r["status"], "?")
        print(f"  {icon} {r['pair']:8s} - {r['message']}")

    # Issues
    issues = report.get("issues", [])
    if issues:
        print(f"\n🚨 Issues ({len(issues)}):")
        for i in issues:
            print(f"  • {i}")

    # Valid pairs
    valid = report.get("valid_pairs", [])
    print(f"\n📈 Valid training pairs: {', '.join(valid) if valid else 'NONE'}")

    # Recommendations
    recs = report.get("recommendations", [])
    if recs:
        print("\n💡 Recommendations:")
        for r in recs:
            print(f"  → {r}")

    # News stats
    news = report["sections"].get("news", {})
    if news.get("stats"):
        print(f"\n📰 News: {news['stats'].get('rows', 0):,} rows")

    # Sources
    sources = report["sections"].get("sources", {})
    if sources.get("empty"):
        print(f"\n📁 Empty dirs: {', '.join(sources['empty'])}")

    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run data coverage check")
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "EURGBP", "EURJPY", "GBPJPY", "NZDUSD", "USDCAD", "USDCHF"],
    )
    parser.add_argument("--data-source", default="dukascopy")
    parser.add_argument("--min-years", type=int, default=2)
    parser.add_argument("--expected-years", type=int, default=18)
    args = parser.parse_args()

    report = run_data_coverage_check(args, min_years=args.min_years, expected_years=args.expected_years)
    print_coverage_summary(report)
    save_coverage_report(report)

    # Fail if insufficient pairs
    valid_pairs = report.get("valid_pairs", [])
    if len(valid_pairs) < len(args.pairs):
        import sys

        print(f"ERROR: {len(args.pairs) - len(valid_pairs)} pairs failed coverage requirements.", file=sys.stderr)
        sys.exit(1)
