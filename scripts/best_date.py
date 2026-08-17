"""
best_date.py
============
Find the best representative trading day per year by testing several
candidate dates and scoring them on:
  - completeness  : how many of the 9 pairs have ticks (0-9)
  - consistency   : no pair is more than 5x below the median tick count
  - liquidity     : total tick count (higher = better)

Candidates per year: 3rd Tuesday of March, June, September, December
(mid-quarter dates avoid holiday distortions and year-end low liquidity).
"""

from __future__ import annotations

import sys
import time
from datetime import date, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import statistics

from data.sources import DUKA_PAIR_MAP, DukascopyLoader

PAIRS = list(DUKA_PAIR_MAP.keys())
YEARS = list(range(2010, 2026))

# Session hours only (London + NY overlap) - reduces download by ~54%
SESSION_HOURS = list(range(7, 18))  # 07:00–17:00 UTC  # noqa: RUF003


def third_tuesday(year: int, month: int) -> date:
    """Return the third Tuesday of the given year/month."""
    d = date(year, month, 1)
    while d.weekday() != 1:  # 1 = Tuesday
        d += timedelta(days=1)
    return d + timedelta(weeks=2)


def score_day(results: dict) -> dict:
    """Score a single day's multi-pair results."""
    counts = {p: len(df) if df is not None and not df.empty else 0 for p, df in results.items()}
    non_zero = [v for v in counts.values() if v > 0]
    total = sum(counts.values())
    complete = len(non_zero)  # out of 9

    if len(non_zero) >= 2:
        median = statistics.median(non_zero)
        # consistency: flag pairs that are < 20% of median
        stragglers = sum(1 for v in non_zero if v < median * 0.2)
    else:
        stragglers = 9

    # Score: completeness is paramount, then penalise stragglers, then reward volume
    score = complete * 1000 - stragglers * 100 + min(total // 1000, 500)
    return {"total": total, "complete": complete, "stragglers": stragglers, "score": score, "counts": counts}


def run():
    loader = DukascopyLoader(concurrency=64, max_retries=3, verbose=False)

    # Candidate months: March (3), June (6), September (9), December (12)
    CANDIDATE_MONTHS = [3, 6, 9, 12]

    print()
    print(
        f"Best-date analysis  |  {len(YEARS)} years  |  {len(PAIRS)} pairs  |  session hours {SESSION_HOURS[0]}-{SESSION_HOURS[-1]} UTC"
    )
    print("Candidates per year: 3rd Tuesday of March, June, September, December")
    print()

    best_per_year: dict[int, dict] = {}
    all_results = []

    t0 = time.perf_counter()

    for year in YEARS:
        candidates = [third_tuesday(year, m) for m in CANDIDATE_MONTHS]
        year_best = None

        for d in candidates:
            date_str = str(d)
            results = loader.load_multiple(PAIRS, start=date_str, end=date_str, hours=SESSION_HOURS)
            s = score_day(results)
            s["date"] = d

            flag = "(*)" if year_best is None or s["score"] > year_best["score"] else "   "
            pair_summary = "  ".join(f"{p}={'OK' if s['counts'][p] > 0 else '--'}" for p in PAIRS)
            print(
                f"  {year} {date_str}  complete={s['complete']}/9  total={s['total']:>7,}  score={s['score']:>5}  {flag}  |  {pair_summary}"
            )

            if year_best is None or s["score"] > year_best["score"]:
                year_best = s

        best_per_year[year] = year_best
        all_results.append(year_best)
        print()

    elapsed = time.perf_counter() - t0

    # ── Summary table ────────────────────────────────────────────────────────
    print("=" * 72)
    print("BEST DATE PER YEAR")
    print(f"{'Year':<6} {'Best date':<12} {'Complete':>9} {'Ticks':>9}  Missing pairs")
    print("-" * 72)

    for year in YEARS:
        b = best_per_year[year]
        missing = [p for p in PAIRS if b["counts"].get(p, 0) == 0]
        missing_str = ", ".join(missing) if missing else "none"
        print(f"{year:<6} {b['date']!s:<12} {b['complete']:>7}/9  {b['total']:>9,}  {missing_str}")

    # Overall recommendation
    fully_complete = [y for y, b in best_per_year.items() if b["complete"] == 9]
    print()
    print(f"Years with ALL 9 pairs present: {len(fully_complete)}/{len(YEARS)}")
    if fully_complete:
        print(f"  {fully_complete}")

    # Best single date for ongoing use (most recent year with full coverage)
    for year in reversed(YEARS):
        if best_per_year[year]["complete"] == 9:
            rec = best_per_year[year]
            print(f"\nRecommended sample date: {rec['date']}  ({rec['total']:,} ticks, all 9 pairs)")
            break

    print(f"\nTotal analysis time: {elapsed:.1f}s")
    print()


if __name__ == "__main__":
    run()
