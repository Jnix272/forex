"""
count_ticks.py
==============
Estimates total tick count across all pairs for 2010-2025.

Strategy: Download 1 full trading day (24 hours) per year per pair,
compute average ticks/day, multiply by ~260 trading days/year.

Sample date: second Tuesday of January each year (liquid, post-New-Year).
"""

from __future__ import annotations

import sys
import time
from datetime import date, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.sources import DUKA_PAIR_MAP, DukascopyLoader

PAIRS = list(DUKA_PAIR_MAP.keys())
YEARS = list(range(2010, 2026))   # 2010 inclusive, 2025 inclusive
TRADING_DAYS_PER_YEAR = 260


def second_tuesday_of_january(year: int) -> date:
    """Return the second Tuesday of January for the given year."""
    d = date(year, 1, 1)
    # Advance to first Tuesday (weekday 1)
    while d.weekday() != 1:
        d += timedelta(days=1)
    # Second Tuesday
    d += timedelta(weeks=1)
    return d


def run():
    sample_dates = {y: second_tuesday_of_january(y) for y in YEARS}

    loader = DukascopyLoader(
        concurrency=64,
        max_retries=3,
        verbose=False,
    )

    print()
    print(f"Tick Count Estimate  |  2010-2025  |  {len(PAIRS)} pairs")
    print(f"Sample: 1 trading day per year (second Tuesday of January)")
    print(f"{'Year':<6} {'Date':<12} {'Pairs':<6} {'Ticks/day':>10}  Breakdown")
    print("-" * 72)

    year_totals: dict[int, int] = {}
    pair_year_ticks: dict[str, dict[int, int]] = {p: {} for p in PAIRS}

    t_global = time.perf_counter()

    for year in YEARS:
        sample_date = sample_dates[year]
        date_str    = str(sample_date)

        # Load all pairs for this sample day (all 24 hours)
        results = loader.load_multiple(PAIRS, start=date_str, end=date_str)

        day_total = 0
        breakdown_parts = []
        for pair in PAIRS:
            df = results.get(pair)
            n  = len(df) if df is not None and not df.empty else 0
            pair_year_ticks[pair][year] = n
            day_total += n
            breakdown_parts.append(f"{pair}={n:,}")

        year_totals[year] = day_total
        print(f"{year:<6} {date_str:<12} {len(PAIRS):<6} {day_total:>10,}  "
              + "  ".join(breakdown_parts))

    elapsed = time.perf_counter() - t_global

    # ── Extrapolation ────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("EXTRAPOLATION  (avg ticks/day * 260 trading days/year)")
    print(f"{'Year':<6} {'Days/yr':>8} {'Ticks/day':>10} {'Est. Annual':>14} {'Est. Cumulative':>16}")
    print("-" * 72)

    cumulative = 0
    for year in YEARS:
        tpd   = year_totals[year]
        annual = tpd * TRADING_DAYS_PER_YEAR
        cumulative += annual
        print(f"{year:<6} {TRADING_DAYS_PER_YEAR:>8,} {tpd:>10,} {annual:>14,} {cumulative:>16,}")

    total_ticks = sum(year_totals[y] * TRADING_DAYS_PER_YEAR for y in YEARS)
    avg_per_day = sum(year_totals.values()) / len(YEARS)

    print("=" * 72)
    print()
    print(f"  Estimated total ticks (2010-2025) : {total_ticks:>20,}")
    print(f"  Average ticks/day (all pairs)     : {avg_per_day:>20,.0f}")
    print(f"  Pairs sampled                     : {len(PAIRS):>20}")
    print(f"  Years covered                     : {len(YEARS):>20}")
    print(f"  Sample download time              : {elapsed:>19.1f}s")
    print()

    # ── Per-pair breakdown ────────────────────────────────────────────────────
    print("Per-pair estimated totals:")
    print(f"  {'Pair':<8} {'Avg ticks/day':>14} {'Est. 15yr total':>18}")
    print("  " + "-" * 44)
    for pair in PAIRS:
        avg  = sum(pair_year_ticks[pair].values()) / len(YEARS)
        est  = int(avg * TRADING_DAYS_PER_YEAR * len(YEARS))
        print(f"  {pair:<8} {avg:>14,.0f} {est:>18,}")
    print()


if __name__ == "__main__":
    run()
