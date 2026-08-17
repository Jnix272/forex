"""
scripts/verify_data.py
======================
Data quality verification and auto-repair for the Dukascopy tick cache.

Checks performed:
  1. DUPLICATE DETECTION
     - Duplicate timestamps within each hourly parquet file
     - Removes exact duplicates, keeps first occurrence
  2. MISSING DATA DETECTION
     - Expected weekday session hours vs. files on disk
     - Corrupt / unreadable parquet files
     - Suspiciously small files (< min_ticks threshold)
  3. GAP ANALYSIS
     - Consecutive missing-hour streaks (excluding weekends / holidays)
     - Per-pair coverage summary (% of expected hours with data)
  4. AUTO-REDOWNLOAD
     - Deletes corrupt files, then re-fetches from Dukascopy
     - Re-fetches hours where file is missing but neighbours have data

Usage:
  # Full scan + report (no changes)
  python scripts/verify_data.py

  # Scan specific pairs
  python scripts/verify_data.py --pairs EURUSD GBPUSD

  # Auto-fix: remove duplicates, redownload missing/corrupt
  python scripts/verify_data.py --fix

  # Custom date range
  python scripts/verify_data.py --start 2010-01-01 --end 2025-12-31 --fix

  # Aggressive: also redownload suspiciously small files
  python scripts/verify_data.py --fix --min-ticks 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple  # noqa: UP035

# Project root on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "EURGBP", "NZDUSD", "EURJPY", "GBPJPY"]
DEFAULT_START = "2010-01-01"
DEFAULT_END = "2025-12-31"
DEFAULT_DUKASCOPY_CACHE_DIR = str(_ROOT / "data" / "raw" / "dukascopy")
SESSION_HOURS = list(range(7, 18))  # 07-17 UTC

# Major forex holidays where ALL pairs are expected to have zero data
# (Christmas, New Year's Day). Other partial holidays are handled
# by the neighbour heuristic.
KNOWN_HOLIDAYS_MD = {(12, 25), (1, 1)}
DEFAULT_NEWS_FILE = "data/raw/news/historical_news_combined.parquet"


# ---------------------------------------------------------------------------
# Utility: generate expected hours
# ---------------------------------------------------------------------------
def _expected_hours(
    start: str,
    end: str,
    hours: list[int],
) -> list[datetime]:
    """Generate all weekday x session-hour datetimes in the range."""
    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC)
    end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC)
    result: list[datetime] = []
    current = start_dt
    while current <= end_dt:
        if current.weekday() < 5:  # Mon-Fri
            for h in hours:
                result.append(current.replace(hour=h, minute=0, second=0, microsecond=0))
        current += timedelta(days=1)
    return result


def _is_known_holiday(dt: datetime) -> bool:
    return (dt.month, dt.day) in KNOWN_HOLIDAYS_MD


def _cache_path(cache_dir: Path, pair: str, dt: datetime) -> Path:
    return cache_dir / pair / f"{dt.year}" / f"{dt.month:02d}" / f"{dt.day:02d}_{dt.hour:02d}.parquet"


# ---------------------------------------------------------------------------
# 1. DUPLICATE SCANNER
# ---------------------------------------------------------------------------
def scan_duplicates(
    cache_dir: Path,
    pair: str,
    expected: list[datetime],
    fix: bool = False,
) -> dict:
    """Scan all parquet files for a pair and detect duplicate timestamps."""
    stats = {"files_scanned": 0, "files_with_dupes": 0, "total_dupes": 0, "dupes_removed": 0}

    for dt in expected:
        fpath = _cache_path(cache_dir, pair, dt)
        if not fpath.exists():
            continue
        stats["files_scanned"] += 1
        try:
            df = pd.read_parquet(fpath)
        except Exception:
            continue  # corrupt files handled in missing-data scan

        dupes = df.index.duplicated(keep="first")
        n_dupes = dupes.sum()
        if n_dupes > 0:
            stats["files_with_dupes"] += 1
            stats["total_dupes"] += int(n_dupes)
            if fix:
                df_clean = df[~dupes]
                df_clean.to_parquet(fpath)
                stats["dupes_removed"] += int(n_dupes)

    return stats


# ---------------------------------------------------------------------------
# 2. MISSING / CORRUPT SCANNER
# ---------------------------------------------------------------------------
def scan_missing(
    cache_dir: Path,
    pair: str,
    expected: list[datetime],
    min_ticks: int = 0,
) -> dict:
    """
    Identify missing, corrupt, and suspiciously small files.

    Returns dict with:
      missing     : list of datetimes where file is absent
      corrupt     : list of datetimes where file can't be read
      too_small   : list of datetimes where tick count < min_ticks
      ok          : count of healthy files
      empty_ok    : count of files absent but on known holidays
    """
    missing: list[datetime] = []
    corrupt: list[datetime] = []
    too_small: list[datetime] = []
    ok = 0
    empty_ok = 0

    for dt in expected:
        fpath = _cache_path(cache_dir, pair, dt)
        if not fpath.exists():
            if _is_known_holiday(dt):
                empty_ok += 1
            else:
                missing.append(dt)
            continue

        # Try reading file
        try:
            df = pd.read_parquet(fpath)
        except Exception:
            corrupt.append(dt)
            continue

        if len(df) == 0:
            # Empty parquet: could be legit (no ticks that hour) or failed download
            # Use neighbour heuristic later
            ok += 1
            continue

        if min_ticks > 0 and len(df) < min_ticks:
            too_small.append(dt)
        else:
            ok += 1

    return {
        "missing": missing,
        "corrupt": corrupt,
        "too_small": too_small,
        "ok": ok,
        "empty_ok": empty_ok,
    }


# ---------------------------------------------------------------------------
# 3. GAP ANALYSIS  (consecutive missing streaks)
# ---------------------------------------------------------------------------
def analyse_gaps(
    missing_dts: list[datetime],
    hours: list[int],
) -> list[tuple[datetime, datetime, int]]:
    """
    Find consecutive missing-hour streaks.
    Returns list of (start_dt, end_dt, streak_length).
    """
    if not missing_dts:
        return []

    sorted_dts = sorted(missing_dts)
    hour_set = set(hours)
    gaps: list[tuple[datetime, datetime, int]] = []

    streak_start = sorted_dts[0]
    streak_end = sorted_dts[0]
    streak_len = 1

    for i in range(1, len(sorted_dts)):
        prev = sorted_dts[i - 1]
        curr = sorted_dts[i]

        # Check if curr is the "next expected hour" after prev
        expected_next = prev + timedelta(hours=1)
        # Skip to next session hour / next weekday if needed
        while expected_next.hour not in hour_set or expected_next.weekday() >= 5:
            expected_next += timedelta(hours=1)
            if (expected_next - prev).days > 4:
                break

        if curr == expected_next:
            streak_end = curr
            streak_len += 1
        else:
            if streak_len >= 3:  # Only report gaps >= 3 consecutive hours
                gaps.append((streak_start, streak_end, streak_len))
            streak_start = curr
            streak_end = curr
            streak_len = 1

    # Close final streak
    if streak_len >= 3:
        gaps.append((streak_start, streak_end, streak_len))

    return gaps


# ---------------------------------------------------------------------------
# 4. NEIGHBOUR HEURISTIC: filter missing hours to "suspicious" only
# ---------------------------------------------------------------------------
def _filter_suspicious_missing(
    cache_dir: Path,
    pair: str,
    missing_dts: list[datetime],
) -> list[datetime]:
    """
    Among missing hours, keep only those where at least one neighbour
    (+/- 1 hour, same weekday constraint) has data on disk.
    This avoids trying to redownload hours that legitimately have no data
    (holidays, early closes, etc.).
    """
    suspicious: list[datetime] = []
    for dt in missing_dts:
        has_neighbour = False
        for delta_h in [-1, 1, -2, 2]:
            nb = dt + timedelta(hours=delta_h)
            if nb.weekday() >= 5:
                continue
            fpath = _cache_path(cache_dir, pair, nb)
            if fpath.exists():
                try:
                    df = pd.read_parquet(fpath)
                    if len(df) > 0:
                        has_neighbour = True
                        break
                except Exception:
                    pass
        if has_neighbour:
            suspicious.append(dt)
    return suspicious


# ---------------------------------------------------------------------------
# 5. AUTO-REDOWNLOAD
# ---------------------------------------------------------------------------
def redownload_hours(
    cache_dir: Path,
    pair: str,
    hours_to_fix: list[datetime],
    concurrency: int = 8,
    request_delay: float = 0.1,
) -> dict:
    """
    Delete bad files and re-fetch from Dukascopy.
    Returns stats dict with counts.
    """
    stats = {"deleted": 0, "refetched": 0, "still_missing": 0}

    if not hours_to_fix:
        return stats

    from data.sources import DukascopyLoader

    # Delete existing corrupt/bad files
    for dt in hours_to_fix:
        fpath = _cache_path(cache_dir, pair, dt)
        if fpath.exists():
            try:
                fpath.unlink()
                stats["deleted"] += 1
            except Exception:
                pass

    # Re-fetch using DukascopyLoader
    loader = DukascopyLoader(
        cache_dir=str(cache_dir),
        concurrency=concurrency,
        request_delay=request_delay,
        verbose=True,
    )

    print(f"  [{pair}] Re-fetching {len(hours_to_fix)} hours ...")
    # The loader expects start/end strings but we have specific hours.
    # Use the low-level async method via load() for each batch.
    # Group by date for efficiency.
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    import aiohttp

    from data.sources import _run_dukascopy_async

    async def _refetch_all():
        sem = asyncio.Semaphore(concurrency)
        executor = ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4))
        conn = aiohttp.TCPConnector(
            limit=concurrency + 5,
            limit_per_host=concurrency + 5,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
        )
        async with aiohttp.ClientSession(
            connector=conn,
            headers={"User-Agent": "ForexScaler/2.0 (verify-repair)"},
        ) as sess:
            tasks = []
            for dt in hours_to_fix:
                tasks.append(loader._load_hour_async(sess, sem, pair, dt, executor))
            results = await asyncio.gather(*tasks, return_exceptions=True)
        executor.shutdown(wait=True)
        return results

    results = _run_dukascopy_async(_refetch_all())

    for dt, res in zip(hours_to_fix, results, strict=False):  # noqa: B007
        if isinstance(res, Exception) or (isinstance(res, pd.DataFrame) and res.empty):
            stats["still_missing"] += 1
        else:
            stats["refetched"] += 1

    return stats


# ---------------------------------------------------------------------------
# MAIN SCAN + REPORT
# ---------------------------------------------------------------------------
def run_verification(
    pairs: list[str],
    start: str,
    end: str,
    hours: list[int],
    cache_dir: str,
    fix: bool = False,
    min_ticks: int = 0,
    concurrency: int = 8,
    request_delay: float = 0.1,
) -> None:
    cache = Path(cache_dir)
    expected = _expected_hours(start, end, hours)
    total_expected = len(expected)

    print("\n" + "=" * 72)
    print("  Forex Scaling Model -- Data Verification & Repair")
    print("=" * 72)
    print(f"  Pairs      : {', '.join(pairs)}")
    print(f"  Range      : {start}  ->  {end}")
    print(f"  Hours/day  : {len(hours)}  ({hours[0]:02d}:00 - {hours[-1]:02d}:59 UTC)")
    print(f"  Expected   : {total_expected:,} hour-slots per pair")
    print(f"  Cache      : {cache_dir}")
    print(f"  Mode       : {'FIX (repair + redownload)' if fix else 'SCAN ONLY (dry run)'}")
    if min_ticks > 0:
        print(f"  Min ticks  : {min_ticks} (files below this flagged as too-small)")
    print("=" * 72)

    grand_dupes = 0
    grand_missing = 0
    grand_corrupt = 0
    grand_small = 0
    grand_refetched = 0

    for pair in pairs:
        t0 = time.perf_counter()
        print(f"\n  --- {pair} {'=' * 55}")

        # 1. Duplicate scan
        dup_stats = scan_duplicates(cache, pair, expected, fix=fix)
        if dup_stats["files_with_dupes"] > 0:
            print(
                f"  [Duplicates] {dup_stats['files_with_dupes']} files with "
                f"{dup_stats['total_dupes']:,} duplicate rows",
                end="",
            )
            if fix:
                print(f"  -> REMOVED {dup_stats['dupes_removed']:,}")
            else:
                print("  (use --fix to remove)")
        else:
            print(f"  [Duplicates] CLEAN ({dup_stats['files_scanned']:,} files scanned)")
        grand_dupes += dup_stats["total_dupes"]

        # 2. Missing / corrupt scan
        miss_stats = scan_missing(cache, pair, expected, min_ticks=min_ticks)
        n_missing = len(miss_stats["missing"])
        n_corrupt = len(miss_stats["corrupt"])
        n_small = len(miss_stats["too_small"])
        n_ok = miss_stats["ok"]
        coverage = (n_ok / total_expected * 100) if total_expected > 0 else 0

        print(
            f"  [Coverage]   {n_ok:,}/{total_expected:,} OK "
            f"({coverage:.1f}%)  |  {miss_stats['empty_ok']} holiday-skipped"
        )

        if n_corrupt > 0:
            print(f"  [Corrupt]    {n_corrupt} unreadable parquet files", end="")
            if fix:
                print("  -> will redownload")
            else:
                print("  (use --fix to repair)")

        if n_missing > 0:
            # Apply neighbour heuristic
            suspicious = _filter_suspicious_missing(cache, pair, miss_stats["missing"])
            legit_empty = n_missing - len(suspicious)
            print(
                f"  [Missing]    {n_missing} total  |  {len(suspicious)} suspicious "
                f"(neighbours have data)  |  {legit_empty} likely no-data"
            )
        else:
            suspicious = []
            print("  [Missing]    0 (all expected hours accounted for)")

        if n_small > 0:
            print(f"  [Too-small]  {n_small} files with < {min_ticks} ticks", end="")
            if fix:
                print("  -> will redownload")
            else:
                print("  (use --fix to repair)")

        grand_missing += len(suspicious)
        grand_corrupt += n_corrupt
        grand_small += n_small

        # 3. Gap analysis
        all_problem_dts = sorted(set(miss_stats["missing"]) | set(miss_stats["corrupt"]))
        gaps = analyse_gaps(all_problem_dts, hours)
        if gaps:
            big_gaps = [g for g in gaps if g[2] >= 11]  # >= 1 full day
            if big_gaps:
                print(f"  [Gaps]       {len(gaps)} streaks  |  {len(big_gaps)} are >= 1 trading day:")
                for gs, ge, gl in big_gaps[:5]:
                    print(
                        f"               {gs.strftime('%Y-%m-%d %H:00')} -> "
                        f"{ge.strftime('%Y-%m-%d %H:00')}  ({gl} hours)"
                    )
                if len(big_gaps) > 5:
                    print(f"               ... and {len(big_gaps) - 5} more")

        # 4. Auto-fix
        if fix:
            to_redownload = miss_stats["corrupt"] + miss_stats["too_small"] + suspicious
            if to_redownload:
                to_redownload = sorted(set(to_redownload))
                rd_stats = redownload_hours(
                    cache,
                    pair,
                    to_redownload,
                    concurrency=concurrency,
                    request_delay=request_delay,
                )
                print(
                    f"  [Repair]     Deleted: {rd_stats['deleted']}  |  "
                    f"Re-fetched: {rd_stats['refetched']}  |  "
                    f"Still missing: {rd_stats['still_missing']}"
                )
                grand_refetched += rd_stats["refetched"]
            else:
                print("  [Repair]     Nothing to fix")

        elapsed = time.perf_counter() - t0
        print(f"  [{pair}] scan complete in {elapsed:.1f}s")

    # Summary
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("-" * 72)
    print(f"  Total duplicates found  : {grand_dupes:,}")
    print(f"  Total suspicious missing: {grand_missing:,}")
    print(f"  Total corrupt files     : {grand_corrupt:,}")
    print(f"  Total too-small files   : {grand_small:,}")
    if fix:
        print(f"  Total re-fetched        : {grand_refetched:,}")
    issues = grand_dupes + grand_missing + grand_corrupt + grand_small
    if issues == 0:
        print("\n  ALL CLEAN -- no data quality issues found.")
    elif not fix:
        print(f"\n  {issues:,} issues found. Run with --fix to auto-repair.")
    else:
        remaining = grand_missing + grand_corrupt + grand_small - grand_refetched
        if remaining <= 0:
            print("\n  ALL ISSUES REPAIRED.")
        else:
            print(f"\n  {remaining:,} issues remain (likely no data available from Dukascopy).")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Historical news verification
# ---------------------------------------------------------------------------
def _month_range(start_ym: str, end_ym: str) -> list[str]:
    sy, sm = [int(x) for x in start_ym.split("-")]
    ey, em = [int(x) for x in end_ym.split("-")]
    months: list[str] = []
    while (sy, sm) <= (ey, em):
        months.append(f"{sy:04d}-{sm:02d}")
        sm += 1
        if sm == 13:
            sy += 1
            sm = 1
    return months


def _pair_currency_sql(pair: str) -> str:
    p = pair.upper().replace("/", "").replace("_", "")
    currencies = {"GLOBAL", "ALL"}
    if len(p) >= 6:
        currencies.update([p[:3], p[3:6]])
    else:
        currencies.add(p)
    return ", ".join("'" + c.replace("'", "''") + "'" for c in sorted(currencies))


def verify_news_dataset(news_file: str, pairs: list[str]) -> None:
    """Verify month coverage for the pair-scoped historical news Parquet."""
    try:
        import duckdb
    except Exception as exc:
        raise SystemExit(f"DuckDB is required for --verify-news: {exc}") from exc

    path = Path(news_file)
    print("\n" + "=" * 72)
    print("  HISTORICAL NEWS VERIFICATION")
    print("-" * 72)
    print(f"  File : {path}")
    print(f"  Pairs: {', '.join(pairs)}")

    if not path.exists():
        raise SystemExit(f"News file not found: {path}")
    if path.stat().st_size == 0:
        raise SystemExit(f"News file is zero bytes: {path}")

    con = duckdb.connect(database=":memory:")
    rel = f"read_parquet('{str(path).replace(chr(39), chr(39) + chr(39))}')"

    total_rows, min_ts, max_ts = con.execute(
        f"""
        SELECT count(*), min(timestamp_utc), max(timestamp_utc)
        FROM {rel}
        WHERE timestamp_utc IS NOT NULL
        """
    ).fetchone()
    print(f"  Rows : {total_rows:,}")
    print(f"  Span : {min_ts} -> {max_ts}")

    ccy_rows = con.execute(
        f"""
        SELECT upper(coalesce(currency, '')) AS currency, count(*) AS rows
        FROM {rel}
        GROUP BY 1
        ORDER BY rows DESC
        """
    ).fetchall()
    print("  Currencies:")
    for currency, rows in ccy_rows[:20]:
        print(f"    {currency or '<blank>':>8}: {rows:,}")

    print("\n  Pair month coverage:")
    for pair in pairs:
        months = [
            row[0]
            for row in con.execute(
                f"""
                SELECT DISTINCT strftime(try_cast(timestamp_utc AS TIMESTAMP), '%Y-%m') AS ym
                FROM {rel}
                WHERE timestamp_utc IS NOT NULL
                  AND upper(coalesce(currency, '')) IN ({_pair_currency_sql(pair)})
                  AND try_cast(timestamp_utc AS TIMESTAMP) IS NOT NULL
                ORDER BY ym
                """
            ).fetchall()
            if row[0]
        ]
        if not months:
            print(f"    {pair}: no rows")
            continue
        expected = set(_month_range(months[0], months[-1]))
        missing = sorted(expected - set(months))
        pair_rows = con.execute(
            f"""
            SELECT count(*)
            FROM {rel}
            WHERE upper(coalesce(currency, '')) IN ({_pair_currency_sql(pair)})
            """
        ).fetchone()[0]
        print(f"    {pair}: rows={pair_rows:,} {months[0]}..{months[-1]} present={len(months)} missing={len(missing)}")
        if missing:
            shown = ", ".join(missing[:80])
            suffix = " ..." if len(missing) > 80 else ""
            print(f"      missing: {shown}{suffix}")
    con.close()
    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify data quality and auto-repair Dukascopy tick cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS, metavar="PAIR", help="FX pairs to verify")
    p.add_argument("--start", default=DEFAULT_START, help="Start date YYYY-MM-DD")
    p.add_argument("--end", default=DEFAULT_END, help="End date YYYY-MM-DD")
    p.add_argument("--full-day", action="store_true", help="Check all 24 hours (default: session only 07-17 UTC)")
    p.add_argument(
        "--tick-cache", default=DEFAULT_DUKASCOPY_CACHE_DIR, help="Root directory for raw Parquet tick cache"
    )
    p.add_argument("--fix", action="store_true", help="Auto-fix: remove duplicates + redownload missing/corrupt")
    p.add_argument("--min-ticks", type=int, default=0, help="Flag files with fewer than N ticks as too-small (0=off)")
    p.add_argument("--concurrency", type=int, default=8, help="Concurrent downloads for repair")
    p.add_argument("--request-delay", type=float, default=0.1, help="Delay between requests during repair")
    p.add_argument(
        "--verify-news", action="store_true", help="Verify historical news Parquet coverage instead of tick cache"
    )
    p.add_argument(
        "--news-file", default=DEFAULT_NEWS_FILE, help="Historical news Parquet to verify with --verify-news"
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    pairs = [p.upper().replace("/", "") for p in args.pairs]
    if args.verify_news:
        verify_news_dataset(args.news_file, pairs)
        return

    hours = list(range(0, 24)) if args.full_day else SESSION_HOURS

    run_verification(
        pairs=pairs,
        start=args.start,
        end=args.end,
        hours=hours,
        cache_dir=args.tick_cache,
        fix=args.fix,
        min_ticks=args.min_ticks,
        concurrency=args.concurrency,
        request_delay=args.request_delay,
    )


if __name__ == "__main__":
    main()
