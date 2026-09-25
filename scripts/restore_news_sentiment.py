"""Restore FinBERT sentiment scores into the historical news archive.

The 2026-09-05 rewrite of ``historical_news_combined.parquet`` dropped the
``sentiment_score`` column (~20M headlines scored on 2026-08-07), so every
dataset build got constant-zero sentiment. The scores still exist in
``historical_news_combined.raw_backup.parquet``. This joins them back on the
trimmed headline text, the same key the original scoring pipeline used, and
writes the result atomically. URL-only headlines stay unscored
(docs/NEWS_DATA_GUIDE.md section 0).

    python scripts/restore_news_sentiment.py --dry-run
    python scripts/restore_news_sentiment.py
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import duckdb

NEWS_DIR = Path("data/raw/news")
CURRENT = NEWS_DIR / "historical_news_combined.parquet"
SOURCE = NEWS_DIR / "historical_news_combined.raw_backup.parquet"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--current", type=Path, default=CURRENT)
    ap.add_argument("--source", type=Path, default=SOURCE, help="Parquet that still has sentiment_score")
    ap.add_argument("--dry-run", action="store_true", help="Report coverage only; write nothing")
    ap.add_argument("--memory-limit", default="3GB")
    args = ap.parse_args()

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{args.memory_limit}'")
    con.execute("SET preserve_insertion_order=false")
    cur, src = args.current.as_posix(), args.source.as_posix()

    cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{cur}')").fetchall()]
    if "sentiment_score" in cols:
        print(f"{cur} already has sentiment_score; nothing to do.")
        return 0

    # One score per headline (identical text got identical FinBERT scores).
    con.execute(
        f"""
        CREATE TEMP TABLE scores AS
        SELECT trim(CAST(headline AS VARCHAR)) AS h, any_value(sentiment_score) AS sentiment_score
        FROM read_parquet('{src}')
        WHERE sentiment_score IS NOT NULL
          AND lower(substring(trim(CAST(headline AS VARCHAR)), 1, 5)) NOT IN ('http:', 'https')
        GROUP BY 1
        """
    )
    joined = f"""
        SELECT n.*, s.sentiment_score
        FROM read_parquet('{cur}') n
        LEFT JOIN scores s ON trim(CAST(n.headline AS VARCHAR)) = s.h
    """
    total, scored = con.execute(
        f"SELECT count(*), count(sentiment_score) FROM ({joined})"
    ).fetchone()
    print(f"rows={total:,}  scored={scored:,} ({scored / max(total, 1):.1%})")
    by_year = con.execute(
        f"""SELECT year(timestamp_utc) y, count(*) n, count(sentiment_score) s
            FROM ({joined}) GROUP BY 1 ORDER BY 1"""
    ).fetchall()
    for y, n, s in by_year:
        print(f"  {y}: {s:>10,} / {n:>10,} scored ({s / max(n, 1):.0%})")
    if args.dry_run:
        return 0

    tmp = args.current.with_suffix(".restore_tmp.parquet")
    con.execute(f"COPY ({joined}) TO '{tmp.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    backup = args.current.with_suffix(".pre_sentiment_restore.bak")
    shutil.copy2(args.current, backup)
    os.replace(tmp, args.current)
    print(f"wrote {args.current} (previous version kept at {backup})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
