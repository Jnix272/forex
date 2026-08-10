#!/usr/bin/env python3
"""One-time migration from raw/compact parquet to consolidated DuckDB."""

import sys
import time
from pathlib import Path

import duckdb


def migrate_to_duckdb(
    compact_dir: str = "data/compact/dukascopy",
    output_path: str = "data/store/forex_ticks.duckdb",
    pairs: list[str] = None,
) -> str:
    """Migrate compact daily parquet to single DuckDB file."""
    compact = Path(compact_dir)
    granularity = "daily"

    # Discover available pairs
    pair_dir = compact / f"granularity={granularity}"
    if not pair_dir.exists():
        print(f"ERROR: Compact dir not found: {pair_dir}")
        print("Run first: python scripts/download_data.py")
        sys.exit(1)

    all_pairs = sorted(
        p.name.replace("pair=", "")
        for p in pair_dir.iterdir()
        if p.is_dir() and p.name.startswith("pair=")
    )

    if pairs:
        pairs = [p for p in pairs if p in set(all_pairs)]
    else:
        pairs = all_pairs

    if not pairs:
        print(f"No pairs found in {pair_dir}")
        sys.exit(1)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Check if DB exists
    if output.exists():
        conn = duckdb.connect(str(output))
        existing = set(row[0] for row in conn.execute("SELECT DISTINCT pair FROM ticks").fetchall())
        conn.close()
        if set(pairs).issubset(existing):
            print(f"All {len(pairs)} pairs already in DuckDB. Done.")
            return str(output)
        print(f"Incremental: {len(existing)} pairs present, adding {len(set(pairs) - existing)}")
    else:
        existing = set()

    conn = duckdb.connect(str(output))
    conn.execute("SET memory_limit = '8GB'")
    conn.execute("SET threads = 4")

    total_before = 0
    try:
        total_before = conn.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]
    except Exception:
        pass

    for i, pair in enumerate(pairs, 1):
        if pair in existing:
            continue

        glob_path = str(compact / f"granularity={granularity}" / f"pair={pair}" / "year=*" / "month=*" / "day=*" / "ticks.parquet")

        t0 = time.time()
        print(f"[{i}/{len(pairs)}] {pair}... ", end="", flush=True)

        if total_before == 0 and not existing:
            conn.execute(f"""
                CREATE TABLE ticks AS
                SELECT timestamp, bid, ask, mid, spread, volume, pair, source
                FROM read_parquet('{glob_path}', hive_partitioning=true)
                ORDER BY timestamp
            """)
        else:
            conn.execute(f"""
                INSERT INTO ticks
                SELECT timestamp, bid, ask, mid, spread, volume, pair, source
                FROM read_parquet('{glob_path}', hive_partitioning=true)
            """)

        new_ticks = conn.execute("SELECT COUNT(*) FROM ticks").fetchone()[0] - total_before
        total_before += new_ticks
        elapsed = time.time() - t0
        print(f"{new_ticks:>12,} ticks ({elapsed:.0f}s)")

    # Index
    print("\nCreating indexes...")
    for sql, name in [
        ("CREATE INDEX IF NOT EXISTS idx_ticks_pair ON ticks(pair)", "pair"),
        ("CREATE INDEX IF NOT EXISTS idx_ticks_ts ON ticks(timestamp)", "ts"),
        ("CREATE INDEX IF NOT EXISTS idx_ticks_pair_ts ON ticks(pair, timestamp)", "pair_ts"),
    ]:
        t0 = time.time()
        conn.execute(sql)
        print(f"  idx_{name}: {time.time() - t0:.0f}s")

    # Report
    stats = conn.execute("""
        SELECT pair, COUNT(*), MIN(timestamp)::VARCHAR[:10], MAX(timestamp)::VARCHAR[:10]
        FROM ticks GROUP BY pair ORDER BY pair
    """).fetchall()
    total = conn.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]
    conn.close()

    size_gb = output.stat().st_size / 1e9
    print(f"\n{'='*50}")
    print(f"Migration complete: {total:,} ticks in {len(stats)} pairs")
    print(f"Size: {size_gb:.2f} GB")
    print(f"Path: {output}")
    for s in stats:
        print(f"  {s[0]:8s}: {s[1]:>12,}  ({s[2]} → {s[3]})")

    return str(output)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Migrate parquet ticks to DuckDB")
    ap.add_argument("--compact-dir", default="data/compact/dukascopy")
    ap.add_argument("--output", default="data/store/forex_ticks.duckdb")
    ap.add_argument("--pairs", nargs="+", default=None)
    args = ap.parse_args()

    migrate_to_duckdb(
        compact_dir=args.compact_dir,
        output_path=args.output,
        pairs=args.pairs,
    )