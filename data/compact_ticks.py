"""
Build consolidated DuckDB from compact daily parquet partitions.

Replaces 189K individual files (raw hourly + compact daily) with a single
DuckDB database for 10-50× faster reads.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import duckdb


def build_forex_duckdb(
    compact_dir: str = "data/compact/dukascopy",
    output_path: str = "data/store/forex_ticks.duckdb",
    pairs: Optional[list[str]] = None,
    only_missing: bool = False,
) -> str:
    """
    Build single DuckDB database from compact daily parquet partitions.

    Args:
        compact_dir: Path to compact daily parquet files (Hive-partitioned)
        output_path: Where to write the .duckdb file
        pairs: Pairs to include (default: all found in compact_dir)
        only_missing: Only add pairs not already in the DuckDB (incremental)

    Returns:
        Path to the built DuckDB file
    """
    compact = Path(compact_dir)
    granularity = "daily"  # compact files are daily

    # Discover available pairs
    pair_dir = compact / f"granularity={granularity}"
    if not pair_dir.exists():
        raise FileNotFoundError(f"Compact directory not found: {pair_dir}")

    all_pairs = sorted(
        p.name.replace("pair=", "")
        for p in pair_dir.iterdir()
        if p.is_dir() and p.name.startswith("pair=")
    )

    if pairs:
        pairs = [p for p in pairs if p in set(all_pairs)]
        if not pairs:
            raise ValueError(f"None of the requested pairs found in {pair_dir}")
    else:
        pairs = all_pairs

    print(f"[DuckDB] Building from {len(pairs)} pairs: {', '.join(pairs)}")
    print(f"[DuckDB] Source: {compact_dir}")
    print(f"[DuckDB] Target: {output_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(output_path))

    # Memory budget: 12GB for sort
    conn.execute("SET memory_limit = '12GB'")
    conn.execute("SET threads = 8")

    # Build glob for parquet files
    glob_path = str(compact / f"granularity={granularity}" / "pair=*" / "year=*" / "month=*" / "day=*" / "ticks.parquet")

    if only_missing and Path(output_path).exists():
        # Check which pairs are already in the DB
        existing = set(
            row[0]
            for row in conn.execute("SELECT DISTINCT pair FROM ticks").fetchall()
        )
        pairs = [p for p in pairs if p not in existing]
        if not pairs:
            print("[DuckDB] All pairs already present — nothing to do")
            conn.close()
            return output_path
        print(f"[DuckDB] Incremental: adding {len(pairs)} new pairs")

        # Create temp table for new pairs, then merge
        pair_filter = ", ".join(f"'{p}'" for p in pairs)
        conn.execute(f"""
            CREATE TEMP TABLE new_ticks AS
            SELECT 
                timestamp,
                bid, ask, mid, spread, volume,
                pair, source
            FROM read_parquet('{glob_path}', hive_partitioning=true)
            WHERE pair IN ({pair_filter})
            ORDER BY pair, timestamp
        """)
        n = conn.execute("SELECT COUNT(*) FROM new_ticks").fetchone()[0]
        print(f"[DuckDB] New ticks: {n:,} rows")
        conn.execute("INSERT INTO ticks SELECT * FROM new_ticks")
        conn.execute("DROP TABLE new_ticks")
    else:
        # Full build
        conn.execute(f"""
            CREATE TABLE ticks AS
            SELECT 
                timestamp,
                bid, ask, mid, spread, volume,
                pair, source
            FROM read_parquet('{glob_path}', hive_partitioning=true)
            ORDER BY pair, timestamp
        """)

        # Create indexes
        conn.execute("CREATE INDEX idx_ticks_pair ON ticks(pair)")
        conn.execute("CREATE INDEX idx_ticks_ts ON ticks(timestamp)")
        conn.execute("CREATE INDEX idx_ticks_pair_ts ON ticks(pair, timestamp)")

    # Report stats
    stats = conn.execute("""
        SELECT pair,
               COUNT(*) as ticks,
               MIN(timestamp) as ts_start,
               MAX(timestamp) as ts_end
        FROM ticks
        GROUP BY pair
        ORDER BY pair
    """).fetchall()

    total = conn.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]

    print(f"\n[DuckDB] Complete — {total:,} ticks across {len(stats)} pairs")
    for row in stats:
        print(f"  {row[0]:8s}: {row[1]:>12,} ticks  ({row[2][:10]} → {row[3][:10]})")

    conn.close()

    # Report file size
    size_gb = os.path.getsize(output_path) / (1024**3)
    print(f"\n[DuckDB] File size: {size_gb:.2f} GB")
    print(f"[DuckDB] Path: {output_path}")

    return output_path


def verify_duckdb(
    db_path: str = "data/store/forex_ticks.duckdb",
    sample_queries: bool = True,
) -> dict:
    """Verify DuckDB integrity and performance."""
    import time

    if not Path(db_path).exists():
        return {"status": "MISSING", "message": f"File not found: {db_path}"}

    conn = duckdb.connect(str(db_path))
    report = {"status": "OK", "issues": [], "stats": {}, "performance": {}}

    # Basic integrity
    total = conn.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]
    if total == 0:
        report["issues"].append("Table 'ticks' is empty")
        report["status"] = "ERROR"
    else:
        report["stats"]["total_ticks"] = total

    # Check for bad data
    nulls = conn.execute("""
        SELECT COUNT(*) FROM ticks 
        WHERE bid IS NULL OR ask IS NULL OR bid > ask OR bid <= 0 OR ask <= 0
    """).fetchone()[0]
    if nulls > 0:
        report["issues"].append(f"{nulls} rows with bad bid/ask values")
    else:
        report["stats"]["bad_rows"] = 0

    # Pair coverage
    pairs = conn.execute("""
        SELECT pair, COUNT(*) as n, COUNT(DISTINCT DATE_TRUNC('year', timestamp)) as years
        FROM ticks GROUP BY pair ORDER BY pair
    """).fetchall()
    report["stats"]["pairs"] = {
        row[0]: {"ticks": row[1], "years": row[2]}
        for row in pairs
    }

    # Performance benchmark
    if sample_queries:
        import time
        # Benchmark: query 1 month of data
        start = time.time()
        conn.execute("""
            SELECT * FROM ticks
            WHERE pair = 'EURUSD' AND timestamp BETWEEN '2024-01-01' AND '2024-02-01'
        """).fetchall()
        query_ms = (time.time() - start) * 1000
        report["performance"]["month_query_ms"] = f"{query_ms:.1f}"

        # Benchmark: query 1 year
        start = time.time()
        result = conn.execute("""
            SELECT * FROM ticks
            WHERE pair = 'EURUSD' AND timestamp_utc BETWEEN '2024-01-01' AND '2025-01-01'
        """).fetchall()
        query_ms = (time.time() - start) * 1000
        report["performance"]["year_query_ms"] = f"{query_ms:.1f}"

    conn.close()
    return report


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build consolidated DuckDB from compact parquet")
    ap.add_argument("--compact-dir", default="data/compact/dukascopy")
    ap.add_argument("--output", default="data/store/forex_ticks.duckdb")
    ap.add_argument("--pairs", nargs="+", default=None,
                    help="Pairs to include (default: all)")
    ap.add_argument("--incremental", action="store_true",
                    help="Only add pairs not already in the DuckDB")
    ap.add_argument("--verify", action="store_true",
                    help="Verify after build")

    args = ap.parse_args()

    build_forex_duckdb(
        compact_dir=args.compact_dir,
        output_path=args.output,
        pairs=args.pairs,
        only_missing=args.incremental,
    )

    if args.verify:
        report = verify_duckdb(args.output)
        print(f"\nVerification: {report['status']}")
        for issue in report.get("issues", []):
            print(f"  ⚠ {issue}")
        if report.get("performance"):
            perf = report["performance"]
            print(f"  Month query: {perf.get('month_query_ms', 'N/A')}ms")
            print(f"  Year query: {perf.get('year_query_ms', 'N/A')}ms")