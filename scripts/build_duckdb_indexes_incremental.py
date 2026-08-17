"""Incrementally build ticks indexes on a memory-constrained box.

A one-shot ``CREATE INDEX`` on the full 1.74B-row ``ticks`` table builds the
entire ART index in the DuckDB buffer pool, leaves zero free blocks, and fails
commit with "could not allocate block" for any ``memory_limit`` that fits in
RAM.

Instead we:
  1. create an empty ``ticks_rebuild`` with the same schema,
  2. create all three indexes on the empty table (instant, no memory pressure),
  3. copy the data over batch-by-batch per (pair, month) so each INSERT is a
     small transaction - DuckDB maintains the ART index incrementally, bounded
     by the batch size,
  4. swap ``ticks_rebuild`` back to ``ticks``.

Re-runnable: step 3 skips a finished (pair, month) window via the row count on
the rebuild table.

Usage:
    python scripts/build_duckdb_indexes_incremental.py \
        [--db data/store/forex_ticks.duckdb] \
        [--memory-limit 5GB] \
        [--threads 1]
"""

from __future__ import annotations

import argparse
import time

import duckdb

DEFAULT_DB = "data/store/forex_ticks.duckdb"
_SPILL_DIR = ".duckdb_spill"

INDEXES = [
    ("idx_ticks_pair", "CREATE INDEX IF NOT EXISTS idx_ticks_pair ON ticks_rebuild(pair)"),
    ("idx_ticks_ts", "CREATE INDEX IF NOT EXISTS idx_ticks_ts ON ticks_rebuild(timestamp)"),
    ("idx_ticks_pair_ts", "CREATE INDEX IF NOT EXISTS idx_ticks_pair_ts ON ticks_rebuild(pair, timestamp)"),
]


def build_incremental(db_path: str, memory_limit: str, threads: int) -> dict:
    output = db_path
    import pathlib

    spill = pathlib.Path(output).parent / _SPILL_DIR
    spill.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(output), config={"temp_directory": str(spill)})
    conn.execute(f"SET memory_limit = '{memory_limit}'")
    conn.execute(f"SET threads = {int(threads)}")
    conn.execute("SET preserve_insertion_order = false")

    # 1. Empty schema copy.
    conn.execute("CREATE TABLE IF NOT EXISTS ticks_rebuild AS SELECT * FROM ticks WHERE FALSE")

    # 2. Indexes on the empty table - instant.
    for name, sql in INDEXES:
        t0 = time.time()
        conn.execute(sql)
        conn.execute("CHECKPOINT")
        print(f"  {name}: created in {time.time() - t0:.1f}s", flush=True)

    # 3. Batch copy per (pair, month), skipping windows already loaded.
    pairs = [r[0] for r in conn.execute("SELECT DISTINCT pair FROM ticks ORDER BY pair").fetchall()]
    t_min = conn.execute("SELECT min(timestamp) FROM ticks").fetchone()[0]
    t_max = conn.execute("SELECT max(timestamp) FROM ticks").fetchone()[0]

    total = 0
    t_copy = time.time()
    for pair in pairs:
        for year in range(t_min.year, t_max.year + 1):
            for month in range(1, 13):
                lo = f"{year:04d}-{month:02d}-01"
                hi = f"{year:04d}-{month + 1:02d}-01" if month < 12 else f"{year + 1:04d}-01-01"
                have = conn.execute(
                    "SELECT count(*) FROM ticks_rebuild WHERE pair = ? AND timestamp >= ? AND timestamp < ?",
                    [pair, lo, hi],
                ).fetchone()[0]
                if have:
                    total += have
                    continue
                conn.execute(
                    "INSERT INTO ticks_rebuild SELECT * FROM ticks WHERE pair = ? AND timestamp >= ? AND timestamp < ?",
                    [pair, lo, hi],
                )
                added = conn.execute(
                    "SELECT count(*) FROM ticks_rebuild WHERE pair = ? AND timestamp >= ? AND timestamp < ?",
                    [pair, lo, hi],
                ).fetchone()[0]
                total += added
                conn.execute("CHECKPOINT")
                print(f"  {pair} {lo}: +{added:,} ({total:,} rows) {time.time() - t_copy:.0f}s", flush=True)

    # 4. Swap empty table for the migrated one.
    src_n = conn.execute("SELECT count(*) FROM ticks").fetchone()[0]
    rebuild_n = conn.execute("SELECT count(*) FROM ticks_rebuild").fetchone()[0]
    print(f"\nticks={src_n:,}  ticks_rebuild={rebuild_n:,}")
    if src_n and src_n == rebuild_n:
        conn.execute("DROP TABLE ticks")
        conn.execute("ALTER TABLE ticks_rebuild RENAME TO ticks")
        conn.execute("CHECKPOINT")
        print("swapped ticks_rebuild -> ticks", flush=True)
    else:
        print("row counts differ - NOT swapping", flush=True)

    conn.close()

    # 5. Report final index state.
    check = duckdb.connect(str(output), read_only=True)
    idx = [
        r[0]
        for r in check.execute(
            "SELECT index_name FROM duckdb_indexes() WHERE table_name = 'ticks' ORDER BY index_name"
        ).fetchall()
    ]
    print("indexes:", idx, flush=True)
    check.close()
    return {"indexes": idx}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Incremental (index-first) DuckDB index build")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--memory-limit", default="5GB")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    build_incremental(args.db, args.memory_limit, args.threads)
