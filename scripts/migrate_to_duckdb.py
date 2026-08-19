#!/usr/bin/env python3
"""Incremental, auto-refreshing migration from compact parquet to DuckDB.

Discovers pairs under ``data/compact/dukascopy/granularity=daily/pair=*`` and
ingests them into a single consolidated ``data/store/forex_ticks.duckdb``.

The migration is idempotent and safe to re-run:
  * Pairs already present AND whose compact parquet is not newer than the last
    migration are skipped (fast no-op).
  * Pairs that were re-downloaded (compact parquet changed) are refreshed via
    DELETE + INSERT.
  * New pairs are appended incrementally.

Run ``--force`` to rebuild every pair, or ``--dry-run`` to see planned actions
without touching the database.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import duckdb

_DEFAULT_COMPACT_DIR = "data/compact/dukascopy"
_DEFAULT_OUTPUT = "data/store/forex_ticks.duckdb"
_MANIFEST_NAME = ".migrate_manifest.json"
_SPILL_DIR_NAME = ".duckdb_spill"

GRANULARITY = "daily"


def _pair_parquet_max_mtime(pair_dir: Path) -> float:
    """Newest mtime of any ticks.parquet under a compacted pair directory."""
    latest = 0.0
    for f in pair_dir.rglob("ticks.parquet"):
        try:
            latest = max(latest, f.stat().st_mtime)
        except OSError:
            continue
    return latest


def _load_manifest(output: Path) -> dict[str, float]:
    path = output.parent / _MANIFEST_NAME
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        return {str(k): float(v) for k, v in data.items() if isinstance(v, (int, float))}
    except Exception:
        return {}


def _save_manifest(output: Path, manifest: dict[str, float]) -> None:
    path = output.parent / _MANIFEST_NAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _cleanup_temp_storage(output: Path) -> None:
    """Remove stale DuckDB spill dirs left next to the database by older runs."""
    leftover_tmp = output.parent / f"{output.name}.tmp"
    if leftover_tmp.is_dir():
        shutil.rmtree(leftover_tmp, ignore_errors=True)
    spill_dir = output.parent / _SPILL_DIR_NAME
    if spill_dir.is_dir():
        shutil.rmtree(spill_dir, ignore_errors=True)


def _discover_pairs(compact: Path) -> list[str]:
    pair_dir = compact / f"granularity={GRANULARITY}"
    if not pair_dir.exists():
        print(f"ERROR: Compact dir not found: {pair_dir}")
        print("Run first: python scripts/download_data.py")
        return []
    all_pairs = sorted(
        p.name.replace("pair=", "") for p in pair_dir.iterdir() if p.is_dir() and p.name.startswith("pair=")
    )
    return all_pairs


def migrate_to_duckdb(
    compact_dir: str = _DEFAULT_COMPACT_DIR,
    output_path: str = _DEFAULT_OUTPUT,
    pairs: list[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    """Migrate compact daily parquet into a single consolidated DuckDB file.

    Returns a machine-readable summary for programmatic callers.
    """
    compact = Path(compact_dir)
    all_pairs = _discover_pairs(compact)
    if not all_pairs:
        sys.exit(1)

    if pairs:
        requested = [p.upper().replace("/", "") for p in pairs]
        pairs = [p for p in requested if p in set(all_pairs)]
        if not pairs:
            print(f"No requested pairs found in {compact / f'granularity={GRANULARITY}'}")
            sys.exit(1)
    else:
        pairs = all_pairs

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(output)

    # Get DB pairs
    db_pairs = set()
    if output.exists():
        try:
            with duckdb.connect(str(output), read_only=True) as c:
                db_pairs = {r[0] for r in c.execute("SELECT DISTINCT pair FROM ticks").fetchall()}
        except Exception:
            pass

    # Work out the planned action per pair without touching the main connection yet.
    planned: dict[str, str] = {}
    pair_max_mtimes: dict[str, float] = {}
    for pair in pairs:
        src_mtime = _pair_parquet_max_mtime(compact / f"granularity={GRANULARITY}" / f"pair={pair}")
        pair_max_mtimes[pair] = src_mtime
        last = manifest.get(pair)

        if pair not in db_pairs:
            planned[pair] = "add"
        elif last is None:
            # Pair is in DB but missing from manifest (e.g. legacy/corrupt) - refresh
            planned[pair] = "refresh"
        elif src_mtime > last:
            planned[pair] = "refresh"
        else:
            planned[pair] = "skip"

    if force:
        planned = dict.fromkeys(pairs, "refresh")

    n_skip = sum(1 for v in planned.values() if v == "skip")
    print(f"{'=' * 60}")
    print(f"  DuckDB migration  ({len(pairs)} pairs, {n_skip} up-to-date)")
    print(f"  Compact : {compact / f'granularity={GRANULARITY}'}")
    print(f"  Output  : {output}")
    print(f"  Mode    : {'DRY-RUN' if dry_run else ('FORCE' if force else 'INCREMENTAL')}")
    print(f"{'=' * 60}")

    if dry_run:
        for pair in pairs:
            print(f"  {pair:<8} -> {planned[pair]:>8}  (parquet mtime {pair_max_mtimes[pair]:.0f})")
        print(f"\nDry run complete. {n_skip} skipped, {sum(1 for v in planned.values() if v != 'skip')} to process.")
        return {"dry_run": True, "pairs": planned}

    if not force and n_skip == len(pairs):
        # Nothing changed - short-circuit without touching the (possibly large) DB.
        for pair in pairs:
            print(f"  {pair:<8} - up to date")
        print("\nAll pairs up to date - nothing to do.")
        return {
            "db_path": str(output),
            "up_to_date": True,
            "pairs": {p: {"action": "skip", "ticks": 0} for p in pairs},
        }

    _cleanup_temp_storage(output)
    spill_dir = output.parent / _SPILL_DIR_NAME
    spill_dir.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(output), config={"temp_directory": str(spill_dir)})
    conn.execute("SET memory_limit = '6GB'")
    conn.execute("SET threads = 1")
    conn.execute("SET preserve_insertion_order = false")

    try:
        table_exists = bool(
            conn.execute(
                "SELECT COUNT(*) FROM duckdb_tables() WHERE schema_name='main' AND table_name='ticks'"
            ).fetchone()[0]
        )

        if force:
            conn.execute("DROP TABLE IF EXISTS ticks")
            table_exists = False

        if not table_exists:
            first_pair = next((p for p in pairs if planned[p] != "skip"), pairs[0])
            first_glob = _pair_glob(compact, first_pair)
            conn.execute(f"""
                CREATE TABLE ticks AS
                SELECT timestamp, bid, ask, mid, spread, volume, pair, source
                FROM read_parquet('{first_glob}', hive_partitioning=true)
                WHERE 1 = 0
            """)

        summary: dict[str, dict] = {}
        for i, pair in enumerate(pairs, 1):
            action = planned[pair]
            if action == "skip":
                print(f"[{i}/{len(pairs)}] {pair} - up to date")
                summary[pair] = {"action": "skip", "ticks": 0}
                manifest[pair] = pair_max_mtimes[pair]
                continue

            _pair_glob(compact, pair)
            t0 = time.time()
            print(f"[{i}/{len(pairs)}] {pair} ({action})... ", end="", flush=True)

            before = conn.execute("SELECT COUNT(*) FROM ticks WHERE pair = ?", [pair]).fetchone()[0]

            if table_exists and before > 0:
                conn.execute("DELETE FROM ticks WHERE pair = ?", [pair])

            pair_path = compact / f"granularity={GRANULARITY}" / f"pair={pair}"
            year_dirs = sorted([d.name for d in pair_path.iterdir() if d.is_dir() and d.name.startswith("year=")])

            for y_dir in year_dirs:
                month_dirs = sorted(
                    [d.name for d in (pair_path / y_dir).iterdir() if d.is_dir() and d.name.startswith("month=")]
                )
                for m_dir in month_dirs:
                    m_glob = str(pair_path / y_dir / m_dir / "day=*" / "ticks.parquet")
                    conn.execute(f"""
                        INSERT INTO ticks
                        SELECT timestamp, bid, ask, mid, spread, volume, pair, source
                        FROM read_parquet('{m_glob}', hive_partitioning=true)
                    """)
                # Force WAL flush to disk to prevent OOM
                conn.execute("CHECKPOINT")

            after = conn.execute("SELECT COUNT(*) FROM ticks WHERE pair = ?", [pair]).fetchone()[0]
            elapsed = time.time() - t0
            print(f"{after:>12,} ticks ({elapsed:.0f}s)")

            manifest[pair] = pair_max_mtimes[pair]
            summary[pair] = {"action": action, "ticks": after}

        conn.execute("CHECKPOINT")
    finally:
        conn.close()
        _save_manifest(output, manifest)
        _cleanup_temp_storage(output)

    # Indexes
    print("\nCreating indexes...")
    conn = duckdb.connect(str(output), config={"temp_directory": str(spill_dir)})
    conn.execute("SET memory_limit = '12GB'")
    conn.execute("SET threads = 1")
    try:
        for sql, name in [
            ("CREATE INDEX IF NOT EXISTS idx_ticks_pair ON ticks(pair)", "pair"),
            ("CREATE INDEX IF NOT EXISTS idx_ticks_ts ON ticks(timestamp)", "ts"),
            ("CREATE INDEX IF NOT EXISTS idx_ticks_pair_ts ON ticks(pair, timestamp)", "pair_ts"),
        ]:
            t0 = time.time()
            conn.execute(sql)
            # Flush each index build to disk so memory is released back to the OS
            # before the next build starts. Without this, three concurrent in-flight
            # index builds on 1.74B rows push a 16GB box into swap death.
            conn.execute("CHECKPOINT")
            print(f"  idx_{name}: {time.time() - t0:.0f}s")
        stats = conn.execute(
            """
            SELECT pair, COUNT(*),
                   (MIN(timestamp) AT TIME ZONE 'UTC')::DATE::VARCHAR,
                   (MAX(timestamp) AT TIME ZONE 'UTC')::DATE::VARCHAR
            FROM ticks GROUP BY pair ORDER BY pair
            """
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]
    finally:
        conn.close()
        _cleanup_temp_storage(output)

    size_gb = output.stat().st_size / 1e9
    print(f"\n{'=' * 50}")
    print(f"Migration complete: {total:,} ticks in {len(stats)} pairs")
    print(f"Size: {size_gb:.2f} GB")
    print(f"Path: {output}")
    for right in stats:
        print(f"  {right[0]:8s}: {right[1]:>12,}  ({right[2]} → {right[3]})")

    return {
        "db_path": str(output),
        "total_ticks": total,
        "pairs": summary,
    }


def build_indexes(
    output_path: str = _DEFAULT_OUTPUT,
    *,
    memory_limit: str = "3GB",
    threads: int = 1,
    only: list[str] | None = None,
    drop_existing: bool = False,
) -> dict:
    """Build (or rebuild) the ticks indexes on an existing consolidated DB.

    Safe to run after `migrate_to_duckdb` has finished - does not touch the row
    data, only the secondary indexes. Designed to be re-runnable: an index that
    already exists is skipped unless `drop_existing=True`.

    Memory strategy
    ---------------
    Each index on a 1.74B-row table is built in a single DuckDB statement that
    holds a duplicate of the index column(s) plus the row IDs in memory until it
    can be flushed. Building all three in one connection can triple that working
    set. To stay within RAM on a 16GB box:

      * One connection per index (released + spilled before the next starts)
      * `memory_limit` capped low (default 3GB) - DuckDB will spill to
        `temp_directory` rather than OOM
      * `threads = 1` to bound the parallel-sort memory
      * `CHECKPOINT` + `gc()` after each build to release memory back to the OS
        before starting the next index

    Parameters
    ----------
    output_path : path to the consolidated forex_ticks.duckdb
    memory_limit : DuckDB memory_limit, e.g. '3GB', '2GB'
    threads : DuckDB thread count for the build (1 = lowest peak RAM)
    only : list of index names to build - 'pair', 'ts', 'pair_ts'.
           None = build all three in the recommended order.
    drop_existing : if True, DROP INDEX first so a previously-failed/partial
                    index is rebuilt from scratch instead of being skipped.
    """
    output = Path(output_path)
    if not output.exists():
        print(f"[build_indexes] DB not found: {output}")
        sys.exit(1)

    spill_dir = output.parent / _SPILL_DIR_NAME
    spill_dir.mkdir(parents=True, exist_ok=True)

    all_indexes = [
        ("idx_ticks_pair", "CREATE INDEX IF NOT EXISTS idx_ticks_pair    ON ticks(pair)", "pair"),
        ("idx_ticks_ts", "CREATE INDEX IF NOT EXISTS idx_ticks_ts       ON ticks(timestamp)", "ts"),
        ("idx_ticks_pair_ts", "CREATE INDEX IF NOT EXISTS idx_ticks_pair_ts  ON ticks(pair, timestamp)", "pair_ts"),
    ]
    if only:
        wanted = set(only)
        all_indexes = [i for i in all_indexes if i[2] in wanted]
        if not all_indexes:
            print(f"[build_indexes] no matching indexes for --only {only}")
            sys.exit(1)

    print(f"\nBuilding indexes on {output}")
    print(f"  memory_limit={memory_limit}  threads={threads}  spill_dir={spill_dir}")
    print(f"  indexes: {[i[2] for i in all_indexes]}  drop_existing={drop_existing}")

    results = []
    for idx_name, create_sql, short_name in all_indexes:
        # Open a fresh connection per index so the previous index's working
        # memory is fully released before we start the next one.
        conn = duckdb.connect(str(output), config={"temp_directory": str(spill_dir)})
        conn.execute(f"SET memory_limit = '{memory_limit}'")
        conn.execute(f"SET threads = {int(threads)}")
        # Spill-to-disk for sorts rather than OOM-kill. DuckDB's default is
        # temp_directory/memory_limit, this just makes the policy explicit.
        conn.execute("SET preserve_insertion_order = false")
        try:
            if drop_existing:
                conn.execute(f"DROP INDEX IF EXISTS {idx_name}")
            t0 = time.time()
            conn.execute(create_sql)
            conn.execute("CHECKPOINT")  # flush + free
            elapsed = time.time() - t0
            print(f"  idx_{short_name}: {elapsed:.0f}s")
            results.append({"name": idx_name, "elapsed_s": round(elapsed, 1)})
        except Exception as e:
            print(f"  idx_{short_name}: FAILED - {e}")
            results.append({"name": idx_name, "error": str(e)})
        finally:
            conn.close()
            _cleanup_temp_storage(output)

    # Summary
    print("\nFinal index state:")
    conn = duckdb.connect(str(output), read_only=True)
    try:
        idx_rows = conn.execute(
            "SELECT index_name FROM duckdb_indexes() WHERE table_name = 'ticks' ORDER BY index_name"
        ).fetchall()
        for r in idx_rows:
            print(f"  {r[0]}")
    finally:
        conn.close()
    return {"db_path": str(output), "indexes": results}


def _pair_glob(compact: Path, pair: str) -> str:
    return str(
        compact / f"granularity={GRANULARITY}" / f"pair={pair}" / "year=*" / "month=*" / "day=*" / "ticks.parquet"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Migrate parquet ticks to DuckDB")
    ap.add_argument("--compact-dir", default=_DEFAULT_COMPACT_DIR)
    ap.add_argument("--output", default=_DEFAULT_OUTPUT)
    ap.add_argument("--pairs", nargs="+", default=None)
    ap.add_argument("--force", action="store_true", help="Rebuild every pair")
    ap.add_argument("--dry-run", action="store_true", help="Show planned actions without touching the DB")
    ap.add_argument(
        "--indexes-only", action="store_true", help="Skip ingestion and only build the ticks indexes on an existing DB"
    )
    ap.add_argument("--index-memory", default="3GB", help="DuckDB memory_limit for index builds (default 3GB)")
    ap.add_argument(
        "--index-threads",
        type=int,
        default=1,
        help="DuckDB thread count for index builds (default 1 = lowest peak RAM)",
    )
    ap.add_argument(
        "--index-only",
        action="append",
        choices=["pair", "ts", "pair_ts"],
        help="Build only the named index(es); may be repeated. Default = all three",
    )
    ap.add_argument(
        "--rebuild-indexes",
        action="store_true",
        help="Drop existing indexes before rebuilding (useful after a failed/partial build)",
    )
    args = ap.parse_args()

    if args.indexes_only:
        build_indexes(
            output_path=args.output,
            memory_limit=args.index_memory,
            threads=args.index_threads,
            only=args.index_only,
            drop_existing=args.rebuild_indexes,
        )
    else:
        migrate_to_duckdb(
            compact_dir=args.compact_dir,
            output_path=args.output,
            pairs=args.pairs,
            force=args.force,
            dry_run=args.dry_run,
        )
