"""One-off audit probe for the dataset-building audit."""
import ast
import os
import sys
from pathlib import Path

print("=" * 70)
print("1) BUILTIN_FEATURES count")
print("=" * 70)
with open("data/feature_definitions.py", encoding="utf-8", errors="replace") as f:
    tree = ast.parse(f.read())
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id == "BUILTIN_FEATURES":
                print(f"   BUILTIN_FEATURES entries: {len(node.value.elts)}")
                # also check for any other *FEATURES lists
    if isinstance(node, ast.AnnAssign):
        if isinstance(node.target, ast.Name) and node.target.id == "BUILTIN_FEATURES":
            print(f"   BUILTIN_FEATURES entries (AnnAssign): {len(node.value.elts)}")

print()
print("=" * 70)
print("2) DuckDB hour distribution for EURUSD Jan 2008")
print("=" * 70)
import duckdb
conn = duckdb.connect("data/store/forex_ticks.duckdb", read_only=True)
rows = conn.execute(
    "SELECT EXTRACT(hour FROM timestamp) as h, COUNT(*) as c "
    "FROM ticks WHERE pair='EURUSD' "
    "AND timestamp >= TIMESTAMP '2008-01-01' "
    "AND timestamp < TIMESTAMP '2008-02-01' "
    "GROUP BY h ORDER BY h"
).fetchall()
print("   DuckDB EURUSD Jan 2008 hour distribution:")
for h, c in rows:
    print(f"   hour {h:2d}: {c:>10,} ticks")
total = sum(c for _, c in rows)
print(f"   ---- Total: {total:,} ticks")
conn.close()

print()
print("=" * 70)
print("3) Parquet cache hour distribution for EURUSD Jan 2008")
print("=" * 70)
import polars as pl
# Hive partitioning is zero-padded, so month=01 day=01 etc.
files = sorted(Path("data/compact/dukascopy/granularity=daily/pair=EURUSD/year=2008/month=01/day=01/ticks.parquet").parent.glob("day=*/ticks.parquet"))
if not files:
    # try glob across all days
    files = sorted(Path("data/compact/dukascopy/granularity=daily/pair=EURUSD/year=2008/month=01").rglob("ticks.parquet"))
print(f"   Found {len(files)} parquet files for EURUSD/2008/01")
if files:
    df = pl.scan_parquet([str(f) for f in files]).collect()
    hours = (
        df.with_columns(pl.col("timestamp").dt.hour().alias("hour"))
        .group_by("hour")
        .agg(pl.len().alias("count"))
        .sort("hour")
    )
    print("   Parquet cache EURUSD Jan 2008 hour distribution:")
    print(hours)
    print(f"   Total rows: {df.height:,}")

print()
print("=" * 70)
print("4) Check DuckDB for presence of non-session hours (outside 07-17)")
print("=" * 70)
conn = duckdb.connect("data/store/forex_ticks.duckdb", read_only=True)
off = conn.execute(
    "SELECT EXTRACT(hour FROM timestamp) as h, COUNT(*) as c "
    "FROM ticks WHERE pair='EURUSD' "
    "AND timestamp >= TIMESTAMP '2008-01-01' "
    "AND timestamp < TIMESTAMP '2008-02-01' "
    "AND (EXTRACT(hour FROM timestamp) < 7 OR EXTRACT(hour FROM timestamp) >= 18) "
    "GROUP BY h ORDER BY h"
).fetchall()
if off:
    print("   Non-session hours FOUND in DuckDB:")
    for h, c in off:
        print(f"   hour {h:2d}: {c:>10,} ticks")
else:
    print("   No non-session hours found — DuckDB Jan 2008 EURUSD is session-only (07-17 UTC)")
conn.close()
