"""Build DuckDB pair-by-pair for visibility."""

from pathlib import Path

import duckdb

compact = Path("data/compact/dukascopy")

# All available pairs
pairs = sorted(
    p.name.replace("pair=", "")
    for p in (compact / "granularity=daily").iterdir()
    if p.is_dir() and p.name.startswith("pair=")
)

output = Path("data/store/forex_ticks.duckdb")
output.parent.mkdir(parents=True, exist_ok=True)

conn = duckdb.connect(str(output))
conn.execute("SET memory_limit = '8GB'")
conn.execute("SET threads = 4")

total_ticks = 0

for i, pair in enumerate(pairs, 1):
    glob_path = str(compact / f"granularity=daily/pair={pair}/year=*/month=*/day=*/ticks.parquet")

    print(f"[{i}/{len(pairs)}] {pair}... ", end="", flush=True)

    if i == 1:
        # First pair: CREATE table
        conn.execute(f"""
            CREATE TABLE ticks AS
            SELECT timestamp, bid, ask, mid, spread, volume, pair, source
            FROM read_parquet('{glob_path}', hive_partitioning=true)
            ORDER BY timestamp
        """)
    else:
        # Subsequent pairs: INSERT
        conn.execute(f"""
            INSERT INTO ticks
            SELECT timestamp, bid, ask, mid, spread, volume, pair, source
            FROM read_parquet('{glob_path}', hive_partitioning=true)
            ORDER BY timestamp
        """)

    count = conn.execute("SELECT COUNT(*) FROM ticks").fetchone()[0] - total_ticks
    total_ticks += count
    print(f"{count:>12,} ticks")

# Create indexes
print("\nCreating indexes...")
conn.execute("CREATE INDEX idx_ticks_pair ON ticks(pair)")
print("  idx_pair done")
conn.execute("CREATE INDEX idx_ticks_ts ON ticks(timestamp)")
print("  idx_ts done")
conn.execute("CREATE INDEX idx_ticks_pair_ts ON ticks(pair, timestamp)")
print("  idx_pair_ts done")

# Report
print("\n=== DuckDB Complete ===")
stats = conn.execute("""
    SELECT pair, COUNT(*) as ticks, MIN(timestamp)::VARCHAR as start, MAX(timestamp)::VARCHAR as end
    FROM ticks GROUP BY pair ORDER BY pair
""").fetchall()

for row in stats:
    print(f"  {row[0]:8s}: {row[1]:>12,} ticks  ({row[2][:10]} → {row[3][:10]})")

total = conn.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]
conn.close()

import os  # noqa: E402

size_mb = os.path.getsize(str(output)) / (1024**2)
print(f"\nTotal: {total:,} ticks")
print(f"Size: {size_mb:.0f} MB")
print(f"Path: {output}")
print(f"✓ Done - {len(pairs)} pairs in 1 database")
