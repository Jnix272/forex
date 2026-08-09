"""
Feature Store Core
==================
Persistent feature registry (SQLite) + materialized feature storage (Parquet).
Supports eager batch, incremental, and on-demand materialization strategies.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from data.feature_definitions import (
    BUILTIN_FEATURES,
    REGISTRY,
    FeatureRegistry,
    FeatureSource,
    FeatureSpec,
    FeatureType,
    MaterializationStrategy,
)

# ════════════════════════════════════════════════════════════════════════════
# CONTENT HASHING (for deduplication / versioning)
# ════════════════════════════════════════════════════════════════════════════

def compute_feature_hash(spec: FeatureSpec, params: dict = None) -> str:
    """Deterministic hash of feature definition + parameters."""
    content = {
        "name": spec.name,
        "transformation": spec.transformation,
        "params": params or spec.params,
        "dependencies": sorted(spec.dependencies),
        "version": spec.version,
    }
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:16]


def compute_data_hash(df: pl.DataFrame, feature_col: str) -> str:
    """Hash of feature values for data versioning."""
    # Use a sample for large dataframes
    sample = df[feature_col].head(10000).to_numpy()
    return hashlib.sha256(sample.tobytes()).hexdigest()[:16]


# ════════════════════════════════════════════════════════════════════════════
# FEATURE STORE
# ════════════════════════════════════════════════════════════════════════════

class FeatureStore:
    """
    Persistent feature store with:
    - SQLite registry (metadata, lineage, materialization tracking)
    - Parquet files for feature values (partitioned by date)
    - Thread-safe operations
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        root: str | Path = "data/feature_store",
        registry: FeatureRegistry = None,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry = registry or REGISTRY

        # Paths
        self.db_path = self.root / "registry.db"
        self.data_root = self.root / "features"
        self.data_root.mkdir(parents=True, exist_ok=True)

        # Optional OHLCV frame for bar-backed materializers (set via set_bars /
        # materialize(..., bars=...)). Macro features can run without it.
        self._bars: pl.DataFrame | None = None
        self._bars_path: Path | None = None

        # Thread safety
        self._lock = threading.RLock()

        # Initialize
        self._init_db()
        self._sync_registry()

    def set_bars(self, bars: pl.DataFrame | None) -> None:
        """Attach an OHLCV frame used by subsequent ``materialize`` / ``_compute_feature`` calls."""
        self._bars = bars

    def set_bars_path(self, path: str | Path | None) -> None:
        """Optional parquet/CSV of OHLCV bars loaded on demand when ``_bars`` is unset."""
        self._bars_path = Path(path) if path else None

    # ──────────────────────────────────────────────────────────────────────
    # DATABASE INITIALIZATION
    # ──────────────────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────────────
    # CONNECTION HELPER
    # ──────────────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        """Open a connection with WAL mode, normal sync, and FK enforcement.

        SQLite's PRAGMA foreign_keys is *per-connection* and defaults to OFF.
        All code must obtain connections through this helper so that FK checks
        are consistently enforced on every read/write operation.
        """
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            with self._connect() as conn:
                # Schema version — seed on first init, never overwrite
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS schema_version (
                        version INTEGER PRIMARY KEY,
                        updated_at TEXT
                    )
                """)
                conn.execute(
                    "INSERT OR IGNORE INTO schema_version (version, updated_at) VALUES (?, ?)",
                    (self.SCHEMA_VERSION, datetime.now(UTC).isoformat()),
                )

                # Feature definitions
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS features (
                        name TEXT PRIMARY KEY,
                        feature_type TEXT NOT NULL,
                        description TEXT,
                        source TEXT NOT NULL,
                        transformation TEXT NOT NULL,
                        dependencies TEXT,           -- JSON array
                        params TEXT,                 -- JSON object
                        version INTEGER DEFAULT 1,
                        tags TEXT,                   -- JSON array
                        owner TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        deprecated INTEGER DEFAULT 0,
                        content_hash TEXT NOT NULL,
                        UNIQUE(name, content_hash)
                    )
                """)

                # Materialization tracking (which time ranges are computed)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS materializations (
                        feature_name TEXT NOT NULL,
                        start_ts TEXT NOT NULL,      -- ISO format
                        end_ts TEXT NOT NULL,
                        path TEXT NOT NULL,          -- relative to data_root
                        rows INTEGER NOT NULL,
                        data_hash TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        strategy TEXT NOT NULL,      -- MaterializationStrategy
                        PRIMARY KEY (feature_name, start_ts, end_ts),
                        FOREIGN KEY (feature_name) REFERENCES features(name)
                    )
                """)

                # Lineage / dependency graph.
                # NOTE: Only `downstream` carries a FK back to the features
                # table, because `upstream` entries may legitimately be raw
                # OHLCV column names (high, low, volume, …) or intermediate
                # signals (adx_14, rsi_14, trend_regime, …) that are inputs
                # consumed by the materializers but are NOT themselves entries
                # in the feature registry.  Adding a FK on upstream would
                # produce hundreds of violations on every fresh init.
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS lineage (
                        downstream TEXT NOT NULL,
                        upstream TEXT NOT NULL,
                        PRIMARY KEY (downstream, upstream),
                        FOREIGN KEY (downstream) REFERENCES features(name)
                    )
                """)

                # Materialization job queue
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS job_queue (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        feature_name TEXT NOT NULL,
                        start_ts TEXT NOT NULL,
                        end_ts TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        priority INTEGER DEFAULT 0,
                        status TEXT DEFAULT 'pending',  -- pending, running, done, failed
                        created_at TEXT NOT NULL,
                        started_at TEXT,
                        finished_at TEXT,
                        error TEXT,
                        FOREIGN KEY (feature_name) REFERENCES features(name)
                    )
                """)

                # Stats / monitoring
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS feature_stats (
                        feature_name TEXT NOT NULL,
                        ts TEXT NOT NULL,              -- computation timestamp
                        mean REAL,
                        std REAL,
                        min REAL,
                        max REAL,
                        null_count INTEGER,
                        skew REAL,
                        kurtosis REAL,
                        PRIMARY KEY (feature_name, ts),
                        FOREIGN KEY (feature_name) REFERENCES features(name)
                    )
                """)

    def _sync_registry(self) -> None:
        """Upsert builtin features into registry DB."""
        with self._lock:
            with self._connect() as conn:
                now = datetime.now(UTC).isoformat()

                for spec in self.registry.all():
                    content_hash = compute_feature_hash(spec)
                    conn.execute("""
                        INSERT INTO features (
                            name, feature_type, description, source, transformation,
                            dependencies, params, version, tags, owner,
                            created_at, updated_at, deprecated, content_hash
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(name, content_hash) DO UPDATE SET
                            updated_at = excluded.updated_at,
                            description = excluded.description,
                            params = excluded.params,
                            tags = excluded.tags
                    """, (
                        spec.name, spec.feature_type.value, spec.description, spec.source.value,
                        spec.transformation, json.dumps(spec.dependencies), json.dumps(spec.params),
                        spec.version, json.dumps(spec.tags), spec.owner,
                        spec.created_at or now, now, int(spec.deprecated), content_hash
                    ))

                    # Lineage — upstream may be a raw column name (high, low, volume, …)
                    # that is not itself a registered feature; only downstream has a FK.
                    for dep in spec.dependencies:
                        conn.execute("""
                            INSERT OR IGNORE INTO lineage (downstream, upstream)
                            VALUES (?, ?)
                        """, (spec.name, dep))

    # ──────────────────────────────────────────────────────────────────────
    # FEATURE REGISTRY QUERIES
    # ──────────────────────────────────────────────────────────────────────

    def get_feature(self, name: str) -> FeatureSpec | None:
        """Get feature spec from DB (includes runtime params)."""
        with self._lock:
            with self._connect() as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM features WHERE name = ? AND deprecated = 0", (name,)
                ).fetchone()

            if not row:
                return None

            return FeatureSpec(
                name=row["name"],
                feature_type=FeatureType(row["feature_type"]),
                description=row["description"],
                source=FeatureSource(row["source"]),
                transformation=row["transformation"],
                dependencies=json.loads(row["dependencies"] or "[]"),
                params=json.loads(row["params"] or "{}"),
                version=row["version"],
                tags=json.loads(row["tags"] or "[]"),
                owner=row["owner"],
                created_at=row["created_at"],
                deprecated=bool(row["deprecated"]),
            )

    def list_features(
        self, source: FeatureSource = None, tag: str = None, deprecated: bool = False
    ) -> list[FeatureSpec]:
        """List features with optional filters."""
        with self._lock:
            with self._connect() as conn:
                conn.row_factory = sqlite3.Row

                query = "SELECT * FROM features WHERE deprecated = ?"
                params = [int(deprecated)]

                if source:
                    query += " AND source = ?"
                    params.append(source.value)

                rows = conn.execute(query, params).fetchall()

            features = []
            for row in rows:
                spec = FeatureSpec(
                    name=row["name"],
                    feature_type=FeatureType(row["feature_type"]),
                    description=row["description"],
                    source=FeatureSource(row["source"]),
                    transformation=row["transformation"],
                    dependencies=json.loads(row["dependencies"] or "[]"),
                    params=json.loads(row["params"] or "{}"),
                    version=row["version"],
                    tags=json.loads(row["tags"] or "[]"),
                    owner=row["owner"],
                    created_at=row["created_at"],
                    deprecated=bool(row["deprecated"]),
                )
                if tag and tag not in spec.tags:
                    continue
                features.append(spec)
            return features

    def get_lineage(self, feature_name: str, direction: str = "both") -> dict[str, list[str]]:
        """Get upstream/downstream dependencies."""
        with self._lock:
            with self._connect() as conn:
                result = {"upstream": [], "downstream": []}

                if direction in ("upstream", "both"):
                    rows = conn.execute(
                        "SELECT upstream FROM lineage WHERE downstream = ?", (feature_name,)
                    ).fetchall()
                    result["upstream"] = [r[0] for r in rows]

                if direction in ("downstream", "both"):
                    rows = conn.execute(
                        "SELECT downstream FROM lineage WHERE upstream = ?", (feature_name,)
                    ).fetchall()
                    result["downstream"] = [r[0] for r in rows]

            return result

    # ──────────────────────────────────────────────────────────────────────
    # MATERIALIZATION
    # ──────────────────────────────────────────────────────────────────────

    def _get_feature_path(self, feature_name: str, start: datetime, end: datetime) -> Path:
        """Generate Parquet path for feature time range."""
        date_str = start.strftime("%Y%m")
        feature_dir = self.data_root / feature_name
        feature_dir.mkdir(parents=True, exist_ok=True)
        return feature_dir / f"{feature_name}_{date_str}.parquet"

    def is_materialized(
        self, feature_name: str, start: datetime, end: datetime
    ) -> bool:
        """Check if feature is materialized for time range."""
        with self._lock:
            with self._connect() as conn:
                row = conn.execute("""
                    SELECT 1 FROM materializations
                    WHERE feature_name = ? AND start_ts <= ? AND end_ts >= ?
                """, (feature_name, start.isoformat(), end.isoformat())).fetchone()
            return row is not None

    def get_materialized_ranges(self, feature_name: str) -> list[tuple[datetime, datetime]]:
        """Get all materialized time ranges for a feature."""
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute("""
                    SELECT start_ts, end_ts FROM materializations
                    WHERE feature_name = ? ORDER BY start_ts
                """, (feature_name,)).fetchall()
            return [(datetime.fromisoformat(r[0]), datetime.fromisoformat(r[1])) for r in rows]

    def load_feature(
        self, feature_name: str, start: datetime, end: datetime
    ) -> pl.DataFrame | None:
        """Load materialized feature data for time range."""
        if not self.is_materialized(feature_name, start, end):
            return None

        # Find all parquet files covering the range
        ranges = self.get_materialized_ranges(feature_name)
        relevant = [r for r in ranges if r[0] <= end and r[1] >= start]

        if not relevant:
            return None

        dfs = []
        for r_start, r_end in relevant:
            path = self._get_feature_path(feature_name, r_start, r_end)
            if path.exists():
                df = pl.read_parquet(path)
                dfs.append(df)

        if not dfs:
            return None

        combined = pl.concat(dfs)
        # Filter to requested range
        if "timestamp_utc" in combined.columns:
            combined = combined.filter(
                (pl.col("timestamp_utc") >= start) & (pl.col("timestamp_utc") <= end)
            )
        return combined.sort("timestamp_utc")

    def materialize(
        self,
        feature_names: str | list[str],
        start: datetime,
        end: datetime,
        strategy: MaterializationStrategy = MaterializationStrategy.EAGER_BATCH,
        force: bool = False,
        bars: pl.DataFrame | None = None,
    ) -> dict[str, pl.DataFrame]:
        """
        Materialize features for time range.
        Returns dict of feature_name -> DataFrame with timestamp_utc + feature column.

        Pass ``bars`` (or call ``set_bars`` first) for OHLCV-backed features.
        Macro/cross-asset features load from the external panel when bars are absent.
        """
        if bars is not None:
            self.set_bars(bars)
        if isinstance(feature_names, str):
            feature_names = [feature_names]

        # Resolve dependencies
        all_names = self.registry.resolve_dependencies(feature_names)

        results = {}
        for name in all_names:
            if name in results:
                continue

            spec = self.get_feature(name)
            if not spec:
                raise ValueError(f"Feature '{name}' not registered")

            # Check if already materialized
            if not force and self.is_materialized(name, start, end):
                df = self.load_feature(name, start, end)
                if df is not None:
                    results[name] = df
                    continue

            # Compute feature
            df = self._compute_feature(spec, start, end)

            if df is not None and not df.is_empty():
                # Store
                self._store_materialization(name, df, start, end, strategy)
                results[name] = df

        return {k: results[k] for k in feature_names if k in results}

    def _load_bars_frame(self) -> pl.DataFrame | None:
        if self._bars is not None and not self._bars.is_empty():
            return self._bars
        path = self._bars_path
        if path is None:
            # Convention: optional bars.parquet next to the store root
            candidate = self.root / "bars.parquet"
            path = candidate if candidate.is_file() else None
        if path is None or not Path(path).is_file():
            return None
        path = Path(path)
        if path.suffix.lower() == ".csv":
            df = pl.read_csv(path, try_parse_dates=True)
        else:
            df = pl.read_parquet(path)
        if "timestamp_utc" not in df.columns and "timestamp" in df.columns:
            df = df.rename({"timestamp": "timestamp_utc"})
        self._bars = df
        return df

    def _resolve_bars(self, start: datetime, end: datetime, requires_ohlcv: bool) -> pl.DataFrame:
        df = self._load_bars_frame()
        if df is not None and not df.is_empty() and "timestamp_utc" in df.columns:
            return df.filter(
                (pl.col("timestamp_utc") >= start) & (pl.col("timestamp_utc") <= end)
            ).sort("timestamp_utc")
        if requires_ohlcv:
            raise RuntimeError(
                "FeatureStore: OHLCV bars required but none attached. "
                "Call set_bars(df), materialize(..., bars=df), or place bars.parquet under the store root."
            )
        # Scaffold daily timestamps so macro materializers can still align.
        import pandas as pd
        idx = pd.date_range(start=start, end=end, freq="1D", tz="UTC")
        return pl.DataFrame({"timestamp_utc": idx})

    def _compute_feature(self, spec: FeatureSpec, start: datetime, end: datetime) -> pl.DataFrame | None:
        """Compute feature via the typed materializer registry (wired, not a placeholder)."""
        from data.feature_materializers import get_materializer

        mat = get_materializer(spec, self)
        bars = self._resolve_bars(start, end, requires_ohlcv=bool(mat.requires_ohlcv))
        out = mat.compute(spec, bars, start, end)
        if out is None or out.is_empty():
            raise RuntimeError(
                f"FeatureStore: materializer for {spec.name!r} returned empty "
                f"(source={getattr(spec, 'source', None)}). Check bars / cross-asset panel."
            )
        return out

    def _store_materialization(
        self,
        feature_name: str,
        df: pl.DataFrame,
        start: datetime,
        end: datetime,
        strategy: MaterializationStrategy,
    ) -> None:
        """Store feature values as Parquet and record in DB."""
        path = self._get_feature_path(feature_name, start, end)

        # Ensure timestamp_utc column exists
        if "timestamp_utc" not in df.columns:
            raise ValueError("Feature DataFrame must have 'timestamp_utc' column")

        # Write Parquet
        df.write_parquet(path, compression="zstd")

        # Compute stats
        feature_col = [c for c in df.columns if c != "timestamp_utc"][0]
        vals = df[feature_col].to_numpy()
        stats = {
            "mean": float(np.nanmean(vals)) if len(vals) > 0 else None,
            "std": float(np.nanstd(vals)) if len(vals) > 0 else None,
            "min": float(np.nanmin(vals)) if len(vals) > 0 else None,
            "max": float(np.nanmax(vals)) if len(vals) > 0 else None,
            "null_count": int(df[feature_col].null_count()),
            "skew": float(np.nan) if len(vals) < 3 else float(
                np.nanmean((vals - np.nanmean(vals))**3) / (np.nanstd(vals)**3 + 1e-12)
            ),
            "kurtosis": float(np.nan) if len(vals) < 4 else float(
                np.nanmean((vals - np.nanmean(vals))**4) / (np.nanstd(vals)**4 + 1e-12) - 3
            ),
        }

        data_hash = compute_data_hash(df, feature_col)

        with self._lock:
            with self._connect() as conn:
                now = datetime.now(UTC).isoformat()

                conn.execute("""
                    INSERT OR REPLACE INTO materializations
                    (feature_name, start_ts, end_ts, path, rows, data_hash, created_at, strategy)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feature_name, start.isoformat(), end.isoformat(),
                    str(path.relative_to(self.root)), len(df), data_hash, now, strategy.value
                ))

                conn.execute("""
                    INSERT OR REPLACE INTO feature_stats
                    (feature_name, ts, mean, std, min, max, null_count, skew, kurtosis)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feature_name, now,
                    stats["mean"], stats["std"], stats["min"], stats["max"],
                    stats["null_count"], stats["skew"], stats["kurtosis"]
                ))

    # ──────────────────────────────────────────────────────────────────────
    # JOB QUEUE (for async materialization)
    # ──────────────────────────────────────────────────────────────────────

    def enqueue_materialization(
        self,
        feature_name: str,
        start: datetime,
        end: datetime,
        strategy: MaterializationStrategy = MaterializationStrategy.EAGER_BATCH,
        priority: int = 0,
    ) -> int:
        """Add materialization job to queue."""
        with self._lock:
            with self._connect() as conn:
                now = datetime.now(UTC).isoformat()
                cursor = conn.execute("""
                    INSERT INTO job_queue (feature_name, start_ts, end_ts, strategy, priority, created_at, status)
                    VALUES (?, ?, ?, ?, ?, ?, 'pending')
                """, (feature_name, start.isoformat(), end.isoformat(), strategy.value, priority, now))
                job_id = cursor.lastrowid
            return job_id

    def get_pending_jobs(self, limit: int = 10) -> list[dict]:
        """Get next jobs to process."""
        with self._lock:
            with self._connect() as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute("""
                    SELECT * FROM job_queue
                    WHERE status = 'pending'
                    ORDER BY priority DESC, created_at ASC
                    LIMIT ?
                """, (limit,)).fetchall()
            return [dict(r) for r in rows]

    def mark_job_started(self, job_id: int) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE job_queue SET status = 'running', started_at = ? WHERE id = ?",
                    (datetime.now(UTC).isoformat(), job_id)
                )

    def mark_job_done(self, job_id: int, error: str = None) -> None:
        with self._lock:
            with self._connect() as conn:
                status = 'failed' if error else 'done'
                conn.execute("""
                    UPDATE job_queue SET status = ?, finished_at = ?, error = ?
                    WHERE id = ?
                """, (status, datetime.now(UTC).isoformat(), error, job_id))

    # ──────────────────────────────────────────────────────────────────────
    # INCREMENTAL / ON-DEMAND HELPERS
    # ──────────────────────────────────────────────────────────────────────

    def get_latest_timestamp(self, feature_name: str) -> datetime | None:
        """Get latest materialized timestamp for feature."""
        ranges = self.get_materialized_ranges(feature_name)
        if not ranges:
            return None
        return max(r[1] for r in ranges)

    def needs_incremental_update(
        self, feature_name: str, lookback_bars: int = 100
    ) -> tuple[bool, datetime | None, datetime | None]:
        """
        Check if feature needs incremental update.
        Returns (needs_update, start, end) for incremental range.

        Uses attached bars (or bars.parquet) when present; otherwise compares
        latest materialization to ``now`` with a ``lookback_bars``-minute pad.
        """
        latest = self.get_latest_timestamp(feature_name)
        end = datetime.now(UTC)
        if latest is None:
            return True, None, end  # Full materialization needed

        bars = self._load_bars_frame()
        if bars is not None and not bars.is_empty() and "timestamp_utc" in bars.columns:
            bars_end = bars["timestamp_utc"].max()
            if bars_end is not None and bars_end > latest:
                # Recompute a short overlap window for rolling features.
                from datetime import timedelta
                start = latest - timedelta(minutes=max(0, int(lookback_bars)))
                return True, start, bars_end

        if end > latest:
            from datetime import timedelta
            start = latest - timedelta(minutes=max(0, int(lookback_bars)))
            return True, start, end
        return False, None, None

    # ──────────────────────────────────────────────────────────────────────
    # MAINTENANCE
    # ──────────────────────────────────────────────────────────────────────

    def vacuum(self) -> None:
        """Compact database."""
        with self._lock:
            with self._connect() as conn:
                conn.execute("VACUUM")

    def get_storage_stats(self) -> dict:
        """Get storage usage statistics."""
        total_size = 0
        feature_sizes = {}

        for feature_dir in self.data_root.iterdir():
            if feature_dir.is_dir():
                size = sum(f.stat().st_size for f in feature_dir.glob("*.parquet"))
                feature_sizes[feature_dir.name] = size
                total_size += size

        return {
            "total_bytes": total_size,
            "total_mb": total_size / (1024 * 1024),
            "feature_count": len(feature_sizes),
            "per_feature_mb": {k: v / (1024 * 1024) for k, v in feature_sizes.items()},
        }


# ════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

_DEFAULT_STORE: FeatureStore | None = None


def get_feature_store(root: str | Path = None) -> FeatureStore:
    """Get or create global feature store instance."""
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        _DEFAULT_STORE = FeatureStore(root or "data/feature_store")
    return _DEFAULT_STORE


def materialize_features(
    feature_names: str | list[str],
    start: datetime,
    end: datetime,
    strategy: MaterializationStrategy = MaterializationStrategy.EAGER_BATCH,
) -> dict[str, pl.DataFrame]:
    """Convenience function for materialization."""
    store = get_feature_store()
    return store.materialize(feature_names, start, end, strategy)


if __name__ == "__main__":
    # Demo
    store = FeatureStore("data/feature_store_test")

    # Register a test feature
    from data.feature_definitions import FeatureSource, FeatureSpec, FeatureType, MaterializationStrategy
    test_spec = FeatureSpec(
        name="test_feature",
        feature_type=FeatureType.NUMERIC,
        description="Test feature",
        source=FeatureSource.PRICE,
        transformation="close / close.shift(1) - 1",
        dependencies=[],
        params={},
        tags=["test"],
    )

    print(f"Feature store initialized at {store.root}")
    print(f"Builtin features: {len(BUILTIN_FEATURES)}")
    print(f"Storage stats: {store.get_storage_stats()}")
