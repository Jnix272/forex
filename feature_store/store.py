"""
Feature Store Implementation
============================
Base feature store and implementations.
"""

from __future__ import annotations

import hashlib
import json
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import polars as pl

ParquetCompression = Literal["lz4", "uncompressed", "snappy", "gzip", "brotli", "zstd"]


def _as_datetime(value: Any) -> datetime | None:
    """Normalize timestamp-like values to a datetime instance."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_pydatetime"):
        try:
            converted = value.to_pydatetime()
            if isinstance(converted, datetime):
                return converted
        except Exception:
            pass
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


@dataclass
class FeatureVersion:
    """Feature version metadata"""

    version: str
    created_at: datetime
    schema_hash: str
    data_hash: str
    n_rows: int
    n_cols: int
    pairs: list[str]
    date_range: tuple[datetime, datetime] | None
    description: str = ""
    tags: dict[str, str] = field(default_factory=dict)
    lineage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "schema_hash": self.schema_hash,
            "data_hash": self.data_hash,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "pairs": self.pairs,
            "date_range": [d.isoformat() for d in self.date_range] if self.date_range else None,
            "description": self.description,
            "tags": self.tags,
            "lineage": self.lineage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureVersion:
        date_range_raw = data.get("date_range")
        normalized_range: tuple[datetime, datetime] | None = None
        if isinstance(date_range_raw, list) and len(date_range_raw) == 2:
            start = _as_datetime(date_range_raw[0])
            end = _as_datetime(date_range_raw[1])
            if start is not None and end is not None:
                normalized_range = (start, end)

        return cls(
            version=data["version"],
            created_at=datetime.fromisoformat(data["created_at"]),
            schema_hash=data["schema_hash"],
            data_hash=data["data_hash"],
            n_rows=data["n_rows"],
            n_cols=data["n_cols"],
            pairs=data["pairs"],
            date_range=normalized_range,
            description=data.get("description", ""),
            tags=data.get("tags", {}),
            lineage=data.get("lineage", {}),
        )


class FeatureStore(ABC):
    """Abstract base class for feature stores"""

    @abstractmethod
    def write(
        self,
        features: pl.DataFrame,
        version: str,
        pair: str | None = None,
        description: str = "",
        tags: dict[str, str] | None = None,
        lineage: dict | None = None,
    ) -> FeatureVersion:
        """Write features to store"""
        pass

    @abstractmethod
    def read(
        self,
        version: str,
        pair: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        columns: list[str] | None = None,
    ) -> pl.DataFrame:
        """Read features from store"""
        pass

    @abstractmethod
    def read_latest(
        self,
        pair: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        columns: list[str] | None = None,
    ) -> tuple[pl.DataFrame, FeatureVersion]:
        """Read latest version of features"""
        pass

    @abstractmethod
    def list_versions(self, pair: str | None = None) -> list[FeatureVersion]:
        """List available versions"""
        pass

    @abstractmethod
    def delete_version(self, version: str, pair: str | None = None) -> bool:
        """Delete a version"""
        pass

    @abstractmethod
    def get_version(self, version: str, pair: str | None = None) -> FeatureVersion | None:
        """Get version metadata"""
        pass


class ParquetFeatureStore(FeatureStore):
    """
    Parquet-based feature store with partitioning.

    Stores features as partitioned Parquet files:
    {base_path}/{version}/{pair}/year=YYYY/month=MM/day=DD/features.parquet
    """

    def __init__(
        self,
        base_path: str | Path,
        partition_cols: list[str] | None = None,
        compression: ParquetCompression = "zstd",
    ):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.partition_cols = partition_cols or ["pair", "year", "month", "day"]
        self.compression: ParquetCompression = compression
        self._metadata_file = self.base_path / "_metadata.json"
        self._lock = threading.Lock()
        self._load_metadata()

    def _load_metadata(self):
        """Load version metadata"""
        if self._metadata_file.exists():
            with open(self._metadata_file) as f:
                data = json.load(f)
                self.versions = {k: FeatureVersion.from_dict(v) for k, v in data.items()}
        else:
            self.versions = {}

    def _save_metadata(self):
        """Save version metadata"""
        with open(self._metadata_file, "w") as f:
            json.dump({k: v.to_dict() for k, v in self.versions.items()}, f, indent=2, default=str)

    def _compute_hashes(self, df: pl.DataFrame) -> tuple[str, str]:
        """Compute schema and data hashes"""
        schema_str = json.dumps([(c, str(df.schema[c])) for c in sorted(df.columns)], sort_keys=True)
        schema_hash = hashlib.sha256(schema_str.encode()).hexdigest()[:16]

        sample = df.head(min(1000, len(df)))
        data_str = sample.write_csv()
        data_hash = hashlib.sha256(data_str.encode()).hexdigest()[:16]

        return schema_hash, data_hash

    def _add_partition_columns(self, df: pl.DataFrame, pair: str | None) -> pl.DataFrame:
        """Add partition columns for writing"""
        if "timestamp_utc" not in df.columns:
            return df

        return df.with_columns(
            [
                pl.lit(pair or "unknown").alias("pair"),
                pl.col("timestamp_utc").dt.year().alias("year"),
                pl.col("timestamp_utc").dt.month().alias("month"),
                pl.col("timestamp_utc").dt.day().alias("day"),
            ]
        )

    def write(
        self,
        features: pl.DataFrame,
        version: str,
        pair: str | None = None,
        description: str = "",
        tags: dict[str, str] | None = None,
        lineage: dict | None = None,
    ) -> FeatureVersion:
        """Write features to parquet store"""
        if len(features) == 0:
            raise ValueError("Cannot write empty DataFrame")

        schema_hash, data_hash = self._compute_hashes(features)

        # Check if version exists
        with self._lock:
            if version in self.versions:
                raise ValueError(f"Version {version} already exists")

            # Add partition columns
            df = self._add_partition_columns(features, pair)

            # Determine pairs in data
            pairs = [pair] if pair else df["pair"].unique().to_list()

            # Write partitioned parquet
            version_path = self.base_path / version
            version_path.mkdir(parents=True, exist_ok=True)

            df.write_parquet(
                version_path,
                partition_by=self.partition_cols,
                compression=self.compression,
            )

            # Get date range
            date_range: tuple[datetime, datetime] | None = None
            if "timestamp_utc" in features.columns:
                ts_min = _as_datetime(features["timestamp_utc"].min())
                ts_max = _as_datetime(features["timestamp_utc"].max())
                if ts_min is not None and ts_max is not None:
                    date_range = (ts_min, ts_max)

            # Create version metadata
            version_meta = FeatureVersion(
                version=version,
                created_at=datetime.now(),
                schema_hash=schema_hash,
                data_hash=data_hash,
                n_rows=len(features),
                n_cols=len(features.columns),
                pairs=pairs,
                date_range=date_range,
                description=description,
                tags=tags or {},
                lineage=lineage or {},
            )

            self.versions[version] = version_meta
            self._save_metadata()

            return version_meta

    def read(
        self,
        version: str,
        pair: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        columns: list[str] | None = None,
    ) -> pl.DataFrame:
        """Read features from parquet store"""
        version_path = self.base_path / version
        if not version_path.exists():
            raise ValueError(f"Version {version} not found")

        # Build lazy frame
        lf = pl.scan_parquet(version_path)

        # Apply filters
        if pair:
            lf = lf.filter(pl.col("pair") == pair)

        if start_date or end_date:
            if "timestamp_utc" not in lf.columns:
                # Try to reconstruct from partition columns
                pass
            if start_date:
                lf = lf.filter(pl.col("timestamp_utc") >= start_date)
            if end_date:
                lf = lf.filter(pl.col("timestamp_utc") <= end_date)

        # Select columns
        if columns:
            # Ensure partition columns are included for filtering
            select_cols = columns + [c for c in self.partition_cols if c not in columns]
            lf = lf.select(select_cols)

        return lf.collect()

    def read_latest(
        self,
        pair: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        columns: list[str] | None = None,
    ) -> tuple[pl.DataFrame, FeatureVersion]:
        """Read latest version"""
        versions = self.list_versions(pair)
        if not versions:
            raise ValueError("No versions available")

        latest = max(versions, key=lambda v: v.created_at)
        df = self.read(latest.version, pair, start_date, end_date, columns)
        return df, latest

    def list_versions(self, pair: str | None = None) -> list[FeatureVersion]:
        """List available versions"""
        versions = list(self.versions.values())

        if pair:
            versions = [v for v in versions if pair in v.pairs]

        return sorted(versions, key=lambda v: v.created_at, reverse=True)

    def delete_version(self, version: str, pair: str | None = None) -> bool:
        """Delete a version"""
        with self._lock:
            if version not in self.versions:
                return False

            # If pair specified and version has multiple pairs, don't delete
            if pair and len(self.versions[version].pairs) > 1:
                # Would need partial delete - not implemented
                return False

            # Delete parquet files
            import shutil

            version_path = self.base_path / version
            if version_path.exists():
                shutil.rmtree(version_path)

            del self.versions[version]
            self._save_metadata()
            return True

    def get_version(self, version: str, pair: str | None = None) -> FeatureVersion | None:
        """Get version metadata"""
        v = self.versions.get(version)
        if v and pair and pair not in v.pairs:
            return None
        return v


class DeltaFeatureStore(FeatureStore):
    """
    Delta Lake-based feature store (placeholder for future implementation).

    Would provide ACID transactions, time travel, and schema evolution.
    """

    def __init__(self, base_path: str | Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        raise NotImplementedError("Delta Lake store not yet implemented")

    def write(
        self,
        features: pl.DataFrame,
        version: str,
        pair: str | None = None,
        description: str = "",
        tags: dict[str, str] | None = None,
        lineage: dict | None = None,
    ) -> FeatureVersion:
        raise NotImplementedError()

    def read(
        self,
        version: str,
        pair: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        columns: list[str] | None = None,
    ) -> pl.DataFrame:
        raise NotImplementedError()

    def read_latest(
        self,
        pair: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        columns: list[str] | None = None,
    ) -> tuple[pl.DataFrame, FeatureVersion]:
        raise NotImplementedError()

    def list_versions(self, pair: str | None = None) -> list[FeatureVersion]:
        raise NotImplementedError()

    def delete_version(self, version: str, pair: str | None = None) -> bool:
        raise NotImplementedError()

    def get_version(self, version: str, pair: str | None = None) -> FeatureVersion | None:
        raise NotImplementedError()


# Convenience function
def create_feature_store(
    store_type: str = "parquet", base_path: str | Path = "./feature_store", **kwargs
) -> FeatureStore:
    """Create a feature store instance"""
    if store_type == "parquet":
        return ParquetFeatureStore(base_path, **kwargs)
    elif store_type == "delta":
        return DeltaFeatureStore(base_path, **kwargs)
    else:
        raise ValueError(f"Unknown store type: {store_type}")
