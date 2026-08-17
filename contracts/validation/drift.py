"""
Schema Drift Detection
======================
Detects and reports schema drift across pipeline runs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


@dataclass
class DriftReport:
    """Report of schema drift detection"""

    stage: str
    pair: str | None
    timestamp: datetime
    schema_hash: str
    data_hash: str
    reference_schema_hash: str | None
    reference_data_hash: str | None
    drift_detected: bool
    drift_type: str | None  # "schema", "data", "both", "none"
    added_columns: list[str] = field(default_factory=list)
    removed_columns: list[str] = field(default_factory=list)
    type_changes: dict[str, tuple[str, str]] = field(default_factory=dict)  # col -> (old_type, new_type)
    psi_scores: dict[str, float] = field(default_factory=dict)
    high_psi_columns: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "pair": self.pair,
            "timestamp": self.timestamp.isoformat(),
            "schema_hash": self.schema_hash,
            "data_hash": self.data_hash,
            "reference_schema_hash": self.reference_schema_hash,
            "reference_data_hash": self.reference_data_hash,
            "drift_detected": self.drift_detected,
            "drift_type": self.drift_type,
            "added_columns": self.added_columns,
            "removed_columns": self.removed_columns,
            "type_changes": {k: list(v) for k, v in self.type_changes.items()},
            "psi_scores": self.psi_scores,
            "high_psi_columns": self.high_psi_columns,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DriftReport:
        return cls(
            stage=data["stage"],
            pair=data["pair"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            schema_hash=data["schema_hash"],
            data_hash=data["data_hash"],
            reference_schema_hash=data.get("reference_schema_hash"),
            reference_data_hash=data.get("reference_data_hash"),
            drift_detected=data["drift_detected"],
            drift_type=data["drift_type"],
            added_columns=data.get("added_columns", []),
            removed_columns=data.get("removed_columns", []),
            type_changes={k: tuple(v) for k, v in data.get("type_changes", {}).items()},
            psi_scores=data.get("psi_scores", {}),
            high_psi_columns=data.get("high_psi_columns", []),
            details=data.get("details", {}),
        )


class SchemaDriftDetector:
    """Detects schema and data drift across pipeline runs"""

    def __init__(
        self,
        reference_dir: str | Path,
        psi_threshold: float = 0.2,
        min_samples_for_psi: int = 1000,
    ):
        self.reference_dir = Path(reference_dir)
        self.psi_threshold = psi_threshold
        self.min_samples_for_psi = min_samples_for_psi
        self.reference_dir.mkdir(parents=True, exist_ok=True)

    def _get_reference_path(self, stage: str, pair: str | None) -> Path:
        """Get path for reference schema/data"""
        pair_str = pair or "global"
        return self.reference_dir / f"{stage}_{pair_str}_reference.json"

    def _compute_schema_signature(self, df: pl.DataFrame) -> tuple[str, dict]:
        """Compute schema signature"""
        schema_info = {
            "columns": {col: str(df.schema[col]) for col in sorted(df.columns)},
            "n_cols": len(df.columns),
        }
        schema_str = json.dumps(schema_info, sort_keys=True)
        schema_hash = hashlib.sha256(schema_str.encode()).hexdigest()[:16]
        return schema_hash, schema_info

    def _compute_data_signature(self, df: pl.DataFrame, sample_size: int = 10000) -> tuple[str, dict]:
        """Compute data signature"""
        sample = df.head(min(sample_size, len(df)))
        data_str = sample.write_csv()
        data_hash = hashlib.sha256(data_str.encode()).hexdigest()[:16]

        # Column statistics for drift detection
        stats = {}
        for col in df.columns:
            if df.schema[col].is_numeric():
                vals = df[col].drop_nulls().to_numpy()
                if len(vals) > 0:
                    stats[col] = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "min": float(np.min(vals)),
                        "max": float(np.max(vals)),
                        "null_pct": float(df[col].null_count() / len(df)),
                    }

        return data_hash, stats

    def _compute_psi(self, current: np.ndarray, reference_stats: dict) -> float:
        """Compute Population Stability Index against reference"""
        if "bin_edges" not in reference_stats:
            return 0.0

        bin_edges = np.array(reference_stats["bin_edges"])
        ref_dist = np.array(reference_stats["bin_dist"])

        current_hist, _ = np.histogram(current, bins=bin_edges)
        current_dist = current_hist / current_hist.sum() if current_hist.sum() > 0 else np.zeros_like(current_hist)

        # Avoid division by zero
        eps = 1e-10
        current_dist = np.clip(current_dist, eps, 1.0)
        ref_dist = np.clip(ref_dist, eps, 1.0)

        psi = np.sum((current_dist - ref_dist) * np.log(current_dist / ref_dist))
        return float(psi)

    def load_reference(self, stage: str, pair: str | None) -> dict | None:
        """Load reference signature from disk"""
        ref_path = self._get_reference_path(stage, pair)
        if not ref_path.exists():
            return None

        with open(ref_path) as f:
            return json.load(f)

    def save_reference(self, stage: str, pair: str | None, reference: dict):
        """Save reference signature to disk"""
        ref_path = self._get_reference_path(stage, pair)
        with open(ref_path, "w") as f:
            json.dump(reference, f, indent=2)

    def detect_drift(
        self,
        df: pl.DataFrame,
        stage: str,
        pair: str | None = None,
        save_as_reference: bool = False,
    ) -> DriftReport:
        """
        Detect drift against reference.

        Args:
            df: DataFrame to check
            stage: Pipeline stage name
            pair: Currency pair
            save_as_reference: If True, save this as new reference

        Returns:
            DriftReport with drift analysis
        """
        # Compute current signatures
        schema_hash, schema_info = self._compute_schema_signature(df)
        data_hash, data_stats = self._compute_data_signature(df)

        # Load reference
        reference = self.load_reference(stage, pair)

        if reference is None:
            # No reference - create one if requested
            if save_as_reference:
                reference = {
                    "schema_hash": schema_hash,
                    "schema_info": schema_info,
                    "data_hash": data_hash,
                    "data_stats": data_stats,
                    "created_at": datetime.now().isoformat(),
                }
                self.save_reference(stage, pair, reference)

            return DriftReport(
                stage=stage,
                pair=pair,
                timestamp=datetime.now(),
                schema_hash=schema_hash,
                data_hash=data_hash,
                reference_schema_hash=None,
                reference_data_hash=None,
                drift_detected=False,
                drift_type="none",
                details={
                    "message": "No reference available, baseline created"
                    if save_as_reference
                    else "No reference available"
                },
            )

        # Compare schemas
        ref_schema = reference.get("schema_info", {})
        ref_columns = set(ref_schema.get("columns", {}).keys())
        curr_columns = set(schema_info["columns"].keys())

        added_columns = sorted(curr_columns - ref_columns)
        removed_columns = sorted(ref_columns - curr_columns)

        type_changes = {}
        for col in curr_columns & ref_columns:
            if schema_info["columns"][col] != ref_schema["columns"][col]:
                type_changes[col] = (ref_schema["columns"][col], schema_info["columns"][col])

        schema_drift = bool(added_columns or removed_columns or type_changes)

        # Compare data distributions (PSI)
        ref_data_stats = reference.get("data_stats", {})
        psi_scores = {}
        high_psi_columns = []

        for col, _stats in data_stats.items():
            if col in ref_data_stats:
                # Build reference bin edges from reference stats
                ref_mean = ref_data_stats[col].get("mean", 0)
                ref_std = ref_data_stats[col].get("std", 1)

                # Create 10 bins from reference distribution
                bin_edges = np.linspace(
                    ref_mean - 4 * ref_std,
                    ref_mean + 4 * ref_std,
                    11,
                )

                ref_dist, _ = np.histogram([], bins=bin_edges)  # Empty, will use reference stats
                # Actually compute reference distribution from stats
                # For simplicity, use normal distribution approximation
                from scipy import stats as scipy_stats

                ref_dist = scipy_stats.norm.cdf(bin_edges[1:], ref_mean, ref_std) - scipy_stats.norm.cdf(
                    bin_edges[:-1], ref_mean, ref_std
                )

                ref_data_stats[col]["bin_edges"] = bin_edges.tolist()
                ref_data_stats[col]["bin_dist"] = ref_dist.tolist()

                # Current distribution
                current_vals = df[col].drop_nulls().to_numpy()
                if len(current_vals) >= self.min_samples_for_psi:
                    psi = self._compute_psi(current_vals, ref_data_stats[col])
                    psi_scores[col] = psi
                    if psi > self.psi_threshold:
                        high_psi_columns.append(col)

        data_drift = len(high_psi_columns) > 0

        # Determine drift type
        if schema_drift and data_drift:
            drift_type = "both"
        elif schema_drift:
            drift_type = "schema"
        elif data_drift:
            drift_type = "data"
        else:
            drift_type = "none"

        drift_detected = drift_type != "none"

        report = DriftReport(
            stage=stage,
            pair=pair,
            timestamp=datetime.now(),
            schema_hash=schema_hash,
            data_hash=data_hash,
            reference_schema_hash=reference.get("schema_hash"),
            reference_data_hash=reference.get("data_hash"),
            drift_detected=drift_detected,
            drift_type=drift_type,
            added_columns=added_columns,
            removed_columns=removed_columns,
            type_changes=type_changes,
            psi_scores=psi_scores,
            high_psi_columns=high_psi_columns,
            details={
                "n_columns_current": len(curr_columns),
                "n_columns_reference": len(ref_columns),
                "psi_threshold": self.psi_threshold,
            },
        )

        # Save as new reference if requested
        if save_as_reference:
            reference = {
                "schema_hash": schema_hash,
                "schema_info": schema_info,
                "data_hash": data_hash,
                "data_stats": data_stats,
                "updated_at": datetime.now().isoformat(),
            }
            self.save_reference(stage, pair, reference)

        return report

    def check_pipeline_drift(
        self,
        stage_dataframes: dict[str, pl.DataFrame],
        pair: str | None = None,
        save_as_reference: bool = False,
    ) -> dict[str, DriftReport]:
        """Check drift for multiple pipeline stages"""
        reports = {}
        for stage, df in stage_dataframes.items():
            reports[stage] = self.detect_drift(df, stage, pair, save_as_reference)
        return reports
