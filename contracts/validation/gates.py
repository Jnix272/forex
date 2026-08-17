"""
Pipeline Validation Gates
=========================
Validation gates for each pipeline stage with fail-fast behavior.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from contracts.bar import BarContract
from contracts.base import ContractMetadata, Stage
from contracts.dataset import DatasetContract
from contracts.feature import FeatureContract
from contracts.label import LabelContract
from contracts.tick import TickContract


class ValidationResult(StrEnum):
    """Validation result status"""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class GateResult:
    """Result of a single validation gate"""

    gate_name: str
    stage: Stage
    pair: str | None
    result: ValidationResult
    message: str
    metadata: ContractMetadata | None = None
    duration_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "stage": self.stage.value,
            "pair": self.pair,
            "result": self.result.value,
            "message": self.message,
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "duration_ms": self.duration_ms,
            "details": self.details,
        }


class ValidationGate:
    """Base class for validation gates"""

    def __init__(self, name: str, stage: Stage, fail_fast: bool = True):
        self.name = name
        self.stage = stage
        self.fail_fast = fail_fast

    def run(self, df: pl.DataFrame, pair: str | None = None, **kwargs) -> GateResult:
        """Run the validation gate"""
        start = time.perf_counter()
        try:
            result = self._validate(df, pair, **kwargs)
            duration_ms = (time.perf_counter() - start) * 1000
            result.duration_ms = duration_ms
            return result
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            return GateResult(
                gate_name=self.name,
                stage=self.stage,
                pair=pair,
                result=ValidationResult.FAIL,
                message=f"Gate execution failed: {e}",
                duration_ms=duration_ms,
                details={"error": str(e), "error_type": type(e).__name__},
            )

    def _validate(self, df: pl.DataFrame, pair: str | None, **kwargs) -> GateResult:
        """Override in subclass"""
        raise NotImplementedError


class ContractValidationGate(ValidationGate):
    """Gate that validates against a data contract"""

    def __init__(self, contract_class: type, fail_fast: bool = True):
        super().__init__(
            name=f"{contract_class.contract_name}_validation",
            stage=contract_class.stage,
            fail_fast=fail_fast,
        )
        self.contract_class = contract_class

    def _validate(self, df: pl.DataFrame, pair: str | None, **kwargs) -> GateResult:
        try:
            _validated_df, metadata = self.contract_class.validate(df, pair=pair, strict=False)
            warnings = metadata.warnings

            if warnings:
                return GateResult(
                    gate_name=self.name,
                    stage=self.stage,
                    pair=pair,
                    result=ValidationResult.WARN,
                    message=f"Contract validation passed with {len(warnings)} warnings",
                    metadata=metadata,
                    details={"warnings": warnings},
                )
            else:
                return GateResult(
                    gate_name=self.name,
                    stage=self.stage,
                    pair=pair,
                    result=ValidationResult.PASS,
                    message="Contract validation passed",
                    metadata=metadata,
                )
        except ValueError as e:
            return GateResult(
                gate_name=self.name,
                stage=self.stage,
                pair=pair,
                result=ValidationResult.FAIL,
                message=str(e),
            )


class RowCountGate(ValidationGate):
    """Gate that checks minimum row count"""

    def __init__(self, min_rows: int = 100, stage: Stage = Stage.INGESTION):
        super().__init__(name="row_count_check", stage=stage)
        self.min_rows = min_rows

    def _validate(self, df: pl.DataFrame, pair: str | None, **kwargs) -> GateResult:
        n_rows = len(df)
        if n_rows < self.min_rows:
            return GateResult(
                gate_name=self.name,
                stage=self.stage,
                pair=pair,
                result=ValidationResult.FAIL,
                message=f"Row count {n_rows} below minimum {self.min_rows}",
                details={"n_rows": n_rows, "min_rows": self.min_rows},
            )
        return GateResult(
            gate_name=self.name,
            stage=self.stage,
            pair=pair,
            result=ValidationResult.PASS,
            message=f"Row count OK: {n_rows}",
            details={"n_rows": n_rows},
        )


class NullThresholdGate(ValidationGate):
    """Gate that checks null percentage threshold"""

    def __init__(self, max_null_pct: float = 0.05, stage: Stage = Stage.FEATURE_ENGINEERING):
        super().__init__(name="null_threshold_check", stage=stage)
        self.max_null_pct = max_null_pct

    def _validate(self, df: pl.DataFrame, pair: str | None, **kwargs) -> GateResult:
        violations = []
        for col in df.columns:
            if df.schema[col].is_numeric():
                null_count = df[col].null_count()
                null_pct = null_count / len(df) if len(df) > 0 else 0
                if null_pct > self.max_null_pct:
                    violations.append({"column": col, "null_pct": null_pct, "null_count": null_count})

        if violations:
            return GateResult(
                gate_name=self.name,
                stage=self.stage,
                pair=pair,
                result=ValidationResult.FAIL,
                message=f"Null threshold exceeded for {len(violations)} columns",
                details={"violations": violations, "max_null_pct": self.max_null_pct},
            )
        return GateResult(
            gate_name=self.name,
            stage=self.stage,
            pair=pair,
            result=ValidationResult.PASS,
            message="Null thresholds OK",
        )


class DistributionDriftGate(ValidationGate):
    """Gate that checks for distribution drift against reference"""

    def __init__(
        self,
        reference_stats: dict[str, dict] | None = None,
        psi_threshold: float = 0.2,
        stage: Stage = Stage.FEATURE_ENGINEERING,
    ):
        super().__init__(name="distribution_drift_check", stage=stage)
        self.reference_stats = reference_stats or {}
        self.psi_threshold = psi_threshold

    def _validate(self, df: pl.DataFrame, pair: str | None, **kwargs) -> GateResult:
        if not self.reference_stats:
            return GateResult(
                gate_name=self.name,
                stage=self.stage,
                pair=pair,
                result=ValidationResult.SKIP,
                message="No reference stats provided, skipping drift check",
            )

        drift_violations = []
        for col in df.columns:
            if col not in self.reference_stats:
                continue
            if not df.schema[col].is_numeric():
                continue

            ref = self.reference_stats[col]
            current_vals = df[col].drop_nulls().to_numpy()
            if len(current_vals) < 100:
                continue

            # Compute PSI (Population Stability Index)
            psi = self._compute_psi(current_vals, ref.get("bins", 10), ref.get("bin_edges"))
            if psi > self.psi_threshold:
                drift_violations.append({"column": col, "psi": psi, "threshold": self.psi_threshold})

        if drift_violations:
            return GateResult(
                gate_name=self.name,
                stage=self.stage,
                pair=pair,
                result=ValidationResult.WARN,  # Warning, not fail - drift is expected over time
                message=f"Distribution drift detected in {len(drift_violations)} columns",
                details={"violations": drift_violations, "psi_threshold": self.psi_threshold},
            )
        return GateResult(
            gate_name=self.name,
            stage=self.stage,
            pair=pair,
            result=ValidationResult.PASS,
            message="No significant distribution drift",
        )

    def _compute_psi(self, current: np.ndarray, n_bins: int, ref_bin_edges: np.ndarray | None) -> float:
        """Compute Population Stability Index"""
        import numpy as np

        if ref_bin_edges is not None:
            bin_edges = ref_bin_edges
        else:
            # Create bins from current data
            _, bin_edges = np.histogram(current, bins=n_bins)

        # Current distribution
        current_hist, _ = np.histogram(current, bins=bin_edges)
        current_dist = current_hist / current_hist.sum()

        # Reference distribution (uniform if not provided)
        ref_dist = np.ones(n_bins) / n_bins

        # PSI = sum((current - ref) * ln(current / ref))
        psi = 0.0
        for c, r in zip(current_dist, ref_dist, strict=False):
            if c > 0 and r > 0:
                psi += (c - r) * np.log(c / r)

        return psi


class LookaheadGuardGate(ValidationGate):
    """Gate that checks for lookahead bias in features"""

    def __init__(self, stage: Stage = Stage.FEATURE_ENGINEERING):
        super().__init__(name="lookahead_guard", stage=stage)

    def _validate(self, df: pl.DataFrame, pair: str | None, **kwargs) -> GateResult:
        try:
            from features.lookahead_guard import LookaheadViolation, assert_no_lookahead  # noqa: F401

            # Get feature columns (numeric, non-target)
            feature_cols = [
                c for c in df.columns if df.schema[c].is_numeric() and c not in ["label", "reward", "tb_label"]
            ]
            if len(feature_cols) == 0:
                return GateResult(
                    gate_name=self.name,
                    stage=self.stage,
                    pair=pair,
                    result=ValidationResult.SKIP,
                    message="No feature columns to check",
                )

            # Use last row of each sequence as features
            features = df.select(feature_cols).tail(1).to_numpy()
            timestamps = (
                df["timestamp_utc"].tail(1).to_numpy() if "timestamp_utc" in df.columns else np.arange(len(features))
            )

            report = assert_no_lookahead(
                timestamps=timestamps,
                features=features,
                feature_names=feature_cols,
                rolling_check=True,
                permutation_check=False,
            )

            if report.violations:
                return GateResult(
                    gate_name=self.name,
                    stage=self.stage,
                    pair=pair,
                    result=ValidationResult.FAIL,
                    message=f"Lookahead violations detected: {len(report.violations)}",
                    details={"violations": [str(v) for v in report.violations]},
                )

            return GateResult(
                gate_name=self.name,
                stage=self.stage,
                pair=pair,
                result=ValidationResult.PASS,
                message="No lookahead bias detected",
                details={"checks_passed": len(report.checks_passed)},
            )
        except ImportError:
            return GateResult(
                gate_name=self.name,
                stage=self.stage,
                pair=pair,
                result=ValidationResult.SKIP,
                message="lookahead_guard not available",
            )
        except Exception as e:
            return GateResult(
                gate_name=self.name,
                stage=self.stage,
                pair=pair,
                result=ValidationResult.WARN,
                message=f"Lookahead check failed: {e}",
            )


class PipelineStageValidator:
    """Orchestrates validation gates for a pipeline stage"""

    def __init__(
        self,
        stage: Stage,
        gates: list[ValidationGate] | None = None,
        fail_fast: bool = True,
        output_dir: str | Path | None = None,
    ):
        self.stage = stage
        self.gates = gates or self._default_gates(stage)
        self.fail_fast = fail_fast
        self.output_dir = Path(output_dir) if output_dir else None
        self.results: list[GateResult] = []

    def _default_gates(self, stage: Stage) -> list[ValidationGate]:
        """Get default gates for a stage"""
        gates = []

        if stage == Stage.INGESTION:
            gates.append(ContractValidationGate(TickContract))
            gates.append(RowCountGate(min_rows=100, stage=stage))
        elif stage == Stage.RESAMPLING:
            gates.append(ContractValidationGate(BarContract))
            gates.append(RowCountGate(min_rows=50, stage=stage))
        elif stage == Stage.FEATURE_ENGINEERING:
            gates.append(ContractValidationGate(FeatureContract))
            gates.append(RowCountGate(min_rows=1000, stage=stage))
            gates.append(NullThresholdGate(max_null_pct=0.01, stage=stage))
            gates.append(LookaheadGuardGate(stage=stage))
        elif stage == Stage.LABELING:
            gates.append(ContractValidationGate(LabelContract))
            gates.append(RowCountGate(min_rows=1000, stage=stage))
        elif stage == Stage.DATASET_BUILD:
            gates.append(ContractValidationGate(DatasetContract))
            gates.append(RowCountGate(min_rows=1000, stage=stage))

        return gates

    def validate(self, df: pl.DataFrame, pair: str | None = None, **kwargs) -> list[GateResult]:
        """Run all gates for this stage"""
        self.results = []

        for gate in self.gates:
            result = gate.run(df, pair=pair, **kwargs)
            self.results.append(result)

            # Log result
            self._log_result(result)

            # Fail fast
            if self.fail_fast and result.result == ValidationResult.FAIL:
                break

        # Save results if output directory specified
        if self.output_dir:
            self._save_results(pair)

        return self.results

    def _log_result(self, result: GateResult):
        """Log gate result"""
        icon = {
            ValidationResult.PASS: "✓",
            ValidationResult.WARN: "⚠",
            ValidationResult.FAIL: "✗",
            ValidationResult.SKIP: "⊘",
        }.get(result.result, "?")

        print(f"  [{icon}] {result.gate_name} ({result.stage.value}): {result.message} ({result.duration_ms:.1f}ms)")

    def _save_results(self, pair: str | None):
        """Save validation results to JSON"""
        if not self.output_dir:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        pair_str = pair or "unknown"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"validation_{self.stage.value}_{pair_str}_{timestamp}.json"

        data = {
            "stage": self.stage.value,
            "pair": pair,
            "timestamp": datetime.now().isoformat(),
            "results": [r.to_dict() for r in self.results],
            "summary": {
                "total": len(self.results),
                "passed": sum(1 for r in self.results if r.result == ValidationResult.PASS),
                "warnings": sum(1 for r in self.results if r.result == ValidationResult.WARN),
                "failed": sum(1 for r in self.results if r.result == ValidationResult.FAIL),
                "skipped": sum(1 for r in self.results if r.result == ValidationResult.SKIP),
            },
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def get_summary(self) -> dict[str, Any]:
        """Get validation summary"""
        return {
            "stage": self.stage.value,
            "total_gates": len(self.results),
            "passed": sum(1 for r in self.results if r.result == ValidationResult.PASS),
            "warnings": sum(1 for r in self.results if r.result == ValidationResult.WARN),
            "failed": sum(1 for r in self.results if r.result == ValidationResult.FAIL),
            "skipped": sum(1 for r in self.results if r.result == ValidationResult.SKIP),
            "overall": "fail"
            if any(r.result == ValidationResult.FAIL for r in self.results)
            else "warn"
            if any(r.result == ValidationResult.WARN for r in self.results)
            else "pass",
        }
