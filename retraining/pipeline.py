"""
Full Pipeline Integration (Phase 4)
====================================
End-to-end wiring: Feature Store → Drift Detection → Retrain Orchestrator.
Provides the `FullPipeline` class that coordinates the complete lifecycle.

Flow:
  1. Initialize FeatureStore (load or create registry DB)
  2. Materialize features for a time range
  3. Run drift check (baseline vs live)
  4. Evaluate retrain triggers
  5. Conditionally launch retraining
  6. Validate and promote new model
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from data.feature_materializers import materialize_feature_set
from data.feature_store import FeatureStore
from monitoring.drift_detection import (
    DriftReport,
    DriftSeverity,
    DriftTracker,
    schedule_drift_check,
)
from retraining.orchestrator import (
    RetrainConfig,
    RetrainOrchestrator,
)

# ════════════════════════════════════════════════════════════════════════════
# Pipeline Configuration
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    # Feature Store
    feature_store_root: str = "data/feature_store"
    feature_names: list[str] = field(default_factory=lambda: [
        # Names MUST match exact aliases from HAELTFeatureBuilder.build()
        "close", "ret_5", "ret_20",   # lag_returns() -> ret_{w}, NOT log_ret_{w}
        "atr_6", "atr_20",
        "vol_20",                      # rolling_volatility(20) -> vol_20, NOT rolling_vol_20
        "ofi",                         # order_flow_imbalance() -> ofi, NOT ofi_20
        "obi_proxy",
        "time_sin", "time_cos",        # HAELTFeatureBuilder temporal -> time_sin/time_cos
    ])

    # Drift Detection
    drift_baseline_days: int = 90
    drift_live_days: int = 7
    drift_psi_bins: int = 10
    drift_auto_schedule: bool = True
    drift_schedule_hours: int = 24

    # Retrain Orchestrator
    retrain_enabled: bool = True
    retrain_model_family: str = "haelt"
    retrain_dry_run: bool = False  # Production mode — retraining will execute
    retrain_extras: list[str] = field(default_factory=lambda: [
        "--epochs", "40", "--batch-size", "128",
    ])

    # Materialization
    auto_materialize: bool = True
    materialize_bars: int = 500_000

    # Monitoring
    log_every_step: bool = True


# ════════════════════════════════════════════════════════════════════════════
# Full Pipeline
# ════════════════════════════════════════════════════════════════════════════

class FullPipeline:
    """
    End-to-end pipeline integrating all phases.

    Usage:
        pipeline = FullPipeline(config)
        result = pipeline.run(bars)
        print(result["drift_report"].drift_detected)
        print(result["retrain_result"])
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.store = FeatureStore(root=self.config.feature_store_root)
        self.drift_tracker = DriftTracker(self.store)
        self.orchestrator = RetrainOrchestrator(
            self.store,
            config=RetrainConfig(
                enable_drift_trigger=True,
                enable_scheduled_trigger=True,
                schedule_interval_days=7,
                psi_threshold=0.2,
                drift_feature_frac=0.3,
                promote_on_complete=True,
                model_root=Path("checkpoints"),
            ),
        )
        self._log("Pipeline initialized")

    # ──────────────────────────────────────────────────────────────────────
    # Step 1: Materialize Features
    # ──────────────────────────────────────────────────────────────────────

    def materialize(
        self, bars: pl.DataFrame, start: datetime, end: datetime
    ) -> dict[str, pl.DataFrame]:
        """Materialize configured features from raw bars."""
        if not self.config.auto_materialize:
            return {}
        self._log(f"Materializing {len(self.config.feature_names)} features...")
        result = materialize_feature_set(
            self.store, self.config.feature_names, bars, start, end,
        )
        self._log(f"  Materialized {len(result)} features")
        return result

    # ──────────────────────────────────────────────────────────────────────
    # Step 2: Check Drift
    # ──────────────────────────────────────────────────────────────────────

    def check_drift(self, as_of: datetime = None) -> DriftReport:
        """Run scheduled drift check using FeatureStore data."""
        self._log("Running drift check...")
        report = schedule_drift_check(
            self.store,
            feature_names=self.config.feature_names,
            baseline_window_days=self.config.drift_baseline_days,
            live_window_days=self.config.drift_live_days,
            as_of=as_of,
        )
        self._log(
            f"  Drift: {report.n_drifted}/{report.n_features} features "
            f"(PSI max={report.psi_max:.4f})"
        )
        return report

    # ──────────────────────────────────────────────────────────────────────
    # Step 3: Evaluate Triggers & Retrain
    # ──────────────────────────────────────────────────────────────────────

    def evaluate_and_retrain(
        self, drift_report: DriftReport
    ) -> dict[str, Any]:
        """Evaluate all triggers and conditionally retrain."""
        if not self.config.retrain_enabled:
            return {"retrain_skipped": True, "reason": "retrain disabled in config"}

        self._log("Evaluating retrain triggers...")
        should, reason, context = self.orchestrator.should_retrain(
            self.config.retrain_model_family,
        )

        if not should:
            self._log(f"  No trigger: {reason}")
            return {"retrain_skipped": True, "reason": reason}

        self._log(f"  Trigger: {reason}")

        # Augment with drift context
        if drift_report.drift_detected:
            context["drift_n_features"] = drift_report.n_features
            context["drift_psi_max"] = drift_report.psi_max

        result = self.orchestrator.retrain(
            family=self.config.retrain_model_family,
            reason=reason,
            extras=self.config.retrain_extras,
            dry_run=self.config.retrain_dry_run,
        )
        self._log(f"  Retrain result: {result.get('status', 'unknown')}")
        return result

    # ──────────────────────────────────────────────────────────────────────
    # Step 4: Full Run
    # ──────────────────────────────────────────────────────────────────────

    def run(
        self,
        bars: pl.DataFrame = None,
        start: datetime = None,
        end: datetime = None,
        skip_materialize: bool = False,
    ) -> dict[str, Any]:
        """
        Execute full pipeline end-to-end.

        Args:
            bars: Raw OHLCV bars for materialization (optional).
            start: Start datetime for materialization.
            end: End datetime for materialization.
            skip_materialize: Skip materialization step.

        Returns:
            Dict with keys: materialization, drift_report, retrain_result, status.
        """
        result: dict[str, Any] = {
            "status": "running",
            "materialization": {},
            "drift_report": None,
            "retrain_result": None,
            "errors": [],
        }

        # Step 1: Materialize
        if bars is not None and not skip_materialize:
            try:
                mat_start = start or datetime.now(UTC) - timedelta(days=1)
                mat_end = end or datetime.now(UTC)
                result["materialization"] = self.materialize(bars, mat_start, mat_end)
            except Exception as e:
                self._log(f"Materialization error: {e}")
                result["errors"].append(str(e))

        # Step 2: Drift Check
        drift_report = None
        try:
            drift_as_of = end if end is not None else None
            drift_report = self.check_drift(as_of=drift_as_of)
            result["drift_report"] = {
                "drift_detected": drift_report.drift_detected,
                "n_drifted": drift_report.n_drifted,
                "n_features": drift_report.n_features,
                "psi_max": drift_report.psi_max,
                "severity": drift_report.overall_severity.value,
                "reasons": drift_report.reasons,
            }
        except Exception as e:
            self._log(f"Drift check error: {e}")
            result["errors"].append(str(e))

        # Step 3: Retrain
        try:
            retrain_result = self.evaluate_and_retrain(
                drift_report or DriftReport(
                    feature_results=[], psi_max=0.0, ks_min_pvalue=1.0, ks_max_stat=0.0,
                    n_drifted=0, n_features=0, overall_severity=DriftSeverity.NONE,
                    drift_detected=False, baseline_time="", live_time="",
                )
            )
            result["retrain_result"] = retrain_result
        except Exception as e:
            self._log(f"Retrain error: {e}")
            result["errors"].append(str(e))

        result["status"] = "error" if result["errors"] else "complete"
        return result

    def status(self) -> dict[str, Any]:
        """Get full pipeline status summary."""
        return {
            "feature_store": {
                "root": str(self.store.root),
                "feature_count": len(self.store.list_features()),
                "storage": self.store.get_storage_stats(),
            },
            "orchestrator": self.orchestrator.get_status(),
            "config": asdict(self.config),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if self.config.log_every_step:
            print(f"[Pipeline] {msg}")


# ════════════════════════════════════════════════════════════════════════════
# Convenience: one-shot pipeline run
# ════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    config: PipelineConfig = None,
    bars: pl.DataFrame = None,
    **kwargs,
) -> dict[str, Any]:
    """Convenience: create pipeline, run, return result."""
    pipeline = FullPipeline(config)
    return pipeline.run(bars=bars, **kwargs)


# ════════════════════════════════════════════════════════════════════════════
# Config loader from run.yaml
# ════════════════════════════════════════════════════════════════════════════

def load_config_from_yaml(yaml_path: str = "config/run.yaml") -> PipelineConfig:
    """Load pipeline config from YAML, merging with defaults."""
    try:
        import yaml
        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return PipelineConfig()

    fs_cfg = data.get("feature_store", {})
    drift_cfg = data.get("drift_detection", {})
    retrain_cfg = data.get("retraining", {})

    return PipelineConfig(
        feature_store_root=fs_cfg.get("root", "data/feature_store"),
        drift_baseline_days=drift_cfg.get("baseline_window_days", 90),
        drift_live_days=drift_cfg.get("live_window_days", 7),
        drift_auto_schedule=drift_cfg.get("auto_schedule", True),
        drift_schedule_hours=drift_cfg.get("schedule_hours", 24),
        retrain_enabled=retrain_cfg.get("enabled", True),
        retrain_model_family=retrain_cfg.get("model_family", "haelt"),
        retrain_dry_run=retrain_cfg.get("dry_run", False),
        retrain_extras=retrain_cfg.get("extras", []),
    )


if __name__ == "__main__":
    # Demo
    print("Pipeline module loaded. Available components:")
    print("  FullPipeline")
    print("  PipelineConfig")
    print("  run_pipeline()")
    print("  load_config_from_yaml()")
