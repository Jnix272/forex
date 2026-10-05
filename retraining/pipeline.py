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
from feature_store.config import (
    FeatureStoreSettings,
    materialize_with_settings,
    monitored_features,
    open_feature_store,
)
from feature_store.polars_store import FeatureStore
from monitoring.drift_detection import (
    KS_PVALUE_THRESHOLD,
    KS_STAT_THRESHOLD,
    DriftReport,
    DriftSeverity,
    DriftTracker,
    run_drift_check,
)
from retraining.orchestrator import (
    RetrainConfig,
    RetrainOrchestrator,
)


def _apply_drift_thresholds(
    report: DriftReport, psi_threshold: float, ks_pvalue_threshold: float, ks_stat_threshold: float
) -> DriftReport:
    """Re-derive per-feature ``drifted`` flags with configured thresholds.

    monitoring.drift_detection hardcodes PSI > 0.2; this applies
    drift_detection.psi_threshold / ks_* from run.yaml.
    """
    for r in report.feature_results:
        ks_drift = r.ks_pvalue < ks_pvalue_threshold and r.ks_stat >= ks_stat_threshold
        r.drifted = bool(r.psi > psi_threshold or ks_drift)
    drifted = sorted(r.feature_name for r in report.feature_results if r.drifted)
    report.n_drifted = len(drifted)
    report.drift_detected = bool(drifted)
    report.reasons = [m for m in report.reasons if not m.startswith(("Drift detected", "PSI max"))]
    if drifted:
        report.reasons.append(f"Drift detected in {len(drifted)}/{len(report.feature_results)} features: {drifted}")
    if report.psi_max > psi_threshold:
        report.reasons.append(f"PSI max {report.psi_max:.4f} > {psi_threshold}")
    return report


def schedule_drift_check(
    store: FeatureStore,
    feature_names: list[str],
    baseline_window_days: int = 90,
    live_window_days: int = 7,
    as_of: datetime | None = None,
    *,
    psi_bins: int = 10,
    psi_threshold: float = 0.2,
    ks_pvalue_threshold: float = KS_PVALUE_THRESHOLD,
    ks_stat_threshold: float = KS_STAT_THRESHOLD,
) -> DriftReport:
    """Store-backed drift check honouring all drift_detection thresholds.

    Same windowing as monitoring.drift_detection.schedule_drift_check, but
    psi_bins / psi_threshold / ks_* are applied *before* results are persisted,
    so the orchestrator's drift trigger (which reads persisted flags) sees them.
    """
    now = as_of or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    live_end = now
    live_start = now - timedelta(days=live_window_days)
    baseline_end = live_start - timedelta(days=1)
    baseline_start = baseline_end - timedelta(days=baseline_window_days)

    tracker = DriftTracker(store)
    baseline_data, live_data, missing = {}, {}, []
    for name in feature_names:
        bl = store.load_feature(name, baseline_start, baseline_end)
        lv = store.load_feature(name, live_start, live_end)
        if bl is None or len(bl) == 0 or lv is None or len(lv) == 0:
            missing.append(name)
            continue
        col = next(c for c in bl.columns if c != "timestamp_utc")
        baseline_data[name] = bl[col].to_numpy()
        live_data[name] = lv[col].to_numpy()

    report = run_drift_check(
        baseline_data,
        live_data,
        baseline_time=baseline_start.isoformat(),
        live_time=live_start.isoformat(),
        psi_bins=int(psi_bins),
        ks_pvalue_threshold=float(ks_pvalue_threshold),
        ks_stat_threshold=float(ks_stat_threshold),
    )
    report = _apply_drift_thresholds(report, float(psi_threshold), float(ks_pvalue_threshold), float(ks_stat_threshold))
    tracker._persist_report(report)
    if missing:
        report.reasons.append(f"Features not found in store: {missing}")
    return report

# ════════════════════════════════════════════════════════════════════════════
# Pipeline Configuration
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    # Feature Store
    feature_store_root: str = "data/feature_store"
    feature_names: list[str] = field(
        default_factory=lambda: [
            # Names MUST match exact aliases from HAELTFeatureBuilder.build()
            "close",
            "ret_5",
            "ret_20",  # lag_returns() -> ret_{w}, NOT log_ret_{w}
            "atr_6",
            "atr_20",
            "vol_20",  # rolling_volatility(20) -> vol_20, NOT rolling_vol_20
            "ofi",  # order_flow_imbalance() -> ofi, NOT ofi_20
            "obi_proxy",
            "time_sin",
            "time_cos",  # HAELTFeatureBuilder temporal -> time_sin/time_cos
        ]
    )

    # Feature store (run.yaml feature_store.*)
    feature_store_enabled: bool = True
    feature_store_registry_db: str = "registry.db"
    feature_store_data_root: str = "features"
    feature_store_compression: str = "zstd"
    feature_store_default_strategy: str = "eager_batch"
    feature_store_incremental_lookback_bars: int = 100
    feature_store_job_queue_enabled: bool = False
    feature_store_job_queue_max_workers: int = 1
    feature_store_job_queue_poll_interval_sec: float = 60.0

    # Drift Detection (run.yaml drift_detection.*)
    drift_enabled: bool = True
    drift_baseline_days: int = 90
    drift_live_days: int = 7
    drift_psi_bins: int = 10
    drift_psi_threshold: float = 0.2
    drift_ks_pvalue_threshold: float = 0.05
    drift_ks_stat_threshold: float = 0.05
    # None = feature_names; raw price levels are always excluded.
    drift_monitored_features: list[str] | None = None
    drift_auto_schedule: bool = True
    drift_schedule_hours: int = 24

    # Retrain Orchestrator (run.yaml retraining.*)
    retrain_enabled: bool = True
    retrain_model_family: str = "haelt"
    retrain_dry_run: bool = False  # Production mode - retraining will execute
    retrain_model_root: str = "checkpoints"
    retrain_min_interval_days: int = 1
    retrain_schedule_interval_days: int = 7
    retrain_drift_feature_frac: float = 0.3
    retrain_promote_on_complete: bool = True
    retrain_extras: list[str] = field(
        default_factory=lambda: [
            "--epochs",
            "40",
            "--batch-size",
            "128",
        ]
    )

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
        c = self.config
        self.fs_settings = FeatureStoreSettings(
            enabled=c.feature_store_enabled,
            root=c.feature_store_root,
            registry_db=c.feature_store_registry_db,
            data_root=c.feature_store_data_root,
            compression=c.feature_store_compression,
            default_strategy=c.feature_store_default_strategy,
            incremental_lookback_bars=c.feature_store_incremental_lookback_bars,
            auto_materialize=c.auto_materialize,
            job_queue_enabled=c.feature_store_job_queue_enabled,
            job_queue_max_workers=c.feature_store_job_queue_max_workers,
            job_queue_poll_interval_sec=c.feature_store_job_queue_poll_interval_sec,
        )
        self.store = open_feature_store(self.fs_settings)
        self.drift_tracker = DriftTracker(self.store)
        self.orchestrator = RetrainOrchestrator(
            self.store,
            config=RetrainConfig(
                enable_drift_trigger=bool(c.drift_enabled),
                enable_scheduled_trigger=True,
                min_interval_days=int(c.retrain_min_interval_days),
                schedule_interval_days=int(c.retrain_schedule_interval_days),
                psi_threshold=float(c.drift_psi_threshold),
                drift_feature_frac=float(c.retrain_drift_feature_frac),
                promote_on_complete=bool(c.retrain_promote_on_complete),
                model_root=Path(c.retrain_model_root),
            ),
        )
        self._log("Pipeline initialized")

    def drift_features(self) -> list[str]:
        src = self.config.drift_monitored_features
        return monitored_features(
            src if src else self.config.feature_names,
            source="drift_detection.monitored_features" if src else "PipelineConfig.feature_names",
        )

    # ──────────────────────────────────────────────────────────────────────
    # Step 1: Materialize Features
    # ──────────────────────────────────────────────────────────────────────

    def materialize(self, bars: pl.DataFrame, start: datetime, end: datetime) -> dict[str, pl.DataFrame]:
        """Materialize configured features from raw bars."""
        if not (self.config.auto_materialize and self.config.feature_store_enabled):
            return {}
        self._log(f"Materializing {len(self.config.feature_names)} features...")
        if self.fs_settings.job_queue_enabled:
            result = materialize_with_settings(self.store, self.fs_settings, self.config.feature_names, bars, start, end)
        else:
            result = materialize_feature_set(
                self.store,
                self.config.feature_names,
                bars,
                start,
                end,
            )
        self._log(f"  Materialized {len(result)} features")
        return result

    # ──────────────────────────────────────────────────────────────────────
    # Step 2: Check Drift
    # ──────────────────────────────────────────────────────────────────────

    def check_drift(self, as_of: datetime | None = None) -> DriftReport:
        """Run scheduled drift check using FeatureStore data."""
        if not self.config.drift_enabled:
            self._log("Drift check disabled (drift_detection.enabled=false)")
            return DriftReport(
                feature_results=[],
                psi_max=0.0,
                ks_min_pvalue=1.0,
                ks_max_stat=0.0,
                n_drifted=0,
                n_features=0,
                overall_severity=DriftSeverity.NONE,
                drift_detected=False,
                baseline_time="",
                live_time="",
                reasons=["drift detection disabled"],
            )
        self._log("Running drift check...")
        report = schedule_drift_check(
            self.store,
            feature_names=self.drift_features(),
            baseline_window_days=self.config.drift_baseline_days,
            live_window_days=self.config.drift_live_days,
            as_of=as_of,
            psi_bins=self.config.drift_psi_bins,
            psi_threshold=self.config.drift_psi_threshold,
            ks_pvalue_threshold=self.config.drift_ks_pvalue_threshold,
            ks_stat_threshold=self.config.drift_ks_stat_threshold,
        )
        self._log(f"  Drift: {report.n_drifted}/{report.n_features} features (PSI max={report.psi_max:.4f})")
        return report

    # ──────────────────────────────────────────────────────────────────────
    # Step 3: Evaluate Triggers & Retrain
    # ──────────────────────────────────────────────────────────────────────

    def evaluate_and_retrain(self, drift_report: DriftReport) -> dict[str, Any]:
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
        start: datetime | None = None,
        end: datetime | None = None,
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
                drift_report
                or DriftReport(
                    feature_results=[],
                    psi_max=0.0,
                    ks_min_pvalue=1.0,
                    ks_max_stat=0.0,
                    n_drifted=0,
                    n_features=0,
                    overall_severity=DriftSeverity.NONE,
                    drift_detected=False,
                    baseline_time="",
                    live_time="",
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

        with open(yaml_path, encoding="utf-8-sig") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return PipelineConfig()

    fs = FeatureStoreSettings.from_dict(
        {"enabled": True, "auto_materialize": True, **(data.get("feature_store") or {})}
    )
    drift_cfg = data.get("drift_detection") or {}
    retrain_cfg = data.get("retraining") or {}
    monitored = drift_cfg.get("monitored_features")

    return PipelineConfig(
        feature_store_root=fs.root,
        feature_store_enabled=fs.enabled,
        feature_store_registry_db=fs.registry_db,
        feature_store_data_root=fs.data_root,
        feature_store_compression=fs.compression,
        feature_store_default_strategy=fs.default_strategy,
        feature_store_incremental_lookback_bars=fs.incremental_lookback_bars,
        feature_store_job_queue_enabled=fs.job_queue_enabled,
        feature_store_job_queue_max_workers=fs.job_queue_max_workers,
        feature_store_job_queue_poll_interval_sec=fs.job_queue_poll_interval_sec,
        auto_materialize=fs.auto_materialize,
        drift_enabled=bool(drift_cfg.get("enabled", True)),
        drift_baseline_days=int(drift_cfg.get("baseline_window_days", 90)),
        drift_live_days=int(drift_cfg.get("live_window_days", 7)),
        drift_psi_bins=int(drift_cfg.get("psi_bins", 10)),
        drift_psi_threshold=float(drift_cfg.get("psi_threshold", 0.2)),
        drift_ks_pvalue_threshold=float(drift_cfg.get("ks_pvalue_threshold", 0.05)),
        drift_ks_stat_threshold=float(drift_cfg.get("ks_stat_threshold", 0.05)),
        drift_monitored_features=monitored_features(monitored) if monitored else None,
        drift_auto_schedule=drift_cfg.get("auto_schedule", True),
        drift_schedule_hours=drift_cfg.get("schedule_hours", 24),
        retrain_enabled=retrain_cfg.get("enabled", True),
        retrain_model_family=retrain_cfg.get("model_family", "haelt"),
        retrain_dry_run=retrain_cfg.get("dry_run", False),
        retrain_extras=retrain_cfg.get("extras", []),
        retrain_model_root=str(retrain_cfg.get("model_root", "checkpoints")),
        retrain_min_interval_days=int(retrain_cfg.get("min_interval_days", 1)),
        retrain_schedule_interval_days=int(retrain_cfg.get("schedule_interval_days", 7)),
        retrain_drift_feature_frac=float(retrain_cfg.get("drift_feature_frac", 0.3)),
        retrain_promote_on_complete=bool(retrain_cfg.get("promote_on_complete", True)),
    )


if __name__ == "__main__":
    # Demo
    print("Pipeline module loaded. Available components:")
    print("  FullPipeline")
    print("  PipelineConfig")
    print("  run_pipeline()")
    print("  load_config_from_yaml()")
