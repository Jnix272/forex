"""
Retrain Orchestrator (Phase 3)
==============================
Automatic retraining lifecycle: trigger detection, model versioning,
training job orchestration, and promotion validation.

Flow:
  1. DriftTracker detects feature drift → triggers RetrainOrchestrator
  2. Orchestrator increments model version, creates versioned checkpoint dir
  3. Launches training subprocess (train_gpu.py or train_xgboost.py)
  4. Validates new model against promotion gates
  5. Registers in ModelRegistry, promotes to production
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from data.feature_store import FeatureStore

# ════════════════════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_MODEL_ROOT = Path("checkpoints")
DEFAULT_REGISTRY_FILE = "model_registry.json"
TRAINING_SCRIPT = "training/train_gpu.py"
XGB_TRAINING_SCRIPT = "training/train_xgboost.py"

# Minimum number of days between retrains for the same model family
MIN_RETRAIN_INTERVAL_DAYS = 1

# Promotion gate thresholds
MIN_SHARPE = 0.5
MIN_PROFIT_FACTOR = 1.3
MAX_DRAWDOWN = 0.15


class ModelFamily(Enum):
    HAELT = "haelt"
    TFT = "tft"
    TRANSFORMER = "transformer"
    MAMBA = "mamba"
    GNN = "gnn"
    EXPERT = "expert"
    XGBOOST = "xgboost"
    ENSEMBLE = "ensemble"


class RetrainReason(Enum):
    SCHEDULED = "scheduled"
    DRIFT_DETECTED = "drift_detected"
    PERFORMANCE_DEGRADED = "performance_degraded"
    MANUAL = "manual"
    INITIAL = "initial"


class ModelStatus(Enum):
    TRAINING = "training"
    READY = "ready"
    PROMOTED = "promoted"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


# ════════════════════════════════════════════════════════════════════════════
# Data Classes
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class ModelRecord:
    """Single model version record."""

    version: int
    family: str
    status: ModelStatus
    checkpoint_dir: str
    created_at: str
    reason: str
    training_cmd: str
    metrics: dict[str, Any] = field(default_factory=dict)
    drift_report: dict[str, Any] | None = None
    promoted_at: str | None = None
    rollback_at: str | None = None
    notes: str = ""


@dataclass
class RetrainConfig:
    """Configuration for the retrain orchestrator."""

    model_root: Path = DEFAULT_MODEL_ROOT
    registry_file: str = DEFAULT_REGISTRY_FILE
    min_interval_days: int = MIN_RETRAIN_INTERVAL_DAYS
    enable_drift_trigger: bool = True
    enable_scheduled_trigger: bool = True
    enable_performance_trigger: bool = False
    schedule_interval_days: int = 7
    max_versions_kept: int = 10
    psi_threshold: float = 0.2
    drift_feature_frac: float = 0.3  # Fraction of features drifted to trigger
    xgboost_enabled: bool = True
    promote_on_complete: bool = False


# ════════════════════════════════════════════════════════════════════════════
# Model Registry
# ════════════════════════════════════════════════════════════════════════════


class ModelRegistry:
    """
    Versioned model registry backed by JSON file.
    Tracks training runs, promotion status, and metadata.
    """

    def __init__(self, root: str | Path = DEFAULT_MODEL_ROOT):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._path = self.root / DEFAULT_REGISTRY_FILE
        self._records: list[ModelRecord] = []
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            data = json.loads(self._path.read_text())
            self._records = [ModelRecord(**r) for r in data.get("models", [])]

    def _save(self) -> None:
        data = {
            "registry_version": 1,
            "updated_at": datetime.now(UTC).isoformat(),
            "models": [asdict(r) for r in self._records],
        }
        self._path.write_text(json.dumps(data, indent=2, default=str))

    def next_version(self, family: str) -> int:
        """Get next version number for a model family."""
        versions = [r.version for r in self._records if r.family == family and r.status != ModelStatus.FAILED]
        return max(versions, default=0) + 1

    def register(
        self,
        family: str,
        reason: str,
        checkpoint_dir: str,
        training_cmd: str = "",
        metrics: dict | None = None,
    ) -> ModelRecord:
        """Register a new model training run."""
        version = self.next_version(family)
        record = ModelRecord(
            version=version,
            family=family,
            status=ModelStatus.TRAINING,
            checkpoint_dir=str(checkpoint_dir),
            created_at=datetime.now(UTC).isoformat(),
            reason=reason,
            training_cmd=training_cmd,
            metrics=metrics or {},
        )
        self._records.append(record)
        self._save()
        return record

    def update_status(
        self,
        family: str,
        version: int,
        status: ModelStatus,
        metrics: dict | None = None,
        notes: str | None = None,
    ) -> None:
        """Update model status and optional metrics."""
        for r in self._records:
            if r.family == family and r.version == version:
                r.status = status
                if metrics:
                    r.metrics.update(metrics)
                if status == ModelStatus.PROMOTED:
                    r.promoted_at = datetime.now(UTC).isoformat()
                elif status == ModelStatus.ROLLED_BACK:
                    r.rollback_at = datetime.now(UTC).isoformat()
                if notes:
                    r.notes = notes
                self._save()
                return
        raise KeyError(f"Model {family} v{version} not found")

    def get_latest(self, family: str, status: ModelStatus = None) -> ModelRecord | None:
        """Get latest version of a model family, optionally filtered by status."""
        candidates = [r for r in self._records if r.family == family]
        if status:
            candidates = [r for r in candidates if r.status == status]
        return max(candidates, key=lambda r: r.version) if candidates else None

    def get_production(self) -> ModelRecord | None:
        """Get the currently promoted production model."""
        for r in sorted(self._records, key=lambda x: x.version, reverse=True):
            if r.status == ModelStatus.PROMOTED:
                return r
        return None

    def list_models(self, family: str | None = None) -> list[ModelRecord]:
        """List all records, optionally filtered by family."""
        if family:
            return [r for r in self._records if r.family == family]
        return list(self._records)

    def get_version(self, family: str, version: int) -> ModelRecord | None:
        for r in self._records:
            if r.family == family and r.version == version:
                return r
        return None

    def rollback(self, family: str) -> ModelRecord | None:
        """Rollback to previous promoted version."""
        promoted = sorted(
            [r for r in self._records if r.family == family and r.status == ModelStatus.PROMOTED],
            key=lambda x: x.version,
            reverse=True,
        )
        if len(promoted) < 2:
            return None
        current = promoted[0]
        target = promoted[1]
        self.update_status(family, current.version, ModelStatus.ROLLED_BACK)
        self.update_status(family, target.version, ModelStatus.PROMOTED)
        return target

    def cleanup_old_versions(self, keep: int = 10) -> int:
        """Remove old checkpoint directories beyond the keep limit."""
        families = {r.family for r in self._records}
        removed = 0
        for family in families:
            versions = sorted(
                [r for r in self._records if r.family == family],
                key=lambda x: x.version,
                reverse=True,
            )
            for r in versions[keep:]:
                ckpt_dir = Path(r.checkpoint_dir)
                if ckpt_dir.exists():
                    shutil.rmtree(ckpt_dir)
                    removed += 1
        return removed


# ════════════════════════════════════════════════════════════════════════════
# Promotion Gate
# ════════════════════════════════════════════════════════════════════════════


def check_promotion_gates(metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate model metrics against promotion thresholds using the strict PromotionGate.
    Returns (passed, reasons).
    """
    from validation.promotion_gate import PromotionGate

    gate = PromotionGate()

    res = gate.evaluate(
        sharpe=metrics.get("val_sharpe", 0.0),
        profit_factor=metrics.get("profit_factor", 0.0),
        max_drawdown=metrics.get("max_drawdown", 1.0),
        n_trades=metrics.get("n_trades", 0),
        gross_pnl=metrics.get("gross_pnl", 0.0) if metrics.get("gross_pnl") is not None else 1.0,
        transaction_costs=metrics.get("transaction_costs", 0.0),
        n_obs=metrics.get("n_obs", 1),
        n_backtest_trials=metrics.get("n_backtest_trials", 1),
        backtest_sharpe_std=metrics.get("backtest_sharpe_std", 0.0),
        regime_pnl=metrics.get("regime_pnl", {}),
        skewness=metrics.get("skewness", 0.0),
        kurtosis=metrics.get("kurtosis", 3.0),
        turnover_rate=metrics.get("turnover_rate"),
        avg_latency_ms=metrics.get("avg_latency_ms"),
    )

    return res["promoted"], res["reasons"]


# ════════════════════════════════════════════════════════════════════════════
# Retrain Orchestrator
# ════════════════════════════════════════════════════════════════════════════


class RetrainOrchestrator:
    """
    High-level coordinator for the retraining lifecycle.

    Responsibilities:
    - Monitor drift (via DriftTracker)
    - Decide when to retrain (drift/schedule/performance triggers)
    - Launch training jobs via subprocess
    - Validate and promote new models
    - Maintain versioned model registry
    """

    def __init__(
        self,
        feature_store: FeatureStore,
        config: RetrainConfig = None,
    ):
        self.store = feature_store
        self.config = config or RetrainConfig()
        self.registry = ModelRegistry(root=self.config.model_root)

    # ──────────────────────────────────────────────────────────────────────
    # Trigger Evaluation
    # ──────────────────────────────────────────────────────────────────────

    def should_retrain(self, family: str = "haelt", force: bool = False) -> tuple[bool, str, dict[str, Any]]:
        """
        Evaluate all retrain triggers.
        Returns (should_retrain, reason, context).
        """
        now = datetime.now(UTC)
        latest = self.registry.get_latest(family, status=ModelStatus.PROMOTED)

        if force:
            return True, RetrainReason.MANUAL.value, {"forced": True}

        # Initial training (no promoted model exists)
        if latest is None:
            return True, RetrainReason.INITIAL.value, {"initial": True}

        # Scheduled retrain
        if self.config.enable_scheduled_trigger:
            last_time = datetime.fromisoformat(latest.promoted_at or latest.created_at)
            if now - last_time > timedelta(days=self.config.schedule_interval_days):
                return (
                    True,
                    RetrainReason.SCHEDULED.value,
                    {
                        "last_retrain": last_time.isoformat(),
                        "interval_days": self.config.schedule_interval_days,
                    },
                )

        # Drift-triggered retrain
        if self.config.enable_drift_trigger:
            drift_summary = self._check_drift_trigger()
            if drift_summary.get("should_retrain", False):
                return True, RetrainReason.DRIFT_DETECTED.value, drift_summary

        return False, "", {}

    def _check_drift_trigger(self) -> dict[str, Any]:
        """Check if drift severity warrants retraining."""
        from monitoring.drift_detection import DriftTracker

        tracker = DriftTracker(self.store)
        summary = tracker.get_drift_summary(since=datetime.now(UTC) - timedelta(days=7))

        n_drifted = summary.get("n_drifted", 0)
        n_total = summary.get("total_checks", 0)
        frac = n_drifted / max(n_total, 1)

        should_retrain = frac >= self.config.drift_feature_frac

        return {
            "should_retrain": should_retrain,
            "n_drifted": n_drifted,
            "n_checks": n_total,
            "drift_fraction": frac,
            "threshold": self.config.drift_feature_frac,
            "drift_rate": summary.get("drift_rate", 0),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Training Execution
    # ──────────────────────────────────────────────────────────────────────

    def _build_checkpoint_dir(self, family: str, version: int) -> Path:
        """Build versioned checkpoint directory path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{family}_v{version:03d}_{timestamp}"
        return self.config.model_root / dir_name

    def _build_training_cmd(
        self, family: str, version: int, checkpoint_dir: Path, extras: list[str] | None = None
    ) -> list[str]:
        """Build the training command for subprocess."""
        if family == ModelFamily.XGBOOST.value:
            script = XGB_TRAINING_SCRIPT
        else:
            script = TRAINING_SCRIPT

        cmd = [
            sys.executable,
            script,
            "--model",
            family,
            "--checkpoint-dir",
            str(checkpoint_dir),
        ]

        if extras:
            cmd.extend(extras)

        return cmd

    def retrain(
        self,
        family: str = "haelt",
        reason: str = "manual",
        extras: list[str] | None = None,
        timeout_seconds: int = 86400,  # 24h default
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """
        Execute a full retrain lifecycle:
          1. Register new model version
          2. Create checkpoint directory
          3. Launch training (unless dry_run)
          4. Validate/promote on success
          5. Return result

        Returns dict with version, status, checkpoint_dir, training_output, etc.
        """
        version = self.registry.next_version(family)
        ckpt_dir = self._build_checkpoint_dir(family, version)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        training_cmd = " ".join(self._build_training_cmd(family, version, ckpt_dir, extras))

        record = self.registry.register(
            family=family,
            reason=reason,
            checkpoint_dir=str(ckpt_dir),
            training_cmd=training_cmd,
            metrics={"started_at": datetime.now(UTC).isoformat()},
        )

        result = {
            "version": version,
            "family": family,
            "checkpoint_dir": str(ckpt_dir),
            "status": ModelStatus.TRAINING.value,
            "reason": reason,
            "training_cmd": training_cmd,
        }

        if dry_run:
            result["dry_run"] = True
            result["status"] = "dry_run"
            return result

        # Launch training
        cmd = self._build_training_cmd(family, version, ckpt_dir, extras)
        print(f"[Retrain] Launching: {' '.join(cmd)}")

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
            output = proc.stdout + "\n" + proc.stderr
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            return self._fail_training(record, result, "Training timed out")
        except Exception as e:
            return self._fail_training(record, result, f"Training failed: {e}")

        result["returncode"] = returncode
        result["training_output"] = output[-5000:] if len(output) > 5000 else output

        if returncode != 0:
            return self._fail_training(record, result, f"Non-zero exit: {returncode}")

        # Extract metrics from training output (simplified)
        metrics = self._extract_metrics(output)
        result["metrics"] = metrics

        # Promotion gate
        if self.config.promote_on_complete:
            passed, reasons = check_promotion_gates(metrics)
            if passed:
                self.registry.update_status(
                    family,
                    version,
                    ModelStatus.PROMOTED,
                    metrics=metrics,
                    notes="Promoted by orchestrator",
                )
                result["status"] = ModelStatus.PROMOTED.value
                result["promotion_reasons"] = []
            else:
                self.registry.update_status(
                    family,
                    version,
                    ModelStatus.READY,
                    metrics=metrics,
                    notes=f"Gate failed: {'; '.join(reasons)}",
                )
                result["status"] = ModelStatus.READY.value
                result["promotion_reasons"] = reasons
        else:
            self.registry.update_status(family, version, ModelStatus.READY, metrics=metrics)
            result["status"] = ModelStatus.READY.value

        return result

    def _fail_training(self, record: ModelRecord, result: dict, error: str) -> dict:
        self.registry.update_status(
            record.family,
            record.version,
            ModelStatus.FAILED,
            notes=error,
        )
        result["status"] = ModelStatus.FAILED.value
        result["error"] = error
        return result

    def _extract_metrics(self, output: str) -> dict[str, Any]:
        """
        Extract key metrics from training stdout.
        Parses log lines like:
          "Best val_sharpe: 1.23"
          "val_loss: 0.456"
        """
        metrics = {}
        import re

        patterns = {
            "val_sharpe": r"val[_\s]sharpe[:\s]+([-]?\d+\.?\d*)",
            "val_loss": r"val[_\s]loss[:\s]+([-]?\d+\.?\d*)",
            "train_loss": r"train[_\s]loss[:\s]+([-]?\d+\.?\d*)",
            "best_sharpe": r"best[_\s]sharpe[:\s]+([-]?\d+\.?\d*)",
            "profit_factor": r"profit[_\s]factor[:\s]+([-]?\d+\.?\d*)",
            "max_drawdown": r"max[_\s]drawdown[:\s]+([-]?\d+\.?\d*)",
            "accuracy": r"acc[:\s]+([-]?\d+\.?\d*)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                metrics[key] = float(match.group(1))

        return metrics

    # ──────────────────────────────────────────────────────────────────────
    # Convenience / Scheduling
    # ──────────────────────────────────────────────────────────────────────

    def check_and_retrain(
        self,
        family: str = "haelt",
        extras: list[str] | None = None,
        force: bool = False,
        dry_run: bool = False,
    ) -> dict | None:
        """
        Check triggers and retrain if needed. Returns result dict or None.
        """
        should, reason, context = self.should_retrain(family, force=force)
        if not should:
            return {
                "should_retrain": False,
                "reason": reason,
                "context": context,
            }
        result = self.retrain(
            family=family,
            reason=reason,
            extras=extras,
            dry_run=dry_run,
        )
        result["trigger_context"] = context
        return result

    def get_status(self, family: str | None = None) -> dict:
        """Get orchestrator status summary."""
        latest = None
        for f in [family] if family else [e.value for e in ModelFamily]:
            rec = self.registry.get_latest(f)
            if rec:
                latest = rec
        production = self.registry.get_production()

        return {
            "latest_model": asdict(latest) if latest else None,
            "production_model": asdict(production) if production else None,
            "total_models": len(self.registry.list_models(family)),
            "config": asdict(self.config),
        }


# ════════════════════════════════════════════════════════════════════════════
# Drift→Retrain integration convenience
# ════════════════════════════════════════════════════════════════════════════


def auto_retrain_on_drift(
    store: FeatureStore,
    family: str = "haelt",
    config: RetrainConfig = None,
    dry_run: bool = False,
) -> dict:
    """
    One-shot: check drift for all materialized features and trigger retrain if needed.
    """
    orchestrator = RetrainOrchestrator(store, config)
    return orchestrator.check_and_retrain(family=family, dry_run=dry_run)


if __name__ == "__main__":
    # Demo
    from data.feature_store import get_feature_store

    store = get_feature_store()
    orch = RetrainOrchestrator(store)
    status = orch.get_status()
    print(f"Latest model: {status['latest_model']}")
    print(f"Production: {status['production_model']}")
    print(f"Total models: {status['total_models']}")
