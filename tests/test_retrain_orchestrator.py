"""
Tests for Phase 3 — Retrain Orchestrator: ModelRegistry, promotion gates,
trigger evaluation, and orchestrator lifecycle.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from data.feature_store import FeatureStore
from retraining.orchestrator import (
    ModelRegistry,
    ModelStatus,
    RetrainConfig,
    RetrainOrchestrator,
    RetrainReason,
    check_promotion_gates,
)

# ════════════════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def tmp_registry(tmp_path) -> ModelRegistry:
    return ModelRegistry(root=tmp_path / "models")


@pytest.fixture
def tmp_store(tmp_path) -> FeatureStore:
    return FeatureStore(root=tmp_path / "feature_store")


@pytest.fixture
def seeded_registry(tmp_path) -> ModelRegistry:
    """Registry with a few historical models."""
    reg = ModelRegistry(root=tmp_path / "seeded_models")
    for i in range(1, 4):
        reg.register(
            family="haelt",
            reason=RetrainReason.INITIAL.value,
            checkpoint_dir=str(tmp_path / f"models/haelt_v{i:03d}"),
        )
        reg.update_status("haelt", i, ModelStatus.PROMOTED, metrics={"val_sharpe": 0.8 + i * 0.1})
    return reg


# ════════════════════════════════════════════════════════════════════════════
# ModelRegistry
# ════════════════════════════════════════════════════════════════════════════

class TestModelRegistry:
    def test_register_creates_record(self, tmp_registry):
        r = tmp_registry.register("haelt", "initial", "/tmp/ckpt")
        assert r.version == 1
        assert r.family == "haelt"
        assert r.status == ModelStatus.TRAINING
        assert r.created_at is not None

    def test_next_version_increments(self, tmp_registry):
        tmp_registry.next_version("haelt")
        tmp_registry.register("haelt", "initial", "/tmp/ckpt1")
        tmp_registry.next_version("haelt")
        tmp_registry.register("haelt", "scheduled", "/tmp/ckpt2")
        v3 = tmp_registry.next_version("haelt")
        assert v3 == 3

    def test_next_version_ignores_failed(self, tmp_registry):
        tmp_registry.register("haelt", "initial", "/tmp/ckpt1")
        tmp_registry.register("haelt", "test", "/tmp/ckpt2")
        tmp_registry.update_status("haelt", 2, ModelStatus.FAILED)
        assert tmp_registry.next_version("haelt") == 2  # reuse version

    def test_get_latest(self, tmp_registry):
        tmp_registry.register("haelt", "v1", "/ckpt1")
        tmp_registry.register("haelt", "v2", "/ckpt2")
        latest = tmp_registry.get_latest("haelt")
        assert latest.version == 2

    def test_get_latest_by_status(self, tmp_registry):
        tmp_registry.register("tft", "v1", "/ckpt1")
        tmp_registry.register("tft", "v2", "/ckpt2")
        tmp_registry.update_status("tft", 1, ModelStatus.PROMOTED)
        promoted = tmp_registry.get_latest("tft", status=ModelStatus.PROMOTED)
        assert promoted is not None
        assert promoted.version == 1

    def test_get_production(self, seeded_registry):
        prod = seeded_registry.get_production()
        assert prod is not None
        assert prod.status == ModelStatus.PROMOTED
        assert prod.version == 3  # latest promoted

    def test_list_models(self, tmp_registry):
        tmp_registry.register("haelt", "v1", "/ckpt1")
        tmp_registry.register("haelt", "v2", "/ckpt2")
        tmp_registry.register("tft", "v1", "/ckpt3")
        assert len(tmp_registry.list_models()) == 3
        assert len(tmp_registry.list_models(family="haelt")) == 2
        assert len(tmp_registry.list_models(family="tft")) == 1

    def test_get_version(self, tmp_registry):
        tmp_registry.register("haelt", "test", "/ckpt")
        r = tmp_registry.get_version("haelt", 1)
        assert r is not None
        assert r.version == 1
        assert tmp_registry.get_version("haelt", 99) is None

    def test_update_status(self, tmp_registry):
        tmp_registry.register("haelt", "test", "/ckpt")
        tmp_registry.update_status("haelt", 1, ModelStatus.PROMOTED, metrics={"val_sharpe": 1.5})
        r = tmp_registry.get_version("haelt", 1)
        assert r.status == ModelStatus.PROMOTED
        assert r.metrics["val_sharpe"] == 1.5
        assert r.promoted_at is not None

    def test_update_status_raises_on_missing(self, tmp_registry):
        with pytest.raises(KeyError):
            tmp_registry.update_status("haelt", 999, ModelStatus.PROMOTED)

    def test_rollback(self, seeded_registry):
        target = seeded_registry.rollback("haelt")
        assert target is not None
        assert target.version == 2
        new_prod = seeded_registry.get_production()
        assert new_prod.version == 2
        rolled = seeded_registry.get_version("haelt", 3)
        assert rolled.status == ModelStatus.ROLLED_BACK

    def test_rollback_no_previous(self, tmp_registry):
        tmp_registry.register("haelt", "only", "/ckpt")
        tmp_registry.update_status("haelt", 1, ModelStatus.PROMOTED)
        assert tmp_registry.rollback("haelt") is None

    def test_persistence_across_reload(self, tmp_path):
        reg1 = ModelRegistry(root=tmp_path / "persist")
        reg1.register("haelt", "v1", "/ckpt1")
        reg1.register("haelt", "v2", "/ckpt2")
        reg2 = ModelRegistry(root=tmp_path / "persist")
        assert len(reg2.list_models()) == 2

    def test_cleanup_old_versions(self, tmp_path):
        for i in range(1, 6):
            ckpt = tmp_path / f"models/haelt_v{i:03d}"
            ckpt.mkdir(parents=True)
            (ckpt / "model.pt").write_text("dummy")
        reg = ModelRegistry(root=tmp_path / "models")
        for i in range(1, 6):
            reg.register("haelt", "test", str(tmp_path / f"models/haelt_v{i:03d}"))
        n_removed = reg.cleanup_old_versions(keep=2)
        assert n_removed >= 3


# ════════════════════════════════════════════════════════════════════════════
# Promotion Gates
# ════════════════════════════════════════════════════════════════════════════

class TestPromotionGates:
    def test_passes_good_metrics(self):
        passed, reasons = check_promotion_gates({
            "val_sharpe": 2.0,  # Higher to pass PSR
            "profit_factor": 1.5,
            "max_drawdown": 0.08,
            "val_loss": 0.5,
            "n_trades": 1000,
            "gross_pnl": 5000.0,
            "transaction_costs": 800.0,
            "n_backtest_trials": 1,
            "regime_pnl": {"trending": 0.5, "neutral": 0.3, "mean_rev": 0.2},
            "n_obs": 1000,
        })
        print(f"passed={passed}, reasons={reasons}")
        assert passed
        # reasons contains all gates with status, check that all passed
        assert all("✓" in r for r in reasons)

    def test_fails_low_sharpe(self):
        passed, reasons = check_promotion_gates({
            "val_sharpe": 0.3,
            "profit_factor": 1.5,
            "max_drawdown": 0.08,
            "n_trades": 1000,
            "gross_pnl": 5000.0,
            "transaction_costs": 800.0,
            "n_backtest_trials": 1,
            "regime_pnl": {"trending": 0.5, "neutral": 0.3, "mean_rev": 0.2},
            "n_obs": 1000,
        })
        print(f"passed={passed}, reasons={reasons}")
        assert not passed
        # Check that at least one sharpe-related gate failed
        assert any("sharpe" in r.lower() and "✗" in r for r in reasons)

    def test_fails_high_drawdown(self):
        passed, reasons = check_promotion_gates({
            "val_sharpe": 1.0,
            "profit_factor": 1.5,
            "max_drawdown": 0.25,
        })
        assert not passed
        assert any("drawdown" in r for r in reasons)

    def test_fails_multiple_gates(self):
        passed, reasons = check_promotion_gates({
            "val_sharpe": 0.2,
            "profit_factor": 0.8,
            "max_drawdown": 0.30,
        })
        assert not passed
        assert len(reasons) >= 2

    def test_empty_metrics_fails(self):
        passed, reasons = check_promotion_gates({})
        assert not passed


# ════════════════════════════════════════════════════════════════════════════
# RetrainOrchestrator
# ════════════════════════════════════════════════════════════════════════════

class TestRetrainOrchestrator:
    def _orch(self, tmp_store, tmp_path):
        """Helper: create orchestrator with isolated model root."""
        return RetrainOrchestrator(
            tmp_store, config=RetrainConfig(model_root=tmp_path / "models")
        )

    def test_should_retrain_initial(self, tmp_store, tmp_path):
        orch = self._orch(tmp_store, tmp_path)
        should, reason, ctx = orch.should_retrain("haelt")
        assert should
        assert reason == RetrainReason.INITIAL.value

    def test_should_retrain_force(self, tmp_store, tmp_path):
        orch = self._orch(tmp_store, tmp_path)
        orch.registry.register("haelt", "initial", "/tmp/ckpt")
        orch.registry.update_status("haelt", 1, ModelStatus.PROMOTED)
        should, reason, ctx = orch.should_retrain("haelt", force=True)
        assert should
        assert reason == RetrainReason.MANUAL.value

    def test_should_not_retrain_recently(self, tmp_store, tmp_path):
        orch = self._orch(tmp_store, tmp_path)
        orch.registry.register("haelt", "initial", "/tmp/ckpt")
        orch.registry.update_status("haelt", 1, ModelStatus.PROMOTED)
        orch.registry.get_version("haelt", 1).promoted_at = datetime.now(UTC).isoformat()
        should, reason, ctx = orch.should_retrain("haelt")
        assert not should

    def test_retrain_dry_run(self, tmp_store, tmp_path):
        orch = self._orch(tmp_store, tmp_path)
        result = orch.retrain(family="haelt", reason="test", dry_run=True)
        assert result["dry_run"]
        assert result["version"] == 1
        assert result["status"] == "dry_run"

    def test_retrain_creates_checkpoint_dir(self, tmp_store, tmp_path):
        config = RetrainConfig(model_root=tmp_path / "models_orch")
        orch = RetrainOrchestrator(tmp_store, config=config)
        result = orch.retrain(family="haelt", reason="test", dry_run=True)
        ckpt_dir = Path(result["checkpoint_dir"])
        assert ckpt_dir.parent.exists()

    def test_build_training_cmd(self, tmp_store, tmp_path):
        orch = self._orch(tmp_store, tmp_path)
        cmd = orch._build_training_cmd("haelt", 1, Path("/tmp/ckpt"))
        assert "train_gpu.py" in " ".join(cmd)
        assert "--model" in cmd
        assert "haelt" in cmd

    def test_build_training_cmd_xgboost(self, tmp_store, tmp_path):
        orch = self._orch(tmp_store, tmp_path)
        cmd = orch._build_training_cmd("xgboost", 1, Path("/tmp/ckpt"))
        assert "train_xgboost.py" in " ".join(cmd)

    def test_check_and_retrain_no_trigger(self, tmp_store, tmp_path):
        orch = self._orch(tmp_store, tmp_path)
        orch.registry.register("haelt", "initial", "/tmp/ckpt")
        orch.registry.update_status("haelt", 1, ModelStatus.PROMOTED)
        orch.registry.get_version("haelt", 1).promoted_at = datetime.now(UTC).isoformat()
        result = orch.check_and_retrain("haelt")
        assert not result["should_retrain"]

    def test_check_and_retrain_with_force(self, tmp_store, tmp_path):
        orch = self._orch(tmp_store, tmp_path)
        result = orch.check_and_retrain("haelt", force=True, dry_run=True)
        assert result["dry_run"]

    def test_get_status_empty(self, tmp_store, tmp_path):
        orch = self._orch(tmp_store, tmp_path)
        status = orch.get_status()
        assert status["latest_model"] is None
        assert status["production_model"] is None
        assert status["total_models"] == 0

    def test_get_status_with_models(self, tmp_store, tmp_path):
        orch = self._orch(tmp_store, tmp_path)
        orch.registry.register("haelt", "v1", "/tmp/ckpt")
        status = orch.get_status("haelt")
        assert status["total_models"] == 1
        assert status["latest_model"] is not None

    def test_extract_metrics(self, tmp_store, tmp_path):
        orch = self._orch(tmp_store, tmp_path)
        output = """
        Training run complete
        val_sharpe: 1.234
        val_loss: 0.456
        profit_factor: 1.89
        max_drawdown: 0.087
        """
        metrics = orch._extract_metrics(output)
        assert metrics.get("val_sharpe") == 1.234
        assert metrics.get("val_loss") == 0.456
        assert metrics.get("profit_factor") == 1.89
        assert metrics.get("max_drawdown") == 0.087

    def test_extract_metrics_empty(self, tmp_store, tmp_path):
        orch = self._orch(tmp_store, tmp_path)
        assert orch._extract_metrics("no metrics here") == {}


# ════════════════════════════════════════════════════════════════════════════
# Drift + Retrain Integration
# ════════════════════════════════════════════════════════════════════════════

class TestDriftRetrainIntegration:
    def test_auto_retrain_on_drift_no_data(self, tmp_store, tmp_path):
        from retraining.orchestrator import auto_retrain_on_drift
        result = auto_retrain_on_drift(
            tmp_store, config=RetrainConfig(model_root=tmp_path / "drift_retrain"),
            dry_run=True,
        )
        assert result["dry_run"]
