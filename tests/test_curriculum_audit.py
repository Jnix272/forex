"""Tests for curriculum/FEATURE_MASK audits and run.yaml parseability."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from config.curriculum_audit import (
    audit_built_dataset_schema,
    audit_curriculum_feature_groups,
    audit_required_market_columns,
    audit_settings_yaml_curriculum_drift,
)
from config.feature_mask import FEATURE_MASK
from config.settings import CURRICULUM


def test_run_yaml_parses_and_strategy_block_loads():
    path = Path("config/run.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    assert isinstance(raw, dict)
    strat = raw["strategy"]
    assert isinstance(strat, dict)
    assert strat["mode"] == "scalping"
    assert float(strat["profit_target_atr"]) == pytest.approx(1.2)
    assert float(strat["stop_loss_atr"]) == pytest.approx(0.8)
    assert strat["bar_freq"] == "5m"
    assert "lookahead_bars" in strat


def test_apply_yaml_config_loads_strategy_defaults(tmp_path):
    import argparse

    from training.gpu_cli import _YAML_MAP, _apply_yaml_config

    p = argparse.ArgumentParser()
    # Dest names used by strategy keys
    p.add_argument("--strategy-mode", dest="strategy_mode", default="swing")
    p.add_argument("--profit-target-atr", dest="profit_target_atr", type=float, default=9.9)
    p.add_argument("--stop-loss-atr", dest="stop_loss_atr", type=float, default=9.9)
    p.add_argument("--bar-freq", dest="bar_freq", default="5min")
    p.add_argument("--lookahead-bars", dest="lookahead_bars", type=int, default=1)
    _apply_yaml_config(p, "config/run.yaml")
    args, _ = p.parse_known_args([])
    assert args.strategy_mode == "scalping"
    assert args.profit_target_atr == pytest.approx(1.2)
    assert args.stop_loss_atr == pytest.approx(0.8)
    assert "strategy.profit_target_atr" in _YAML_MAP


def test_apply_yaml_config_fails_hard_on_bad_indent(tmp_path):
    import argparse

    from training.gpu_cli import _apply_yaml_config

    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "strategy:\n  mode: scalping\n   profit_target_atr: 1.2\n",
        encoding="utf-8",
    )
    p = argparse.ArgumentParser()
    with pytest.raises(RuntimeError, match="YAML parse failed"):
        _apply_yaml_config(p, str(bad))


def test_curriculum_audit_detects_missing_overlap_orphan():
    schema = ["spread_pips", "ofi", "ret_5"]
    groups = {
        "execution_cost": {
            "always_on": False,
            "epoch_unfreeze": 2,
            "features": ["spread_pips", "ghost_feature", "ofi"],
        },
        "momentum": {
            "always_on": True,
            "features": ["ofi", "ret_5"],
        },
    }
    mask = {"spread_pips": True, "ofi": True, "ret_5": True, "sentiment_raw": True}
    report = audit_curriculum_feature_groups(schema=schema, feature_groups=groups, feature_mask=mask)
    assert "execution_cost" in report["missing_from_schema"]
    assert "ghost_feature" in report["missing_from_schema"]["execution_cost"]
    assert "ofi" in report["overlapping"]
    assert "sentiment_raw" in report["orphans_always_on"]
    assert report["warnings"]


def test_market_schema_audit_warns_without_spread_in_mask():
    mask = dict.fromkeys(("spread_pips", "spread_avg", "atr_6", "atr_20"), False)
    mask["atr_6"] = True
    report = audit_required_market_columns(feature_mask=mask)
    assert any("spread" in w.lower() for w in report["warnings"])


def test_feature_mask_covers_primary_atr_and_spread():
    assert FEATURE_MASK.get("atr_6") is True
    assert FEATURE_MASK.get("spread_pips") is True


def test_settings_yaml_curriculum_drift_detects_epoch_mismatch():
    settings = {
        "seq_schedule": [{"epoch_start": 0, "seq_len": 80}],
        "difficulty_schedule": [{"epoch_start": 0, "max_difficulty": 2}],
        "chunk_early_stop_patience": 3,
        "chunk_early_stop_min_batches": 50,
        "difficulty_spread_threshold": 1.5,
        "difficulty_spread_threshold_hard": 2.0,
        "feature_groups": {
            "cross_asset": {"always_on": False, "epoch_unfreeze": 6},
            "macro": {"always_on": False, "epoch_unfreeze": 12},
        },
    }
    yaml_cur = {
        "seq_schedule": [{"epoch_start": 0, "seq_len": 80}],
        "difficulty_schedule": [{"epoch_start": 0, "max_difficulty": 2}],
        "chunk_early_stop_patience": 3,
        "chunk_early_stop_min_batches": 50,
        "difficulty_spread_threshold": 1.5,
        "difficulty_spread_threshold_hard": 2.0,
        "feature_groups": {
            "cross_asset": {"always_on": False, "epoch_unfreeze": 8, "features": ["a"]},
            "macro": {"always_on": False, "epoch_unfreeze": 12, "features": ["b"]},
            "news": {"always_on": False, "epoch_unfreeze": 10, "features": ["c"]},
        },
    }
    report = audit_settings_yaml_curriculum_drift(settings, yaml_cur, yaml_path="config/run.yaml")
    assert report["errors"]
    assert any("cross_asset" in e for e in report["errors"])
    assert "news" in report["only_yaml_groups"]
    assert any("news" in w for w in report["warnings"])


def test_settings_aligned_with_run_yaml_curriculum():
    """settings.CURRICULUM schedule stubs must match active config/run.yaml."""
    raw = yaml.safe_load(Path("config/run.yaml").read_text(encoding="utf-8-sig"))
    curr = raw["curriculum"]
    report = audit_settings_yaml_curriculum_drift(CURRICULUM, curr, yaml_path="config/run.yaml")
    assert report["errors"] == [], report["errors"]
    assert report["only_settings_groups"] == []
    assert report["only_yaml_groups"] == []


def test_built_dataset_schema_gate_errors_on_missing_curriculum_feature():
    groups = {
        "execution_cost": {
            "always_on": False,
            "epoch_unfreeze": 2,
            "features": ["spread_pips", "ghost_cost"],
        },
        "microstructure": {
            "always_on": True,
            "features": ["ofi"],
        },
    }
    report = audit_built_dataset_schema(
        feature_names=["ofi", "spread_pips", "atr_6", "mid_close"],
        feature_groups=groups,
        feature_mask={"ofi": True, "spread_pips": True, "atr_6": True, "ghost_cost": True},
    )
    assert report["errors"]
    assert any("ghost_cost" in e or "curriculum" in e.lower() for e in report["errors"])
    assert "execution_cost" in report["missing_from_schema"]


def test_built_dataset_schema_gate_ok_when_curriculum_covered():
    groups = {
        "microstructure": {"always_on": True, "features": ["ofi"]},
        "execution_cost": {"always_on": False, "features": ["spread_pips"]},
    }
    report = audit_built_dataset_schema(
        feature_names=["ofi", "spread_pips", "atr_6", "mid_close"],
        feature_groups=groups,
        feature_mask={"ofi": True, "spread_pips": True, "atr_6": True},
    )
    assert report["errors"] == [], report["errors"]
    assert report["missing_from_schema"] == {}


def test_enforce_dataset_feature_schema_raises_when_gated(tmp_path):
    from types import SimpleNamespace

    from training.dataset_builder import _enforce_dataset_feature_schema

    args = SimpleNamespace(
        config="config/run.yaml",
        curriculum={
            "feature_groups": {
                "macro": {"always_on": False, "features": ["sentiment_raw", "missing_fb"]},
            }
        },
        integrity_gate=True,
        feature_schema_gate=True,
        # Keep args aligned with YAML for the args_yaml part so only schema fails.
        batch_size=128,
        epochs=2,
        seq_len=80,
        loss="sharpe_huber",
        grad_clip=0.75,
        patience=6,
        weight_decay=0.001,
        sharpe_annualization_factor=325.0,
        bar_freq="1min",
        strategy_mode="scalping",
        profit_target_atr=1.2,
        stop_loss_atr=0.8,
        lookahead_bars=15,
    )
    with pytest.raises(RuntimeError, match="feature-schema gate failed"):
        _enforce_dataset_feature_schema(
            args,
            ["sentiment_raw", "atr_6", "mid_close", "spread_pips"],
            tmp_path / "cache",
            phase="test",
        )
    audit_path = tmp_path / "cache_feature_schema_audit.json"
    assert audit_path.is_file()
    import json

    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    assert "parts" in payload
    assert "built_schema" in payload["parts"]
    assert "settings_yaml" in payload["parts"]
    assert "args_yaml" in payload["parts"]


def test_settings_yaml_section_mismatch_flags_critical_keys():
    from config.config_mismatch_audit import audit_settings_yaml_section_mismatches

    # Real settings.TRAINING vs a fake YAML that drifts on critical keys
    report = audit_settings_yaml_section_mismatches(
        {"training": {"seq_len": 12, "loss": "mse", "sharpe_annualization_factor": 1.0}},
        yaml_path="fake.yaml",
        sections={"training": "TRAINING"},
    )
    assert report["errors"], report
    assert any("seq_len" in e or "loss" in e for e in report["errors"])


def test_settings_aligned_on_critical_keys_with_run_yaml():
    from config.config_mismatch_audit import (
        audit_settings_yaml_section_mismatches,
        load_yaml_config,
    )

    raw = load_yaml_config("config/run.yaml")
    report = audit_settings_yaml_section_mismatches(raw, yaml_path="config/run.yaml")
    assert report["errors"] == [], report["errors"]
    # Non-critical drift (epochs/batch_size/…) may still warn
    assert isinstance(report["warnings"], list)


def test_strategy_vs_labeling_aligned_on_run_yaml():
    from config.config_mismatch_audit import (
        audit_strategy_vs_labeling,
        load_yaml_config,
    )

    raw = load_yaml_config("config/run.yaml")
    report = audit_strategy_vs_labeling(raw, yaml_path="config/run.yaml")
    assert report["errors"] == [], report["errors"]


def test_ubuntu_profile_pretrain_epochs_not_critical():
    from config.config_mismatch_audit import (
        audit_settings_yaml_section_mismatches,
        load_yaml_config,
    )

    raw = load_yaml_config("config/run_ubuntu.yaml")
    report = audit_settings_yaml_section_mismatches(raw, yaml_path="config/run_ubuntu.yaml")
    # Hardware-scaled pretrain.epochs must not fail closed vs settings stubs.
    assert not any("pretrain.epochs" in e for e in report["errors"]), report["errors"]
    # Strategy ↔ LABELING still fail-closed on profile YAMLs.
    assert not any("strategy." in e for e in report["errors"]), report["errors"]
