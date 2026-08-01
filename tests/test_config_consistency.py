"""
Tests for configuration consistency across YAML, Python dicts, and model profiles.
Catches naming mismatches, dead config, missing keys, and cross-file drift.
"""
from __future__ import annotations

import pytest
import yaml
from pathlib import Path

from config.settings import (
    TRAINING, DATA, LABELING, PRETRAIN, CROSS_ASSET, BACKTEST, LIVE_RISK, CURRICULUM,
)
from config.models import MODELS, SUPPORTED_SUPERVISED
from config.feature_mask import FEATURE_MASK
from config.strategy_profiles import STRATEGY_PROFILES


# ---------------------------------------------------------------------------
# 1. YAML ↔ Python consistency
# ---------------------------------------------------------------------------

_YAML_DIR = Path("config/models")

@pytest.fixture(scope="module")
def yaml_configs() -> dict:
    configs = {}
    for p in _YAML_DIR.glob("*.yaml"):
        with p.open("r", encoding="utf-8") as f:
            configs[p.stem] = yaml.safe_load(f) or {}
    return configs


def test_every_supported_model_has_python_profile():
    for name in SUPPORTED_SUPERVISED:
        assert name in MODELS, f"MODELS dict missing profile for supported model '{name}'"


def test_every_python_profile_is_supported():
    for name in MODELS:
        assert name in SUPPORTED_SUPERVISED, (
            f"MODELS contains '{name}' but it's not in SUPPORTED_SUPERVISED"
        )


def test_every_supported_model_has_yaml(yaml_configs):
    for name in SUPPORTED_SUPERVISED:
        assert name in yaml_configs, f"Missing YAML config file config/models/{name}.yaml"


def test_yaml_model_names_match_filename(yaml_configs):
    for stem, cfg in yaml_configs.items():
        model_name = (cfg.get("model") or {}).get("name", "").lower()
        assert model_name == stem, (
            f"{stem}.yaml declares model.name='{model_name}', expected '{stem}'"
        )


def test_all_python_profiles_have_required_keys():
    required = {"dropout", "seq_len"}
    for name, profile in MODELS.items():
        for key in required:
            if key == "seq_len" and name == "gnn":
                continue
            assert key in profile, f"MODELS['{name}'] missing required key '{key}'"


def test_no_python_profiles_use_legacy_key_names():
    legacy_keys = {
        "n_transformer_layers": "use 'num_layers'",
        "attention_head_size": "use 'nhead'",
        "correlation_threshold": "removed (dead config)",
    }
    for name, profile in MODELS.items():
        for bad_key, suggestion in legacy_keys.items():
            assert bad_key not in profile, (
                f"MODELS['{name}'] has legacy key '{bad_key}' — {suggestion}"
            )


def test_consistent_num_layers_key():
    for name, profile in MODELS.items():
        if "num_layers" in profile:
            assert "n_transformer_layers" not in profile
            assert "n_layers" not in profile


# ---------------------------------------------------------------------------
# 2. PRETRAIN config
# ---------------------------------------------------------------------------

def test_pretrain_has_explicit_loss():
    assert "pretrain_loss" in PRETRAIN, "PRETRAIN should have explicit 'pretrain_loss' key"
    assert PRETRAIN["pretrain_loss"] in ("huber", "mse", "cross_entropy")


def test_pretrain_epochs_positive():
    assert PRETRAIN["pretrain_epochs"] > 0


def test_pretrain_lr_sane():
    assert 1e-6 < PRETRAIN["pretrain_lr"] < 1e-2


# ---------------------------------------------------------------------------
# 3. TRAINING config
# ---------------------------------------------------------------------------

def test_training_loss_is_known():
    valid = {"cross_entropy", "sharpe_huber", "huber", "mse", "focal"}
    assert TRAINING["loss"] in valid, f"Unknown loss: {TRAINING['loss']}"


def test_warmup_less_than_half_epochs():
    warmup = TRAINING.get("lr_warmup_epochs", 3)
    epochs = TRAINING.get("epochs", 40)
    assert warmup < epochs * 0.5, (
        f"Warmup ({warmup}) >= 50% of epochs ({epochs})"
    )


def test_patience_reachable():
    warmup = TRAINING.get("lr_warmup_epochs", 3)
    epochs = TRAINING.get("epochs", 40)
    patience = TRAINING.get("patience", 10)
    effective = epochs - warmup
    assert patience < effective, (
        f"patience ({patience}) >= post-warmup epochs ({effective}); early stopping can never fire"
    )


def test_batch_size_positive():
    assert TRAINING["batch_size"] > 0


def test_lr_comes_from_yaml_not_training_dict():
    """LR is set via YAML/CLI, not in the TRAINING dict. Verify it's absent here."""
    assert "lr" not in TRAINING, (
        "LR should not be in TRAINING dict — it comes from YAML/CLI config"
    )


# ---------------------------------------------------------------------------
# 4. LABELING config
# ---------------------------------------------------------------------------

def test_labeling_has_atr_keys():
    assert "profit_target_atr" in LABELING
    assert "stop_loss_atr" in LABELING
    assert "lookahead_bars" in LABELING


def test_labeling_reward_risk_ratio():
    tp = LABELING["profit_target_atr"]
    sl = LABELING["stop_loss_atr"]
    ratio = tp / sl
    assert 1.0 <= ratio <= 5.0, f"TP/SL ATR ratio {ratio:.2f} is unusual"


# ---------------------------------------------------------------------------
# 5. DATA config
# ---------------------------------------------------------------------------

def test_data_pairs_not_empty():
    assert len(DATA["pairs"]) >= 1


def test_data_pairs_uppercase():
    for pair in DATA["pairs"]:
        assert pair == pair.upper(), f"Pair '{pair}' should be uppercase"


def test_data_has_resolution():
    assert "resolution" in DATA


# ---------------------------------------------------------------------------
# 6. CROSS_ASSET config
# ---------------------------------------------------------------------------

def test_cross_asset_has_assets():
    assert len(CROSS_ASSET) >= 1, "CROSS_ASSET should have at least one asset"


# ---------------------------------------------------------------------------
# 7. BACKTEST config
# ---------------------------------------------------------------------------

def test_backtest_stop_and_tp_defined():
    assert "stop_pips" in BACKTEST
    assert "take_profit_pips" in BACKTEST
    assert BACKTEST["stop_pips"] > 0
    assert BACKTEST["take_profit_pips"] > 0


def test_backtest_tp_gt_stop():
    assert BACKTEST["take_profit_pips"] > BACKTEST["stop_pips"], (
        "Take profit should exceed stop loss for positive expectancy"
    )


# ---------------------------------------------------------------------------
# 8. LIVE_RISK config
# ---------------------------------------------------------------------------

def test_live_risk_has_session_limits():
    assert "session_limits" in LIVE_RISK
    for session in ("asia", "london", "ny", "off"):
        assert session in LIVE_RISK["session_limits"], f"Missing session '{session}'"


def test_live_risk_kelly_fraction_bounded():
    kf = LIVE_RISK.get("kelly_fraction", 0.25)
    assert 0.0 < kf <= 0.5, f"Kelly fraction {kf} should be in (0, 0.5]"


def test_live_risk_drawdown_thresholds_ordered():
    soft = LIVE_RISK.get("soft_drawdown_reduce", 0.05)
    hard = LIVE_RISK.get("max_drawdown_halt", 0.10)
    assert soft < hard, f"soft_dd ({soft}) should be < hard_dd ({hard})"


# ---------------------------------------------------------------------------
# 9. FEATURE_MASK config
# ---------------------------------------------------------------------------

def test_feature_mask_not_empty():
    assert len(FEATURE_MASK) > 50, "Feature mask should have >50 features"


def test_feature_mask_values_are_bool():
    for key, val in FEATURE_MASK.items():
        assert isinstance(val, bool), f"FEATURE_MASK['{key}'] = {val!r} is not bool"


# ---------------------------------------------------------------------------
# 10. STRATEGY_PROFILES config
# ---------------------------------------------------------------------------

def test_strategy_profiles_have_required_keys():
    required = {"seq_len", "checkpoint_dir", "lookahead_bars"}
    for name, profile in STRATEGY_PROFILES.items():
        for key in required:
            assert key in profile, (
                f"STRATEGY_PROFILES['{name}'] missing required key '{key}'"
            )


def test_strategy_seq_len_within_curriculum():
    for name, profile in STRATEGY_PROFILES.items():
        assert profile["seq_len"] <= 120, (
            f"Strategy '{name}' seq_len={profile['seq_len']} exceeds max curriculum (120)"
        )


# ---------------------------------------------------------------------------
# 11. Config __init__.py re-exports
# ---------------------------------------------------------------------------

def test_config_package_reexports():
    from config import TRAINING as T, MODELS as M, FEATURE_MASK as FM
    assert T is TRAINING
    assert M is MODELS
    assert FM is FEATURE_MASK


# ---------------------------------------------------------------------------
# 12. CURRICULUM config
# ---------------------------------------------------------------------------

def test_curriculum_has_seq_schedule():
    assert "seq_schedule" in CURRICULUM, "CURRICULUM must define a seq_schedule"
    assert isinstance(CURRICULUM["seq_schedule"], list)
    assert len(CURRICULUM["seq_schedule"]) >= 1


def test_curriculum_seq_schedule_epochs_ascending():
    schedule = CURRICULUM["seq_schedule"]
    epochs = [s.get("epoch_start", 0) for s in schedule if isinstance(s, dict)]
    assert epochs == sorted(epochs), f"seq_schedule epochs not ascending: {epochs}"
