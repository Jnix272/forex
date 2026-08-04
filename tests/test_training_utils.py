"""
Tests for training utilities: config validation, profile normalization,
YAML mapping, and model resolution.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from config.models import MODELS, SUPPORTED_SUPERVISED
from training.config_validate import (
    _effective_max_seq_len,
    _parse_pretrain_ablation_models,
    collect_config_issues,
    estimate_run_minutes,
    resolve_models_to_train,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        model="haelt", epochs=40, batch_size=212, lr=2.5e-5,
        patience=7, lr_warmup_epochs=3, seq_len=60, loss="sharpe_huber",
        pair="EURUSD", pairs=None, checkpoint_dir="checkpoints",
        data_cache="data/processed", walk_forward_cv=False,
        walk_forward_folds=5, all_models=False, models="",
        resume=False, retrain_completed_models=False,
        pretrain=True, pretrain_epochs=30, pretrain_method="byol",
        pretrain_ablation="auto", pretrain_ablation_models="",
        train_ensemble=False, rl_train=False, rl_episodes=500,
        rl_all_models=False, model_profile=True, strategy_mode="scalping",
        data_start="2008-01-01", data_end="2025-12-31",
        quick_mode=False, pair_embed_dim=16, grad_accum_steps=4,
        training_memory=False, drift_gate=False,
        teacher_model=None, distill_weight=0, config="config/run.yaml",
        curriculum=None, dry_tune=True, auto_tune=True,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# 1. Config validation — error detection
# ---------------------------------------------------------------------------

class TestCollectConfigIssues:
    def test_valid_config_produces_no_errors(self):
        args = _make_args()
        models, skipped = resolve_models_to_train(args)
        errors, warnings, info = collect_config_issues(args, models, skipped)
        assert len(errors) == 0, f"Unexpected errors: {errors}"

    def test_zero_epochs_produces_error(self):
        # config_validate uses `int(getattr(args, 'epochs', 40) or 40)` so 0 → 40
        # Use a negative value to trigger the error
        args = _make_args(epochs=-1)
        errors, _, _ = collect_config_issues(args, ["haelt"], [])
        assert any("epochs" in e for e in errors)

    def test_negative_lr_produces_error(self):
        args = _make_args(lr=-0.001)
        errors, _, _ = collect_config_issues(args, ["haelt"], [])
        assert any("lr" in e for e in errors)

    def test_lr_above_one_produces_error(self):
        args = _make_args(lr=1.5)
        errors, _, _ = collect_config_issues(args, ["haelt"], [])
        assert any("lr" in e for e in errors)

    def test_negative_batch_produces_error(self):
        args = _make_args(batch_size=-1)
        errors, _, _ = collect_config_issues(args, ["haelt"], [])
        assert any("batch_size" in e for e in errors)

    def test_negative_patience_produces_error(self):
        args = _make_args(patience=-1)
        errors, _, _ = collect_config_issues(args, ["haelt"], [])
        assert any("patience" in e for e in errors)

    def test_warmup_exceeds_epochs_produces_error(self):
        args = _make_args(lr_warmup_epochs=50, epochs=40)
        errors, _, _ = collect_config_issues(args, ["haelt"], [])
        assert any("warmup" in e.lower() for e in errors)

    def test_warmup_gt_half_epochs_produces_warning(self):
        args = _make_args(lr_warmup_epochs=25, epochs=40)
        _, warnings, _ = collect_config_issues(args, ["haelt"], [])
        assert any("warmup" in w.lower() for w in warnings)

    def test_patience_unreachable_produces_warning(self):
        args = _make_args(lr_warmup_epochs=3, epochs=10, patience=8)
        _, warnings, _ = collect_config_issues(args, ["haelt"], [])
        assert any("patience" in w.lower() for w in warnings)

    def test_unknown_model_produces_error(self):
        args = _make_args(model="nonexistent_model")
        errors, _, _ = collect_config_issues(args, ["haelt"], [])
        assert any("nonexistent_model" in e.lower() for e in errors)

    def test_unknown_loss_produces_warning(self):
        args = _make_args(loss="imaginary_loss")
        _, warnings, _ = collect_config_issues(args, ["haelt"], [])
        assert any("imaginary_loss" in w for w in warnings)

    def test_empty_model_queue_produces_error(self):
        args = _make_args()
        errors, _, _ = collect_config_issues(args, [], [])
        assert any("No models" in e for e in errors)

    def test_all_skipped_produces_warning(self):
        args = _make_args()
        _, warnings, _ = collect_config_issues(args, [], [("haelt", "done")])
        assert any("complete" in w.lower() for w in warnings)

    def test_multi_pair_without_embed_dim_warns(self):
        args = _make_args(pairs="EURUSD,GBPUSD,USDJPY", pair_embed_dim=0)
        _, warnings, _ = collect_config_issues(args, ["haelt"], [])
        assert any("pair_embed_dim" in w for w in warnings)


# ---------------------------------------------------------------------------
# 2. Model resolution
# ---------------------------------------------------------------------------

class TestResolveModelsToTrain:
    def test_single_model_default(self):
        args = _make_args(model="haelt", all_models=False)
        models, skipped = resolve_models_to_train(args)
        assert models == ["haelt"]
        assert skipped == []

    def test_all_models_returns_full_set(self):
        args = _make_args(all_models=True)
        models, _ = resolve_models_to_train(args)
        assert set(models) == SUPPORTED_SUPERVISED

    def test_filtered_models(self):
        args = _make_args(all_models=True, models="haelt,tft")
        models, _ = resolve_models_to_train(args)
        assert set(models) == {"haelt", "tft"}

    def test_unknown_model_raises(self):
        args = _make_args(all_models=True, models="haelt,fake_model")
        with pytest.raises(ValueError, match="Unknown model"):
            resolve_models_to_train(args)


# ---------------------------------------------------------------------------
# 3. Runtime estimation
# ---------------------------------------------------------------------------

class TestEstimateRunMinutes:
    def test_returns_expected_keys(self):
        args = _make_args()
        est = estimate_run_minutes(args, ["haelt"])
        expected = {
            "pretrain_min", "supervised_min", "ablation_min",
            "post_min", "total_min", "avg_sup_epochs", "folds", "n_models",
        }
        assert set(est.keys()) == expected

    def test_total_gt_zero(self):
        args = _make_args()
        est = estimate_run_minutes(args, ["haelt"])
        assert est["total_min"] > 0

    def test_more_models_more_time(self):
        args = _make_args()
        one = estimate_run_minutes(args, ["haelt"])
        three = estimate_run_minutes(args, ["haelt", "tft", "transformer"])
        assert three["total_min"] > one["total_min"]

    def test_walk_forward_multiplies_time(self):
        args_no_wf = _make_args(walk_forward_cv=False)
        args_wf = _make_args(walk_forward_cv=True, walk_forward_folds=5)
        est_no = estimate_run_minutes(args_no_wf, ["haelt"])
        est_wf = estimate_run_minutes(args_wf, ["haelt"])
        assert est_wf["supervised_min"] > est_no["supervised_min"]


# ---------------------------------------------------------------------------
# 4. Pretrain ablation model parsing
# ---------------------------------------------------------------------------

class TestParsePretrainAblationModels:
    def test_none_returns_defaults(self):
        result = _parse_pretrain_ablation_models(None)
        assert result == {"tft", "transformer", "haelt"}

    def test_empty_string_returns_defaults(self):
        result = _parse_pretrain_ablation_models("")
        assert result == {"tft", "transformer", "haelt"}

    def test_none_string_disables(self):
        result = _parse_pretrain_ablation_models("none")
        assert result == set()

    def test_false_string_disables(self):
        result = _parse_pretrain_ablation_models("false")
        assert result == set()

    def test_comma_separated_parsed(self):
        result = _parse_pretrain_ablation_models("haelt,mamba")
        assert result == {"haelt", "mamba"}

    def test_list_input(self):
        result = _parse_pretrain_ablation_models(["tft", "gnn"])
        assert result == {"tft", "gnn"}


# ---------------------------------------------------------------------------
# 5. Effective max seq_len
# ---------------------------------------------------------------------------

class TestEffectiveMaxSeqLen:
    def test_no_curriculum_returns_at_least_base(self):
        args = _make_args(seq_len=60, curriculum=None)
        result = _effective_max_seq_len(args)
        assert result >= 60, f"Expected >= 60, got {result}"

    def test_curriculum_schedule_increases_max(self):
        cur = {
            "seq_schedule": [
                {"epoch_start": 0, "seq_len": 30},
                {"epoch_start": 5, "seq_len": 60},
                {"epoch_start": 10, "seq_len": 120},
            ]
        }
        args = _make_args(seq_len=60, curriculum=cur)
        assert _effective_max_seq_len(args) == 120


# ---------------------------------------------------------------------------
# 6. Profile normalization
# ---------------------------------------------------------------------------

class TestNormalizeArchitectureProfile:
    @pytest.fixture(autouse=True)
    def _import(self):
        from training.train_gpu import _normalize_architecture_profile
        self.normalize = _normalize_architecture_profile

    def test_haelt_maps_lstm_hidden(self):
        profile = {"lstm_hidden": 128, "d_model": 256, "nhead": 8, "num_layers": 3}
        out = self.normalize(profile, "haelt")
        assert out["hidden_size"] == 256  # lstm_hidden * 2
        assert out["d_model"] == 256
        assert out["nhead"] == 8
        assert out["num_layers"] == 3

    def test_haelt_legacy_n_transformer_layers(self):
        profile = {"n_transformer_layers": 4}
        out = self.normalize(profile, "haelt")
        assert out["num_layers"] == 4

    def test_haelt_prefers_num_layers_over_legacy(self):
        profile = {"num_layers": 3, "n_transformer_layers": 5}
        out = self.normalize(profile, "haelt")
        assert out["num_layers"] == 3

    def test_tft_maps_nhead(self):
        profile = {"hidden_size": 128, "nhead": 4, "lstm_layers": 2}
        out = self.normalize(profile, "tft")
        assert out["nhead"] == 4
        assert out["hidden_size"] == 128
        assert out["num_layers"] == 2

    def test_tft_legacy_attention_head_size(self):
        profile = {"attention_head_size": 4}
        out = self.normalize(profile, "tft")
        assert out["nhead"] == 4

    def test_tft_prefers_nhead_over_legacy(self):
        profile = {"nhead": 4, "attention_head_size": 8}
        out = self.normalize(profile, "tft")
        assert out["nhead"] == 4

    def test_gnn_maps_hidden_channels(self):
        profile = {"hidden_channels": 64, "num_layers": 3, "heads": 4, "node_features": 32}
        out = self.normalize(profile, "gnn")
        assert out["hidden_size"] == 64
        assert out["num_layers"] == 3
        assert out["nhead"] == 4
        assert out["node_features"] == 32

    def test_transformer_generic_mapping(self):
        profile = {"d_model": 128, "nhead": 8, "num_layers": 3}
        out = self.normalize(profile, "transformer")
        assert out["d_model"] == 128
        assert out["nhead"] == 8
        assert out["num_layers"] == 3

    def test_expert_generic_mapping(self):
        profile = {"d_model": 128, "nhead": 8, "num_layers": 4}
        out = self.normalize(profile, "expert")
        assert out["d_model"] == 128

    def test_common_fields_always_mapped(self):
        for model_name in SUPPORTED_SUPERVISED:
            profile = MODELS[model_name]
            out = self.normalize(profile, model_name)
            if "learning_rate" in profile:
                assert "lr" in out
            if "dropout" in profile:
                assert "dropout" in out
            if "seq_len" in profile:
                assert "seq_len" in out


# ---------------------------------------------------------------------------
# 7. YAML model config files parse and match Python profiles
# ---------------------------------------------------------------------------

class TestYamlModelConfigAlignment:
    @classmethod
    @pytest.fixture(scope="class")
    def yaml_configs(cls):
        import yaml
        configs = {}
        for p in Path("config/models").glob("*.yaml"):
            with p.open("r", encoding="utf-8") as f:
                configs[p.stem] = yaml.safe_load(f) or {}
        return configs

    def test_yaml_and_python_dropout_agree(self, yaml_configs):
        for name in SUPPORTED_SUPERVISED:
            if name not in yaml_configs:
                continue
            yaml_d = (yaml_configs[name].get("model") or {}).get("dropout")
            py_d = MODELS[name].get("dropout")
            if yaml_d is not None and py_d is not None:
                assert abs(yaml_d - py_d) < 1e-6, (
                    f"{name} dropout mismatch: YAML={yaml_d}, Python={py_d}"
                )

    def test_yaml_seq_len_matches_training(self, yaml_configs):
        for name in SUPPORTED_SUPERVISED:
            if name not in yaml_configs:
                continue
            yaml_seq = (yaml_configs[name].get("training") or {}).get("seq_len")
            py_seq = MODELS[name].get("seq_len")
            if yaml_seq is not None and py_seq is not None:
                assert yaml_seq == py_seq, (
                    f"{name} seq_len mismatch: YAML={yaml_seq}, Python={py_seq}"
                )
