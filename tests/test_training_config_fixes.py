"""Regression tests for the training config / loss / curriculum audit fixes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


# ── Fix 1 / 10: config precedence and generic CLI detection ─────────────────


@pytest.fixture
def run_yaml_args(monkeypatch):
    from training.cli import parse_args

    monkeypatch.setattr(sys, "argv", ["train_gpu", "--config", "config/run.yaml"])
    monkeypatch.chdir(ROOT)
    return parse_args()


def test_run_yaml_lr_dropout_survive_model_profile(run_yaml_args):
    from training.cli.profile import _model_build_args

    args = run_yaml_args
    assert args.lr == pytest.approx(3e-5)
    assert args.dropout == pytest.approx(0.35)
    assert {"lr", "dropout"} <= set(args._yaml_explicit_keys)
    eff = _model_build_args(args, args.model)
    assert eff.lr == pytest.approx(3e-5)
    assert eff.dropout == pytest.approx(0.35)


def test_run_yaml_early_stop_and_flags(run_yaml_args):
    args = run_yaml_args
    assert args.early_stop_metric == "cost_sharpe"
    assert args.early_stop_min_delta == pytest.approx(0.02)
    assert args.mt_w_quantile == pytest.approx(0.2)
    assert args.use_volatility_sampler is False
    assert args.curriculum_manager is False


def test_run_yaml_direction_gate_thresholds(run_yaml_args):
    args = run_yaml_args
    assert args.direction_max_pred_class_share == pytest.approx(0.80)
    assert args.direction_min_pred_class_share == pytest.approx(0.05)
    assert args.direction_min_recall == pytest.approx(0.05)
    assert args.direction_min_true_class_share == pytest.approx(0.15)


def test_model_profile_keeps_yaml_keys_only_for_primary_model():
    from config.models import architecture_config
    from training.cli.profile import _apply_model_profile

    prof = architecture_config("haelt")
    prof_lr = prof.get("lr", prof.get("learning_rate"))
    if prof_lr is None:
        pytest.skip("haelt profile has no lr")
    yaml_lr = float(prof_lr) * 7.0

    def _ns(primary):
        return argparse.Namespace(
            lr=yaml_lr, dropout=0.35, model="haelt",
            _yaml_explicit_keys=frozenset({"lr"}),
            _cli_profile_overrides=frozenset(),
            _primary_model=primary,
        )

    kept = _apply_model_profile(_ns("haelt"), "haelt")
    assert kept.lr == pytest.approx(yaml_lr)
    other = _apply_model_profile(_ns("tft"), "haelt")
    assert other.lr == pytest.approx(float(prof_lr))


def test_generic_cli_override_detection():
    from training.cli.helpers import _collect_cli_profile_overrides

    p = argparse.ArgumentParser()
    p.add_argument("--learning-rate", dest="lr", type=float)
    p.add_argument("--early-stop-metric")
    p.add_argument("--pretrain", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dropout", type=float)
    got = _collect_cli_profile_overrides(
        p, ["--learning-rate=1e-4", "--no-pretrain", "--early-stop-m", "sharpe", "--", "--dropout", "0.1"]
    )
    assert got == frozenset({"lr", "pretrain", "early_stop_metric"})


# ── Fix 11: BooleanOptionalAction flags ─────────────────────────────────────


@pytest.mark.parametrize(
    "flag,dest",
    [
        ("pretrain", "pretrain"),
        ("rl-train", "rl_train"),
        ("train-ensemble", "train_ensemble"),
        ("deploy-ensemble", "deploy_ensemble"),
        ("curriculum-manager", "curriculum_manager"),
        ("use-mixup", "use_mixup"),
    ],
)
def test_boolean_optional_flags(monkeypatch, flag, dest):
    from training.cli import parse_args

    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(sys, "argv", ["train_gpu", "--config", "config/run.yaml", f"--no-{flag}"])
    assert getattr(parse_args(), dest) is False
    monkeypatch.setattr(sys, "argv", ["train_gpu", "--config", "config/run.yaml", f"--{flag}"])
    args = parse_args()
    assert getattr(args, dest) is True
    assert dest in args._cli_profile_overrides


# ── Fix 2 / 15: early-stop keys, aliases, dead keys ─────────────────────────


def _yaml_parser(tmp_path, text):
    from training.cli.sync import _apply_yaml_config

    cfg = tmp_path / "run.yaml"
    cfg.write_text(text, encoding="utf-8")
    p = argparse.ArgumentParser()
    _apply_yaml_config(p, str(cfg))
    return p.parse_args([])


def test_training_patience_alias_and_canonical_wins(tmp_path, capsys):
    ns = _yaml_parser(tmp_path, "training:\n  patience: 4\n  early_stop_metric: sharpe\n  early_stop_min_delta: 0.01\n")
    assert ns.early_stop_patience == 4
    assert ns.early_stop_metric == "sharpe"
    assert ns.early_stop_min_delta == pytest.approx(0.01)

    ns = _yaml_parser(tmp_path, "training:\n  patience: 4\n  early_stop_patience: 9\n")
    assert ns.early_stop_patience == 9
    assert "conflicts with training.early_stop_patience" in capsys.readouterr().out


def test_yaml_new_mappings(tmp_path):
    ns = _yaml_parser(
        tmp_path,
        "training:\n  huber_delta: 0.7\n  mixup_alpha: 0.3\n  mixup_prob: 0.4\n  ewc_lambda_ramp: 1.0\n"
        "multitask:\n  w_quantile: 0.15\n"
        "curriculum:\n  manager:\n    enabled: true\n    mode: self_paced\n",
    )
    assert ns.huber_delta == pytest.approx(0.7)
    assert ns.mixup_alpha == pytest.approx(0.3)
    assert ns.mixup_prob == pytest.approx(0.4)
    assert ns.ewc_lambda_ramp == pytest.approx(1.0)
    assert ns.mt_w_quantile == pytest.approx(0.15)
    assert ns.curriculum_manager is True
    assert ns.curriculum_manager_mode == "self_paced"


def test_dead_yaml_keys_warn_once(tmp_path, capsys):
    from training.cli.sync import _warn_unused_yaml_keys

    cfg = {"training": {"num_classes": 3}}
    assert "training.num_classes" in _warn_unused_yaml_keys(cfg, str(tmp_path / "a.yaml"))
    assert "training.num_classes is set but unused" in capsys.readouterr().out
    _warn_unused_yaml_keys(cfg, str(tmp_path / "a.yaml"))
    assert "training.num_classes" not in capsys.readouterr().out


def test_early_stop_metric_direction():
    from training.train_utils import early_stop_metric_maximizes

    assert early_stop_metric_maximizes("cost_sharpe")
    assert early_stop_metric_maximizes("sharpe")
    assert not early_stop_metric_maximizes("val_loss")
    assert not early_stop_metric_maximizes("auto")


def test_fold_history_summary_cost_sharpe_higher_is_better():
    from training.pretrain_runner import _fold_history_summary

    folds = [{"best_metric": 0.4}, {"best_metric": 1.2}]
    assert _fold_history_summary(folds, "cost_sharpe")["best_metric"] == pytest.approx(1.2)
    assert _fold_history_summary(folds, "val_loss")["best_metric"] == pytest.approx(0.4)


def test_supervised_loop_uses_early_stop_metric():
    src = (ROOT / "training" / "supervised_loop.py").read_text(encoding="utf-8")
    assert "early_stop_metric" in src
    assert "_es_min_delta" in src


# ── Fix 1: training-memory never nudges explicit keys ───────────────────────


def test_training_memory_skips_protected_keys(tmp_path):
    from training.training_memory import TrainingMemory

    mem = TrainingMemory(tmp_path / "mem.json")
    mem._data = {"total_runs": 3, "recommended_lr": 1e-3, "recommended_dropout": 0.1}
    args = SimpleNamespace(lr=3e-5, dropout=0.35, _yaml_explicit_keys=frozenset({"lr"}),
                           _cli_profile_overrides=frozenset({"dropout"}))
    mem.apply_to_args(args)
    assert args.lr == pytest.approx(3e-5)
    assert args.dropout == pytest.approx(0.35)

    free = SimpleNamespace(lr=3e-5, dropout=0.35)
    mem.apply_to_args(free)
    assert free.lr != pytest.approx(3e-5)


def test_train_gpu_applies_memory_per_model_only():
    src = (ROOT / "training" / "train_gpu.py").read_text(encoding="utf-8")
    assert "_train_memory.apply_to_args(args)" not in src


# ── Fix 3 / 7: curriculum suffix mask and label-magnitude gate ──────────────


def test_curriculum_mask_matches_pair_suffix():
    src = (ROOT / "training" / "supervised_loop.py").read_text(encoding="utf-8")
    assert "_schema_positions" in src
    assert '"::"' in src


def test_label_gate_follows_difficulty_schedule():
    from training.curriculum import label_gate_max_tier

    sched = [{"epoch_start": 0, "max_difficulty": 0}, {"epoch_start": 3, "max_difficulty": 2}]
    assert label_gate_max_tier(0, 40, sched) == 0
    assert label_gate_max_tier(5, 40, sched) == 2
    assert label_gate_max_tier(1, 40, sched, active_stage=1) == 1
    assert label_gate_max_tier(0, 40, None) == 0
    assert label_gate_max_tier(15, 40, None) == 1
    assert label_gate_max_tier(25, 40, None) is None


# ── Fix 13: SI lambda range reachable ───────────────────────────────────────


def test_dynamic_si_lambda_reaches_bounds():
    from training.train_utils import dynamic_si_lambda

    assert dynamic_si_lambda(1.0, 0.0, 0.1, 2.0) == pytest.approx(2.0)
    assert dynamic_si_lambda(1.0, 1e6, 0.1, 2.0) == pytest.approx(0.1, abs=1e-6)
    assert dynamic_si_lambda(1.0, float("nan"), 0.1, 2.0) == pytest.approx(2.0)
    assert dynamic_si_lambda(0.5, 3.0) == pytest.approx(0.5)


# ── Fix 6: feature ablation ─────────────────────────────────────────────────


def test_keep_groups_keeps_ungrouped_and_warns_unknown(capsys):
    from training.feature_ablation import _build_feature_ablation_mask

    schema = ["EUR_USD::rsi", "EUR_USD::cot_net", "EUR_USD::hour_sin", "GBP_USD::rsi"]
    groups = {"technical": {"features": ["rsi"]}, "cot": {"features": ["cot_net"]}}
    cfg = {"enabled": True, "keep_groups": ["technical", "nope"], "drop_features": ["ghost"]}
    mask, report = _build_feature_ablation_mask(schema, groups, cfg, len(schema))
    assert mask.tolist() == [1.0, 0.0, 1.0, 1.0]
    assert report["unknown_groups"] == ["nope"]
    assert report["unknown_features"] == ["ghost"]
    assert "[FeatureAblation] WARN" in capsys.readouterr().out


def test_ablation_mask_applied_each_epoch_and_persisted():
    src = (ROOT / "training" / "supervised_loop.py").read_text(encoding="utf-8")
    assert "_feature_ablation_mask_list" in src
    assert 'ckpt_meta["feature_ablation"]' in src or "\"feature_ablation\"" in src


# ── Fix 12: loss wiring ─────────────────────────────────────────────────────


def _loss_args(**kw):
    base = dict(multitask=True, per_pair_heads=False, loss="sharpe_huber", sharpe_weight=0.3,
                mt_w_quantile=0.11, mt_entropy_weight=0.02, huber_delta=0.6, bar_freq="1h",
                mt_w_ret=0.5, mt_w_conf=0.3, mt_focal_gamma=0.0, mt_class_balance_weight=0.0,
                label_smoothing=0.05)
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.mark.parametrize("per_pair", [False, True])
def test_build_criterion_multitask_wiring(per_pair):
    from training.loop_losses import build_criterion

    crit = build_criterion(_loss_args(per_pair_heads=per_pair), torch.device("cpu"))
    assert crit.w_quantile == pytest.approx(0.11)
    assert crit.entropy_weight == pytest.approx(0.02)
    inner = getattr(crit, "single_loss", crit)
    assert inner.w_sharpe == pytest.approx(0.3)


def test_entropy_bonus_lowers_loss_for_uncertain_logits():
    from training.loop_losses import _apply_direction_entropy_bonus

    crit = SimpleNamespace(entropy_weight=0.5)
    loss = torch.tensor(1.0)
    out = _apply_direction_entropy_bonus(loss, (torch.zeros(8),), crit, direction_only=False)
    assert float(out) == pytest.approx(1.0 - 0.5 * np.log(2.0), rel=1e-5)
    assert float(_apply_direction_entropy_bonus(loss, (torch.zeros(8),), crit, True)) == 1.0


def test_unsupported_option_warnings(capsys):
    from training.cli.helpers import _warn_unsupported_training_options

    args = SimpleNamespace(multitask=True, loss="huber", use_volatility_sampler=True,
                           direction_weight=0.5, sharpe_weight=0.2, mt_direction_weight_floor=0.3,
                           _yaml_explicit_keys=frozenset({"direction_weight", "sharpe_weight"}))
    msgs = _warn_unsupported_training_options(args)
    assert len(msgs) == 4
    assert capsys.readouterr().out.count("[Config] WARN") == 4


# ── Fix 5: mixup ────────────────────────────────────────────────────────────


def test_mixup_sample_pairing():
    from training.train_utils import MixupBatch

    np.random.seed(0)
    lam, perm = MixupBatch(alpha=0.2, p=1.0).sample_pairing(16)
    assert 0.5 <= lam <= 1.0
    assert sorted(perm.tolist()) == list(range(16))
    assert MixupBatch(alpha=0.0, p=1.0).sample_pairing(16) is None
    assert MixupBatch(alpha=0.2, p=1.0).sample_pairing(1) is None


def test_build_train_loss_with_mixup(monkeypatch):
    import training.loop_losses as ll
    from training.train_utils import MixupBatch

    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1)
    xb = torch.randn(8, 4)
    yb = torch.randn(8)
    crit = torch.nn.MSELoss(reduction="none")

    def _run():
        _, loss = ll._build_train_loss(
            model, xb, yb, None, None, None, crit, False, False, False,
            None, 0.0, None, 0.0, None, 0.0, None, None, None, 1,
        )
        return loss

    monkeypatch.setattr(ll, "_MIXUP", None)
    plain = _run()
    class _FixedMix(MixupBatch):
        def sample_pairing(self, bsz, device=None):
            return 0.6, torch.arange(bsz - 1, -1, -1)

    monkeypatch.setattr(ll, "_MIXUP", _FixedMix())
    mixed = _run()
    xm = 0.6 * xb + 0.4 * xb.flip(0)
    pm = model(xm).squeeze(-1)
    expected = 0.6 * ((pm - yb) ** 2).mean() + 0.4 * ((pm - yb.flip(0)) ** 2).mean()
    assert float(mixed) == pytest.approx(float(expected), rel=1e-4)
    assert torch.isfinite(mixed).all()
    assert not torch.allclose(plain, mixed)


# ── Fix 9: EWC anchor ───────────────────────────────────────────────────────


def test_ewc_set_anchor_copies_warm_start_weights():
    from training.supervised_loop import _ewc_set_anchor

    model = torch.nn.Linear(3, 2)
    anchor = {f"_orig_mod.{k}": torch.full_like(v, 0.25) for k, v in model.state_dict().items()}
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(1.0)

    class _EwcStub(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.params = dict(m.named_parameters())
            for n, p in self.params.items():
                self.register_buffer(f"saved_{n.replace('.', '_')}", p.clone().detach())

    ewc = _EwcStub(model)
    assert _ewc_set_anchor(ewc, anchor) == 2
    assert torch.allclose(ewc.saved_weight, torch.full_like(ewc.saved_weight, 0.25))


def test_ewc_engages_on_warm_start_not_resume():
    src = (ROOT / "training" / "supervised_loop.py").read_text(encoding="utf-8")
    assert "_warm_started" in src
    assert "ewc_lambda_ramp" in src


# ── Fix 14: pretrain handoff patience / ablation ────────────────────────────


def test_pretrain_handoff_patience_and_ablation_guard():
    pre = (ROOT / "training" / "pretrain_runner.py").read_text(encoding="utf-8")
    assert "_stale_epochs >= handoff_patience" in pre
    gpu = (ROOT / "training" / "train_gpu.py").read_text(encoding="utf-8")
    assert "early_stop_metric_maximizes" in gpu
