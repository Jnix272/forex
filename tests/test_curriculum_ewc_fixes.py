"""Regression tests for curriculum-manager idempotency, EWC Fisher objective,
supervised-loop curriculum mask mapping, and nested YAML key loading."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from training.curriculum import (
    CurriculumManager,
    CurriculumManagerConfig,
    DifficultyCurriculumConfig,
    LossWeightingConfig,
)
from training.ewc import ElasticWeightConsolidation

ROOT = Path(__file__).resolve().parents[1]


def _lw_manager(n: int = 20) -> CurriculumManager:
    cfg = CurriculumManagerConfig(mode="loss_weighting", loss_weighting=LossWeightingConfig(ema_decay=0.5))
    return CurriculumManager(cfg, n_samples=n)


def test_get_sample_weights_without_losses_returns_cached_weights():
    mgr = _lw_manager()
    losses = np.linspace(0.1, 3.0, 20)
    w = mgr.get_sample_weights(losses=losses)
    assert not np.allclose(w, 1.0)
    np.testing.assert_allclose(mgr.get_sample_weights(), w)
    np.testing.assert_allclose(mgr.get_sample_weights(), w)


def test_loss_weighting_ema_advances_once_per_update():
    mgr = _lw_manager()
    calls = {"n": 0}
    orig = mgr.loss_weighting.compute_weights

    def _counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    mgr.loss_weighting.compute_weights = _counting
    mgr.update(0, losses=np.ones(20))
    assert calls["n"] == 1

    cfg = CurriculumManagerConfig(mode="combined", loss_weighting=LossWeightingConfig())
    combined = CurriculumManager(cfg, n_samples=20)
    calls["n"] = 0
    orig_c = combined.loss_weighting.compute_weights

    def _counting_c(*a, **k):
        calls["n"] += 1
        return orig_c(*a, **k)

    combined.loss_weighting.compute_weights = _counting_c
    combined.update(0, losses=np.ones(20))
    assert calls["n"] == 1


def test_update_without_miner_stats_keeps_freeze_counter():
    cfg = CurriculumManagerConfig(mode="difficulty", difficulty=DifficultyCurriculumConfig())
    mgr = CurriculumManager(cfg, n_samples=20, difficulty_scores=np.linspace(0, 1, 20))
    assert mgr.difficulty_curriculum is not None
    mgr._freeze_counter = 1
    mgr.update(3, losses=np.ones(20))
    assert mgr._freeze_counter == 1
    mgr.update(4, losses=np.ones(20), forgetting_rate=0.0, easy_ratio=0.0)
    assert mgr._freeze_counter == 0


class _PairHead(nn.Module):
    def __init__(self, n_feat: int = 4, n_pairs: int = 2):
        super().__init__()
        self.lin = nn.Linear(n_feat, n_pairs)

    def forward(self, x):
        return self.lin(x[:, -1, :])


def test_ewc_batch_loss_fn_drives_fisher():
    torch.manual_seed(0)
    model = _PairHead()
    x = torch.randn(64, 3, 4)
    y_cls = torch.randint(0, 3, (64, 2)).float()
    ds = TensorDataset(x, torch.randn(64, 2), y_cls)
    seen = {"batches": 0}

    def _batch_loss(m, batch):
        seen["batches"] += 1
        xb, _, yc = batch
        logits = m(xb)
        trade = yc != 1
        if not bool(trade.any()):
            return None, 0
        loss = nn.functional.binary_cross_entropy_with_logits(logits[trade], (yc[trade] == 2).float())
        return loss, int(xb.shape[0])

    ewc = ElasticWeightConsolidation(model, ds, torch.device("cpu"), max_samples=64, batch_loss_fn=_batch_loss)
    assert seen["batches"] >= 2
    assert ewc.last_samples_processed == 64
    assert float(ewc.fisher_lin_weight.sum()) > 0.0


def test_ewc_batch_loss_fn_skips_empty_batches():
    model = _PairHead()
    ds = TensorDataset(torch.randn(40, 3, 4), torch.zeros(40, 2))
    ewc = ElasticWeightConsolidation(
        model, ds, torch.device("cpu"), max_samples=40, batch_loss_fn=lambda m, b: (None, 0)
    )
    assert ewc.last_samples_processed == 0
    assert float(ewc.fisher_lin_weight.abs().sum()) == 0.0


def test_supervised_loop_curriculum_mask_maps_positions_to_rows():
    src = (ROOT / "training" / "supervised_loop.py").read_text(encoding="utf-8")
    assert "get_inclusion_mask(ep)" not in src
    assert "np.intersect1d(ep_train_idx, train_idx[_cm_mask])" in src
    assert "len(_cm_mask) == len(ep_train_idx)" not in src
    assert "huber_loss(outputs.squeeze(), y.float())" not in src
    assert "batch_loss_fn=_ewc_batch_loss" in src


def _si_epoch(si, model, x, y, steps: int = 5):
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(steps):
        opt.zero_grad()
        nn.functional.mse_loss(model(x), y).backward()
        si.pre_step()
        opt.step()
        si.post_step()
    si.update_omega()


def test_si_omega_decay_bounds_importance():
    from training.synaptic_intelligence import SynapticIntelligence

    x, y = torch.randn(32, 3, 4), torch.randn(32, 2)
    totals = {}
    for decay in (1.0, 0.5):
        torch.manual_seed(0)
        model = _PairHead()
        si = SynapticIntelligence(model, omega_decay=decay)
        for _ in range(6):
            _si_epoch(si, model, x, y)
        totals[decay] = float(si.omega_lin_weight.sum())
    assert 0.0 < totals[0.5] < totals[1.0]


def test_si_state_roundtrip_excludes_model_weights():
    from training.synaptic_intelligence import SynapticIntelligence

    torch.manual_seed(0)
    model = _PairHead()
    si = SynapticIntelligence(model, omega_decay=0.9)
    _si_epoch(si, model, torch.randn(32, 3, 4), torch.randn(32, 2))
    state = si.get_state()
    assert state and all(k.startswith(("omega_", "path_integral_", "cached_params_")) for k in state)
    assert not any(k.startswith("model.") for k in state)

    fresh = SynapticIntelligence(_PairHead(), omega_decay=0.9)
    assert fresh.load_state(state) == len(state)
    torch.testing.assert_close(fresh.omega_lin_weight, si.omega_lin_weight)
    torch.testing.assert_close(fresh.cached_params_lin_weight, si.cached_params_lin_weight)
    assert fresh.load_state({"omega_lin_weight": torch.zeros(9, 9)}) == 0


def test_supervised_loop_si_wiring():
    src = (ROOT / "training" / "supervised_loop.py").read_text(encoding="utf-8")
    assert 'if bool(getattr(args, "si_dynamic", False)):' in src
    assert '"si_state": _si.get_state() if _si is not None else None' in src
    assert '_resume_si_state = ck.get("si_state")' in src


def test_yaml_explicit_keys_beat_training_profile(tmp_path):
    from training.cli.profile import _apply_training_profile
    from training.cli.sync import _apply_yaml_config

    cfg = tmp_path / "run.yaml"
    cfg.write_text(
        "training:\n  si_lambda: 0.5\n  adversarial:\n    enabled: false\n",
        encoding="utf-8",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--si-lambda", type=float, default=1.0)
    parser.add_argument("--ewc-lambda", type=float, default=1.0)
    _apply_yaml_config(parser, str(cfg))
    args = parser.parse_args([])
    assert {"si_lambda", "enable_adversarial"} <= set(args._yaml_explicit_keys)

    log: list[str] = []
    _apply_training_profile(args, "haelt", frozenset(), log)
    assert args.si_lambda == 0.5
    assert args.enable_adversarial is False
    # Keys the YAML did not set still come from the profile.
    from config.model_training_profile import get_training_profile

    assert args.ewc_lambda == get_training_profile("haelt").ewc_lambda

    cli_args = parser.parse_args(["--si-lambda", "2.0"])
    _apply_training_profile(cli_args, "haelt", frozenset({"si_lambda"}), [])
    assert cli_args.si_lambda == 2.0


def test_yaml_nested_keys_load(tmp_path):
    from training.cli.sync import _apply_yaml_config

    cfg = tmp_path / "run.yaml"
    cfg.write_text(
        "training:\n"
        "  adversarial:\n"
        "    enabled: false\n"
        "    steps: 7\n"
        "curriculum:\n"
        "  self_paced:\n"
        "    enabled: true\n"
        "    models: [haelt]\n",
        encoding="utf-8",
    )
    parser = argparse.ArgumentParser()
    _apply_yaml_config(parser, str(cfg))
    ns = parser.parse_args([])
    assert ns.enable_adversarial is False
    assert ns.adversarial_steps == 7
    assert ns.use_self_paced is True
    assert ns.self_paced_models == ["haelt"]
