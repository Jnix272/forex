"""Pretrain resume reuse, held-out collapse gate, and hard-example span format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

import training.pretrain_runner as pr

ROOT = Path(__file__).resolve().parents[1]


class _TinyBackbone(nn.Module):
    def __init__(self, n_features: int, d_model: int = 8):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(n_features, d_model, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.layers(x.transpose(1, 2))


class _ConstBackbone(nn.Module):
    """Ignores its input: every window maps to the same embedding (collapsed)."""

    def __init__(self, d_model: int = 8):
        super().__init__()
        self.bias = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return self.bias.expand(x.shape[0], -1) + 0.0 * x.sum(dim=(1, 2), keepdim=False)[:, None]


class _TinyModel(nn.Module):
    def __init__(self, backbone: nn.Module, d_model: int = 8):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(d_model, 3)

    def forward(self, x):
        return self.head(self.backbone(x))


def _args(tmp: Path, **overrides) -> argparse.Namespace:
    base = dict(
        model="transformer",
        checkpoint_dir=str(tmp / "ckpt"),
        seq_len=8,
        pretrain=True,
        pretrain_method="byol",
        pretrain_epochs=1,
        pretrain_batch=8,
        pretrain_sample_windows=64,
        pretrain_blocks_per_epoch=1,
        pretrain_regime=False,
        pretrain_max_epochs=0,
        pretrain_min_epochs=0,
        pretrain_handoff_min_delta=0.0,
        pretrain_handoff_loss=float("-inf"),
        pretrain_projection_dim=8,
        pretrain_pred_dim=8,
        pretrain_ema_decay=0.99,
        pretrain_lr=1e-3,
        pretrain_temperature=0.5,
        pretrain_augmentations=None,
        force_pretrain=False,
        resume=False,
        use_multi_task_pretrainer=False,
        seed=42,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture
def tiny_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "_promotion_holdout_n", lambda n, args: 0)
    monkeypatch.setattr(pr, "_embargo_bars", lambda args: 0)
    monkeypatch.setattr(pr, "_trainable_max_index", lambda n, args: n)
    monkeypatch.setattr(pr, "_load_diff_array", lambda cp, n: None)
    rng = np.random.default_rng(0)
    cache_path = str(tmp_path / "cache")
    np.save(cache_path + "_X.npy", rng.standard_normal((160, 8, 4)).astype(np.float32))
    np.save(cache_path + "_y.npy", rng.standard_normal(160).astype(np.float32))
    return cache_path


def _report(tmp: Path) -> dict:
    return json.loads((tmp / "ckpt" / "pretrain_report.json").read_text(encoding="utf-8"))


def test_reuse_blocker_requires_matching_completed_report(tmp_path):
    ckpt = tmp_path / "enc.pt"
    ckpt.write_bytes(b"weights")
    good = {
        "status": "completed",
        "quality_gate_result": "passed",
        "scaled_inputs": True,
        "method": "byol",
        "cache_path": "c.zarr",
        "trainable_windows_used_by_pretrain": 100,
        "checkpoint_sha256": pr._sha256(str(ckpt)),
    }
    kw = dict(cache_path="c.zarr", trainable_windows=100, method="byol")
    assert pr._pretrain_reuse_blocker(good, str(ckpt), **kw) is None
    assert pr._pretrain_reuse_blocker({}, str(ckpt), **kw)
    for key, bad in (
        ("status", "started"),
        ("quality_gate_result", "warning_low_uniformity"),
        ("scaled_inputs", False),
        ("method", "tscl"),
        ("cache_path", "other.zarr"),
        ("trainable_windows_used_by_pretrain", 99),
        ("checkpoint_sha256", "0" * 64),
    ):
        assert pr._pretrain_reuse_blocker({**good, key: bad}, str(ckpt), **kw), key


def test_resume_reuses_encoder_and_keeps_report_loadable(tmp_path, tiny_cache):
    from training.supervised_loop import _load_pretrained_encoder

    device = torch.device("cpu")
    model = _TinyModel(_TinyBackbone(4))
    assert pr.run_pretrain(model, tiny_cache, n_features=4, args=_args(tmp_path), device=device) is model
    first = _report(tmp_path)
    assert first["status"] == "completed" and first["quality_gate_result"] == "passed"

    resumed = _TinyModel(_TinyBackbone(4))
    out = pr.run_pretrain(resumed, tiny_cache, n_features=4, args=_args(tmp_path, resume=True), device=device)
    assert out is resumed
    second = _report(tmp_path)
    assert second["status"] == "completed"
    assert second["quality_gate_result"] == "passed"
    assert second["checkpoint_sha256"] == first["checkpoint_sha256"]
    assert "resume_reused_at" in second

    fresh = _TinyModel(_TinyBackbone(4))
    assert _load_pretrained_encoder(fresh, _args(tmp_path, resume=True), device) is True


def test_resume_reruns_when_window_changed(tmp_path, tiny_cache, monkeypatch):
    device = torch.device("cpu")
    pr.run_pretrain(_TinyModel(_TinyBackbone(4)), tiny_cache, n_features=4, args=_args(tmp_path), device=device)
    monkeypatch.setattr(pr, "_trainable_max_index", lambda n, args: n - 40)
    pr.run_pretrain(
        _TinyModel(_TinyBackbone(4)), tiny_cache, n_features=4, args=_args(tmp_path, resume=True), device=device
    )
    rep = _report(tmp_path)
    assert "resume_reused_at" not in rep
    assert rep["trainable_windows_used_by_pretrain"] == int((160 - 40) * 0.9)


def test_byol_collapse_is_discarded(tmp_path, tiny_cache, monkeypatch):
    sentinel = nn.Linear(1, 1)
    monkeypatch.setattr(pr, "build_model", lambda *a, **k: sentinel)
    out = pr.run_pretrain(
        _TinyModel(_ConstBackbone()), tiny_cache, n_features=4, args=_args(tmp_path), device=torch.device("cpu")
    )
    assert out is sentinel
    rep = _report(tmp_path)
    assert rep["status"] == "discarded"
    assert rep["quality_gate_result"] == "failed_embedding_collapse"
    assert not (tmp_path / "ckpt" / "contrastive_encoder.pt").exists()


def test_pretrain_lr_schedule_warms_up_and_decays():
    lrs = [pr._pretrain_lr_for_epoch(1e-3, e, 12) for e in range(12)]
    assert lrs[0] < lrs[1] < lrs[2] == pytest.approx(1e-3)
    assert all(a >= b for a, b in zip(lrs[2:], lrs[3:]))
    assert lrs[-1] == pytest.approx(5e-5)
    assert pr._pretrain_lr_for_epoch(1e-3, 0, 1) == pytest.approx(1e-3)


def test_outer_loop_drives_trainer_lr(tmp_path, tiny_cache, monkeypatch):
    seen: list[float] = []
    orig = pr.BYOLTrainer.pretrain

    def _spy(self, X, *a, **k):
        seen.append(self.opt.param_groups[0]["lr"])
        return orig(self, X, *a, **k)

    monkeypatch.setattr(pr.BYOLTrainer, "pretrain", _spy)
    pr.run_pretrain(
        _TinyModel(_TinyBackbone(4)),
        tiny_cache,
        n_features=4,
        args=_args(tmp_path, pretrain_epochs=4),
        device=torch.device("cpu"),
    )
    assert seen == pytest.approx([pr._pretrain_lr_for_epoch(1e-3, e, 4) for e in range(4)])


def test_state_transfer_fraction():
    a, b = _TinyBackbone(4), _TinyBackbone(4)
    assert pr._state_transfer_fraction(a.state_dict(), b) == 1.0
    assert pr._state_transfer_fraction(_TinyBackbone(5).state_dict(), b) < 1.0
    assert pr._state_transfer_fraction({}, b) == 0.0


def test_multi_task_skips_untransferable_backbone(tmp_path):
    ckpt = tmp_path / "contrastive_encoder.pt"
    args = _args(tmp_path, pretrain_epochs=1)
    windows = np.random.default_rng(0).standard_normal((32, 8, 4)).astype(np.float32)
    out = pr._run_multi_task_pretrain(
        _TinyModel(_TinyBackbone(4)), windows, str(ckpt), 4, args, torch.device("cpu")
    )
    assert out is None
    assert not ckpt.exists()


def test_heldout_gate_error_blocks_transfer(tmp_path, tiny_cache, monkeypatch):
    import pretrain.loss_scaling as ls

    def _boom(*a, **k):
        raise RuntimeError("gate exploded")

    monkeypatch.setattr(ls, "normalized_mse_loss", _boom)
    pr.run_pretrain(
        _TinyModel(_TinyBackbone(4)),
        tiny_cache,
        n_features=4,
        args=_args(tmp_path, pretrain_method="masked"),
        device=torch.device("cpu"),
    )
    rep = _report(tmp_path)
    assert rep["quality_gate_result"] == "unverified_heldout_gate_error"
    assert "gate exploded" in rep["heldout_gate_error"]


def test_adversarial_vulnerability_scores_checked_against_features():
    src = (ROOT / "training" / "supervised_loop.py").read_text(encoding="utf-8")
    assert "len(_vuln) != int(n_features)" in src


def test_hard_example_spans_are_single_windows():
    src = (ROOT / "training" / "pretrain_runner.py").read_text(encoding="utf-8")
    assert "(int(i), 1) for i in _valid_he" in src
    assert "(i, i + 1)" not in src
    assert 'getattr(trainer, "train_loader"' not in src
