from __future__ import annotations

import argparse

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from pretrain.contrastive import (
    BYOLTrainer,
    MaskedReconstructionTrainer,
    RegimeAwareTSCLTrainer,
    TSCLTrainer,
)
from pretrain.extended_trainers import (
    ClusterContrastiveTrainer,
    DriftContrastiveTrainer,
    ForecastPretextTrainer,
    VAESeqTrainer,
)
from training.train_gpu import (
    _apply_yaml_config,
    _make_pretrain_span_plan,
    _normalize_pretrain_method,
    _select_pretrain_trainer_class,
)


def test_yaml_pretrain_knobs_override_defaults(tmp_path):
    cfg = tmp_path / "run.yaml"
    cfg.write_text(
        """
pretrain:
  enabled: true
  method: tscl
  lr: 0.0003
  batch: 64
  projection_dim: 96
  pred_dim: 48
  ema_decay: 0.99
  sample_windows: 2048
  blocks_per_epoch: 3
  mask_prob: 0.3
  recon_hidden_dim: 128
""",
        encoding="utf-8",
    )
    parser = argparse.ArgumentParser()

    _apply_yaml_config(parser, str(cfg))
    args = parser.parse_args([])

    assert args.pretrain is True
    assert args.pretrain_method == "tscl"
    assert args.pretrain_lr == pytest.approx(0.0003)
    assert args.pretrain_batch == 64
    assert args.pretrain_projection_dim == 96
    assert args.pretrain_pred_dim == 48
    assert args.pretrain_ema_decay == pytest.approx(0.99)
    assert args.pretrain_sample_windows == 2048
    assert args.pretrain_blocks_per_epoch == 3
    assert args.pretrain_mask_prob == pytest.approx(0.3)
    assert args.pretrain_recon_hidden_dim == 128


def test_pretrain_span_plan_covers_multiple_timeline_regions():
    spans = _make_pretrain_span_plan(
        1_000,
        200,
        max_spans=5,
        rng=np.random.default_rng(3),
    )
    starts = [s for s, _ in spans]

    assert sum(length for _, length in spans) == 200
    assert len(spans) > 1
    assert max(starts) - min(starts) > 300


def test_pretrain_span_plan_uses_difficulty_buckets():
    diff = np.concatenate([
        np.zeros(300, dtype=np.uint8),
        np.ones(300, dtype=np.uint8),
        np.full(300, 2, dtype=np.uint8),
    ])
    spans = _make_pretrain_span_plan(
        len(diff),
        180,
        diff=diff,
        max_spans=6,
        rng=np.random.default_rng(11),
    )
    idx = np.concatenate([np.arange(start, start + length) for start, length in spans])

    assert sum(length for _, length in spans) == 180
    assert set(np.unique(diff[idx])).issuperset({0, 1})
    assert 2 in set(np.unique(diff[idx]))


class TinyEncoder(nn.Module):
    def __init__(self, in_features: int = 4, d_model: int = 8):
        super().__init__()
        self.proj = nn.Linear(in_features, d_model)

    def forward(self, x):
        return self.proj(x.mean(dim=1))


class ConstantEncoder(nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[0], 8, device=x.device)


def test_byol_checkpoint_and_diagnostics_are_finite(tmp_path):
    rng = np.random.default_rng(5)
    X = rng.standard_normal((32, 8, 4)).astype(np.float32)
    ckpt = tmp_path / "encoder.pt"
    trainer = BYOLTrainer(
        TinyEncoder(),
        d_model=8,
        proj_dim=8,
        pred_dim=8,
        lr=1e-3,
        device="cpu",
        seed=7,
    )

    history = trainer.pretrain(X, epochs=1, batch_size=8, checkpoint_path=str(ckpt))
    diag = trainer.diagnostics(X)

    assert ckpt.exists()
    assert np.isfinite(history["loss"][0])
    assert np.isfinite(diag["embed_std"])
    assert np.isfinite(diag["align"])
    assert np.isfinite(diag["unif"])


def test_byol_diagnostics_detect_clear_collapse():
    X = np.ones((16, 8, 4), dtype=np.float32)
    trainer = BYOLTrainer(
        ConstantEncoder(),
        d_model=8,
        proj_dim=8,
        pred_dim=8,
        device="cpu",
    )

    diag = trainer.diagnostics(X)

    assert diag["collapsed"] is True
    assert diag["embed_std"] < 0.005


def test_pretrain_method_selection():
    assert _select_pretrain_trainer_class("byol", False) is BYOLTrainer
    assert _select_pretrain_trainer_class("masked", False) is MaskedReconstructionTrainer
    assert _select_pretrain_trainer_class("tscl", False) is TSCLTrainer
    assert _select_pretrain_trainer_class("tscl", True) is RegimeAwareTSCLTrainer
    assert _select_pretrain_trainer_class("vae", False) is VAESeqTrainer
    assert _select_pretrain_trainer_class("autoencoder", False) is VAESeqTrainer
    assert _select_pretrain_trainer_class("cluster", False) is ClusterContrastiveTrainer
    assert _select_pretrain_trainer_class("forecast", False) is ForecastPretextTrainer
    assert _select_pretrain_trainer_class("drift", False) is DriftContrastiveTrainer
    assert _normalize_pretrain_method("autoencoder") == "vae"
    assert _normalize_pretrain_method("drift_pretrain") == "drift"


@pytest.mark.parametrize(
    "trainer_cls,extra",
    [
        (VAESeqTrainer, {"seq_len": 8, "n_features": 4, "latent_dim": 8, "hidden_dim": 16}),
        (ForecastPretextTrainer, {"seq_len": 8, "n_features": 4, "horizon": 2, "hidden_dim": 16}),
        (DriftContrastiveTrainer, {}),
        (ClusterContrastiveTrainer, {"proj_dim": 8, "n_clusters": 2}),
    ],
)
def test_extended_pretrain_smoke(tmp_path, trainer_cls, extra):
    rng = np.random.default_rng(21)
    X = rng.standard_normal((24, 8, 4)).astype(np.float32)
    ckpt = tmp_path / f"{trainer_cls.__name__}.pt"
    kwargs = dict(
        encoder=TinyEncoder(),
        d_model=8,
        lr=1e-3,
        device="cpu",
        seed=3,
    )
    kwargs.update(extra)
    trainer = trainer_cls(**kwargs)
    history = trainer.pretrain(X, epochs=1, batch_size=8, checkpoint_path=str(ckpt))
    diag = trainer.diagnostics(X)
    assert ckpt.exists()
    assert np.isfinite(history["loss"][0])
    assert np.isfinite(diag.get("embed_std", diag.get("align", 0.0)))


def test_masked_reconstruction_checkpoint_and_diagnostics(tmp_path):
    rng = np.random.default_rng(9)
    X = rng.standard_normal((32, 8, 4)).astype(np.float32)
    ckpt = tmp_path / "masked_encoder.pt"
    trainer = MaskedReconstructionTrainer(
        TinyEncoder(),
        d_model=8,
        seq_len=8,
        n_features=4,
        hidden_dim=16,
        mask_prob=0.25,
        lr=1e-3,
        device="cpu",
        seed=13,
    )

    history = trainer.pretrain(X, epochs=1, batch_size=8, checkpoint_path=str(ckpt))
    diag = trainer.diagnostics(X)

    assert ckpt.exists()
    assert np.isfinite(history["loss"][0])
    assert np.isfinite(diag["masked_mse"])
    assert np.isfinite(diag["embed_std"])
