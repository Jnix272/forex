"""
Tests for multi-task pretraining (Improvement #10):
Contrastive, masked reconstruction, forecast, VAE, drift, domain adaptation.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from pretrain.multi_task import (
    CORALLoss,
    DomainDiscriminator,
    MMDLoss,
    MultiTaskPretrainConfig,
    MultiTaskPretrainer,
    TimeSeriesAugmenter,
    adapt_encoder_to_target,
    byol_loss,
    create_multi_task_pretrainer,
    domain_adversarial_loss,
    drift_loss,
    forecast_loss,
    grad_reverse,
    masked_reconstruction_loss,
    nt_xent_loss,
    pretrain_multi_task,
    vae_loss,
)

# ════════════════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_data():
    """Generate synthetic time-series data for testing."""
    rng = np.random.default_rng(0)
    n = 200
    seq_len = 30
    n_features = 5
    # Create data with some structure (trend + noise)
    t = np.linspace(0, 10, seq_len)
    base = np.sin(t)[:, None] + np.cos(2 * t)[:, None]
    X = rng.normal(0, 0.5, (n, seq_len, 3)) + base[:seq_len, :3]
    return X.astype(np.float32)


@pytest.fixture
def domain_data():
    """Generate data with two domains."""
    rng = np.random.default_rng(1)
    n1 = 100
    n2 = 100
    seq_len = 20
    n_features = 4
    # Domain 0: lower mean
    X1 = rng.normal(0, 0.5, (n1, seq_len, 4))
    # Domain 2: higher mean
    X2 = rng.normal(2, 0.5, (n2, seq_len, 4))
    X = np.vstack([X1, X2]).astype(np.float32)
    labels = np.hstack([np.zeros(n1), np.ones(n2)]).astype(np.int64)
    return X, labels


# ════════════════════════════════════════════════════════════════════════════
# Augmentations
# ════════════════════════════════════════════════════════════════════════════

def test_augmenter_basic():
    """Test TimeSeriesAugmenter basic functionality."""
    aug = TimeSeriesAugmenter(
        jitter_std=0.03,
        scale_range=(0.9, 1.1),
        feature_drop_p=0.1,
        crop_ratio=(0.7, 1.0),
        seed=0,
    )
    x = np.random.randn(10, 20, 5).astype(np.float32)
    v1, v2 = aug.augment_pair(x)
    assert v1.shape == x.shape
    assert v2.shape == x.shape
    assert not np.array_equal(v1, x)  # should be different
    assert not np.array_equal(v2, x)
    assert not np.array_equal(v1, v2)  # should be different from each other


def test_augmenter_deterministic():
    """Test augmenter is deterministic with same seed."""
    aug1 = TimeSeriesAugmenter(seed=42)
    aug2 = TimeSeriesAugmenter(seed=42)
    x = np.random.randn(5, 10, 3).astype(np.float32)
    v1a, v1b = aug1.augment_pair(x)
    v2a, v2b = aug2.augment_pair(x)
    assert np.array_equal(v1a, v2a)
    assert np.array_equal(v1b, v2b)


# ════════════════════════════════════════════════════════════════════════════
# Gradient Reversal
# ═════════════════════════════════════════════════════════════════════════════

def test_grad_reverse():
    """Test gradient reversal function."""
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = grad_reverse(x, 2.0)
    loss = y.sum()
    loss.backward()
    assert torch.allclose(x.grad, -2.0 * torch.ones_like(x.grad))


# ════════════════════════════════════════════════════════════════════════════
# Task Losses
# ════════════════════════════════════════════════════════════════════════════

def test_nt_xent_loss():
    """Test NT-Xent contrastive loss."""
    z1 = torch.randn(4, 16)
    z2 = torch.randn(4, 16)
    loss = nt_xent_loss(z1, z2, temperature=0.5)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_nt_xent_loss_perfect_match():
    """Test NT-Xent with perfect positive pairs."""
    z = F.normalize(torch.randn(4, 16), dim=-1)
    loss = nt_xent_loss(z, z, temperature=0.5)
    # Perfect match should give very low loss
    assert loss.item() < 1.0


def test_byol_loss():
    """Test BYOL loss."""
    p1 = torch.randn(4, 16)
    p2 = torch.randn(4, 16)
    z1 = torch.randn(4, 16)
    z2 = torch.randn(4, 16)
    loss = byol_loss(p1, p2, z1, z2)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_masked_reconstruction_loss():
    """Test masked reconstruction loss."""
    B, T, F = 4, 10, 3
    recon = torch.randn(B, T, F)
    target = torch.randn(B, T, F)
    mask = torch.zeros(B, T, F, dtype=torch.bool)
    mask[:, 2:5, :] = True

    loss = masked_reconstruction_loss(recon, target, mask)
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_vae_loss():
    """Test VAE loss."""
    B, T, F = 4, 10, 3
    recon = torch.randn(B, T, F)
    target = torch.randn(B, T, F)
    mu = torch.randn(4, 16)
    logvar = torch.randn(4, 16)

    loss, recon_loss, kl = vae_loss(torch.randn(4, 10, 3), torch.randn(4, 10, 3), mu, logvar, beta=0.001)
    assert loss.ndim == 0
    assert recon_loss.ndim == 0
    assert kl.ndim == 0


def test_forecast_loss():
    """Test forecast loss."""
    pred = torch.randn(4, 5, 3)
    target = torch.randn(4, 5, 3)
    loss = forecast_loss(pred, target)
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_drift_loss():
    """Test drift loss."""
    clean = torch.randn(8, 16)
    drift = torch.randn(8, 16)
    loss = drift_loss(clean, drift, margin=1.0)
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_domain_adversarial_loss():
    """Test domain adversarial loss."""
    features = torch.randn(10, 16, requires_grad=True)
    domain_labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    disc = nn.Linear(16, 2)
    loss = domain_adversarial_loss(features, domain_labels, disc, lambda_=1.0)
    assert loss.ndim == 0
    assert loss.item() >= 0


# ════════════════════════════════════════════════════════════════════════════
# Domain Adaptation Losses
# ════════════════════════════════════════════════════════════════════════════

def test_mmd_loss():
    """Test MMD loss."""
    source = torch.randn(10, 8)
    target = torch.randn(10, 8)
    mmd = MMDLoss(kernel="rbf", gamma=1.0)
    loss = mmd(source, target)
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_mmd_loss_same_distribution():
    """Test MMD with same distribution (should be near 0)."""
    X = torch.randn(20, 8)
    idx = torch.randperm(20)
    source = X[:10]
    target = X[10:]
    mmd = MMDLoss(kernel="rbf", gamma=1.0)
    loss = mmd(source, target)
    assert loss.item() < 1.0  # Should be small for same distribution


def test_coral_loss():
    """Test CORAL loss."""
    source = torch.randn(20, 8)
    target = torch.randn(20, 8)
    coral = CORALLoss()
    loss = coral(source, target)
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_coral_loss_same():
    """Test CORAL with same covariance."""
    X = torch.randn(20, 8)
    source = X[:10]
    target = X[10:]
    coral = CORALLoss()
    loss = coral(source, target)
    # Same covariance should give small loss
    assert loss.item() < 1.0


# ════════════════════════════════════════════════════════════════════════════
# Domain Discriminator
# ════════════════════════════════════════════════════════════════════════════

def test_domain_discriminator():
    """Test DomainDiscriminator forward pass."""
    disc = DomainDiscriminator(128, 3, hidden_dim=64)
    x = torch.randn(8, 128)
    logits = disc(x)
    assert logits.shape == (8, 3)


# ════════════════════════════════════════════════════════════════════════════
# MultiTaskPretrainer
# ═════════════════════════════════════════════════════════════════════════════

def test_multitask_pretrainer_init():
    """Test MultiTaskPretrainer initialization."""
    config = MultiTaskPretrainConfig(
        seq_len=20,
        n_features=5,
        d_model=64,
        use_contrastive=True,
        use_masked_recon=True,
        use_forecast=True,
        use_domain_adaptation=True,
        n_domains=2,
        device="cpu",
        seed=0,
    )
    trainer = MultiTaskPretrainer(config)

    assert hasattr(trainer, "encoder")
    assert hasattr(trainer, "heads")
    assert "contrastive_proj" in trainer.heads
    assert "masked_decoder" in trainer.heads
    assert "forecast" in trainer.heads
    assert trainer.discriminator is not None


def test_multitask_step():
    """Test single training step."""
    config = MultiTaskPretrainConfig(
        seq_len=20,
        n_features=5,
        d_model=64,
        use_contrastive=True,
        use_masked_recon=True,
        use_forecast=True,
        device="cpu",
        seed=0,
        lr=1e-3,
        contrastive_weight=1.0,
        masked_recon_weight=1.0,
        forecast_weight=1.0,
    )
    trainer = MultiTaskPretrainer(config)

    x = torch.randn(8, 20, 5)
    losses = trainer.step(x)

    assert "contrastive" in trainer.history
    assert "masked_recon" in trainer.history
    assert "forecast" in trainer.history
    assert "total" in trainer.history


def test_multitask_with_domain_adaptation():
    """Test MultiTaskPretrainer with domain adaptation."""
    config = MultiTaskPretrainConfig(
        seq_len=20,
        n_features=5,
        d_model=64,
        use_contrastive=True,
        use_domain_adaptation=True,
        n_domains=2,
        da_method="dann",
        device="cpu",
        seed=0,
        lr=1e-3,
        contrastive_weight=1.0,
    )
    trainer = MultiTaskPretrainer(config)

    x = torch.randn(8, 20, 5)
    dom_labels = torch.randint(0, 2, (8,))
    losses = trainer.step(x, dom_labels)

    assert "contrastive" in trainer.history
    assert "domain" in trainer.history


def test_multitask_pretrain():
    """Test full pretraining run."""
    config = MultiTaskPretrainConfig(
        seq_len=15,
        n_features=4,
        d_model=64,
        use_contrastive=True,
        use_masked_recon=True,
        use_forecast=False,
        use_domain_adaptation=False,
        device="cpu",
        seed=0,
        lr=1e-3,
        epochs=3,
        batch_size=16,
        contrastive_weight=1.0,
        masked_recon_weight=1.0,
    )

    rng = np.random.default_rng(0)
    X = np.random.randn(64, 15, 4).astype(np.float32)

    trainer = MultiTaskPretrainer(config)
    history = trainer.pretrain(X, epochs=3, batch_size=16, silent=True)

    assert "loss" in history
    assert len(history["loss"]) == 3
    assert "contrastive" in trainer.history
    assert "masked_recon" in trainer.history


def test_multitask_with_domain_labels():
    """Test pretraining with domain labels."""
    config = MultiTaskPretrainConfig(
        seq_len=15,
        n_features=4,
        d_model=64,
        use_contrastive=True,
        use_domain_adaptation=True,
        n_domains=2,
        da_method="dann",
        device="cpu",
        seed=0,
        lr=1e-3,
        epochs=2,
        batch_size=16,
        contrastive_weight=1.0,
        da_weight=0.5,
    )

    rng = np.random.default_rng(0)
    X = np.random.randn(64, 15, 4).astype(np.float32)
    domain_labels = np.array([0] * 32 + [1] * 32)

    trainer = MultiTaskPretrainer(config)
    history = trainer.pretrain(X, domain_labels=domain_labels, epochs=2, batch_size=16, silent=True)

    assert "domain" in trainer.history
    assert len(history["domain"]) == 2


def test_multitask_gradnorm():
    """Test GradNorm gradient balancing."""
    config = MultiTaskPretrainConfig(
        seq_len=15,
        n_features=4,
        d_model=64,
        use_contrastive=True,
        use_masked_recon=True,
        use_forecast=True,
        device="cpu",
        seed=0,
        lr=1e-3,
        epochs=2,
        batch_size=16,
        use_gradnorm=True,
        gradnorm_alpha=0.5,
        contrastive_weight=1.0,
        masked_recon_weight=1.0,
        forecast_weight=1.0,
    )

    X = np.random.randn(64, 15, 4).astype(np.float32)
    trainer = MultiTaskPretrainer(config)
    history = trainer.pretrain(X, epochs=2, batch_size=16, silent=True)

    assert "gradnorm_weights" in trainer.history
    assert len(trainer.history["gradnorm_weights"]) == 2


def test_diagnostics():
    """Test diagnostics method."""
    config = MultiTaskPretrainConfig(
        seq_len=15,
        n_features=4,
        d_model=64,
        device="cpu",
        seed=0,
    )
    trainer = MultiTaskPretrainer(config)
    X = np.random.randn(10, 15, 4).astype(np.float32)

    diag = trainer.diagnostics(X)
    assert "embed_std" in diag
    assert "collapsed" in diag


def test_save_encoder(tmp_path):
    """Test encoder checkpoint saving."""
    config = MultiTaskPretrainConfig(
        seq_len=15,
        n_features=4,
        d_model=64,
        device="cpu",
        seed=0,
    )
    trainer = MultiTaskPretrainer(config)
    path = tmp_path / "encoder.pt"
    trainer.save_encoder(str(path))
    assert path.exists()


# ════════════════════════════════════════════════════════════════════════════
# Factory Functions
# ════════════════════════════════════════════════════════════════════════════

def test_create_multi_task_pretrainer():
    """Test factory function."""
    X = np.random.randn(50, 20, 5).astype(np.float32)
    trainer = create_multi_task_pretrainer(X, seq_len=20, n_features=5, device="cpu")
    assert isinstance(trainer, MultiTaskPretrainer)


def test_pretrain_multi_task():
    """Test one-shot pretraining function."""
    X = np.random.randn(50, 15, 4).astype(np.float32)
    trainer, history = pretrain_multi_task(
        X,
        seq_len=15,
        n_features=4,
        epochs=2,
        batch_size=16,
        device="cpu",
        silent=True,
    )
    assert isinstance(trainer, MultiTaskPretrainer)
    assert "loss" in history
    assert len(history["loss"]) == 2


# ════════════════════════════════════════════════════════════════════════════
# Domain Adaptation
# ════════════════════════════════════════════════════════════════════════════

def test_adapt_encoder_to_target_dann():
    """Test DANN domain adaptation."""
    encoder = nn.Sequential(
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
    )
    source = np.random.randn(50, 10, 4).astype(np.float32)
    target = np.random.randn(50, 10, 4).astype(np.float32) + 2.0  # shift

    adapted = adapt_encoder_to_target(
        encoder, source, target,
        method="dann", epochs=3, lr=1e-3, device="cpu"
    )
    assert adapted is encoder


def test_adapt_encoder_fine_tune():
    """Test fine-tuning adaptation."""
    encoder = nn.Sequential(
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
    )
    source = np.random.randn(30, 10, 4).astype(np.float32)
    target = np.random.randn(30, 10, 4).astype(np.float32)

    adapted = adapt_encoder_to_target(
        encoder, source, target,
        method="fine_tune", epochs=2, lr=1e-3, device="cpu"
    )
    assert adapted is encoder


# ════════════════════════════════════════════════════════════════════════════
# Integration: Multi-task with domain adaptation + pretraining
# ════════════════════════════════════════════════════════════════════════════

def test_full_pretraining_pipeline():
    """Test full multi-task pretraining with domain adaptation."""
    n = 200
    seq_len = 20
    n_features = 5

    # Source domain
    X_src = np.random.randn(100, 20, 5).astype(np.float32)
    # Target domain (shifted)
    X_tgt = np.random.randn(100, 20, 5).astype(np.float32) + 1.5

    X = np.vstack([X_src, X_tgt])
    domain_labels = np.hstack([np.zeros(100), np.ones(100)]).astype(np.int64)

    config = MultiTaskPretrainConfig(
        seq_len=20,
        n_features=5,
        d_model=64,
        use_contrastive=True,
        use_masked_recon=True,
        use_forecast=False,
        use_domain_adaptation=True,
        n_domains=2,
        da_method="dann",
        device="cpu",
        seed=0,
        epochs=3,
        batch_size=32,
        contrastive_weight=1.0,
        masked_recon_weight=1.0,
        da_weight=0.5,
    )

    trainer, history = pretrain_multi_task(
        X, domain_labels=domain_labels,
        config=config, silent=True,
    )

    assert "contrastive" in trainer.history
    assert "masked_recon" in trainer.history
    assert "domain" in trainer.history
    assert len(history["loss"]) == 3


def test_run_multi_task_pretrain_helper(tmp_path):
    """C3 wiring: _run_multi_task_pretrain trains + saves a usable encoder ckpt."""
    import argparse
    import os

    import torch

    from training.train_gpu import _run_multi_task_pretrain

    windows = np.random.randn(60, 16, 8).astype(np.float32)

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Sequential(
                torch.nn.Linear(8, 16), torch.nn.ReLU(),
                torch.nn.Linear(16, 16),
            )

    model = _Model()
    args = argparse.Namespace(pretrain_epochs=1, pretrain_batch=64, seq_len=16)
    ckpt = str(tmp_path / "contrastive_encoder.pt")
    out = _run_multi_task_pretrain(model, windows, ckpt, 8, args, torch.device("cpu"))
    assert out is not None
    assert os.path.exists(ckpt)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert "model_state" in state
    assert len(state["model_state"]) > 0


def test_run_multi_task_pretrain_graceful_fallback(tmp_path):
    """C3 wiring: invalid pretrain input must not raise — helper returns None."""
    import argparse

    import torch

    from training.train_gpu import _run_multi_task_pretrain

    model = torch.nn.Linear(4, 4)
    args = argparse.Namespace(pretrain_epochs=1, pretrain_batch=4, seq_len=16)
    out = _run_multi_task_pretrain(
        model, [[1.0, 2.0, 3.0, 4.0]],  # not a valid ndarray batch
        str(tmp_path / "x.pt"), 4, args, torch.device("cpu"),
    )
    assert out is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
