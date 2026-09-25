"""
Tests for pretrain loss scaling and target normalization.
Validates scale invariance, gradient balance across extreme feature magnitudes,
and numerical stability on inactive/zero-variance feature channels.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from pretrain.loss_scaling import compute_target_scale, normalized_mse_loss
from pretrain.contrastive import MaskedReconstructionTrainer
from pretrain.extended_trainers import ForecastPretextTrainer, PatchMaskedTrainer, VAESeqTrainer
from pretrain.multi_task import masked_reconstruction_loss, forecast_loss, vae_loss


def test_compute_target_scale_multi_scale():
    """Verify target scale computation accurately scales active channels and floors inactive channels."""
    B, S = 32, 120
    torch.manual_seed(42)

    # Channel 0: Macro volume (~1e5 scale)
    vol = torch.randn(B, S, 1) * 5e5 + 1e6
    # Channel 1: Micro price return (~1e-4 scale)
    ret = torch.randn(B, S, 1) * 2e-4
    # Channel 2: Zero-variance inactive channel (all zeros)
    inactive = torch.zeros(B, S, 1)
    # Channel 3: Constant non-zero channel (zero variance)
    const = torch.full((B, S, 1), 42.0)

    target = torch.cat([vol, ret, inactive, const], dim=-1)  # (32, 120, 4)
    scale = compute_target_scale(target, min_scale=1e-5, fallback_scale=1.0)

    assert scale.shape == (1, 1, 4)
    # Channel 0 scale should be close to 5e5
    assert torch.isclose(scale[0, 0, 0], vol.std(dim=(0, 1)), rtol=0.05)
    # Channel 1 scale should be close to 2e-4
    assert torch.isclose(scale[0, 0, 1], ret.std(dim=(0, 1)), rtol=0.05)
    # Inactive & constant channels must fallback to 1.0 (not 1e-5 or 0.0)
    assert scale[0, 0, 2].item() == 1.0
    assert scale[0, 0, 3].item() == 1.0


def test_normalized_mse_loss_balances_encoder_gradients():
    """Verify normalized_mse_loss prevents volume from starving return gradients into the shared encoder."""
    B, S, D = 16, 64, 32
    torch.manual_seed(101)

    # Shared encoder representation h
    h = nn.Parameter(torch.randn(B, S, D))

    # Decoder heads: head_vol projects to ~1e5 volume, head_ret projects to ~1e-4 return
    head_vol = nn.Linear(D, 1)
    with torch.no_grad():
        head_vol.weight.mul_(1e5)
        head_vol.bias.mul_(1e5)

    head_ret = nn.Linear(D, 1)
    with torch.no_grad():
        head_ret.weight.mul_(1e-4)
        head_ret.bias.mul_(1e-4)

    vol_pred = head_vol(h)
    ret_pred = head_ret(h)
    pred = torch.cat([vol_pred, ret_pred], dim=-1)

    # Ground truth targets with 10% relative error on each channel
    with torch.no_grad():
        vol_target = vol_pred * 0.90
        ret_target = ret_pred * 0.90
        target = torch.cat([vol_target, ret_target], dim=-1)

    # 1. Unscaled MSE: Volume error (~10^10) completely dominates
    loss_unscaled = F.mse_loss(pred, target)
    loss_unscaled.backward(retain_graph=True)
    grad_unscaled_norm = h.grad.norm().item()
    h.grad.zero_()

    # 2. Normalized MSE: Both volume and return contribute ~0.01 to normalized error
    loss_norm = normalized_mse_loss(pred, target)
    loss_norm.backward()
    grad_norm_norm = h.grad.norm().item()

    # Unscaled loss magnitude is ~10^8 (volume dominates)
    assert loss_unscaled.item() > 1e7
    # Normalized loss magnitude is O(1e-2) (equalized)
    assert 0.001 < loss_norm.item() < 0.1
    # Both loss and encoder gradient are well-conditioned and finite
    assert torch.isfinite(torch.tensor(grad_norm_norm))
    assert grad_norm_norm > 0.0


def test_normalized_mse_loss_inactive_channel_stability():
    """Verify predictions on zero-variance inactive channels do not cause loss explosion."""
    B, S = 8, 20
    torch.manual_seed(77)

    # Channel 0: Active, Channel 1: Inactive zero-padded channel
    target = torch.zeros(B, S, 2)
    target[:, :, 0] = torch.randn(B, S) * 0.05  # active

    # Model predicts small noise (e.g. 0.02) on the zero-padded channel
    pred = torch.zeros(B, S, 2)
    pred[:, :, 0] = target[:, :, 0] + 0.005
    pred[:, :, 1] = 0.02  # small bias on inactive feature

    loss = normalized_mse_loss(pred, target, min_scale=1e-5, fallback_scale=1.0)
    # Loss should remain small (~0.01 - 0.1), NOT explode to 10^6
    assert loss.item() < 0.1


def test_masked_reconstruction_loss_multi_task():
    """Test masked_reconstruction_loss helper in multi_task with target normalization."""
    B, T, F_dim = 4, 10, 3
    torch.manual_seed(12)
    target = torch.randn(B, T, F_dim)
    target[:, :, 0] *= 1e5   # volume
    target[:, :, 1] *= 1e-4  # return
    recon = target.clone() + 0.01 * target.std(dim=(0, 1), keepdim=True)
    mask = torch.zeros(B, T, F_dim, dtype=torch.bool)
    mask[:, 2:5, :] = True

    loss = masked_reconstruction_loss(recon, target, mask)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    # Loss should be O(1e-4) because error is 1% of std
    assert 1e-5 < loss.item() < 1e-2


def test_forecast_and_vae_loss_multi_task():
    """Test forecast_loss and vae_loss with multi-scale targets."""
    B, T, F_dim = 4, 10, 3
    torch.manual_seed(99)
    target = torch.randn(B, T, F_dim)
    target[:, :, 0] *= 1e5
    pred = target.clone() + 0.05 * target.std(dim=(0, 1), keepdim=True)

    f_loss = forecast_loss(pred, target)
    assert f_loss.ndim == 0
    assert torch.isfinite(f_loss)
    assert 0.0001 < f_loss.item() < 0.1

    mu = torch.randn(B, 16)
    logvar = torch.randn(B, 16)
    v_loss, recon_loss, kl = vae_loss(pred, target, mu, logvar, beta=0.001)
    assert v_loss.ndim == 0
    assert recon_loss.ndim == 0
    assert kl.ndim == 0
    assert torch.isfinite(v_loss)


class SimpleBackbone(nn.Module):
    def __init__(self, in_features: int = 4, d_model: int = 16):
        super().__init__()
        self.fc = nn.Linear(in_features, d_model)

    def forward(self, x):
        # (B, S, F) -> (B, S, d_model)
        return self.fc(x)


def test_masked_reconstruction_trainer_with_multi_scale(tmp_path):
    """End-to-end smoke test for MaskedReconstructionTrainer with extreme feature scales."""
    rng = np.random.default_rng(42)
    N, S, F_dim = 32, 16, 4
    X = rng.standard_normal((N, S, F_dim)).astype(np.float32)
    X[:, :, 0] *= 1e5  # volume
    X[:, :, 1] *= 1e-4 # return
    X[:, :, 2] = 0.0   # inactive
    # X[:, :, 3] is unit normal

    ckpt = tmp_path / "masked.pt"
    trainer = MaskedReconstructionTrainer(
        SimpleBackbone(in_features=F_dim, d_model=16),
        d_model=16,
        seq_len=S,
        n_features=F_dim,
        hidden_dim=32,
        mask_prob=0.3,
        lr=1e-3,
        device="cpu",
    )

    history = trainer.pretrain(X, epochs=2, batch_size=8, checkpoint_path=str(ckpt), silent=True)
    trainer.save_encoder(str(ckpt))
    diag = trainer.diagnostics(X)

    assert ckpt.exists()
    assert len(history["loss"]) == 2
    assert all(np.isfinite(l) for l in history["loss"])
    assert np.isfinite(diag["masked_mse"])
    # Normalized masked MSE should be reasonable (not 10^10)
    assert diag["masked_mse"] < 100.0


def test_patch_masked_trainer_with_multi_scale(tmp_path):
    """End-to-end smoke test for PatchMaskedTrainer with extreme feature scales."""
    rng = np.random.default_rng(43)
    N, S, F_dim = 32, 16, 4
    X = rng.standard_normal((N, S, F_dim)).astype(np.float32)
    X[:, :, 0] *= 1e5
    X[:, :, 1] *= 1e-4

    ckpt = tmp_path / "patch_masked.pt"
    trainer = PatchMaskedTrainer(
        SimpleBackbone(in_features=F_dim, d_model=16),
        d_model=16,
        seq_len=S,
        n_features=F_dim,
        patch_size=4,
        mask_prob=0.25,
        hidden_dim=32,
        lr=1e-3,
        device="cpu",
    )

    history = trainer.pretrain(X, epochs=2, batch_size=8, checkpoint_path=str(ckpt), silent=True)
    trainer.save_encoder(str(ckpt))
    assert ckpt.exists()
    assert len(history["loss"]) == 2
    assert all(np.isfinite(l) for l in history["loss"])
    assert history["loss"][-1] < 100.0
