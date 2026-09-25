"""Unit tests for TemporalAttentionPooling and EnsembleMetaLearner temporal attention."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from models.ensemble import EnsembleMetaLearner, TemporalAttentionPooling


class DummyBaseModel(nn.Module):
    def __init__(self, in_features: int = 584):
        super().__init__()
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, F) -> use last bar
        return self.fc(x[:, -1, :]).squeeze(-1)


def test_temporal_attention_pooling_shapes_and_weights():
    pool = TemporalAttentionPooling(hidden=64, out_dim=32)
    B, S, F = 4, 120, 584
    x = torch.randn(B, S, F)

    out, weights = pool(x, return_attention=True)
    assert out.shape == (B, 32), f"Expected shape ({B}, 32), got {out.shape}"
    assert weights.shape == (B, S), f"Expected shape ({B}, {S}), got {weights.shape}"
    assert torch.allclose(weights.sum(dim=-1), torch.ones(B), atol=1e-5), "Attention weights must sum to 1.0"
    assert (weights >= 0.0).all(), "Attention weights must be non-negative"


def test_temporal_attention_gradient_flow_all_timesteps():
    pool = TemporalAttentionPooling(hidden=64, out_dim=32)
    B, S, F = 2, 60, 32
    x = torch.randn(B, S, F, requires_grad=True)

    out = pool(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    # Check that gradients flow to the earliest timesteps (e.g. t=0) as well as the latest
    early_grad_norm = x.grad[:, 0, :].norm().item()
    late_grad_norm = x.grad[:, -1, :].norm().item()
    assert early_grad_norm > 1e-7, "Gradients should flow back to early timesteps"
    assert late_grad_norm > 1e-7, "Gradients should flow to latest timesteps"


def test_temporal_attention_pooling_2d_fallback():
    pool = TemporalAttentionPooling(hidden=64, out_dim=32)
    B, F = 4, 584
    x = torch.randn(B, F)

    out = pool(x)
    assert out.shape == (B, 32)


def test_ensemble_meta_learner_with_temporal_attention():
    B, S, F = 4, 120, 584
    base1 = DummyBaseModel(in_features=F)
    base2 = DummyBaseModel(in_features=F)

    ensemble = EnsembleMetaLearner(
        base_models=[base1, base2],
        context_dim=32,
        hidden=64,
        base_names=["base1", "base2"],
    )

    x = torch.randn(B, S, F, requires_grad=True)
    output, weights = ensemble(x)

    assert output.shape == (B,), f"Expected output shape ({B},), got {output.shape}"
    assert weights.shape == (B, 2), f"Expected weights shape ({B}, 2), got {weights.shape}"
    assert torch.allclose(weights.sum(dim=1), torch.ones(B), atol=1e-5), "Model weights must sum to 1.0"

    # Backward pass test
    loss = output.sum()
    loss.backward()
    assert ensemble.meta[0].weight.grad is not None
    assert ensemble.context_enc.query.grad is not None


def test_ensemble_unfreeze_base_heads_and_diversity_loss():
    """Verify that unfreeze_base_heads propagates diversity gradients to base model projection heads."""
    B, S, F = 8, 30, 16
    base1 = DummyBaseModel(in_features=F)
    base2 = DummyBaseModel(in_features=F)

    ensemble = EnsembleMetaLearner(
        base_models=[base1, base2],
        context_dim=16,
        hidden=32,
        base_names=["base1", "base2"],
    )

    unfrozen = ensemble.unfreeze_base_heads(True)
    assert len(unfrozen) == 4, f"Expected 4 unfrozen head parameters (weight+bias x 2), got {len(unfrozen)}"
    assert ensemble._base_heads_unfrozen is True

    x = torch.randn(B, S, F)
    output, weights, preds = ensemble.forward_with_preds(x)
    assert preds.shape == (B, 2)

    # Compute correlation diversity loss
    corr = ensemble.diversity_loss(preds)
    loss = output.sum() + 0.5 * corr
    loss.backward()

    # Check that base model heads received gradients!
    assert base1.fc.weight.grad is not None, "Base1 head must receive diversity gradients"
    assert base2.fc.weight.grad is not None, "Base2 head must receive diversity gradients"

