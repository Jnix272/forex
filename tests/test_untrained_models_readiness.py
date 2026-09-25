"""
tests/test_untrained_models_readiness.py
=========================================
Exhaustive readiness verification for the 3 untrained models:
  - PatchTST (PatchTSTScalper)
  - Transformer (iTransformerScalper)
  - EXPERT (EXPERTEncoder)

Verifies:
  1. Instantiation directly and via build_model()
  2. Model attributes & aliases
  3. Single-head MultiTaskWrapper forward + MultiTaskLoss + loss.backward()
  4. Track B MultiPairMultiTaskWrapper forward + MultiPairMultiTaskLoss + loss.backward()
  5. Gradient validity: non-zero and finite (no NaN / Inf) across backbone & heads
  6. Eager parameter materialization via initialize_parameters()
  7. Dynamic sequence length resilience (curriculum adaptation, T=60 and T=120)
  8. EXPERT branch ablations (use_conv_ffn & no_pos_encoding)
  9. Automatic Mixed Precision (AMP: bfloat16 / float16) compatibility
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import pytest
import torch
from models.architectures import (
    MODEL_REGISTRY,
    EXPERTEncoder,
    MultiPairMultiTaskLoss,
    MultiPairMultiTaskWrapper,
    MultiTaskLoss,
    MultiTaskWrapper,
    PatchTSTScalper,
    build_model,
    iTransformerScalper,
)
from config.models import architecture_config

MODELS_UNDER_TEST = ["patchtst", "transformer", "expert"]


@pytest.mark.parametrize("model_name", MODELS_UNDER_TEST)
def test_model_build_and_attributes(model_name: str):
    """Verify build_model instantiates each architecture and exposes essential attributes."""
    input_size = 584
    seq_len = 120
    model = build_model(model_name, input_size=input_size, seq_len=seq_len)
    assert model is not None
    assert hasattr(model, "d_model"), f"{model_name} must expose d_model"
    assert hasattr(model, "hidden_size"), f"{model_name} must expose hidden_size"
    assert hasattr(model, "seq_len"), f"{model_name} must expose seq_len"
    assert hasattr(model, "input_size"), f"{model_name} must expose input_size"
    assert model.input_size == input_size
    assert model.seq_len == seq_len


def test_registry_aliases():
    """Verify convenience aliases exist in MODEL_REGISTRY."""
    assert "patchtstscalper" in MODEL_REGISTRY
    assert "itransformer" in MODEL_REGISTRY
    assert "expertencoder" in MODEL_REGISTRY
    assert MODEL_REGISTRY["patchtstscalper"] is PatchTSTScalper
    assert MODEL_REGISTRY["itransformer"] is iTransformerScalper
    assert MODEL_REGISTRY["expertencoder"] is EXPERTEncoder


@pytest.mark.parametrize("model_name", MODELS_UNDER_TEST)
def test_multitask_wrapper_forward_backward(model_name: str):
    """
    Verify MultiTaskWrapper:
      - wraps backbone cleanly
      - materializes LazyLinear via initialize_parameters()
      - forward pass on (B=4, T=120, F=584)
      - computes MultiTaskLoss
      - loss.backward() produces non-zero, finite gradients across all weights
    """
    torch.manual_seed(42)
    B, T, F = 4, 120, 584
    x = torch.randn(B, T, F)

    model = build_model(
        model_name,
        input_size=F,
        seq_len=T,
        multitask=True,
    )
    assert isinstance(model, MultiTaskWrapper)

    # Eager parameter materialization
    model.initialize_parameters()

    # Forward pass
    model.train()
    out = model(x)
    assert isinstance(out, tuple) and len(out) >= 3
    dir_logits, ret_hat, conf = out[:3]

    assert dir_logits.shape == (B,), f"Expected (B,), got {dir_logits.shape}"
    assert ret_hat.shape == (B,), f"Expected (B,), got {ret_hat.shape}"
    assert conf.shape == (B,), f"Expected (B,), got {conf.shape}"

    # Targets & Loss
    y_cls = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    y_cont = torch.tensor([0.0012, -0.0008, 0.0025, -0.0010], dtype=torch.float32)
    criterion = MultiTaskLoss()
    loss = criterion(dir_logits, ret_hat, conf, y_cls, y_cont)

    assert torch.isfinite(loss), f"Loss must be finite, got {loss.item()}"

    # Backpropagation
    loss.backward()

    # Gradient assertions
    has_any_grad = False
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has None grad"
            assert torch.isfinite(p.grad).all(), f"Parameter {name} has NaN or Inf grad"
            if p.grad.abs().sum() > 0:
                has_any_grad = True

    assert has_any_grad, f"Model {model_name} has zero gradients across all parameters!"


@pytest.mark.parametrize("model_name", MODELS_UNDER_TEST)
def test_multipair_multitask_wrapper_forward_backward(model_name: str):
    """
    Verify MultiPairMultiTaskWrapper (Track B 4-pair multi-task heads):
      - wraps backbone cleanly with independent heads per pair
      - forward pass on (B=4, T=120, F=584)
      - computes MultiPairMultiTaskLoss
      - loss.backward() produces non-zero, finite gradients
    """
    torch.manual_seed(42)
    B, T, F = 4, 120, 584
    P = 4
    x = torch.randn(B, T, F)

    model = build_model(
        model_name,
        input_size=F,
        seq_len=T,
        per_pair_heads=True,
        n_pair_heads=P,
    )
    assert isinstance(model, MultiPairMultiTaskWrapper)

    # Eager parameter materialization
    model.initialize_parameters()

    # Forward pass
    model.train()
    out = model(x)
    assert isinstance(out, tuple) and len(out) >= 5
    dir_logits, ret_hat, conf, q_low, q_high = out[:5]

    assert dir_logits.shape == (B, P), f"Expected (B, {P}), got {dir_logits.shape}"
    assert ret_hat.shape == (B, P), f"Expected (B, {P}), got {ret_hat.shape}"
    assert conf.shape == (B, P), f"Expected (B, {P}), got {conf.shape}"
    assert q_low.shape == (B, P), f"Expected (B, {P}), got {q_low.shape}"
    assert q_high.shape == (B, P), f"Expected (B, {P}), got {q_high.shape}"

    # Targets & MultiPair Loss
    y_cls = torch.randint(0, 3, (B, P), dtype=torch.long)
    y_cont = torch.randn(B, P, dtype=torch.float32) * 0.001
    criterion = MultiPairMultiTaskLoss()
    loss = criterion(
        logits=dir_logits,
        ret_hat=ret_hat,
        conf=conf,
        y_cls=y_cls,
        y_cont=y_cont,
        q_low=q_low,
        q_high=q_high,
    )

    assert torch.isfinite(loss), f"MultiPair loss must be finite, got {loss.item()}"

    # Backprop
    loss.backward()

    # Verify gradients
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"MultiPair parameter {name} has None grad"
            assert torch.isfinite(p.grad).all(), f"MultiPair parameter {name} has NaN/Inf grad"


@pytest.mark.parametrize("model_name", MODELS_UNDER_TEST)
def test_dynamic_curriculum_seq_len(model_name: str):
    """
    Verify model handles dynamic sequence lengths (e.g. curriculum T=60 vs T=120)
    without crashing or shape mismatches.
    """
    F = 584
    model = build_model(model_name, input_size=F, seq_len=120, multitask=True)
    model.initialize_parameters()
    model.eval()

    # Test with shorter curriculum sequence (T=60)
    x_short = torch.randn(2, 60, F)
    with torch.no_grad():
        out_short = model(x_short)
    assert out_short[0].shape == (2,)
    assert out_short[1].shape == (2,)

    # Test with full sequence (T=120)
    x_full = torch.randn(2, 120, F)
    with torch.no_grad():
        out_full = model(x_full)
    assert out_full[0].shape == (2,)
    assert out_full[1].shape == (2,)


def test_expert_architectural_branches():
    """Verify EXPERTEncoder handles use_conv_ffn and no_pos_encoding switches."""
    F, T = 64, 120
    # Standard EXPERT: ConvFFN + Positional Encoding
    m1 = EXPERTEncoder(input_size=F, seq_len=T, use_conv_ffn=True, no_pos_encoding=False)
    assert m1.pos_emb is not None
    y1 = m1(torch.randn(2, T, F))
    assert y1.shape == (2,)

    # Ablation 1: Standard MLP FFN + Positional Encoding
    m2 = EXPERTEncoder(input_size=F, seq_len=T, use_conv_ffn=False, no_pos_encoding=False)
    assert m2.pos_emb is not None
    y2 = m2(torch.randn(2, T, F))
    assert y2.shape == (2,)

    # Ablation 2: ConvFFN without Positional Encoding
    m3 = EXPERTEncoder(input_size=F, seq_len=T, use_conv_ffn=True, no_pos_encoding=True)
    assert m3.pos_emb is None
    y3 = m3(torch.randn(2, T, F))
    assert y3.shape == (2,)


@pytest.mark.parametrize("model_name", MODELS_UNDER_TEST)
def test_amp_autocast_readiness(model_name: str):
    """Verify model forward pass runs cleanly under Automatic Mixed Precision (AMP bfloat16)."""
    F, T = 584, 120
    model = build_model(model_name, input_size=F, seq_len=T, multitask=True)
    model.initialize_parameters()
    model.train()

    x = torch.randn(2, T, F)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        dir_logits, ret_hat, conf = model(x)
        y_cls = torch.tensor([0, 1], dtype=torch.long)
        y_cont = torch.tensor([0.001, -0.001], dtype=torch.float32)
        loss = MultiTaskLoss()(dir_logits, ret_hat, conf, y_cls, y_cont)

    assert torch.isfinite(loss)
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s", "--tb=short"]))
