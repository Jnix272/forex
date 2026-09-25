import pytest
import torch
import argparse
from models.architectures import build_model, PatchTSTScalper, GLMBaseline, MultiTaskWrapper
from config.models import architecture_config


def test_patchtst_standalone_1d_regression():
    """Verify PatchTST outputs (B,) 1D scalar for CPAR regression."""
    B, T, F_in = 2, 120, 584
    x = torch.randn(B, T, F_in)
    model = PatchTSTScalper(input_size=F_in, seq_len=T, d_model=256, num_classes=1)
    
    assert model.num_classes == 1
    assert model.d_model == 256
    assert model.seq_len == 120
    
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (B,), f"Expected shape ({B},), got {out.shape}"


def test_patchtst_multitask_wrapper():
    """Verify MultiTaskWrapper cleanly wraps PatchTST and pools representations."""
    B, T, F_in = 2, 120, 584
    x = torch.randn(B, T, F_in)
    args = argparse.Namespace(
        hidden_size=256,
        d_model=256,
        nhead=8,
        num_layers=3,
        dropout=0.1,
        seq_len=120,
        loss="huber",
        multitask=True,
        num_classes=1,
        _n_pairs=1,
        _f_per_pair=F_in,
        pair_embed_dim=0,
    )
    model = build_model("patchtst", F_in, args)
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, tuple) and len(out) == 3
    dir_pred, ret_hat, conf = out
    assert ret_hat.shape == (B,)
    assert conf.shape == (B,)


def test_glm_baseline_1d_regression():
    """Verify GLMBaseline defaults to num_classes=1 for CPAR regression."""
    B, T, F_in = 2, 16, 64
    x = torch.randn(B, T, F_in)
    model = GLMBaseline(input_size=F_in, num_classes=1, seq_len=T)
    assert model.num_classes == 1
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (B,), f"Expected shape ({B},), got {out.shape}"


def test_patchtst_config_alignment():
    """Verify PatchTST config profile aligns with run.yaml and 120-bar dataset."""
    cfg = architecture_config("patchtst")
    assert cfg["d_model"] == 256
    assert cfg["seq_len"] == 120
    assert cfg["num_classes"] == 1
    assert cfg["num_layers"] == 3
