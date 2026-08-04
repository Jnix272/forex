"""Smoke: build_model + forward for all 6 supervised architectures (CPU, no training)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODELS = ["haelt", "mamba", "transformer", "tft", "gnn", "expert"]


def _default_args(n_features: int = 80, multitask: bool = True) -> argparse.Namespace:
    """Typical run.yaml dimensions."""
    return argparse.Namespace(
        hidden_size=256,
        d_model=256,
        nhead=8,
        num_layers=3,
        dropout=0.25,
        seq_len=60,
        loss="sharpe_huber",
        multitask=multitask,
        _n_pairs=1,
        _f_per_pair=n_features,
        pair_embed_dim=0,
    )


def test_build_and_forward(n_features: int = 80, multitask: bool = True) -> dict[str, dict]:
    from training.train_gpu import _multitask_head_in, build_model

    args = _default_args(n_features, multitask)
    B, T = 2, args.seq_len
    x = torch.randn(B, T, n_features)
    results: dict[str, dict] = {}

    for name in MODELS:
        row = {"builds": False, "forward_ok": False, "shapes": None, "error": None}
        try:
            model = build_model(name, n_features, args)
            row["builds"] = True
            head_in = _multitask_head_in(name, args, n_features)
            row["head_in"] = head_in
            with torch.no_grad():
                out = model(x)
            if multitask:
                assert isinstance(out, tuple) and len(out) == 3, f"expected 3-tuple, got {type(out)}"
                logits, ret_hat, conf = out
                assert logits.shape == (B, 3), f"logits {logits.shape}"
                assert ret_hat.shape == (B,), f"ret_hat {ret_hat.shape}"
                assert conf.shape == (B,), f"conf {conf.shape}"
                row["shapes"] = (tuple(logits.shape), tuple(ret_hat.shape), tuple(conf.shape))
            else:
                row["shapes"] = tuple(out.shape)
            row["forward_ok"] = True
        except Exception as exc:
            row["error"] = str(exc)
        results[name] = row
    return results


def test_transformer_curriculum_seq_mismatch():
    """iTransformer resamples when curriculum T < build-time seq_len."""
    from models.architectures import iTransformerScalper

    m = iTransformerScalper(input_size=80, seq_len=60, d_model=128, nhead=8, num_layers=2)
    x30 = torch.randn(2, 30, 80)
    out = m(x30)
    assert out.shape == (2,), f"expected (2,), got {out.shape}"
    return {"curriculum_slice_ok": True, "out_shape": tuple(out.shape)}


if __name__ == "__main__":
    print("=== build_model + forward (multitask, n_features=80) ===")
    r = test_build_and_forward(80, multitask=True)
    for name, row in r.items():
        status = "OK" if row["forward_ok"] else "FAIL"
        print(f"  {name:12s} {status:4s}  head_in={row.get('head_in')}  {row.get('shapes') or row.get('error')}")

    failed = [n for n, row in r.items() if not row["forward_ok"]]
    if failed:
        print(f"\nFAILED: {failed}")
        sys.exit(1)

    cur = test_transformer_curriculum_seq_mismatch()
    print("\n=== transformer curriculum seq_len=30 with model built for 60 ===")
    print(f"  {cur}")

    print("\nOK: all model build/forward smoke tests passed")
