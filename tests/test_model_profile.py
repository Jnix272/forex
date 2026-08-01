"""Unit tests for per-architecture profile merging."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.train_gpu import _apply_model_profile


def _approx(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(a - b) <= tol


def _base_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        model="haelt",
        lr=2e-5,
        dropout=0.25,
        hidden_size=256,
        d_model=256,
        nhead=8,
        num_layers=3,
        seq_len=60,
        weight_decay=1e-4,
        batch_size=256,
        model_profile=True,
        _cli_profile_overrides=frozenset(),
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_apply_model_profile_mamba_vs_haelt_lr():
    mamba_args = _base_args(model="mamba")
    haelt_args = _base_args(model="haelt")

    _apply_model_profile(mamba_args, "mamba", enabled=True)
    _apply_model_profile(haelt_args, "haelt", enabled=True)

    assert _approx(mamba_args.lr, 1e-4)
    assert _approx(haelt_args.lr, 3e-4)
    assert mamba_args.lr != haelt_args.lr
    assert _approx(mamba_args.dropout, 0.1)
    assert _approx(haelt_args.dropout, 0.25)


def test_cli_override_preserves_explicit_lr():
    args = _base_args(lr=9e-5, _cli_profile_overrides=frozenset({"lr"}))
    _apply_model_profile(args, "mamba", enabled=True)
    assert _approx(args.lr, 9e-5)


def test_profile_disabled_is_noop():
    args = _base_args(lr=2e-5)
    _apply_model_profile(args, "mamba", enabled=False)
    assert _approx(args.lr, 2e-5)
    assert not getattr(args, "_profile_applied", False)


if __name__ == "__main__":
    test_apply_model_profile_mamba_vs_haelt_lr()
    test_cli_override_preserves_explicit_lr()
    test_profile_disabled_is_noop()
    print("OK: test_model_profile")
