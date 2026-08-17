"""Tests for the dataset builder leakage-prevention checks that were
previously dead code (check_future_leak was always passed None, and
check_label_contamination / assert_fold_isolation were never called).

These tests load the helper functions directly from the dataset_builder
source via AST extraction (so we don't trigger the full training/CUDA
import chain) and verify end-to-end that they now do real work.
"""

from __future__ import annotations

import ast
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure the project root is on sys.path so `data.*` and `features.*` import.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_helpers_from_source():
    """AST-extract the two helper functions we just added and exec them
    in a minimal namespace. This avoids importing the full dataset_builder
    module (which pulls in polars, torch, etc.) and lets the tests run in
    any Python environment with just numpy/pandas/zarr/numcodecs."""
    src = (_REPO_ROOT / "training" / "dataset_builder.py").read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(src)
    wanted = {"_leak_check_features_sample", "_label_contamination_check"}
    extracted: list[ast.FunctionDef] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            extracted.append(node)
    found = {n.name for n in extracted}
    missing = wanted - found
    assert not missing, f"Missing helpers in source: {missing}"

    # Build a minimal module with just the symbols our helpers need.
    module = ast.Module(body=extracted, type_ignores=[])
    ns: dict = {
        "Path": Path,
        "np": np,
        "DatasetManifest": __import__("data.dataset_manifest", fromlist=["DatasetManifest"]).DatasetManifest,
    }
    exec(compile(module, "<extracted>", "exec"), ns)
    return ns["_leak_check_features_sample"], ns["_label_contamination_check"]


_leak_check_features_sample, _label_contamination_check = _load_helpers_from_source()


def _build_minimal_zarr(
    cache_dir: Path, n_samples: int = 256, seq_len: int = 16, n_features: int = 8, leak_factor: float | None = None
):
    """Write a tiny Zarr cache shaped like the real one.

    If ``leak_factor`` is set, we plant a known leak: feature column 0
    equals the forward return (y) scaled by ``leak_factor``. The future-leak
    check should then flag column 0 with |corr| > 0.3.

    Works with both zarr v2 (create_dataset) and v3 (create_array).
    """
    import zarr
    from numcodecs import Blosc

    zarr_v3 = int(getattr(zarr, "__version__", "0").split(".")[0]) >= 3

    rng = np.random.default_rng(42)
    X = rng.normal(0.0, 1.0, size=(n_samples, seq_len, n_features)).astype(np.float32)
    y = rng.normal(0.0, 0.01, size=n_samples).astype(np.float32)
    if leak_factor is not None:
        # Plant a leak: every feature window's last timestep sees the (future)
        # return y, scaled so the correlation is well above 0.3.
        X[:, -1, 0] = leak_factor * y
    if zarr_v3:
        # zarr v3 uses BytesBytesCodec; numcodecs.Blosc only works with v2.
        z = zarr.open_group(str(cache_dir), mode="w")
        z.create_array("X", shape=X.shape, chunks=(64, seq_len, n_features), dtype="float32", overwrite=True)
        z["X"][:] = X
        z.create_array("y", shape=(n_samples,), chunks=(64,), dtype="float32", overwrite=True)
        z["y"][:] = y
        z["X"].attrs["columns"] = [f"f{i}" for i in range(n_features)]
    else:
        compressor = Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE)
        z = zarr.open(str(cache_dir), mode="w")
        z.create_dataset(
            "X",
            shape=X.shape,
            chunks=(64, seq_len, n_features),
            dtype="float32",
            compressor=compressor,
            overwrite=True,
        )
        z["X"][:] = X
        z.create_dataset(
            "y",
            shape=(n_samples,),
            chunks=(64,),
            dtype="float32",
            compressor=compressor,
            overwrite=True,
        )
        z["y"][:] = y
        z["X"].attrs["columns"] = [f"f{i}" for i in range(n_features)]
    return cache_dir


class _Args:
    """Minimal argparse-like namespace for the helpers."""

    seq_len = 16
    execution_delay_bars = 1
    lookahead_bars = 1
    bar_freq = "5min"
    historical_news_mode = "calendar"
    label_method = "rl_reward"
    data_end = "2025-01-01"
    real_data_window_days = 7


def test_leak_check_features_sample_returns_2d_with_names():
    """_leak_check_features_sample should produce a 2D array of last-step
    features plus the feature-name list, not None."""
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "test.zarr"
        _build_minimal_zarr(cache)
        out = _leak_check_features_sample(cache, max_sample=64)
        assert out is not None, "Expected a tuple, got None"
        X_2d, names = out
        assert X_2d.ndim == 2, f"Expected 2D, got {X_2d.ndim}D"
        assert X_2d.shape[0] == 64
        assert X_2d.shape[1] == 8
        assert names == [f"f{i}" for i in range(8)]
        print("OK: leak check feature sampler returns 2D array + names")


def test_check_future_leak_flags_planted_leak():
    """End-to-end: a planted leak (X[:, -1, 0] = factor * y) should be
    detected by check_future_leak when the features are now actually
    passed through (not None)."""
    from data.dataset_manifest import DatasetManifest

    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "test.zarr"
        _build_minimal_zarr(cache, leak_factor=10.0)
        feats = _leak_check_features_sample(cache, max_sample=256)
        assert feats is not None
        X_2d, names = feats
        feat_df = pd.DataFrame(X_2d, columns=names)
        import zarr

        y = np.asarray(
            (
                zarr.open_group(str(cache))
                if int(zarr.__version__.split(".")[0]) >= 3
                else zarr.open(str(cache), mode="r")
            )["y"][:],
            dtype=np.float32,
        )
        flagged = DatasetManifest.check_future_leak(feat_df, y.tolist(), max_abs_corr=0.30)
        assert len(flagged) >= 1, "Planted leak was not detected"
        assert flagged[0]["feature"] == "f0", f"Wrong column flagged: {flagged[0]}"
        assert abs(flagged[0]["corr"]) > 0.3
        print(f"OK: planted leak detected, |corr|={abs(flagged[0]['corr']):.3f}")


def test_check_future_leak_returns_empty_for_clean_data():
    """When no features are correlated with forward returns, the result
    should be an empty list (not a silent skip)."""
    from data.dataset_manifest import DatasetManifest

    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "test.zarr"
        _build_minimal_zarr(cache)  # no leak
        feats = _leak_check_features_sample(cache, max_sample=256)
        X_2d, names = feats
        feat_df = pd.DataFrame(X_2d, columns=names)
        import zarr

        y = np.asarray(
            (
                zarr.open_group(str(cache))
                if int(zarr.__version__.split(".")[0]) >= 3
                else zarr.open(str(cache), mode="r")
            )["y"][:],
            dtype=np.float32,
        )
        flagged = DatasetManifest.check_future_leak(feat_df, y.tolist(), max_abs_corr=0.30)
        assert flagged == [], f"Expected no flags, got {flagged}"
        print("OK: clean data produces no flags")


def test_label_contamination_check_passes_chronological_cache():
    """A chronologically-ordered cache where row index represents time
    should pass the label-contamination check."""
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "test.zarr"
        _build_minimal_zarr(cache)
        result = _label_contamination_check(cache, _Args())
        assert result["ok"] is True
        assert result["violations"] == 0
        assert result["total_checked"] > 0
        print(f"OK: {result['total_checked']} samples, all pass contamination check")


def test_label_contamination_check_detects_violation():
    """If we lie about seq_len and pretend the label is computed from the
    SAME bar as the feature (seq_len=0, delay=1 â†' label_ts == feat_ts),
    the contamination check should fail because feat_ts == label_ts is
    not strictly before."""
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "test.zarr"
        _build_minimal_zarr(cache)
        # seq_len=0 means label_ts = feat_ts + (-1) + 1 = feat_ts â†' violation.
        result = _label_contamination_check(cache, _Args(), seq_len=0)
        assert result["ok"] is False, f"Expected violation, got {result}"
        assert result["violations"] > 0
        print(f"OK: contamination check detected {result['violations']} violations")


def test_assert_fold_isolation_runs_on_normal_split():
    """A normal embargo split should pass the fold isolation check."""
    from features.lookahead_guard import assert_fold_isolation

    train = np.arange(0, 800, dtype=np.int64)
    val = np.arange(820, 1000, dtype=np.int64)  # 20-bar embargo
    # Should not raise
    assert_fold_isolation(train, val, embargo_bars=20)
    print("OK: assert_fold_isolation accepts a valid 20-bar embargo split")


def test_assert_fold_isolation_raises_on_overlap():
    """If val starts before train ends, the check should raise."""
    from features.lookahead_guard import (
        LookaheadViolation,
        assert_fold_isolation,
    )

    train = np.arange(0, 800, dtype=np.int64)
    val = np.arange(700, 900, dtype=np.int64)  # OVERLAPS train
    try:
        assert_fold_isolation(train, val, embargo_bars=20)
    except LookaheadViolation as e:
        print(f"OK: overlap detected: {str(e)[:80]}...")
        return
    raise AssertionError("Expected LookaheadViolation for overlapping fold")


if __name__ == "__main__":
    test_leak_check_features_sample_returns_2d_with_names()
    test_check_future_leak_flags_planted_leak()
    test_check_future_leak_returns_empty_for_clean_data()
    test_label_contamination_check_passes_chronological_cache()
    test_label_contamination_check_detects_violation()
    test_assert_fold_isolation_runs_on_normal_split()
    test_assert_fold_isolation_raises_on_overlap()
    print("\nAll audit fixes verified end-to-end.")
