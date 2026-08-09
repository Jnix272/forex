"""Tests for ``training.cache_integrity._validate_dataset_builder_reader_contract``.

This validator was added in the 2026-08-06 dataset-builder / reader-contract
audit and inspects:

  1. ``multitask`` + ``rl_reward`` requires a real ``pq`` sidecar/array.
  2. ``pq`` value range is ``[0, 1]`` (BCE confidence head is ill-defined
     outside it). The previous reader-side ``min(1, |y|)`` fallback papered
     over bad writers; the new reader honours the real array, so the
     mismatch becomes visible and must fail-stop.
  3. ``y_cls`` ∈ {-1, 0, +1} — a bug that writes raw reward floats into
     ``y_cls`` would silently corrupt classification.
  4. Zarr row chunk size is at least 64 — ``ZarrStreamDataset`` uses it for
     streaming block partitioning; ``chunks[0] == 1`` reintroduces the
     per-row decompress cost it was designed to avoid.
  5. ``scaler.scale_`` shape matches ``X.shape[-1]``.
  6. ``diff`` (curriculum) is ``uint8`` with values in {0, 1, 2}.

Tests build tiny real zarr caches that trigger each check.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from training.cache_integrity import (
    _validate_dataset_builder_reader_contract,
)
from training.gpu_cache_io import (
    ZARR,
    ZARR_FEATURE_DTYPE,
    ZARR_LABEL_DTYPE,
    _zarr_create,
    _zarr_open_group,
    make_training_zarr_compressor,
)


pytestmark = pytest.mark.skipif(not ZARR, reason="zarr not installed")


def _build_cache(
    tmp_path: Path,
    *,
    n_rows: int = 200,
    seq_len: int = 4,
    n_features: int = 3,
    chunk_rows: int = 64,
    with_pq: bool = True,
    pq_array: np.ndarray | None = None,
    with_y_cls: bool = True,
    y_cls_array: np.ndarray | None = None,
    with_diff: bool = True,
    diff_array: np.ndarray | None = None,
    with_scaler: bool = False,
    scaler_scale: np.ndarray | None = None,
    scaler_mean: np.ndarray | None = None,
):
    """Materialise a small zarr cache for the validator to inspect."""
    Comp = make_training_zarr_compressor("lz4", 1) if ZARR else None
    p = tmp_path / "cache.zarr"
    z = _zarr_open_group(str(p), mode="w")

    rs = np.random.RandomState(0)
    X = rs.randn(n_rows, seq_len, n_features).astype(np.float16)
    y = rs.randn(n_rows).astype(np.float32)

    if with_pq:
        pq = (pq_array if pq_array is not None
              else rs.uniform(0.0, 1.0, n_rows).astype(np.float32))
    else:
        pq = None
    if with_y_cls:
        y_cls = (y_cls_array if y_cls_array is not None
                 else rs.choice([-1.0, 0.0, 1.0], n_rows).astype(np.float32))
    else:
        y_cls = None
    if with_diff:
        diff = (diff_array if diff_array is not None
                else rs.choice([0, 1, 2], n_rows).astype(np.uint8))
    else:
        diff = None

    c0 = (chunk_rows,) + X.shape[1:]
    _zarr_create(z, "X", shape=X.shape, chunks=c0,
                 dtype=ZARR_FEATURE_DTYPE, compressor=Comp)
    _zarr_create(z, "y", shape=y.shape, chunks=(chunk_rows,),
                 dtype=ZARR_LABEL_DTYPE, compressor=Comp)
    z["X"][:] = X
    z["y"][:] = y
    if y_cls is not None:
        _zarr_create(z, "y_cls", shape=y_cls.shape, chunks=(chunk_rows,),
                     dtype=ZARR_LABEL_DTYPE, compressor=Comp)
        z["y_cls"][:] = y_cls
    if pq is not None:
        _zarr_create(z, "pq", shape=pq.shape, chunks=(chunk_rows,),
                     dtype=ZARR_LABEL_DTYPE, compressor=Comp)
        z["pq"][:] = pq
    if diff is not None:
        _zarr_create(z, "diff", shape=diff.shape, chunks=(chunk_rows,),
                     dtype="uint8", compressor=Comp)
        z["diff"][:] = diff

    if with_scaler:
        from training.gpu_cache_io import _scaler_npz_path
        sp = _scaler_npz_path(Path(str(p)))
        scale = scaler_scale if scaler_scale is not None else np.ones(n_features, dtype=np.float64)
        mean = scaler_mean if scaler_mean is not None else np.zeros(n_features, dtype=np.float64)
        np.savez(str(sp), scale_=scale, mean_=mean)

    return str(p)


def _args(**kw) -> SimpleNamespace:
    base = dict(
        multitask=False,
        label_method="sign_return",
        ignore_manifest=True,
    )
    base.update(kw)
    return SimpleNamespace(**base)


# ─────────────────────────────────────────────────────────────────────────────
# Happy path: a well-formed cache passes cleanly
# ─────────────────────────────────────────────────────────────────────────────

def test_clean_cache_passes(tmp_path: Path):
    cache = _build_cache(tmp_path, n_rows=200, chunk_rows=64,
                        with_pq=True, with_y_cls=True, with_diff=True,
                        with_scaler=True)
    ok, problems = _validate_dataset_builder_reader_contract(
        cache, _args(multitask=True, label_method="rl_reward"),
    )
    assert ok, problems
    assert problems == []


# ─────────────────────────────────────────────────────────────────────────────
# Check (1) + (2) — pq presence + range
# ─────────────────────────────────────────────────────────────────────────────

def test_multitask_rl_reward_missing_pq_warns(tmp_path: Path):
    """When --multitask + --label-method=rl_reward is on but the cache has
    no ``pq`` array, the validator surfaces a soft warning (the reader falls
    back to ``pq=1.0`` behind the user's back; this is the hint to rebuild)."""
    cache = _build_cache(tmp_path, with_pq=False)
    ok, problems = _validate_dataset_builder_reader_contract(
        cache, _args(multitask=True, label_method="rl_reward"),
    )
    # Soft warning (not a hard fail): reader can still operate safely.
    assert any("no `pq` array" in p for p in problems)


def test_pq_outside_unit_range_is_hard_error(tmp_path: Path):
    """pq values outside [0, 1] make the BCE confidence head ill-defined —
    fail-stop so the bad writer is visible and gets rebuilt."""
    n = 200
    bad_pq = np.linspace(-0.5, 1.7, n).astype(np.float32)  # contains -0.5 and 1.7
    cache = _build_cache(tmp_path, n_rows=n, pq_array=bad_pq)
    ok, problems = _validate_dataset_builder_reader_contract(
        cache, _args(multitask=True, label_method="rl_reward"),
    )
    assert not ok
    assert any("pq range" in p for p in problems)


def test_pq_in_unit_range_passes(tmp_path: Path):
    n = 200
    ok_pq = np.linspace(0.0, 1.0, n).astype(np.float32)
    cache = _build_cache(tmp_path, n_rows=n, pq_array=ok_pq)
    ok, problems = _validate_dataset_builder_reader_contract(
        cache, _args(multitask=True, label_method="rl_reward"),
    )
    assert ok, problems


# ─────────────────────────────────────────────────────────────────────────────
# Check (3) — y_cls ∈ {-1, 0, +1}
# ─────────────────────────────────────────────────────────────────────────────

def test_y_cls_outside_class_set_is_hard_error(tmp_path: Path):
    """If the writer wrote a raw reward (e.g. 0.3, 2.7) into y_cls by mistake,
    classification would silently see stray class ids. Catch it here."""
    n = 200
    bad_y_cls = np.array([0.0, 0.3, 1.5, -0.2, 2.7] * (n // 5) + [0.0] * (n % 5),
                         dtype=np.float32)
    cache = _build_cache(tmp_path, n_rows=n, y_cls_array=bad_y_cls)
    ok, problems = _validate_dataset_builder_reader_contract(
        cache, _args(),
    )
    assert not ok
    assert any("y_cls contains values" in p for p in problems)


def test_y_cls_in_class_set_passes(tmp_path: Path):
    n = 200
    good_y_cls = np.random.RandomState(1).choice([-1.0, 0.0, 1.0], n).astype(np.float32)
    cache = _build_cache(tmp_path, n_rows=n, y_cls_array=good_y_cls)
    ok, problems = _validate_dataset_builder_reader_contract(
        cache, _args(),
    )
    assert ok, problems


# ─────────────────────────────────────────────────────────────────────────────
# Check (4) — zarr row chunk size
# ─────────────────────────────────────────────────────────────────────────────

def test_tiny_zarr_chunk_size_warns(tmp_path: Path):
    """chunk_rows < 64 reintroduces the per-row decompress cost that
    ``ZarrStreamDataset`` was designed to avoid."""
    cache = _build_cache(tmp_path, n_rows=200, chunk_rows=8)
    ok, problems = _validate_dataset_builder_reader_contract(
        cache, _args(),
    )
    assert not ok
    assert any("zarr X row-chunk size" in p for p in problems)


def test_reasonable_zarr_chunk_size_passes(tmp_path: Path):
    cache = _build_cache(tmp_path, n_rows=200, chunk_rows=128)
    ok, problems = _validate_dataset_builder_reader_contract(
        cache, _args(),
    )
    assert ok, problems


# ─────────────────────────────────────────────────────────────────────────────
# Check (5) — scaler.scale_ shape vs X feature dim
# ─────────────────────────────────────────────────────────────────────────────

def test_scaler_shape_mismatch_is_hard_error(tmp_path: Path):
    """If the scaler was fit on a different feature set than the cache (e.g.
    because ``feature_mask`` changed mid-run), `scaler.transform` would crash
    inside the DataLoader worker — opaque mid-epoch failure."""
    cache = _build_cache(tmp_path, n_features=3, with_scaler=True,
                        scaler_scale=np.ones(7, dtype=np.float64))   # 7 != 3
    ok, problems = _validate_dataset_builder_reader_contract(
        cache, _args(),
    )
    assert not ok
    assert any("scaler.scale_ shape" in p for p in problems)


def test_scaler_shape_match_passes(tmp_path: Path):
    cache = _build_cache(tmp_path, n_features=3, with_scaler=True,
                        scaler_scale=np.ones(3, dtype=np.float64))
    ok, problems = _validate_dataset_builder_reader_contract(
        cache, _args(),
    )
    assert ok, problems


# ─────────────────────────────────────────────────────────────────────────────
# Check (6) — diff is uint8 with values in {0, 1, 2}
# ─────────────────────────────────────────────────────────────────────────────

def test_diff_wrong_dtype_is_error(tmp_path: Path):
    """If diff is somehow stored as int32 (legacy bug), the curriculum scaler
    silently reinterprets it. The reader/writer contract requires uint8."""
    # We bypass _build_cache's protection by inserting a manually-typed array.
    from training.gpu_cache_io import _zarr_create, _zarr_open_group
    cache = _build_cache(tmp_path, with_diff=False)
    z = _zarr_open_group(cache, mode="a")
    bad = np.random.RandomState(2).randint(0, 3, 200).astype(np.int32)
    _zarr_create(z, "diff", shape=bad.shape, chunks=(64,),
                 dtype="int32", compressor=None)
    z["diff"][:] = bad
    ok, problems = _validate_dataset_builder_reader_contract(
        cache, _args(),
    )
    assert not ok
    assert any("diff dtype" in p for p in problems)


def test_diff_values_outside_curriculum_set_is_error(tmp_path: Path):
    """A diff array with values like 4, 7, 50 means the curriculum scaler
    will misclassify the row's difficulty stage."""
    n = 200
    bad_diff = np.random.RandomState(3).randint(0, 8, n).astype(np.uint8)  # 0..7
    cache = _build_cache(tmp_path, n_rows=n, diff_array=bad_diff)
    ok, problems = _validate_dataset_builder_reader_contract(
        cache, _args(),
    )
    assert not ok
    assert any("diff contains values" in p for p in problems)


def test_diff_well_formed_passes(tmp_path: Path):
    n = 200
    good_diff = np.random.RandomState(4).choice([0, 1, 2], n).astype(np.uint8)
    cache = _build_cache(tmp_path, n_rows=n, diff_array=good_diff)
    ok, problems = _validate_dataset_builder_reader_contract(
        cache, _args(),
    )
    assert ok, problems
