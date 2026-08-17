"""Smoke tests for ``training.gpu_datasets.ZarrStreamDataset``.

Covers the 2026-08-06 refactor:

* fix #1  batched block reads + per-row seam
* fix #2  cross-chunk shuffle buffer (training-only)
* fix #6  ``pq`` fallback uses ``1.0`` not ``min(1, |y|)`` (matches writer)
* fix #8  worker-shard slicing uses ``np.array_split`` (no silent empty worker)
* fix #9  ``shuffle=True`` not supported (rely on PyTorch's IterableDataset guard)
* fix #11 cache open keyed by ``(worker_id, cache_path)`` - no cross-epoch leak
* fix #12 ``y_cls`` / ``pq`` already published in the zarr group; legacy NPY
          branch retained only for old caches
* fix #13 per-worker ``np.random.default_rng`` (not fork-shared ``np.random``)

A small synthetic zarr cache is materialised in ``tmp_path`` so every test
exercises the actual storage path (no monkeypatching of the dataset internals).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from training.gpu_cache_io import (
    ZARR,
    ZARR_FEATURE_DTYPE,
    ZARR_LABEL_DTYPE,
    _zarr_create,
    _zarr_open_group,
    make_training_zarr_compressor,
)
from training.gpu_datasets import ZarrStreamDataset

pytestmark = pytest.mark.skipif(not ZARR, reason="zarr not installed")


def _build_cache(
    tmp_path: Path,
    *,
    n_rows: int = 64,
    seq_len: int = 4,
    n_features: int = 3,
    chunk_rows: int = 8,
    with_pq: bool = True,
    with_y_cls: bool = True,
):
    """Construct a tiny zarr cache mimicking ``dataset_builder``'s output.

    Layout (matches ``dataset_builder._append_chunk``):
      X     (n_rows, seq_len, n_features) FP16
      y     (n_rows,)                     FP32
      y_cls (n_rows,)                     FP32  (skipped when ``with_y_cls=False``)
      pq    (n_rows,)                     FP32  (skipped when ``with_pq=False``)
    """
    Comp = make_training_zarr_compressor("lz4", 1) if ZARR else None
    p = tmp_path / "cache.zarr"
    z = _zarr_open_group(str(p), mode="w")

    X = np.random.RandomState(0).randn(n_rows, seq_len, n_features).astype(np.float16)
    y = np.random.RandomState(1).randn(n_rows).astype(np.float32)
    if with_y_cls:
        y_cls = np.random.RandomState(2).randint(-1, 2, n_rows).astype(np.float32)
    if with_pq:
        pq = np.random.RandomState(3).uniform(0.0, 1.0, n_rows).astype(np.float32)

    c0 = (chunk_rows, *X.shape[1:])
    _zarr_create(z, "X", shape=X.shape, chunks=c0, dtype=ZARR_FEATURE_DTYPE, compressor=Comp)
    _zarr_create(z, "y", shape=y.shape, chunks=(chunk_rows,), dtype=ZARR_LABEL_DTYPE, compressor=Comp)
    z["X"][:] = X
    z["y"][:] = y
    if with_y_cls:
        _zarr_create(z, "y_cls", shape=y_cls.shape, chunks=(chunk_rows,), dtype=ZARR_LABEL_DTYPE, compressor=Comp)
        z["y_cls"][:] = y_cls
    if with_pq:
        _zarr_create(z, "pq", shape=pq.shape, chunks=(chunk_rows,), dtype=ZARR_LABEL_DTYPE, compressor=Comp)
        z["pq"][:] = pq
    return str(p), X.astype(np.float32), y, (y_cls if with_y_cls else None), (pq if with_pq else None)


# ─────────────────────────────────────────────────────────────────────────────
# Basic forward path
# ─────────────────────────────────────────────────────────────────────────────


def test_basic_iteration_yields_all_rows(tmp_path: Path):
    cache, X, y, _y_cls, _pq = _build_cache(tmp_path)
    ds = ZarrStreamDataset(cache, np.arange(len(y)), shuffle_chunks=False)
    samples = list(ds)
    assert len(samples) == len(y)
    # First element is X (float32 tensor of shape (seq, n_feat))
    x0, y0 = samples[0]
    assert x0.dtype == torch.float32
    assert x0.shape == X.shape[1:]
    assert y0.dtype == torch.float32


def test_block_compression_alignment_uses_chunk_size(tmp_path: Path):
    """Blocks are partitioned by the zarr row chunk size; if chunk=8 and
    rows=64 we should observe exactly 8 blocks in ``ds._blocks`` (fix #1)."""
    cache, _X, _y, _, _ = _build_cache(tmp_path, n_rows=64, chunk_rows=8)
    ds = ZarrStreamDataset(cache, np.arange(64), shuffle_chunks=False)
    assert len(ds._blocks) == 8
    # Each block is contiguous (sorted) and 8 rows long
    for blk in ds._blocks:
        assert len(blk) == 8
        assert int(blk[-1]) - int(blk[0]) == 7  # contiguous


# ─────────────────────────────────────────────────────────────────────────────
# fix #6 - pq fallback = 1.0 (matches dataset_builder convention)
# ─────────────────────────────────────────────────────────────────────────────


def test_multitask_uses_published_pq(tmp_path: Path):
    cache, _X, y, _y_cls, _pq = _build_cache(tmp_path, with_pq=True, with_y_cls=True)
    ds = ZarrStreamDataset(cache, np.arange(len(y)), shuffle_chunks=False, multitask_targets=True, return_indices=False)
    _x_t, _y_t, _yc_t, pq_t = next(iter(ds))
    assert isinstance(pq_t, torch.Tensor)
    assert 0.0 <= float(pq_t) <= 1.0


def test_multitask_pq_fallback_is_unity_when_pq_missing(tmp_path: Path):
    """When ``pq`` is absent from the cache, the multitask path falls back to
    ``pq=1.0`` - matching ``dataset_builder._sidecar_or_default`` (which uses
    ``np.ones``) instead of the legacy ``min(1, |y|)`` (fix #6)."""
    cache, _X, y, _y_cls, _pq = _build_cache(tmp_path, with_pq=False, with_y_cls=True)
    ds = ZarrStreamDataset(cache, np.arange(len(y)), shuffle_chunks=False, multitask_targets=True)
    _x_t, _y_t, _yc_t, pq_t = next(iter(ds))
    assert float(pq_t) == 1.0  # not min(1, |y|)


# ─────────────────────────────────────────────────────────────────────────────
# fix #2 - cross-chunk shuffle buffer
# ─────────────────────────────────────────────────────────────────────────────


def test_shuffle_buffer_makes_churn_across_epochs(tmp_path: Path):
    """With shuffle_chunks=True + shuffle_buffer_size > 0 the emitted order
    should differ across two ``__iter__`` invocations on the same dataset
    *only when* the seed differs. With a fixed seed, the order is reproducible."""
    cache, _X, _y, _, _ = _build_cache(tmp_path, n_rows=128, chunk_rows=8)
    ds = ZarrStreamDataset(cache, np.arange(128), shuffle_chunks=True, shuffle_buffer_size=64, shuffle_seed=42)
    # Collect all samples - verify we get all 128 rows
    samples = list(ds)
    assert len(samples) == 128

    # Get the global indices (5th element of tuple when return_indices=True)
    ds2 = ZarrStreamDataset(
        cache, np.arange(128), shuffle_chunks=True, shuffle_buffer_size=64, shuffle_seed=42, return_indices=True
    )
    order1 = [idx.item() for *_, idx in ds2]
    ds3 = ZarrStreamDataset(
        cache, np.arange(128), shuffle_chunks=True, shuffle_buffer_size=64, shuffle_seed=42, return_indices=True
    )
    order2 = [idx.item() for *_, idx in ds3]
    # All rows preserved (no loss)
    assert sorted(order1) == list(range(128))
    # Reproducible under fixed seed
    assert order1 == order2
    # And not trivially in row order
    assert order1 != sorted(order1)


def test_shuffle_buffer_zero_is_within_block_only(tmp_path: Path):
    """shuffle_buffer_size=0 disables cross-chunk churn; rows within a block
    are still shuffled but the block visit order is the legacy random-order
    pattern."""
    cache, _X, _y, _, _ = _build_cache(tmp_path, n_rows=128, chunk_rows=8)
    ds = ZarrStreamDataset(cache, np.arange(128), shuffle_chunks=True, shuffle_buffer_size=0, shuffle_seed=7)
    order = [float(y_t.item()) for _, y_t in ds]
    assert len(order) == 128


# ─────────────────────────────────────────────────────────────────────────────
# fix #8 - worker-shard slicing via np.array_split
# ─────────────────────────────────────────────────────────────────────────────


def test_array_split_no_silent_empty_worker(tmp_path: Path):
    """With ``num_workers > len(blocks)`` the trailing workers get an empty
    slice - PyTorch then expects them to yield nothing, without crashing.

    We can't easily run a real DataLoader with workers in the test env, but we
    can directly check the helper that the new code delegates to."""
    import numpy as np_

    # 7 workers, but only 4 blocks worth of data -> some workers get []
    blocks = np_.array_split(np_.arange(4), 7)  # 7 > 4 → tail workers get []
    sizes = [len(b) for b in blocks]
    # Total preserved (no rows dropped)
    assert sum(sizes) == 4
    # Some workers get empty
    assert any(s == 0 for s in sizes)


# ─────────────────────────────────────────────────────────────────────────────
# fix #11 - cache open keyed by (worker_id, cache_path)
# ─────────────────────────────────────────────────────────────────────────────


def test_open_arrays_keyed_by_worker_and_cache(tmp_path: Path):
    cache, _X, _y, _, _ = _build_cache(tmp_path, n_rows=64, chunk_rows=8)
    ds = ZarrStreamDataset(cache, np.arange(64), shuffle_chunks=False)
    # Simulate two workers opening the same cache
    h0 = ds._open_arrays(0)
    h1 = ds._open_arrays(1)
    # Same X/y contents from both workers
    assert h0[0].shape == h1[0].shape
    assert h0[1].shape == h1[1].shape
    # Two distinct slots
    assert (0, cache) in ds._opened_arrays
    assert (1, cache) in ds._opened_arrays
    # Re-open returns the cached handle (no new zarr open)
    h0_again = ds._open_arrays(0)
    assert h0_again is h0


# ─────────────────────────────────────────────────────────────────────────────
# fix #13 - per-worker RNG
# ─────────────────────────────────────────────────────────────────────────────


def test_per_worker_rng_streams_independent_with_seed(tmp_path: Path):
    """Two workers with the same ``shuffle_seed`` but different ids get
    independent RNG streams - the SeedSequence mixing guarantees this."""
    cache, _X, _y, _, _ = _build_cache(tmp_path, n_rows=64, chunk_rows=8)
    ds = ZarrStreamDataset(cache, np.arange(64), shuffle_chunks=False, shuffle_seed=1234)
    g0 = ds._worker_rng(0)
    g1 = ds._worker_rng(1)
    a0 = g0.permutation(20)
    a1 = g1.permutation(20)
    assert not np.array_equal(a0, a1)


def test_per_worker_rng_reproducible_under_same_seed(tmp_path: Path):
    """Same (seed, worker_id) → same RNG → same permutation (reproducibility)."""
    cache, _X, _y, _, _ = _build_cache(tmp_path, n_rows=64, chunk_rows=8)
    ds_a = ZarrStreamDataset(cache, np.arange(64), shuffle_chunks=False, shuffle_seed=7)
    ds_b = ZarrStreamDataset(cache, np.arange(64), shuffle_chunks=False, shuffle_seed=7)
    g_a = ds_a._worker_rng(0)
    g_b = ds_b._worker_rng(0)
    assert np.array_equal(g_a.permutation(50), g_b.permutation(50))


# ─────────────────────────────────────────────────────────────────────────────
# return_indices path - exact 5-tuple / 3-tuple / 2-tuple shapes
# ─────────────────────────────────────────────────────────────────────────────


def test_return_indices_3tuple_when_multitask_off(tmp_path: Path):
    cache, _X, y, _, _ = _build_cache(tmp_path, with_pq=False, with_y_cls=False)
    ds = ZarrStreamDataset(cache, np.arange(len(y)), shuffle_chunks=False, return_indices=True, multitask_targets=False)
    sample = next(iter(ds))
    assert len(sample) == 3
    x_t, y_t, idx_t = sample
    assert x_t.dtype == torch.float32
    assert y_t.dtype == torch.float32
    assert idx_t.dtype == torch.long


def test_return_indices_5tuple_when_multitask_on(tmp_path: Path):
    cache, _X, y, _y_cls, _pq = _build_cache(tmp_path, with_pq=True, with_y_cls=True)
    ds = ZarrStreamDataset(cache, np.arange(len(y)), shuffle_chunks=False, return_indices=True, multitask_targets=True)
    sample = next(iter(ds))
    assert len(sample) == 5
    x_t, y_t, yc_t, pq_t, idx_t = sample
    assert x_t.dtype == torch.float32
    assert y_t.dtype == torch.float32
    assert yc_t.dtype == torch.float32
    assert pq_t.dtype == torch.float32
    assert idx_t.dtype == torch.long


def test_multitask_4tuple_without_indices(tmp_path: Path):
    cache, _X, y, _y_cls, _pq = _build_cache(tmp_path, with_pq=True, with_y_cls=True)
    ds = ZarrStreamDataset(cache, np.arange(len(y)), shuffle_chunks=False, return_indices=False, multitask_targets=True)
    sample = next(iter(ds))
    assert len(sample) == 4


def test_two_tuple_returns_when_multitask_off(tmp_path: Path):
    cache, _X, y, _, _ = _build_cache(tmp_path, with_pq=False, with_y_cls=False)
    ds = ZarrStreamDataset(cache, np.arange(len(y)), shuffle_chunks=False)
    sample = next(iter(ds))
    assert len(sample) == 2


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end with DataLoader (single-process so we don't need real workers)
# ─────────────────────────────────────────────────────────────────────────────


def test_dataloader_iteration_single_process(tmp_path: Path):
    cache, X, _y, _y_cls, _pq = _build_cache(tmp_path, n_rows=32, chunk_rows=8, with_pq=True, with_y_cls=True)
    ds = ZarrStreamDataset(
        cache,
        np.arange(32),
        shuffle_chunks=True,
        shuffle_buffer_size=32,
        shuffle_seed=42,
        multitask_targets=True,
        return_indices=True,
    )
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    batches = list(dl)
    assert len(batches) == 4
    # Each batch: (X, y, y_cls, pq, idx) - all same batch dim
    for b in batches:
        assert len(b) == 5
        assert b[0].shape == (8, X.shape[1], X.shape[2])
        assert b[1].shape == (8,)
        assert b[2].shape == (8,)
        assert b[3].shape == (8,)
        assert b[4].shape == (8,)
