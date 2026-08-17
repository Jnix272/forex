"""Zarr compressor tuning + thread prefetch overlap helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from training.gpu_cache_io import (
    ZARR,
    ZARR_FEATURE_DTYPE,
    ZARR_LABEL_DTYPE,
    _zarr_create,
    _zarr_open_group,
    default_zarr_compression,
    make_training_zarr_compressor,
)
from training.gpu_datasets import _ThreadPrefetchLoader, wrap_loader_prefetch


def test_make_training_zarr_compressor_defaults_to_zstd_level_3():
    comp = make_training_zarr_compressor()
    if comp is None:
        # numcodecs unavailable in this env
        return
    auto_cname, auto_clevel = default_zarr_compression()
    cname = getattr(comp, "cname", b"zstd")
    if isinstance(cname, bytes):
        cname = cname.decode()
    assert cname == auto_cname
    assert int(getattr(comp, "clevel", 0)) == auto_clevel


def test_default_zarr_compression_linux_prefers_lz4():
    cname, clevel = default_zarr_compression()
    if os.name == "posix" and sys.platform.startswith("linux"):
        assert cname == "lz4"
        assert clevel == 1
    else:
        assert cname == "zstd"
        assert clevel == 3


def test_make_training_zarr_compressor_override_lz4():
    comp = make_training_zarr_compressor("lz4", 1)
    if comp is None:
        return
    cname = getattr(comp, "cname", "lz4")
    if isinstance(cname, bytes):
        cname = cname.decode()
    assert cname == "lz4"
    assert int(comp.clevel) == 1


def test_zarr_feature_array_uses_float16(tmp_path: Path):
    """Feature tensor X is stored FP16; labels stay FP32."""
    if not ZARR:
        return
    store_path = tmp_path / "feat_f16.zarr"
    z = _zarr_open_group(str(store_path), mode="w")
    X = np.random.randn(4, 8, 3).astype(np.float32)
    y = np.random.randn(4).astype(np.float32)
    _zarr_create(
        z,
        "X",
        shape=X.shape,
        chunks=X.shape,
        dtype=ZARR_FEATURE_DTYPE,
        compressor=None,
    )
    _zarr_create(
        z,
        "y",
        shape=y.shape,
        chunks=y.shape,
        dtype=ZARR_LABEL_DTYPE,
        compressor=None,
    )
    z["X"][:] = np.asarray(X, dtype=ZARR_FEATURE_DTYPE)
    z["y"][:] = y
    assert np.dtype(z["X"].dtype) == np.dtype(np.float16)
    assert np.dtype(z["y"].dtype) == np.dtype(np.float32)
    # Readers upcast to float32 for training
    X_read = np.asarray(z["X"][:], dtype=np.float32)
    assert X_read.dtype == np.float32
    assert X_read.shape == X.shape
    np.testing.assert_allclose(X_read, X.astype(np.float16).astype(np.float32), rtol=1e-3)


def test_wrap_loader_prefetch_single_process():
    """When ``num_workers == 0`` the prefetch thread is the only overlap layer - always wrap."""
    ds = TensorDataset(torch.randn(8, 2), torch.randn(8))
    dl = DataLoader(ds, batch_size=2, num_workers=0)
    args = SimpleNamespace(thread_prefetch_batches=8, num_workers=0)
    wrapped = wrap_loader_prefetch(dl, args)
    assert isinstance(wrapped, _ThreadPrefetchLoader)
    assert wrapped._prefetch == 8
    # idempotent
    assert wrap_loader_prefetch(wrapped, args) is wrapped
    # consumes without hang
    batches = list(wrapped)
    assert len(batches) == 4


def test_wrap_loader_prefetch_skips_when_workers_present():
    """With ``num_workers > 0`` the DataLoader already buffers via worker
    processes; layering the daemon-thread queue would just double-buffer
    pinned tensors (50+ GB on large batch_size on depth 8) without overlap
    benefit. Don't wrap unless the caller opts in via ``force_thread_prefetch``."""
    ds = TensorDataset(torch.randn(8, 2), torch.randn(8))
    dl = DataLoader(ds, batch_size=2, num_workers=4)
    args = SimpleNamespace(thread_prefetch_batches=8, num_workers=4)
    wrapped = wrap_loader_prefetch(dl, args)
    assert wrapped is dl  # no wrapping


def test_wrap_loader_prefetch_force_when_workers_present():
    """Callers can opt back into the daemon-thread queue via
    ``force_thread_prefetch=True`` (e.g. to hide GPU-step jitter on slow disks)."""
    ds = TensorDataset(torch.randn(8, 2), torch.randn(8))
    dl = DataLoader(ds, batch_size=2, num_workers=4)
    args = SimpleNamespace(thread_prefetch_batches=4, num_workers=4, force_thread_prefetch=True)
    wrapped = wrap_loader_prefetch(dl, args)
    assert isinstance(wrapped, _ThreadPrefetchLoader)
    assert wrapped._prefetch == 4
