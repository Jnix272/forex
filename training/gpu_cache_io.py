"""Zarr / NPY cache path helpers and open utilities for GPU training."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Feature tensors (X) on disk: FP16 halves cache size vs FP32 with negligible
# loss for StandardScaler-normalized inputs. Labels / market sidecars stay FP32.
ZARR_FEATURE_DTYPE = np.dtype(np.float16)
ZARR_LABEL_DTYPE = np.dtype(np.float32)

try:
    import numcodecs
    import zarr

    numcodecs.blosc.use_threads = False
    if hasattr(numcodecs.blosc, "set_nthreads"):
        numcodecs.blosc.set_nthreads(1)

    from numcodecs import Blosc as _Blosc

    ZARR = True
    try:
        _ZARR_MAJOR = int(str(getattr(zarr, "__version__", "0")).split(".")[0])
    except Exception:
        _ZARR_MAJOR = 0
    _ZARR_V3 = _ZARR_MAJOR >= 3

    def _zarr_open_group(path: str, mode: str):
        """Open a zarr group, forcing format=2 on zarr v3 for numcodecs compressors."""
        if mode == "w" and not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        if _ZARR_V3:
            return zarr.open_group(path, mode=mode, zarr_format=2)
        return zarr.open_group(path, mode=mode)

    def _zarr_create(
        group,
        name: str,
        *,
        dtype: Any = ZARR_LABEL_DTYPE,
        **kwargs,
    ):
        """Create a named array in a Zarr group.

        ``dtype`` defaults to float32 (labels / sidecars). Pass
        ``ZARR_FEATURE_DTYPE`` (float16) for the feature tensor ``X``.
        """
        kwargs["dtype"] = np.dtype(dtype)
        if hasattr(group, "create_array"):
            return group.create_array(name, **kwargs)
        if hasattr(group, "array"):
            return group.array(name, **kwargs)
        if hasattr(group, "create_dataset"):
            return group.create_dataset(name, **kwargs)
        raise AttributeError("Zarr Group has no supported create method")

except ImportError:
    ZARR = False
    _ZARR_V3 = False
    _Blosc = None  # type: ignore[misc, assignment]

    def _zarr_open_group(path: str, mode: str):  # type: ignore[misc] # pyright: ignore[reportRedeclaration]
        raise ImportError("zarr not installed")

    def _zarr_create(group, name: str, *, dtype: Any = ZARR_LABEL_DTYPE, **kwargs):  # type: ignore[misc] # pyright: ignore[reportRedeclaration]
        raise ImportError("zarr not installed")


# Default training-cache compressor.
# Linux local FS (ext4/xfs/btrfs on NVMe/SSD) is usually not I/O-bound for
# sequential Zarr reads - Blosc+lz4@1 maximizes decompress throughput.
# Other platforms keep zstd@3 for a better ratio on slower/external storage.
_DEFAULT_ZARR_CNAME = "auto"
_DEFAULT_ZARR_CLEVEL: int | None = None
_LINUX_ZARR_CNAME = "lz4"
_LINUX_ZARR_CLEVEL = 1
_FALLBACK_ZARR_CNAME = "zstd"
_FALLBACK_ZARR_CLEVEL = 3


def default_zarr_compression() -> tuple[str, int]:
    """Platform-tuned (cname, clevel) for training-cache Zarr writes.

    Linux: ``lz4`` @ ``1`` - fast decompress on local filesystems.
    Else: ``zstd`` @ ``3`` - stronger ratio when disk/USB I/O dominates.
    """
    if os.name == "posix" and sys.platform.startswith("linux"):
        return _LINUX_ZARR_CNAME, _LINUX_ZARR_CLEVEL
    return _FALLBACK_ZARR_CNAME, _FALLBACK_ZARR_CLEVEL


def make_training_zarr_compressor(
    cname: str | None = None,
    clevel: int | None = None,
    *,
    shuffle: str = "bitshuffle",
):
    """Return a Blosc compressor tuned for training-cache write/read.

    ``cname="auto"`` / ``None`` and ``clevel=None`` select
    :func:`default_zarr_compression` (Linux → lz4@1, else zstd@3).
    Falls back to ``None`` when numcodecs/Blosc is unavailable.
    """
    if _Blosc is None:
        return None
    auto_cname, auto_clevel = default_zarr_compression()
    raw = str(cname if cname is not None else _DEFAULT_ZARR_CNAME).strip().lower()
    codec = auto_cname if (not raw or raw == "auto") else raw
    level = int(auto_clevel if clevel is None else clevel)
    level = max(1, min(level, 9))
    shuffle_map = {
        "bitshuffle": getattr(_Blosc, "BITSHUFFLE", 2),
        "shuffle": getattr(_Blosc, "SHUFFLE", 1),
        "noshuffle": getattr(_Blosc, "NOSHUFFLE", 0),
        "none": getattr(_Blosc, "NOSHUFFLE", 0),
    }
    shuf = shuffle_map.get(str(shuffle).lower(), shuffle_map["bitshuffle"])
    return _Blosc(cname=codec, clevel=level, shuffle=shuf)


def _base_path(cache_path) -> str:
    """Strip .zarr extension to get base name used for NPY/NPZ sidecars."""
    p = str(cache_path)
    if p.endswith(".zarr"):
        return p[:-5]
    return p


def _scaler_npz_path(cache_path: Path) -> Path:
    return Path(_base_path(str(cache_path)) + "_scaler.npz")


def _x_path(cache_path) -> str:
    return _base_path(str(cache_path)) + "_X.npy"


def _y_path(cache_path) -> str:
    return _base_path(str(cache_path)) + "_y.npy"


def _diff_path(cache_path) -> str:
    """NPY sidecar for per-sample difficulty scores (uint8: 0=easy,1=medium,2=hard)."""
    return _base_path(str(cache_path)) + "_diff.npy"


def _pq_path(cache_path) -> str:
    """NPY sidecar for per-sample path-quality scores (float32, range 0-1)."""
    return _base_path(str(cache_path)) + "_pq.npy"


def _y_cls_path(cache_path) -> str:
    """NPY sidecar / zarr array for direction labels {-1,0,+1} when y stores reward."""
    return _base_path(str(cache_path)) + "_y_cls.npy"


def _close_path(cache_path) -> str:
    """Per-sequence mid/close price at the label bar (float32, absolute FX quote)."""
    return _base_path(str(cache_path)) + "_close.npy"


def _atr_path(cache_path) -> str:
    """Per-sequence ATR at the label bar (float32, price units)."""
    return _base_path(str(cache_path)) + "_atr.npy"


def _spread_path(cache_path) -> str:
    """Per-sequence bid-ask spread at the label bar (float32, price units)."""
    return _base_path(str(cache_path)) + "_spread.npy"
