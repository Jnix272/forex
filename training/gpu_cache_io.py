"""Zarr / NPY cache path helpers and open utilities for GPU training.

Backward-compat shim: the implementation moved to :mod:`common.cache_io`
(refactor R1, to break the ``data -> training`` import cycle). All symbols
are re-exported here unchanged so ``training.gpu_cache_io.<name>`` keeps
working and is the *same object* as the canonical one.
"""

from __future__ import annotations

from common.cache_io import (  # noqa: F401
    ZARR,
    ZARR_FEATURE_DTYPE,
    ZARR_LABEL_DTYPE,
    _atr_path,
    _base_path,
    _bet_size_path,
    _close_path,
    _diff_path,
    _pq_path,
    _scaler_npz_path,
    _spread_path,
    _x_path,
    _y_cls_path,
    _y_path,
    _zarr_create,
    _zarr_open_group,
    default_zarr_compression,
    make_training_zarr_compressor,
)
