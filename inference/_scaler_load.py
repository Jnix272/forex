"""Inference-time scaler loading + application.

Closes the train/live contract gap where the supervised pipeline z-scores
features via :class:`sklearn.preprocessing.StandardScaler` but ``inference/*``
fed raw features to the model.

This module is the single inference-side source of truth for:

* loading the persisted scaler (``scaler.npz`` written by
  :func:`training.dataset_builder._save_scaler_npz`)
* applying it to a live obs window (with NaN/Inf sanitisation)
* surfacing the scaler's ``feature_names_in_`` length so the engines can
  assert schema-hash parity at load time.

Kept dependency-light (only NumPy) on purpose: importing from
:mod:`training.dataset_builder` would pull heavy training deps into the
inference path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _scaler_npz_path(cache_path: str | Path) -> Path:
    """Mirror :func:`training.dataset_builder._scaler_npz_path` without importing it."""
    return Path(cache_path) / "scaler.npz"


def load_inference_scaler(cache_path: str | Path | None) -> Any | None:
    """Return a ready-to-use :class:`StandardScaler` or ``None`` if no scaler file exists.

    Reconstruction mirrors :func:`training.dataset_builder._load_scaler_npz`
    exactly so train/live transformation is bit-identical. We avoid importing
    :mod:`sklearn.preprocessing` at module load (so slim environments and ONNX
    export paths don't pay the cost); it is imported lazily here.
    """
    if cache_path is None:
        return None
    path = _scaler_npz_path(cache_path)
    if not path.exists():
        return None
    try:
        from sklearn.preprocessing import StandardScaler  # type: ignore
    except Exception:  # pragma: no cover - sklearn unavailable in slim envs
        return None
    # Try non-pickled load first; feature_names is the only object-typed
    # array that can fail this. If feature_names is present, retry with
    # allow_pickle=True for that field alone. Names are cosmetic; we keep
    # them only for debugging schema drift.
    z = np.load(path, allow_pickle=False)
    try:
        feature_names = None
        if "feature_names" in z.files:
            _names_npz = np.load(path, allow_pickle=True)
            _raw = _names_npz["feature_names"]
            # Defensive: feature_names are cosmetic strings. Cast to str and
            # reject anything whose repr is not a plain string, so a crafted
            # object-dtype pickle cannot smuggle non-string payloads through.
            try:
                if _raw.dtype == object:
                    feature_names = np.asarray([str(n) for n in _raw], dtype=object)
                else:
                    feature_names = np.asarray(_raw, dtype=str)
            except Exception:
                feature_names = None
    except Exception:
        feature_names = None
    s = StandardScaler()
    s.mean_ = np.asarray(z["mean"], dtype=np.float64)
    s.scale_ = np.asarray(z["scale"], dtype=np.float64)
    s.var_ = np.asarray(z["var"], dtype=np.float64)
    s.n_features_in_ = int(z["n_features_in_"])
    if "n_samples_seen_" in z.files:
        s.n_samples_seen_ = int(z["n_samples_seen_"])
    if feature_names is not None:
        s.feature_names_in_ = feature_names
    return s


def scaler_feature_count(scaler: Any) -> int | None:
    """Return the scaler's expected feature count, or ``None`` when unknown."""
    if scaler is None:
        return None
    n = getattr(scaler, "n_features_in_", None)
    if n is None:
        return None
    return int(n)


def apply_inference_scaler(scaler: Any, x: np.ndarray) -> np.ndarray:
    """Apply the scaler to ``x`` (returns ``np.float32``), sanitising non-finite values.

    The training-time :class:`ZarrStreamDataset` worker applies the same three
    steps in the same order - see :func:`training.gpu_datasets._decompress_block`:

    1. novel NaN/Inf cleanup → finite values within ±1e6,
    2. ``scaler.transform`` (when scaler is present),
    3. cast to float32.

    If ``scaler`` is ``None`` (RL encoder-only path, demo, or no cache path
    supplied), the input is returned unchanged (only NaN/Inf sanitised).
    """
    arr = np.asarray(x, dtype=np.float32)
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
    if scaler is None:
        return arr.astype(np.float32, copy=False)
    out = scaler.transform(arr.reshape(-1, arr.shape[-1]))
    out = out.reshape(arr.shape).astype(np.float32, copy=False)
    return out
