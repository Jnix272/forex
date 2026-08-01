"""Helpers for sanitizing NaN/Inf values in arrays, frames, and tensors."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def sanitize_frame(
    df: pd.DataFrame,
    *,
    fill_value: float = 0.0,
    context: str = "frame",
    clip_range: tuple[float, float] = (-20.0, 20.0),
) -> pd.DataFrame:
    cleaned = df.replace([np.inf, -np.inf], np.nan)
    bad_count = int(cleaned.isna().sum().sum())
    if bad_count:
        print(f"[FiniteGuard] Sanitized {bad_count:,} non-finite values in {context}")
    
    cleaned = cleaned.fillna(fill_value)
    if clip_range is not None:
        # Only clip numeric columns
        num_cols = cleaned.select_dtypes(include=[np.number]).columns
        cleaned[num_cols] = cleaned[num_cols].clip(lower=clip_range[0], upper=clip_range[1])
        
    return cleaned


def sanitize_array(
    arr,
    *,
    fill_value: float = 0.0,
    context: str = "array",
    clip_range: tuple[float, float] = (-20.0, 20.0),
) -> np.ndarray:
    clean = np.asarray(arr, dtype=np.float32).copy()
    bad_mask = ~np.isfinite(clean)
    bad_count = int(bad_mask.sum())
    if bad_count:
        print(f"[FiniteGuard] Sanitized {bad_count:,} non-finite values in {context}")
        clean[bad_mask] = np.float32(fill_value)
        
    if clip_range is not None:
        clean = np.clip(clean, clip_range[0], clip_range[1])
        
    return clean


def ensure_finite_tensor(tensor, *, context: str = "tensor") -> None:
    if torch is None:
        return
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"[FiniteGuard] Non-finite values detected in {context}")
