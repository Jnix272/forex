"""Helpers for sanitizing NaN/Inf values in arrays, frames, and tensors."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

# Columns that must never be clipped to the default feature range (prices,
# volumes, spreads, etc.). Matches features.feature_engineering_pl.
_SANITIZE_NO_CLIP = frozenset(
    {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "bid",
        "ask",
        "mid",
        "bid_close",
        "ask_close",
        "spread",
        "spread_pips",
        "expected_latency_ms",
        "timestamp_utc",
        "cot_net_hf",
        "cot_net_comm",
    }
)


def sanitize_frame(
    df: pd.DataFrame,
    *,
    fill_value: float = 0.0,
    context: str = "frame",
    clip_range: tuple[float, float] | None = (-20.0, 20.0),
    no_clip: frozenset[str] | set[str] | None = None,
) -> pd.DataFrame:
    cleaned = df.replace([np.inf, -np.inf], np.nan)
    bad_count = int(cleaned.isna().sum().sum())
    if bad_count:
        print(f"[FiniteGuard] Sanitized {bad_count:,} non-finite values in {context}")

    cleaned = cleaned.fillna(fill_value)
    if clip_range is not None:
        exclude = _SANITIZE_NO_CLIP if no_clip is None else frozenset(no_clip)
        num_cols = [c for c in cleaned.select_dtypes(include=[np.number]).columns if c not in exclude]
        if num_cols:
            cleaned[num_cols] = cleaned[num_cols].clip(lower=clip_range[0], upper=clip_range[1])

    return cleaned


def sanitize_array(
    arr,
    *,
    fill_value: float = 0.0,
    col_medians: np.ndarray | None = None,
    context: str = "array",
    clip_range: tuple[float, float] | None = (-20.0, 20.0),
) -> np.ndarray:
    """Coerce to float32, replace non-finite / non-numeric with fill_value (or per-column
    medians when ``col_medians`` is supplied), optionally clip.

    Parameters
    ----------
    arr :
        Input array-like.
    fill_value :
        Scalar fallback used when ``col_medians`` is None or a column has no
        finite values to compute a median from.
    col_medians :
        1-D array of shape ``(n_cols,)`` with precomputed per-column medians of
        the **finite** training values.  When supplied, each NaN/Inf cell is
        replaced with its column's median instead of the global ``fill_value``.
        This avoids encoding a false ``0`` signal for features where 0 is
        meaningful (MACD, rate-of-change, ATR ratio, etc.).
        Compute with: ``col_medians = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)``
    context :
        Label used in the diagnostic print.
    clip_range :
        Optional ``(lo, hi)`` hard clip applied after fill.

    Empty strings and other non-numeric values (common in mixed Dukascopy
    columns) are coerced via ``pd.to_numeric(..., errors='coerce')`` instead of
    raising ``ValueError`` on ``astype(float32)``.
    """
    raw = np.asarray(arr)
    shape = raw.shape
    try:
        clean = raw.astype(np.float32, copy=True)
    except (ValueError, TypeError):
        flat = pd.to_numeric(
            pd.Series(raw.ravel(), dtype=object),
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        clean = np.nan_to_num(flat, nan=np.float64(fill_value)).astype(np.float32).reshape(shape)

    bad_mask = ~np.isfinite(clean)
    bad_count = int(bad_mask.sum())
    if bad_count:
        print(f"[FiniteGuard] Sanitized {bad_count:,} non-finite values in {context}")
        if col_medians is not None and clean.ndim == 2:
            # Per-column median fill: replace each NaN with that column's median.
            # Fall back to fill_value for any column whose median is itself non-finite.
            medians_f32 = np.asarray(col_medians, dtype=np.float32)
            for col_idx in range(clean.shape[1]):
                col_bad = bad_mask[:, col_idx]
                if col_bad.any():
                    med = medians_f32[col_idx] if col_idx < len(medians_f32) else np.float32(fill_value)
                    clean[col_bad, col_idx] = med if np.isfinite(med) else np.float32(fill_value)
        else:
            clean[bad_mask] = np.float32(fill_value)

    if clip_range is not None:
        clean = np.clip(clean, clip_range[0], clip_range[1])

    return clean


def ensure_finite_tensor(tensor, *, context: str = "tensor") -> None:
    if torch is None:
        return
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"[FiniteGuard] Non-finite values detected in {context}")
