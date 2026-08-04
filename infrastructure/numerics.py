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
_SANITIZE_NO_CLIP = frozenset({
    "open", "high", "low", "close", "volume",
    "bid", "ask", "mid",
    "bid_close", "ask_close", "spread", "spread_pips",
    "expected_latency_ms", "timestamp_utc",
    "cot_net_hf", "cot_net_comm",
})


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
        num_cols = [
            c for c in cleaned.select_dtypes(include=[np.number]).columns
            if c not in exclude
        ]
        if num_cols:
            cleaned[num_cols] = cleaned[num_cols].clip(
                lower=clip_range[0], upper=clip_range[1]
            )

    return cleaned


def sanitize_array(
    arr,
    *,
    fill_value: float = 0.0,
    context: str = "array",
    clip_range: tuple[float, float] | None = (-20.0, 20.0),
) -> np.ndarray:
    """Coerce to float32, replace non-finite / non-numeric with fill_value, optionally clip.

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
        clean = np.nan_to_num(
            flat, nan=np.float64(fill_value)
        ).astype(np.float32).reshape(shape)

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
