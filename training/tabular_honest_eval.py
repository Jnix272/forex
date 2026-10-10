"""Price-based evaluation helpers for tabular model training."""

from __future__ import annotations

import numpy as np


def evaluate_tabular_predictions(
    zarr_root,
    predictions: np.ndarray,
    local_indices: np.ndarray,
    cache_start: int,
    horizon: int,
    task: str,
) -> dict | None:
    """Score predictions on real cache prices, after spread and extra costs.

    `local_indices` index the trainer's trailing data window; `cache_start`
    maps them back to the full Zarr cache. Classification classes are
    encoded short/hold/long as 0/1/2. Regression outputs are signed scores.
    """
    from training.honest_eval import fx_bars_per_year, net_pnl_metrics, pooled_pair_metrics

    if "close_pairs" in zarr_root:
        close = np.asarray(zarr_root["close_pairs"][:], dtype=np.float64)
        spread = np.asarray(zarr_root["spread_pairs"][:], dtype=np.float64) if "spread_pairs" in zarr_root else None
        if close.ndim != 2 or close.shape[1] == 0:
            return None
        names = zarr_root.attrs.get("pairs", "")
        if isinstance(names, bytes):
            names = names.decode("utf-8", errors="replace")
        if isinstance(names, str):
            names = [part.strip() for part in names.split(",") if part.strip()]
        elif isinstance(names, (tuple, list, np.ndarray)):
            names = [str(part) for part in names]
        else:
            names = []
        if len(names) != close.shape[1]:
            names = [f"pair_{i}" for i in range(close.shape[1])]
    elif "close" in zarr_root:
        close = np.asarray(zarr_root["close"][:], dtype=np.float64).reshape(-1)
        spread = np.asarray(zarr_root["spread"][:], dtype=np.float64).reshape(-1) if "spread" in zarr_root else None
        names = None
    else:
        return None

    pred = np.asarray(predictions, dtype=np.float64).squeeze()
    if pred.ndim == 2:
        pred = pred.argmax(axis=1)
    if str(task).lower() == "classification":
        directions = np.where(pred == 2, 1.0, np.where(pred == 0, -1.0, 0.0))
    else:
        directions = np.sign(pred)
    sample_idx = np.asarray(local_indices, dtype=np.int64).reshape(-1) + int(cache_start)
    if len(sample_idx) != len(directions):
        raise ValueError("Prediction count does not match validation sample indices")

    attrs = zarr_root.attrs
    bar_freq = attrs.get("bar_freq", "5min")
    if isinstance(bar_freq, bytes):
        bar_freq = bar_freq.decode("utf-8", errors="replace")
    bars_per_year = fx_bars_per_year(str(bar_freq))
    t_ns = np.asarray(zarr_root["t_ns"][:], dtype=np.int64) if "t_ns" in zarr_root else None
    atr_pairs = np.asarray(zarr_root["atr_pairs"][:], dtype=np.float64) if "atr_pairs" in zarr_root else None

    if close.ndim == 2:
        return pooled_pair_metrics(
            np.repeat(directions[:, None], close.shape[1], axis=1),
            sample_idx,
            close,
            spread,
            horizon,
            pair_names=names,
            bars_per_year=bars_per_year,
            t_ns=t_ns,
            atr_pairs=atr_pairs,
            skip_gap_windows=t_ns is not None,
        )
    return net_pnl_metrics(
        directions,
        sample_idx,
        close,
        spread,
        horizon,
        bars_per_year=bars_per_year,
        t_ns=t_ns,
        skip_gap_windows=t_ns is not None,
    )
