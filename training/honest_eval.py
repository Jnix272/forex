"""Honest validation metric: net PnL on real prices after spread.

The legacy ``dir_sharpe`` / ``cost_sharpe`` in ``loop_epochs.validate_epoch``
multiply ``sign(pred)`` by the CPAR *label* (a synthetic reward with var ~4.6)
and subtract ~6e-4 as "cost" - effectively cost-free and not in price units.
That is how +35 "Sharpe" coexisted with out-of-sample R^2 ~ 0.

This module scores directions against the cache's actual ``close`` and
``spread`` arrays (price units, sample-aligned with the cache index):

    pnl_i = d_i * (close[i + H] - close[i]) - |d_i| * spread[i]

taking one trade every ``H`` samples (non-overlapping) and annualizing with
24h FX bars (288 x 5min per day, 260 days).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

FX_DAYS_PER_YEAR = 260
_FX_BARS_PER_DAY = {
    "1m": 1440, "1min": 1440, "5m": 288, "5min": 288, "15m": 96, "15min": 96,
    "30m": 48, "30min": 48, "1h": 24, "60min": 24, "4h": 6,
}


def fx_bars_per_year(bar_freq: str = "5min") -> int:
    return _FX_BARS_PER_DAY.get(str(bar_freq or "5min").lower(), 288) * FX_DAYS_PER_YEAR


def net_pnl_metrics(
    directions: np.ndarray,
    sample_idx: np.ndarray,
    close: np.ndarray,
    spread: np.ndarray | None,
    horizon: int,
    *,
    bars_per_year: int = 288 * FX_DAYS_PER_YEAR,
    min_trades: int = 30,
) -> dict:
    """Net-of-spread, non-overlapping per-trade metrics.

    directions : per-sample position in {-1, 0, +1} (or signed scores; sign is used)
    sample_idx : cache row index for each direction (ascending)
    close/spread : full cache arrays (price units)
    """
    d = np.sign(np.asarray(directions, dtype=np.float64).reshape(-1))
    idx = np.asarray(sample_idx, dtype=np.int64).reshape(-1)
    close = np.asarray(close, dtype=np.float64).reshape(-1)
    h = max(1, int(horizon))
    out = {"sharpe_net": 0.0, "sharpe_gross": 0.0, "n_trades": 0, "win_rate": 0.0,
           "mean_ret_bps": 0.0, "cost_bps": 0.0,
           "sharpe_net_ci_low": 0.0, "sharpe_net_ci_high": 0.0, "sharpe_net_p_le_0": 1.0}
    if len(d) != len(idx) or len(d) == 0:
        return out

    ok = (idx + h) < len(close)
    d, idx = d[ok], idx[ok]
    # Non-overlapping: one decision every h samples.
    d, idx = d[::h], idx[::h]
    traded = d != 0
    d, idx = d[traded], idx[traded]
    if len(d) < 2:
        return out

    entry = close[idx]
    fwd = close[idx + h]
    valid = np.isfinite(entry) & np.isfinite(fwd) & (entry > 0)
    d, idx, entry, fwd = d[valid], idx[valid], entry[valid], fwd[valid]
    gross = d * (fwd - entry) / entry
    if spread is not None:
        sp = np.nan_to_num(np.asarray(spread, dtype=np.float64).reshape(-1)[idx], nan=0.0)
        cost = np.abs(sp) / entry
    else:
        cost = np.zeros_like(gross)
    net = gross - cost

    n = len(net)
    out["n_trades"] = int(n)
    if n < 2:
        return out
    ann = math.sqrt(bars_per_year / h)
    sd_n, sd_g = net.std(ddof=1), gross.std(ddof=1)
    out["sharpe_net"] = float(net.mean() / sd_n * ann) if sd_n > 0 and n >= min_trades else 0.0
    out["sharpe_gross"] = float(gross.mean() / sd_g * ann) if sd_g > 0 and n >= min_trades else 0.0
    out["win_rate"] = float((net > 0).mean())
    out["mean_ret_bps"] = float(net.mean() * 1e4)
    out["cost_bps"] = float(cost.mean() * 1e4)
    out["_net_returns"] = net  # for pooling across pairs; not a scalar metric
    if n >= min_trades:
        lo, hi, p0 = bootstrap_sharpe_ci(net, ann)
        out["sharpe_net_ci_low"], out["sharpe_net_ci_high"], out["sharpe_net_p_le_0"] = lo, hi, p0
    return out


def bootstrap_sharpe_ci(
    returns: np.ndarray,
    ann: float,
    *,
    n_boot: int = 1000,
    block: int = 5,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Moving-block bootstrap CI for an annualized Sharpe of per-trade returns.

    Returns (ci_low, ci_high, p) where p is the bootstrap share of Sharpe <= 0.
    A point Sharpe of a few hundred non-overlapping trades has a standard error
    of roughly ann / sqrt(n), often larger than the Sharpe itself, so a single
    number is not evidence of skill; ci_low > 0 is the minimum bar.
    """
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    n = len(r)
    if n < 2:
        return 0.0, 0.0, 1.0
    b = max(1, min(int(block), n))
    n_blocks = int(math.ceil(n / b))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - b + 1, size=(int(n_boot), n_blocks))
    idx = (starts[..., None] + np.arange(b)).reshape(int(n_boot), -1)[:, :n]
    sample = r[idx]
    sd = sample.std(axis=1, ddof=1)
    sh = np.where(sd > 0, sample.mean(axis=1) / np.where(sd > 0, sd, 1.0), 0.0) * ann
    lo, hi = np.quantile(sh, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi), float((sh <= 0).mean())


def pooled_pair_metrics(
    directions: np.ndarray,
    sample_idx: np.ndarray,
    close_pairs: np.ndarray,
    spread_pairs: np.ndarray | None,
    horizon: int,
    *,
    pair_names: list[str] | None = None,
    bars_per_year: int = 288 * FX_DAYS_PER_YEAR,
    min_trades: int = 30,
) -> dict:
    """Per-pair honest metrics for (N, P) directions, plus a pooled portfolio.

    Each column is scored against its own pair's close/spread. The pooled Sharpe
    treats every non-overlapping trade from every pair as one return stream
    (equal notional per trade), with a bootstrap CI.
    """
    d = np.asarray(directions, dtype=np.float64)
    n_p = d.shape[1]
    names = pair_names or [f"pair_{k}" for k in range(n_p)]
    per_pair, pooled = {}, []
    for k in range(n_p):
        m = net_pnl_metrics(
            d[:, k], sample_idx, close_pairs[:, k],
            None if spread_pairs is None else spread_pairs[:, k],
            horizon, bars_per_year=bars_per_year, min_trades=min_trades,
        )
        r = m.pop("_net_returns", None)
        if r is not None:
            pooled.append(np.asarray(r))
        per_pair[names[k]] = m
    out = {"sharpe_net": 0.0, "sharpe_gross": 0.0, "n_trades": 0, "win_rate": 0.0, "mean_ret_bps": 0.0,
           "cost_bps": float(np.mean([m["cost_bps"] for m in per_pair.values()])) if per_pair else 0.0,
           "sharpe_net_ci_low": 0.0, "sharpe_net_ci_high": 0.0, "sharpe_net_p_le_0": 1.0,
           "per_pair": per_pair}
    if pooled:
        net = np.concatenate(pooled)
        ann = math.sqrt(bars_per_year / max(1, int(horizon)))
        out["n_trades"] = int(len(net))
        if len(net) >= min_trades and net.std(ddof=1) > 0:
            out["sharpe_net"] = float(net.mean() / net.std(ddof=1) * ann)
            out["win_rate"] = float((net > 0).mean())
            out["mean_ret_bps"] = float(net.mean() * 1e4)
            lo, hi, p0 = bootstrap_sharpe_ci(net, ann)
            out["sharpe_net_ci_low"], out["sharpe_net_ci_high"], out["sharpe_net_p_le_0"] = lo, hi, p0
    return out


def period_balance_weights(cache_path, train_idx, n_samples: int) -> np.ndarray | None:
    """Per-row weights (indexed by global row) so each calendar year of the
    training rows carries equal total weight; mean weight over train_idx is 1.
    Rows without a timestamp (t_ns < 0) keep weight 1. None if the cache has no
    ``t_ns`` array.
    """
    try:
        import zarr  # type: ignore

        root = zarr.open(str(cache_path), mode="r")
        if "t_ns" not in root:
            return None
        t_ns = np.asarray(root["t_ns"][:], dtype=np.int64)
    except Exception:
        return None
    idx = np.asarray(train_idx, dtype=np.int64)
    w = np.ones(int(n_samples), dtype=np.float64)
    t = t_ns[idx]
    ok = t >= 0
    if not ok.any():
        return w
    years = (t[ok] // (365.2425 * 86400 * 1e9)).astype(np.int64)
    uniq, inv, counts = np.unique(years, return_inverse=True, return_counts=True)
    ww = (len(years) / (len(uniq) * counts))[inv]
    w[idx[ok]] = ww
    return w


def load_price_arrays(
    cache_path: str | Path, pairs: bool = False
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Read sample-aligned ``close`` / ``spread`` from a Zarr or NPY cache.

    ``pairs=True`` returns the (N, P) ``close_pairs`` / ``spread_pairs`` instead.
    """
    p = Path(cache_path)
    try:
        import zarr  # type: ignore

        root = zarr.open(str(p), mode="r")
        if pairs:
            if "close_pairs" not in root:
                return None, None
            sp = np.asarray(root["spread_pairs"][:]) if "spread_pairs" in root else None
            return np.asarray(root["close_pairs"][:]), sp
        close = np.asarray(root["close"][:]) if "close" in root else None
        spread = np.asarray(root["spread"][:]) if "spread" in root else None
        return close, spread
    except Exception:
        pass
    close = spread = None
    for name in ("close", "spread"):
        f = p / f"{name}.npy" if p.is_dir() else p.with_name(f"{p.stem}_{name}.npy")
        if f.is_file():
            arr = np.load(f, mmap_mode="r")
            if name == "close":
                close = arr
            else:
                spread = arr
    return close, spread
