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
    out["_gross_returns"] = gross
    out["_costs"] = cost
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
    per_pair, pooled, pooled_gross, pooled_cost = {}, [], [], []
    for k in range(n_p):
        m = net_pnl_metrics(
            d[:, k], sample_idx, close_pairs[:, k],
            None if spread_pairs is None else spread_pairs[:, k],
            horizon, bars_per_year=bars_per_year, min_trades=min_trades,
        )
        r = m.pop("_net_returns", None)
        g = m.pop("_gross_returns", None)
        c = m.pop("_costs", None)
        if r is not None:
            pooled.append(np.asarray(r))
            pooled_gross.append(np.asarray(g))
            pooled_cost.append(np.asarray(c))
        per_pair[names[k]] = m
    out = {"sharpe_net": 0.0, "sharpe_gross": 0.0, "n_trades": 0, "win_rate": 0.0, "mean_ret_bps": 0.0,
           "cost_bps": float(np.mean([m["cost_bps"] for m in per_pair.values()])) if per_pair else 0.0,
           "sharpe_net_ci_low": 0.0, "sharpe_net_ci_high": 0.0, "sharpe_net_p_le_0": 1.0,
           "per_pair": per_pair}
    if pooled:
        net = np.concatenate(pooled)
        out["_net_returns"] = net
        out["_gross_returns"] = np.concatenate(pooled_gross)
        out["_costs"] = np.concatenate(pooled_cost)
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


def trade_stats(net: np.ndarray, gross: np.ndarray | None = None, cost: np.ndarray | None = None) -> dict:
    """Equity-style stats from per-trade net returns (fraction of notional, 1x).

    Profit factor is capped at 100 and 0 when there are no trades; it used to be
    +inf with no losing trades, which the gate treated as a pass.
    """
    net = np.asarray(net, dtype=np.float64).reshape(-1)
    if net.size == 0:
        return {"profit_factor": 0.0, "max_drawdown": 0.0, "gross_pnl": 0.0, "transaction_costs": 0.0, "net_pnl": 0.0}
    equity = 1.0 + np.cumsum(net)
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]
    dd = float(np.max((peak - equity) / np.maximum(peak, 1e-12)))
    wins, losses = net[net > 0].sum(), -net[net < 0].sum()
    pf = float(min(100.0, wins / losses)) if losses > 0 else (100.0 if wins > 0 else 0.0)
    return {
        "profit_factor": pf,
        "max_drawdown": max(0.0, dd),
        # Gross is before the spread cost; the gate's cost ratio divides the
        # summed spread cost by the absolute gross (both in return units).
        "gross_pnl": float(np.sum(gross)) if gross is not None else float(np.sum(net)),
        "transaction_costs": float(np.sum(cost)) if cost is not None else 0.0,
        "net_pnl": float(np.sum(net)),
    }


def holdout_gate_metrics(
    model,
    cache_path: str,
    holdout_idx: np.ndarray,
    *,
    horizon: int,
    bar_freq: str = "5min",
    scaler=None,
    pair_names: list[str] | None = None,
    device=None,
    batch_size: int = 512,
) -> dict:
    """Promotion-gate backtest on the cached holdout rows.

    Uses exactly what training used: the cached X windows, the checkpoint's own
    scaler (clipped like training), and each pair's close/spread. Replaces
    scripts.backtest_model.run_execution_backtest for the gate, which rebuilt
    features with a different pipeline, no scaler, position-padded columns and
    today's headline sentiment on every historical bar.

    Returns gate inputs: sharpe, n_trades, profit_factor, max_drawdown,
    gross_pnl, transaction_costs, returns, periods_per_year (+ per_pair).
    """
    import torch
    import zarr  # type: ignore

    from inference._scaler_load import SCALED_FEATURE_CLIP

    idx = np.sort(np.asarray(holdout_idx, dtype=np.int64))
    root = zarr.open(str(cache_path), mode="r")
    X = root["X"]
    dev = device or torch.device("cpu")
    model.eval()
    rows = []
    with torch.no_grad():
        for i in range(0, len(idx), int(batch_size)):
            b = idx[i : i + int(batch_size)]
            xb = np.asarray(X.get_orthogonal_selection((b, slice(None), slice(None))), dtype=np.float32)
            np.nan_to_num(xb, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
            if scaler is not None:
                shp = xb.shape
                xb = scaler.transform(xb.reshape(-1, shp[-1])).astype(np.float32).reshape(shp)
                np.clip(xb, -SCALED_FEATURE_CLIP, SCALED_FEATURE_CLIP, out=xb)
            out = model(torch.from_numpy(xb).to(dev))
            logit = out[0] if isinstance(out, (tuple, list)) else out
            logit = logit.detach().float().cpu().numpy()
            if logit.ndim == 2 and logit.shape[-1] == 3:  # sell/hold/buy classes
                d = logit.argmax(-1).astype(np.float64) - 1.0
            else:  # scalar or per-pair direction logits
                d = np.sign(logit.reshape(len(b), -1)).astype(np.float64)
                d = d[:, 0] if d.shape[1] == 1 else d
            rows.append(d)
    dirs = np.concatenate(rows, axis=0)
    bpy = fx_bars_per_year(bar_freq)
    h = max(1, int(horizon))
    if dirs.ndim == 2:
        if "close_pairs" not in root:
            raise RuntimeError("per-pair model but cache has no close_pairs; rebuild the cache")
        sp = np.asarray(root["spread_pairs"][:]) if "spread_pairs" in root else None
        m = pooled_pair_metrics(
            dirs, idx, np.asarray(root["close_pairs"][:]), sp, h, pair_names=pair_names, bars_per_year=bpy
        )
    else:
        sp = np.asarray(root["spread"][:]) if "spread" in root else None
        m = net_pnl_metrics(dirs, idx, np.asarray(root["close"][:]), sp, h, bars_per_year=bpy)
    net = np.asarray(m.pop("_net_returns", np.zeros(0)))
    gross = m.pop("_gross_returns", None)
    cost = m.pop("_costs", None)
    return {
        "sharpe": float(m.get("sharpe_net", 0.0)),
        "n_trades": int(m.get("n_trades", 0)),
        "returns": net,
        # Non-overlapping trades are spaced h bars apart -> bars_per_year / h per year.
        "periods_per_year": bpy / h,
        "sharpe_ci_low": float(m.get("sharpe_net_ci_low", 0.0)),
        "sharpe_ci_high": float(m.get("sharpe_net_ci_high", 0.0)),
        "per_pair": {k: v.get("sharpe_net") for k, v in (m.get("per_pair") or {}).items()},
        **trade_stats(net, gross, cost),
    }
