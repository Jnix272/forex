"""Point-in-time execution simulation shared by labeling and validation.

The simulator is deliberately conservative: decisions made on bar ``t`` enter
on a later bar, longs liquidate on bid, shorts liquidate on ask, and an
ambiguous bar that touches both stop and target resolves stop-first.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExecutionScenario:
    slippage_pips: float = 0.5
    commission_pips: float = 0.0
    rejected: bool = False
    fill_fraction: float = 1.0
    latency_ms: float = 50.0


class EmpiricalFillModel:
    """Sample real fill outcomes, with a conservative parametric fallback.

    A calibration CSV may contain ``pair,hour,side,slippage_pips,latency_ms,
    fill_fraction,rejected``. Missing conditioning columns are simply ignored.
    """

    def __init__(self, samples: pd.DataFrame | None = None, seed: int = 17,
                 base_slippage_pips: float = 0.5, randomize: bool = True):
        self.samples = samples.copy() if samples is not None else pd.DataFrame()
        self.seed = int(seed)
        self.base_slippage_pips = float(base_slippage_pips)
        self.randomize = bool(randomize)

    @classmethod
    def from_csv(cls, path: str | Path | None, **kwargs) -> EmpiricalFillModel:
        p = Path(path) if path else None
        return cls(pd.read_csv(p) if p and p.is_file() else None, **kwargs)

    def sample(self, *, timestamp, pair: str, side: str, spread_pips: float,
               volatility_ratio: float = 1.0, latency_ms: float = 50.0) -> ExecutionScenario:
        ts = pd.Timestamp(timestamp)
        pool = self.samples
        if not pool.empty:
            for col, value in (("pair", pair), ("hour", ts.hour), ("side", side)):
                if col in pool.columns:
                    candidate = pool[pool[col].astype(str).str.lower() == str(value).lower()]
                    if not candidate.empty:
                        pool = candidate
            # Stable per-event sampling makes cache rebuilds reproducible.
            key = f"{self.seed}|{int(ts.value)}|{pair}|{side}".encode()
            rng = np.random.default_rng(zlib.crc32(key))
            row = pool.iloc[int(rng.integers(len(pool)))]
            return ExecutionScenario(
                slippage_pips=max(0.0, float(row.get("slippage_pips", self.base_slippage_pips))),
                commission_pips=max(0.0, float(row.get("commission_pips", 0.0))),
                rejected=bool(row.get("rejected", False)),
                fill_fraction=float(np.clip(row.get("fill_fraction", 1.0), 0.0, 1.0)),
                latency_ms=max(0.0, float(row.get("latency_ms", latency_ms))),
            )
        if not self.randomize:
            slip = self.base_slippage_pips
        else:
            key = f"{self.seed}|{int(ts.value)}|{pair}|{side}".encode()
            rng = np.random.default_rng(zlib.crc32(key))
            scale = max(0.15, self.base_slippage_pips * (1 + .35 * max(0, volatility_ratio - 1)
                                                        + .25 * max(0, spread_pips - 1)))
            slip = float(max(0, rng.lognormal(np.log(scale), 0.35)))
        return ExecutionScenario(slippage_pips=slip, latency_ms=latency_ms)


def _series(bars: pd.DataFrame, names: tuple[str, ...], fallback: pd.Series) -> np.ndarray:
    for name in names:
        if name in bars:
            return bars[name].astype(float).to_numpy()
    return fallback.astype(float).to_numpy()


def executable_quotes(bars: pd.DataFrame, pip_size: float, spread_pips: np.ndarray) -> dict[str, np.ndarray]:
    """Return side-specific OHLC, synthesizing quotes only when unavailable."""
    mid = bars["close"].astype(float)
    half = pd.Series(spread_pips * pip_size / 2, index=bars.index)
    high = bars.get("high", mid).astype(float)
    low = bars.get("low", mid).astype(float)
    open_ = bars.get("open", mid).astype(float)
    return {
        "ask_open": _series(bars, ("ask_open", "ask_close"), open_ + half),
        "bid_open": _series(bars, ("bid_open", "bid_close"), open_ - half),
        "ask_high": _series(bars, ("ask_high",), high + half),
        "ask_low": _series(bars, ("ask_low",), low + half),
        "ask_close": _series(bars, ("ask_close",), mid + half),
        "bid_high": _series(bars, ("bid_high",), high - half),
        "bid_low": _series(bars, ("bid_low",), low - half),
        "bid_close": _series(bars, ("bid_close",), mid - half),
    }


def _path_pnl(side: str, entry: float, start: int, end: int, q: dict[str, np.ndarray],
              tp: float, sl: float, pip_size: float) -> tuple[float, float, int]:
    """Executable PnL, MAE and exit offset; ambiguous bars are stop-first."""
    mae = 0.0
    for j in range(start, end):
        if side == "long":
            mae = max(mae, (entry - q["bid_low"][j]) / pip_size)
            stop_hit, target_hit = q["bid_low"][j] <= sl, q["bid_high"][j] >= tp
            if stop_hit: return (sl - entry) / pip_size, mae, j - start + 1
            if target_hit: return (tp - entry) / pip_size, mae, j - start + 1
        else:
            mae = max(mae, (q["ask_high"][j] - entry) / pip_size)
            stop_hit, target_hit = q["ask_high"][j] >= sl, q["ask_low"][j] <= tp
            if stop_hit: return (entry - sl) / pip_size, mae, j - start + 1
            if target_hit: return (entry - tp) / pip_size, mae, j - start + 1
    exit_price = q["bid_close"][end - 1] if side == "long" else q["ask_close"][end - 1]
    pnl = (exit_price - entry) / pip_size if side == "long" else (entry - exit_price) / pip_size
    return pnl, mae, end - start


def realistic_utility_labels(bars: pd.DataFrame, features: pd.DataFrame, *, atr_col: str = "atr_6",
                             lookahead_bars: int = 10, pip_size: float = .0001,
                             execution_delay_bars: int = 1, pair: str = "EURUSD",
                             fill_model: EmpiricalFillModel | None = None,
                             edge_margin_pips: float = .25, rejection_penalty_pips: float = 1.0,
                             latency_col: str | None = None, no_trade_col: str | None = None,
                             profit_target_atr: float = 1.5, stop_loss_atr: float = 1.0,
                             no_trade_threshold: float = 0.67) -> pd.DataFrame:
    """Generate long/hold/short utilities using delayed executable prices."""
    b = bars.reindex(features.index).ffill()
    n = len(b); delay = max(1, int(execution_delay_bars)); horizon = max(1, int(lookahead_bars))
    atr = features.get(atr_col, pd.Series(.0005, index=features.index)).astype(float).to_numpy()
    spread = features.get("spread_pips", pd.Series(.5, index=features.index)).fillna(.5).to_numpy(float)
    latency = features.get(latency_col, pd.Series(50.0, index=features.index)).fillna(50.0).to_numpy(float) if latency_col else np.full(n, 50.0)
    q = executable_quotes(b, pip_size, spread)
    model = fill_model or EmpiricalFillModel(randomize=False)
    long_u = np.full(n, np.nan, np.float32); short_u = long_u.copy(); pq = long_u.copy(); conf = long_u.copy()
    for i in range(n):
        entry_i = i + delay
        end = min(n, entry_i + horizon)
        if entry_i >= n or end <= entry_i: continue
        vol_ratio = atr[i] / max(np.nanmedian(atr[max(0, i-120):i+1]), 1e-12)
        ls = model.sample(timestamp=features.index[i], pair=pair, side="long", spread_pips=spread[i], volatility_ratio=vol_ratio, latency_ms=latency[i])
        ss = model.sample(timestamp=features.index[i], pair=pair, side="short", spread_pips=spread[i], volatility_ratio=vol_ratio, latency_ms=latency[i])
        el = q["ask_open"][entry_i] + ls.slippage_pips * pip_size
        es = q["bid_open"][entry_i] - ss.slippage_pips * pip_size
        tp_dist, sl_dist = profit_target_atr * atr[i], stop_loss_atr * atr[i]
        pl, ml, _ = _path_pnl("long", el, entry_i, end, q, el + tp_dist, el - sl_dist, pip_size)
        ps, ms, _ = _path_pnl("short", es, entry_i, end, q, es - tp_dist, es + sl_dist, pip_size)
        long_u[i] = -rejection_penalty_pips if ls.rejected else (pl - ls.commission_pips) * ls.fill_fraction
        short_u[i] = -rejection_penalty_pips if ss.rejected else (ps - ss.commission_pips) * ss.fill_fraction
        best, mae = (long_u[i], ml) if long_u[i] >= short_u[i] else (short_u[i], ms)
        pq[i] = np.clip(1 - mae / (abs(best) + 1e-3), 0, 1)
        conf[i] = np.clip((best - edge_margin_pips) / max(tp_dist / pip_size, 1e-6), 0, 1)
    hold_u = np.zeros(n, np.float32)
    stack = np.column_stack((short_u, hold_u, long_u))
    valid = np.isfinite(stack).all(axis=1)
    choice = np.where(valid, np.argmax(stack, axis=1), 1)
    best = np.where(valid, np.max(stack, axis=1), np.nan)
    # Abstain unless the winning trade beats hold by a configurable uncertainty margin.
    choice[(best < edge_margin_pips) | ~valid] = 1
    if no_trade_col and no_trade_col in features:
        choice[features[no_trade_col].fillna(0).to_numpy(float) > no_trade_threshold] = 1
    label = np.array([-1, 0, 1], dtype=np.int8)[choice]
    optimal = np.array(["short", "hold", "long"], dtype=object)[choice]
    out = pd.DataFrame({"reward_long": long_u, "reward_short": short_u, "utility_long": long_u,
                        "utility_hold": hold_u, "utility_short": short_u, "reward": best, "label": label,
                        "path_quality": pq, "confidence_target": conf,
                        "no_trade": (choice == 1).astype(np.int8), "optimal_side": optimal}, index=features.index)
    return out.loc[valid]
