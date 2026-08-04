"""
Evaluation Module: Monte Carlo (Improvement #3)
===============================================
Resampling-based robustness testing for backtest results:

  1. block_bootstrap / BlockBootstrap        — fixed-length block resampling
  2. stationary_bootstrap / StationaryBootstrap — Politis–Romano resampling
     (geometric random block lengths)
  3. PathMonteCarlo                          — replay a decision rule over
     bootstrap-resampled market return paths; distribution of final equity,
     max drawdown, Sharpe
  4. TradeSequenceMonteCarlo                 — keep the strategy's trade
     sequence (entry/exit bars + signed size) fixed and resample the market
     path, so realized P&L depends on which market window trades hit
  5. summarize_simulation                    — percentile CIs + tail metrics

Why this matters:
  A single backtest Sharpe of 2.0 may be an artifact of one lucky window.
  Bootstrap resampling shows the *sampling distribution* of performance
  metrics under alternative market paths while respecting the temporal
  dependence structure of returns (blocks) rather than assuming i.i.d.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")


# ═════════════════════════════════════════════════════════════════════════════
# 1. Core resampling primitives (index generators)
# ═════════════════════════════════════════════════════════════════════════════

def _as_float_array(data: Sequence[float]) -> np.ndarray:
    """Coerce input to a clean 1-D float64 array, dropping NaNs."""
    arr = np.asarray(data, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def _validate_length(n: int, min_length: int = 2, name: str = "returns") -> None:
    if n < min_length:
        raise ValueError(f"{name} must contain at least {min_length} observations, got {n}")


def block_bootstrap_indices(
    n: int,
    block_length: int = 20,
    n_bootstraps: int = 1000,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate fixed-length block bootstrap index sets.

    Returns an array of shape (n_bootstraps, n). Each row is a resampling of
    indices ``0..n-1`` built from contiguous blocks of length ``block_length``
    sampled with replacement (López de Prado style). Blocks are concatenated
    and truncated to length ``n``.

    If ``block_length >= n`` the sample degenerates to an i.i.d. bootstrap.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if block_length < 1:
        raise ValueError(f"block_length must be >= 1, got {block_length}")
    if n_bootstraps < 1:
        raise ValueError(f"n_bootstraps must be >= 1, got {n_bootstraps}")

    rng = rng if rng is not None else np.random.default_rng(seed)
    if block_length >= n:
        block_length = n

    n_blocks = int(np.ceil(n / block_length))
    # Start positions of sampled blocks (within [0, n - block_length])
    max_start = n - block_length
    starts = rng.integers(0, max_start + 1, size=(n_bootstraps, n_blocks))

    out = np.empty((n_bootstraps, n), dtype=np.int64)
    for b in range(n_bootstraps):
        idx = (starts[b, :, None] + np.arange(block_length)).ravel()[:n]
        out[b] = idx
    return out


def stationary_bootstrap_indices(
    n: int,
    avg_block_length: int = 20,
    n_bootstraps: int = 1000,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate Politis–Romano stationary bootstrap index sets.

    Random block lengths follow a geometric distribution with mean
    ``avg_block_length``; block starting points are uniform on ``[0, n)``
    with circular wrap. Preserves stationarity of the underlying process.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if avg_block_length < 1:
        raise ValueError(f"avg_block_length must be >= 1, got {avg_block_length}")
    if n_bootstraps < 1:
        raise ValueError(f"n_bootstraps must be >= 1, got {n_bootstraps}")

    rng = rng if rng is not None else np.random.default_rng(seed)
    p = 1.0 / avg_block_length
    out = np.empty((n_bootstraps, n), dtype=np.int64)

    for b in range(n_bootstraps):
        i = 0
        while i < n:
            block_len = int(rng.geometric(p))
            start = int(rng.integers(0, n))
            take = min(block_len, n - i)
            for j in range(take):
                out[b, i] = (start + j) % n
                i += 1
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 2. Polars-native resampling helpers
# ═════════════════════════════════════════════════════════════════════════════

def pl_block_bootstrap(
    series: Any,
    block_length: int = 20,
    n_bootstraps: int = 1000,
    seed: int | None = None,
    prefix: str = "boot",
) -> Any:
    """
    Polars-native block bootstrap.

    Parameters
    ----------
    series : polars.Series
        The return (or metric) series to resample.
    block_length, n_bootstraps, seed : forwarded to block_bootstrap_indices.
    prefix : column name prefix for each bootstrap column.

    Returns
    -------
    polars.DataFrame with one column per bootstrap: ``f"{prefix}_{i}"``.
    Raises ImportError if polars is not installed.
    """
    try:
        import polars as pl
    except ImportError as e:  # pragma: no cover - env guard
        raise ImportError("polars is required for pl_block_bootstrap") from e

    arr = _as_float_array(series)
    _validate_length(len(arr), name="series")
    idx = block_bootstrap_indices(
        len(arr), block_length=block_length, n_bootstraps=n_bootstraps, seed=seed
    )
    cols = {f"{prefix}_{i}": arr[row] for i, row in enumerate(idx)}
    return pl.DataFrame(cols)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Path Monte Carlo — replay a decision rule over resampled paths
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class SimResult:
    """Statistics for a single simulated equity path."""
    final_equity: float
    total_return: float
    max_drawdown: float
    sharpe: float


def _equity_path_from_returns(
    path_returns: np.ndarray,
    initial_equity: float,
) -> np.ndarray:
    """Compound a return path into an equity curve."""
    eq = initial_equity * np.concatenate([[1.0], np.cumprod(1.0 + path_returns)])
    return eq


def _path_stats(
    path_returns: np.ndarray,
    initial_equity: float,
    bars_per_year: float,
) -> SimResult:
    """Compute Sharpe / max drawdown / total return for one return path."""
    eq = _equity_path_from_returns(path_returns, initial_equity)
    total_return = float(eq[-1] / eq[0] - 1.0)

    peak = np.maximum.accumulate(eq)
    max_dd = float(np.max(1.0 - eq / np.maximum(peak, 1e-300)))

    r = path_returns
    if r.size < 2:
        sharpe = 0.0
    else:
        std = r.std(ddof=1)
        if std <= 1e-12:
            sharpe = 0.0
        else:
            sharpe = float(r.mean() / std * np.sqrt(bars_per_year))

    return SimResult(
        final_equity=float(eq[-1]),
        total_return=total_return,
        max_drawdown=max_dd,
        sharpe=sharpe,
    )


def summarize_simulation(
    results: Sequence[SimResult],
    confidence: float = 0.95,
) -> dict[str, Any]:
    """
    Aggregate a list of per-simulation stats into percentile confidence
    intervals plus tail probabilities.

    Returns a dict with ``*_mean``, ``*_5th/50th/95th`` percentiles (or the
    requested confidence band) and probabilities of negative Sharpe / negative
    total return.
    """
    if not results:
        return {"n_simulations": 0}
    sharpe = np.array([r.sharpe for r in results])
    mdd = np.array([r.max_drawdown for r in results])
    rets = np.array([r.total_return for r in results])
    equity = np.array([r.final_equity for r in results])

    lo = (1.0 - confidence) / 2.0
    hi = 1.0 - lo

    def _band(a: np.ndarray) -> list[float]:
        return [
            round(float(np.percentile(a, lo * 100.0)), 4),
            round(float(np.percentile(a, hi * 100.0)), 4),
        ]

    return {
        "confidence": confidence,
        "n_simulations": len(results),
        "final_equity_mean": round(float(equity.mean()), 2),
        "final_equity_ci": [round(float(np.percentile(equity, lo * 100.0)), 2),
                            round(float(np.percentile(equity, hi * 100.0)), 2)],
        "total_return_mean": round(float(rets.mean()), 4),
        "total_return_ci": _band(rets),
        "max_drawdown_mean": round(float(mdd.mean()), 4),
        "max_drawdown_ci": _band(mdd),
        "max_drawdown_5th": round(float(np.percentile(mdd, 5.0)), 4),
        "max_drawdown_median": round(float(np.percentile(mdd, 50.0)), 4),
        "max_drawdown_95th": round(float(np.percentile(mdd, 95.0)), 4),
        "sharpe_mean": round(float(sharpe.mean()), 4),
        "sharpe_ci": _band(sharpe),
        "sharpe_5th": round(float(np.percentile(sharpe, 5.0)), 4),
        "sharpe_median": round(float(np.percentile(sharpe, 50.0)), 4),
        "sharpe_95th": round(float(np.percentile(sharpe, 95.0)), 4),
        "prob_sharpe_negative": round(float(np.mean(sharpe < 0.0)), 4),
        "prob_total_return_negative": round(float(np.mean(rets < 0.0)), 4),
    }


class PathMonteCarlo:
    """
    Replay a decision rule over bootstrap-resampled market return paths.

    The strategy is a callable ``signal_fn(returns) -> positions`` that maps a
    full return path to a position series (e.g. +1 long, 0 flat, -1 short).
    Each bootstrap path is scored with the *same* decision rule, so the
    resulting distribution reflects market-path sampling variability rather
    than a lucky ordering of trades.

    A default buy-and-hold strategy (all positions = +1) is used if no
    strategy is provided, or a fixed position series may be supplied directly.

    Parameters
    ----------
    strategy : Optional[Callable[[np.ndarray], np.ndarray]]
        Maps a return path to positions (same length as the path).
    bootstrap : str
        ``"block"`` or ``"stationary"``.
    block_length : int
        Block length (also the mean block length for stationary bootstrap).
    """

    def __init__(
        self,
        strategy: Callable[[np.ndarray], np.ndarray] | None = None,
        n_simulations: int = 1000,
        confidence: float = 0.95,
        bars_per_year: float = 252.0,
        initial_equity: float = 10_000.0,
        bootstrap: str = "block",
        block_length: int = 20,
        seed: int | None = None,
    ):
        self.strategy = strategy
        self.n_sims = n_simulations
        self.confidence = confidence
        self.bpy = bars_per_year
        self.equity = initial_equity
        if bootstrap not in ("block", "stationary"):
            raise ValueError(f"bootstrap must be 'block' or 'stationary', got {bootstrap!r}")
        self.bootstrap = bootstrap
        self.block_length = block_length
        self.seed = seed

    def _resample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.bootstrap == "block":
            return block_bootstrap_indices(n, self.block_length, 1, rng=rng)[0]
        return stationary_bootstrap_indices(n, self.block_length, 1, rng=rng)[0]

    def run(self, returns: Sequence[float]) -> dict[str, Any]:
        """
        Run the path simulation over ``n_simulations`` resampled return paths.
        """
        arr = _as_float_array(returns)
        _validate_length(len(arr), name="returns")
        rng = np.random.default_rng(self.seed)

        results: list[SimResult] = []
        for _ in range(self.n_sims):
            idx = self._resample(len(arr), rng)
            path = arr[idx]
            if self.strategy is not None:
                pos = np.asarray(self.strategy(path), dtype=np.float64).reshape(-1)
                if len(pos) != len(path):
                    raise ValueError(
                        f"strategy must return a position per bar "
                        f"(expected {len(path)}, got {len(pos)})"
                    )
                path_returns = pos * path
            else:
                path_returns = path
            results.append(_path_stats(path_returns, self.equity, self.bpy))

        summary = summarize_simulation(results, self.confidence)
        summary["method"] = f"path_{self.bootstrap}_bootstrap"
        summary["strategy"] = "buy_and_hold" if self.strategy is None else "custom"
        summary["block_length"] = self.block_length
        return summary


# ═════════════════════════════════════════════════════════════════════════════
# 4. Trade-sequence Monte Carlo — keep trades, resample the market path
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    """A single trade: bar interval plus signed position size (fraction)."""
    entry: int                # entry bar index (inclusive)
    exit: int                 # exit bar index (exclusive)
    size: float = 1.0         # signed fraction of equity (+ long / - short)
    weight: float = 1.0       # per-trade scaling factor
    metadata: dict[str, Any] = field(default_factory=dict)


class TradeSequenceMonteCarlo:
    """
    Preserve the strategy's trade sequence but resample the market path.

    Each trade's realized P&L is recomputed against the resampled return
    window ``path[entry:exit]``, so dependence between consecutive trades
    (entry/exit timing) is preserved while the market itself is resampled.
    This directly tests whether results depend on the specific market window
    the strategy happened to trade through.

    Parameters
    ----------
    bootstrap : str
        ``"block"`` or ``"stationary"``.
    block_length : int
        Block length (mean for stationary bootstrap).
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        confidence: float = 0.95,
        bars_per_year: float = 252.0,
        initial_equity: float = 10_000.0,
        bootstrap: str = "block",
        block_length: int = 20,
        seed: int | None = None,
    ):
        self.n_sims = n_simulations
        self.confidence = confidence
        self.bpy = bars_per_year
        self.equity = initial_equity
        if bootstrap not in ("block", "stationary"):
            raise ValueError(f"bootstrap must be 'block' or 'stationary', got {bootstrap!r}")
        self.bootstrap = bootstrap
        self.block_length = block_length
        self.seed = seed

    @staticmethod
    def from_pnls(trade_pnls: Sequence[float], returns: Sequence[float]) -> list[Trade]:
        """
        Build trades from a flat list of per-trade P&L. Each trade is assigned
        a unit-size bar of length 1 at a sequential index, and the P&L is
        treated as already realized (weight = pnl).
        """
        pnls = _as_float_array(trade_pnls)
        n = len(_as_float_array(returns))
        trades = []
        for i, p in enumerate(pnls):
            trades.append(Trade(entry=i % max(n, 1), exit=(i % max(n, 1)) + 1, size=1.0, weight=float(p)))
        return trades

    def _resample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.bootstrap == "block":
            return block_bootstrap_indices(n, self.block_length, 1, rng=rng)[0]
        return stationary_bootstrap_indices(n, self.block_length, 1, rng=rng)[0]

    def run(
        self,
        market_returns: Sequence[float],
        trades: Sequence[Trade],
    ) -> dict[str, Any]:
        """
        Run the trade-sequence simulation.

        Each simulation resamples ``market_returns`` into a new path and then
        recomputes every trade's P&L over its (fixed) bar window. Trades whose
        window falls outside the path are clamped to the path length.
        """
        arr = _as_float_array(market_returns)
        _validate_length(len(arr), name="market_returns")
        if len(trades) == 0:
            raise ValueError("at least one Trade is required")
        rng = np.random.default_rng(self.seed)

        results: list[SimResult] = []
        for _ in range(self.n_sims):
            path = arr[self._resample(len(arr), rng)]
            pnls = []
            for t in trades:
                lo = int(np.clip(t.entry, 0, len(path) - 1))
                hi = int(np.clip(t.exit, lo + 1, len(path)))
                win = path[lo:hi]
                pnl = t.size * float(np.sum(win)) * t.weight
                pnls.append(pnl)
            path_returns = np.asarray(pnls, dtype=np.float64) / self.equity
            results.append(_path_stats(path_returns, self.equity, self.bpy))

        summary = summarize_simulation(results, self.confidence)
        summary["method"] = f"trade_sequence_{self.bootstrap}_bootstrap"
        summary["n_trades"] = len(trades)
        summary["block_length"] = self.block_length
        return summary


# ═════════════════════════════════════════════════════════════════════════════
# 5. Convenience API + export
# ═════════════════════════════════════════════════════════════════════════════

def monte_carlo_backtest(
    returns: Sequence[float],
    strategy: Callable[[np.ndarray], np.ndarray] | None = None,
    n_simulations: int = 1000,
    confidence: float = 0.95,
    bars_per_year: float = 252.0,
    initial_equity: float = 10_000.0,
    bootstrap: str = "block",
    block_length: int = 20,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    One-call Monte Carlo robustness report for a return series.

    Runs a path simulation; pass a ``strategy`` callable to replay a specific
    decision rule instead of buy-and-hold.
    """
    sim = PathMonteCarlo(
        strategy=strategy,
        n_simulations=n_simulations,
        confidence=confidence,
        bars_per_year=bars_per_year,
        initial_equity=initial_equity,
        bootstrap=bootstrap,
        block_length=block_length,
        seed=seed,
    )
    return sim.run(returns)


def block_bootstrap(
    data: Sequence[float],
    block_length: int = 20,
    n_bootstraps: int = 1000,
    seed: int | None = None,
) -> np.ndarray:
    """Sample ``data`` with the fixed-length block bootstrap.

    Returns a (n_bootstraps, len(data)) array of resampled values.
    """
    arr = _as_float_array(data)
    _validate_length(len(arr), name="data")
    idx = block_bootstrap_indices(len(arr), block_length, n_bootstraps, seed)
    return arr[idx]


def stationary_bootstrap(
    data: Sequence[float],
    avg_block_length: int = 20,
    n_bootstraps: int = 1000,
    seed: int | None = None,
) -> np.ndarray:
    """Sample ``data`` with the Politis–Romano stationary bootstrap.

    Returns a (n_bootstraps, len(data)) array of resampled values.
    """
    arr = _as_float_array(data)
    _validate_length(len(arr), name="data")
    idx = stationary_bootstrap_indices(len(arr), avg_block_length, n_bootstraps, seed)
    return arr[idx]


__all__ = [
    "PathMonteCarlo",
    "SimResult",
    "Trade",
    "TradeSequenceMonteCarlo",
    "block_bootstrap",
    "block_bootstrap_indices",
    "monte_carlo_backtest",
    "pl_block_bootstrap",
    "stationary_bootstrap",
    "stationary_bootstrap_indices",
    "summarize_simulation",
]
