"""
Evaluation Module: Performance Metrics (Improvement #2)
=======================================================
Robust strategy-evaluation metrics beyond raw Sharpe:

  * Probabilistic Sharpe Ratio (PSR) — Bailey & López de Prado; the probability
    that the true Sharpe exceeds a benchmark SR*, adjusting for non-Gaussian
    skew/kurtosis.
  * Deflated Sharpe Ratio (DSR) — Bailey–López de Prado multiple-trials
    correction: inflate the benchmark by the number of trials actually run.
  * Calmar ratio — CAGR / |max drawdown|.
  * Omega ratio — probability-weighted gain/loss ratio above a threshold.
  * Tail ratio — 95th / 5th percentile of returns (upside/downside).
  * Sortino + downside deviation.
  * Minimum backtest length for a given SR to be significant at confidence.

All functions accept a return series and (where relevant) a benchmark Sharpe;
``backtest_metrics`` accepts a backtest-like object exposing
``results_df``/``_trade_pnls`` or a DataFrame with a return column.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from scipy.stats import kurtosis as _kurtosis_fn
    from scipy.stats import norm
    from scipy.stats import skew as _skew_fn
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _as_returns(returns) -> np.ndarray:
    arr = np.asarray(returns, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def _norm_cdf(x: float) -> float:
    if _HAS_SCIPY:
        return float(norm.cdf(x))
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    if _HAS_SCIPY:
        return float(norm.ppf(p))
    # Abramowitz-Stegun approximation for the probit function.
    if not 0.0 < p < 1.0:
        return math.inf if p >= 1.0 else -math.inf
    a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
         1.383577518672690e2, -3.066479806614716e1, 2.506628277459239]
    b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
         6.680131188771972e1, -1.328068155288572e1]
    c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996,
         3.754408661907416]
    plow = 0.02425
    phi = lambda x: math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)  # noqa: E731
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= 1.0 - plow:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def sharpe_ratio(returns, annual_factor: float = 252) -> float:
    """Standard annualised Sharpe (mean/std, ddof=1)."""
    r = _as_returns(returns)
    if r.size < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd <= 1e-12:
        return 0.0
    return float(r.mean() / sd * math.sqrt(annual_factor))


def probabilistic_sharpe_ratio(
    returns,
    benchmark_sharpe: float = 0.0,
    annual_factor: float = 252,
    skewness: float | None = None,
    kurtosis: float | None = None,
) -> float:
    """Probability that the true (annualised) Sharpe exceeds ``benchmark_sharpe``.

    Uses the third / fourth moments of the *per-period* returns to adjust the
    standard error, following Bailey & López de Prado (2012).
    """
    r = _as_returns(returns)
    n = r.size
    if n < 3:
        return 0.5
    sr_period = sharpe_ratio(r, annual_factor=1.0)  # per-period, non-annualised
    sk = skewness if skewness is not None else (float(_skew_fn(r)) if _HAS_SCIPY else 0.0)
    ku = kurtosis if kurtosis is not None else (float(_kurtosis_fn(r, fisher=False)) if _HAS_SCIPY else 3.0)
    if ku < 0.0:
        ku = max(ku, -1e-9)  # guard degenerate samples

    sr_bench = benchmark_sharpe / math.sqrt(annual_factor)  # per-period benchmark
    var_term = (1.0 - sk * sr_period + (ku - 1.0) / 4.0 * sr_period * sr_period) / (n - 1.0)
    se = math.sqrt(max(var_term, 0.0))
    if se <= 0.0:
        return 1.0 if sr_period > sr_bench else 0.0
    return float(_norm_cdf((sr_period - sr_bench) / se))


def deflated_sharpe_ratio(
    returns,
    n_trials: int = 1,
    annual_factor: float = 252,
    variance_of_trials: float | None = None,
) -> float:
    """Deflated Sharpe Ratio (Bailey–López de Prado).

    ``n_trials`` is the number of strategy/trial combinations actually tested
    during model selection. ``variance_of_trials`` is the variance of the
    cross-sectional Sharpe distribution (expected max of N Gaussians when None).
    """
    r = _as_returns(returns)
    n = r.size
    if n < 3:
        return 0.5
    sr = sharpe_ratio(r, annual_factor)

    # Benchmark Sharpe inflation from multiple trials:
    #   E[max of N ~ N(0,1)] ~ sqrt(2 ln N)  (per-period basis).
    if n_trials <= 1:
        sr_benchmark = 0.0
    elif variance_of_trials is not None:
        # Use the supplied cross-sectional variance of trial Sharpes (period basis).
        sr_benchmark = math.sqrt(variance_of_trials / max(n_trials, 1))
    else:
        sr_benchmark = _expected_max_of_normals(n_trials) / math.sqrt(annual_factor)

    return probabilistic_sharpe_ratio(r, benchmark_sharpe=sr_benchmark * math.sqrt(annual_factor),
                                      annual_factor=annual_factor)


def _expected_max_of_normals(n: int) -> float:
    if n <= 1:
        return 0.0
    if n < 100:
        # approximation via recursion E_n = E_{n-1} + pdf / cdf
        mu = 0.0
        for _ in range(2, n + 1):
            mu += _norm_pdf(mu) / _norm_cdf(mu)
        return mu
    return math.sqrt(2.0 * math.log(n)) - (math.log(math.log(n)) + 2.0 * math.log(4.0 * math.pi)) / (2.0 * math.sqrt(2.0 * math.log(n))) + 0.5 / math.sqrt(2.0 * math.log(n))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def max_drawdown(returns) -> float:
    """Maximum drawdown as a positive fraction (e.g. 0.15 == 15%)."""
    r = _as_returns(returns)
    if r.size == 0:
        return 0.0
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    dd = 1.0 - eq / peak
    return float(np.max(dd)) if dd.size else 0.0


def calmar_ratio(returns, annual_factor: float = 252) -> float:
    """CAGR / max drawdown."""
    r = _as_returns(returns)
    if r.size == 0:
        return 0.0
    mdd = max_drawdown(r)
    if mdd <= 1e-9:
        return 0.0
    cagr = float(np.prod(1.0 + r) ** (annual_factor / r.size) - 1.0)
    return float(cagr / mdd)


def downside_deviation(returns, target: float = 0.0) -> float:
    """Downside deviation: sqrt(mean(min(r - target, 0)^2)) over ALL observations
    (standard Sortino-definition denominator)."""
    r = _as_returns(returns)
    if r.size == 0:
        return 0.0
    downside = np.minimum(r - target, 0.0)
    if not np.any(downside):
        return 0.0
    return float(math.sqrt(np.mean(downside ** 2)))


def sortino_ratio(returns, annual_factor: float = 252, target: float = 0.0) -> float:
    r = _as_returns(returns)
    if r.size == 0:
        return 0.0
    dd = downside_deviation(r, target)
    if dd <= 1e-12:
        return 0.0
    return float(r.mean() / dd * math.sqrt(annual_factor))


def omega_ratio(returns, threshold: float = 0.0) -> float:
    """Omega = ∫(1−F(x))dx / ∫F(x)dx above/below the threshold."""
    r = _as_returns(returns)
    if r.size == 0:
        return 1.0
    r = np.sort(r)
    gains = np.sum(r[r > threshold] - threshold)
    losses = np.sum(threshold - r[r < threshold])
    if losses <= 1e-12:
        return math.inf if gains > 0 else 1.0
    return float(gains / losses)


def tail_ratio(returns) -> float:
    """95th percentile / 5th percentile (upside capture vs downside)."""
    r = _as_returns(returns)
    if r.size < 20:
        return 1.0
    lo = float(np.percentile(r, 5.0))
    hi = float(np.percentile(r, 95.0))
    if abs(lo) <= 1e-12:
        return 1.0
    return float(hi / -lo)


def minimum_backtest_length(
    target_sharpe: float,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    confidence: float = 0.95,
    annual_factor: float = 252,
) -> float:
    """Minimum number of observations for ``target_sharpe`` (annualised) to be
    statistically distinguishable from 0 at ``confidence``."""
    z = _norm_ppf(confidence)
    sr_period = target_sharpe / math.sqrt(annual_factor)
    num = (1.0 - skewness * sr_period + (kurtosis - 1.0) / 4.0 * sr_period ** 2)
    denom = sr_period * sr_period
    if denom <= 0:
        return math.inf
    return float(z * z * num / denom + 1.0)


def backtest_metrics(backtest: Any, annual_factor: float = 252) -> dict[str, float]:
    """Compute the full metric suite from a backtest-like object.

    Accepts either an object exposing ``results_df`` (with a return column) /
    ``_trade_pnls`` / ``trades``, or a DataFrame / ndarray of returns.
    """
    returns = _extract_returns(backtest)
    if returns is None or len(returns) == 0:
        return {}
    r = _as_returns(returns)
    n = r.size
    sr = sharpe_ratio(r, annual_factor)
    sk = float(_skew_fn(r)) if _HAS_SCIPY else 0.0
    ku = float(_kurtosis_fn(r, fisher=False)) if _HAS_SCIPY else 3.0
    return {
        "sharpe": round(sr, 4),
        "psr": round(probabilistic_sharpe_ratio(r, benchmark_sharpe=0.0, annual_factor=annual_factor), 4),
        "psr_1": round(probabilistic_sharpe_ratio(r, benchmark_sharpe=1.0, annual_factor=annual_factor), 4),
        "dsr": round(deflated_sharpe_ratio(r, n_trials=1, annual_factor=annual_factor), 4),
        "calmar": round(calmar_ratio(r, annual_factor), 4),
        "omega": round(omega_ratio(r), 4),
        "tail_ratio": round(tail_ratio(r), 4),
        "sortino": round(sortino_ratio(r, annual_factor), 4),
        "downside_dev": round(downside_deviation(r), 6),
        "max_drawdown": round(max_drawdown(r), 4),
        "skewness": round(sk, 4),
        "kurtosis": round(ku, 4),
        "min_backtest_bars": round(minimum_backtest_length(sr, sk, ku), 2),
        "n_obs": n,
    }


def _extract_returns(backtest: Any) -> np.ndarray | None:
    if isinstance(backtest, (np.ndarray, list, tuple)):
        return np.asarray(backtest, dtype=np.float64)
    if hasattr(backtest, "results_df") and backtest.results_df is not None:
        df = backtest.results_df
        for col in ("returns", "return", "net_return", "pnl", "pnl_usd"):
            if col in df.columns:
                return df[col].to_numpy(dtype=np.float64)
    if hasattr(backtest, "_trade_pnls") and backtest._trade_pnls is not None:
        pnls = np.asarray(backtest._trade_pnls, dtype=np.float64)
        return pnls[np.isfinite(pnls)] / 10_000.0 if pnls.size else pnls  # USD -> notional-ish returns
    if hasattr(backtest, "trades") and backtest.trades is not None:
        trades = backtest.trades
        pnls = []
        for t in trades:
            pnls.append(t.get("pnl_usd", t.get("pnl", 0)) if isinstance(t, dict) else getattr(t, "pnl_usd", getattr(t, "pnl", 0)))
        return np.asarray(pnls, dtype=np.float64) / 10_000.0
    if hasattr(backtest, "columns") and "returns" in backtest.columns:
        return backtest["returns"].to_numpy(dtype=np.float64)
    return None


@dataclass
class MetricReport:
    """Typed convenience wrapper around the dict returned by backtest_metrics."""
    values: dict[str, float]

    @property
    def sharpe(self) -> float:
        return float(self.values.get("sharpe", 0.0))

    @property
    def psr(self) -> float:
        return float(self.values.get("psr", 0.0))

    @property
    def dsr(self) -> float:
        return float(self.values.get("dsr", 0.0))

    @property
    def calmar(self) -> float:
        return float(self.values.get("calmar", 0.0))

    def is_significant(self, threshold: float = 0.95) -> bool:
        return self.psr >= threshold


__all__ = [
    "MetricReport",
    "backtest_metrics",
    "calmar_ratio",
    "deflated_sharpe_ratio",
    "downside_deviation",
    "max_drawdown",
    "minimum_backtest_length",
    "omega_ratio",
    "probabilistic_sharpe_ratio",
    "sharpe_ratio",
    "sortino_ratio",
    "tail_ratio",
]
