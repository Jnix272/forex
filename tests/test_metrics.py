"""
Tests for evaluation.metrics (Improvement #2): PSR, DSR, Calmar, Omega, Tail,
Sortino, downside deviation, and minimum backtest length.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from evaluation.metrics import (
    sharpe_ratio,
    probabilistic_sharpe_ratio,
    deflated_sharpe_ratio,
    max_drawdown,
    calmar_ratio,
    downside_deviation,
    sortino_ratio,
    omega_ratio,
    tail_ratio,
    minimum_backtest_length,
    backtest_metrics,
    MetricReport,
)


@pytest.fixture
def returns():
    rng = np.random.default_rng(0)
    return rng.normal(0.0005, 0.01, 1000)


@pytest.fixture
def pos_trend_returns():
    rng = np.random.default_rng(1)
    return rng.normal(0.001, 0.005, 1000)


# ═════════════════════════════════════════════════════════════════════════════
# Sharpe
# ═════════════════════════════════════════════════════════════════════════════

def test_sharpe_zero_variance():
    assert sharpe_ratio(np.zeros(100)) == 0.0


def test_sharpe_known_value():
    # returns with mean 0.001, sd 0.01, 252 annualisation
    rng = np.random.default_rng(3)
    r = rng.normal(0.001, 0.01, 5000)
    sr = sharpe_ratio(r)
    assert sr == pytest.approx(0.001 / 0.01 * math.sqrt(252), rel=0.05)


def test_sharpe_positive_trend(pos_trend_returns):
    assert sharpe_ratio(pos_trend_returns) > 1.0


# ═════════════════════════════════════════════════════════════════════════════
# PSR
# ═════════════════════════════════════════════════════════════════════════════

def test_psr_bounds(returns):
    psr = probabilistic_sharpe_ratio(returns)
    assert 0.0 <= psr <= 1.0


def test_psr_higher_sr_higher_psr():
    rng = np.random.default_rng(4)
    r = rng.normal(0.0003, 0.01, 2000)  # weak but positive drift
    weak = probabilistic_sharpe_ratio(r, benchmark_sharpe=1.0)
    strong = probabilistic_sharpe_ratio(r, benchmark_sharpe=0.0)
    assert strong > weak
    assert strong > probabilistic_sharpe_ratio(r, benchmark_sharpe=2.0)


def test_psr_benchmark_1_matches_sortino_known():
    # A clearly profitable series should beat SR*=0 comfortably.
    rng = np.random.default_rng(5)
    r = rng.normal(0.002, 0.005, 2000)
    assert probabilistic_sharpe_ratio(r, benchmark_sharpe=0.0) > 0.95


def test_psr_skew_kurtosis_explicit():
    rng = np.random.default_rng(6)
    r = rng.normal(0.001, 0.01, 1000)
    p1 = probabilistic_sharpe_ratio(r, skewness=0.0, kurtosis=3.0)
    p2 = probabilistic_sharpe_ratio(r, skewness=-0.5, kurtosis=8.0)
    # Negative skew / fat tails should lower PSR (or not raise it materially)
    assert p2 <= p1 + 1e-6


# ═════════════════════════════════════════════════════════════════════════════
# DSR
# ═════════════════════════════════════════════════════════════════════════════

def test_dsr_single_trial_equals_psr_zero(pos_trend_returns):
    dsr = deflated_sharpe_ratio(pos_trend_returns, n_trials=1)
    psr = probabilistic_sharpe_ratio(pos_trend_returns, benchmark_sharpe=0.0)
    assert dsr == pytest.approx(psr, abs=1e-6)


def test_dsr_many_trials_deflates(pos_trend_returns):
    dsr_1 = deflated_sharpe_ratio(pos_trend_returns, n_trials=1)
    dsr_100 = deflated_sharpe_ratio(pos_trend_returns, n_trials=100)
    assert dsr_100 <= dsr_1 + 1e-9


def test_dsr_bounds(returns):
    dsr = deflated_sharpe_ratio(returns, n_trials=50)
    assert 0.0 <= dsr <= 1.0


# ═════════════════════════════════════════════════════════════════════════════
# Calmar / drawdown / Sortino / downside
# ═════════════════════════════════════════════════════════════════════════════

def test_max_drawdown_known():
    # steady decline -> drawdown approaches ~1
    r = np.full(100, -0.02)
    assert max_drawdown(r) == pytest.approx(1.0 - 0.98 ** 100, abs=1e-6)


def test_max_drawdown_zero_for_flat():
    assert max_drawdown(np.zeros(100)) == 0.0


def test_calmar_positive_for_trend(pos_trend_returns):
    assert calmar_ratio(pos_trend_returns) > 0.0


def test_calmar_zero_mdd():
    assert calmar_ratio(np.zeros(100)) == 0.0


def test_downside_deviation_only_negative():
    r = np.array([0.01, 0.02, -0.03, -0.01, 0.005])
    dd = downside_deviation(r, target=0.0)
    # only negative deviations matter
    expected = math.sqrt(((-0.03) ** 2 + (-0.01) ** 2) / 5.0)
    assert dd == pytest.approx(expected, abs=1e-9)


def test_sortino_positive(pos_trend_returns):
    assert sortino_ratio(pos_trend_returns) > 0.0


def test_sortino_high_for_low_downside():
    r = np.concatenate([np.full(90, 0.001), np.full(10, -0.0001)])
    assert sortino_ratio(r) > sharpe_ratio(r)


# ═════════════════════════════════════════════════════════════════════════════
# Omega / tail
# ═════════════════════════════════════════════════════════════════════════════

def test_omega_symmetric_zero():
    r = np.array([1.0, -1.0, 2.0, -2.0, 0.5, -0.5])
    assert omega_ratio(r, threshold=0.0) == pytest.approx(3.5 / 3.5)


def test_omega_all_gains_inf():
    assert omega_ratio(np.array([0.1, 0.2, 0.3]), threshold=0.0) == math.inf


def test_tail_ratio_positive_skew():
    rng = np.random.default_rng(7)
    # mixture: mostly small negatives, occasional large positives -> heavy right tail
    r = np.concatenate([rng.normal(-0.01, 0.01, 450), rng.exponential(0.1, 50)])
    assert tail_ratio(r) > 1.0


def test_tail_ratio_symmetric():
    r = np.random.default_rng(8).normal(0.0, 1.0, 500)
    assert tail_ratio(r) == pytest.approx(1.0, abs=0.2)


# ═════════════════════════════════════════════════════════════════════════════
# Min backtest length
# ═════════════════════════════════════════════════════════════════════════════

def test_min_backtest_length_monotonic():
    l1 = minimum_backtest_length(target_sharpe=0.5)
    l2 = minimum_backtest_length(target_sharpe=1.0)
    assert l2 < l1


def test_min_backtest_length_known_bounds():
    n = minimum_backtest_length(target_sharpe=2.0, confidence=0.95)
    # SR=2 annualised (0.126 per-period) is roughly detectable in a few hundred bars
    assert 50 < n < 10_000


# ═════════════════════════════════════════════════════════════════════════════
# backtest_metrics integration
# ═════════════════════════════════════════════════════════════════════════════

def test_backtest_metrics_dict(pos_trend_returns):
    res = backtest_metrics(pos_trend_returns)
    for key in ["sharpe", "psr", "dsr", "calmar", "omega", "tail_ratio",
                "sortino", "downside_dev", "max_drawdown", "skewness",
                "kurtosis", "min_backtest_bars", "n_obs"]:
        assert key in res
    assert res["n_obs"] == 1000


def test_backtest_metrics_from_object():
    class FakeBT:
        def __init__(self):
            self.results_df = None
            self._trade_pnls = np.random.default_rng(9).normal(50.0, 80.0, 200)

    res = backtest_metrics(FakeBT())
    assert "sharpe" in res


def test_backtest_metrics_empty():
    assert backtest_metrics([]) == {}


def test_metric_report_accessors(pos_trend_returns):
    report = MetricReport(backtest_metrics(pos_trend_returns))
    assert report.sharpe > 0.0
    assert 0.0 <= report.psr <= 1.0
    assert 0.0 <= report.dsr <= 1.0
    assert isinstance(report.is_significant(), bool)
