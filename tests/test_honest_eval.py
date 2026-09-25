import numpy as np

from training.honest_eval import fx_bars_per_year, net_pnl_metrics


def _prices(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    return 1.1 + np.cumsum(rng.normal(0, 1e-4, n))


def test_perfect_foresight_positive_and_cost_reduces():
    close = _prices()
    h = 12
    idx = np.arange(len(close) - h)
    d = np.sign(close[idx + h] - close[idx])
    free = net_pnl_metrics(d, idx, close, None, h)
    costly = net_pnl_metrics(d, idx, close, np.full(len(close), 1.5e-4), h)
    assert free["sharpe_net"] > 5 and free["win_rate"] > 0.95
    assert costly["mean_ret_bps"] < free["mean_ret_bps"]
    assert costly["cost_bps"] > 1.0


def test_random_directions_lose_to_spread():
    close = _prices(seed=1)
    h = 12
    idx = np.arange(len(close) - h)
    d = np.random.default_rng(2).choice([-1.0, 1.0], len(idx))
    m = net_pnl_metrics(d, idx, close, np.full(len(close), 1.5e-4), h)
    assert m["mean_ret_bps"] < 0
    assert m["n_trades"] == len(range(0, len(idx), h))


def test_mismatched_lengths_return_zero():
    m = net_pnl_metrics(np.ones(5), np.arange(4), _prices(10), None, 1)
    assert m["n_trades"] == 0


def test_fx_bars_per_year():
    assert fx_bars_per_year("5min") == 288 * 260
    assert fx_bars_per_year("15m") == 96 * 260
