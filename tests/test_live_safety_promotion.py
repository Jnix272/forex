from __future__ import annotations

import numpy as np
import pandas as pd

from trading.live_engine import LiveSafetyConfig, LiveSafetyGate, PaperBroker
from validation.promotion_gate import GateConfig, PromotionGate


def _feat_frame(**overrides) -> pd.DataFrame:
    n = 300
    rng = np.random.default_rng(0)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0005, n))
    data = {
        "atr_6": np.abs(np.diff(np.concatenate([[close[0]], close]))),
        "spread_pips": np.full(n, 0.8),
        "adx_14": np.full(n, 12.0),
        "rsi_14": np.full(n, 50.0),
        "ret_5": rng.normal(0, 0.001, n),
    }
    for k, v in overrides.items():
        if isinstance(v, (int, float)):
            data[k] = np.full(n, float(v))
    return pd.DataFrame(data)


def test_no_trade_gate_disabled_by_default():
    from trading.live_guards import NoTradeZoneGate
    gate = NoTradeZoneGate()
    res = gate.check(_feat_frame())
    assert not res.blocked
    assert res.reason == "no_trade_disabled"


def test_no_trade_gate_blocks_high_score():
    from trading.live_guards import NoTradeZoneGate
    gate = NoTradeZoneGate(threshold=0.5, enabled=True)
    df = _feat_frame(no_trade_score=0.95)
    res = gate.check(df)
    assert res.blocked
    assert res.reason == "no_trade_zone"


def test_no_trade_gate_allows_low_score():
    from trading.live_guards import NoTradeZoneGate
    gate = NoTradeZoneGate(threshold=0.5, enabled=True)
    df = _feat_frame(no_trade_score=0.1)
    res = gate.check(df)
    assert not res.blocked


def test_no_trade_gate_heuristic_fallback():
    from trading.live_guards import NoTradeZoneGate
    gate = NoTradeZoneGate(threshold=0.5, enabled=True)
    df = _feat_frame()
    res = gate.check(df)
    # No no_trade_score column -> heuristic path; must not raise
    assert res.reason in ("no_trade_ok", "no_trade_zone", "no_trade_unavailable")


def test_live_safety_blocks_wide_spread():
    gate = LiveSafetyGate(
        LiveSafetyConfig(max_spread_pips=1.0),
        starting_equity=10_000.0,
    )

    result = gate.allow_order(
        pair="EURUSD",
        side="buy",
        lots=0.1,
        bid=1.10000,
        ask=1.10030,
        equity=10_000.0,
    )

    assert not result["ok"]
    assert str(result["reason"]).startswith("spread_too_wide")


def test_live_safety_halts_on_daily_loss_limit():
    gate = LiveSafetyGate(
        LiveSafetyConfig(max_daily_loss_pct=0.05),
        starting_equity=10_000.0,
    )

    result = gate.allow_order(
        pair="EURUSD",
        side="sell",
        lots=0.1,
        bid=1.10000,
        ask=1.10005,
        equity=9_499.0,
    )

    assert not result["ok"]
    assert gate.halted
    assert str(result["reason"]).startswith("daily_loss_limit")


def test_live_safety_rate_limits_orders():
    gate = LiveSafetyGate(
        LiveSafetyConfig(max_orders_per_minute=2),
        starting_equity=10_000.0,
    )

    for t in (1000.0, 1001.0):
        result = gate.allow_order("EURUSD", "buy", 0.1, 1.1, 1.10005, 10_000.0, now=t)
        assert result["ok"]

    blocked = gate.allow_order("EURUSD", "buy", 0.1, 1.1, 1.10005, 10_000.0, now=1002.0)
    assert not blocked["ok"]
    assert blocked["reason"] == "order_rate_limit"


def test_paper_broker_quote_moves_and_accepts_external_quote():
    broker = PaperBroker(initial_equity=10_000.0)
    broker.update_quote(1.23450, 1.23460)

    bid, ask = broker.get_bid_ask("EURUSD")

    assert abs(bid - 1.23450) < 1e-6
    assert abs(ask - 1.23460) < 1e-6


def test_promotion_gate_requires_high_psr():
    gate = PromotionGate(
        GateConfig(
            min_psr=0.95,
            strict_psr=True,
            min_sharpe_per_latency=0.001,
        )
    )

    result = gate.evaluate(
        sharpe=1.6,
        profit_factor=1.8,
        max_drawdown=0.10,
        n_trades=700,
        gross_pnl=10_000.0,
        transaction_costs=1_000.0,
        n_obs=4,
        turnover_rate=2.0,
        avg_latency_ms=50.0,
    )

    assert not result["promoted"]
    assert not result["gates"]["psr_ok"]


def test_promotion_gate_requires_dsr_when_trials_are_reported():
    gate = PromotionGate(
        GateConfig(
            min_psr=0.95,
            min_dsr=0.95,
            strict_psr=True,
            min_sharpe_per_latency=0.001,
        )
    )

    result = gate.evaluate(
        sharpe=1.6,
        profit_factor=1.8,
        max_drawdown=0.10,
        n_trades=700,
        gross_pnl=10_000.0,
        transaction_costs=1_000.0,
        n_obs=50,
        n_backtest_trials=1_000,
        turnover_rate=2.0,
        avg_latency_ms=50.0,
    )

    assert not result["promoted"]
    assert not result["gates"]["dsr_ok"]
