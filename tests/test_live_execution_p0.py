"""
Tests for Phase 1 (P0) Critical Execution Fixes:
1. JPY 3-decimal vs 5-decimal precision formatting in OANDABroker.
2. OANDABroker close_position HTTP 404 already_closed handling.
3. Multi-pair observation buffer isolation and dimensional alignment in _Wrap.
4. Bidirectional position reconciliation loop.
5. Quote-currency PnL conversion to USD for USDJPY / USDCAD.
"""

import json
import numpy as np
import pytest
from unittest.mock import MagicMock
import pandas as pd
from trading.live_engine import OANDABroker, LiveTradingEngine, PaperBroker


def test_oanda_jpy_and_standard_precision(monkeypatch):
    monkeypatch.setenv("OANDA_API_KEY", "dummy")
    monkeypatch.setenv("OANDA_ACCOUNT_ID", "dummy")
    broker = OANDABroker()

    # 1. USDJPY price formatting must use 3 decimal places
    assert broker._format_price(157.732145, "USDJPY") == "157.732"
    assert broker._format_price(157.7, "USD_JPY") == "157.700"

    # 2. Standard FX (EURUSD, GBPUSD, USDCAD) must use 5 decimal places
    assert broker._format_price(1.1023456, "EURUSD") == "1.10235"
    assert broker._format_price(1.331456, "GBP_USD") == "1.33146"
    assert broker._format_price(1.408221, "USDCAD") == "1.40822"


def test_oanda_market_order_brackets_formatting(monkeypatch):
    monkeypatch.setenv("OANDA_API_KEY", "dummy")
    monkeypatch.setenv("OANDA_ACCOUNT_ID", "dummy")
    broker = OANDABroker()

    captured_requests = []

    class MockResp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "orderFillTransaction": {
                        "id": "100",
                        "price": "157.750",
                        "units": "10000",
                    }
                }
            ).encode("utf-8")

    def mock_urlopen(req, **kwargs):
        captured_requests.append(json.loads(req.data.decode("utf-8")))
        return MockResp()

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    # Test USDJPY order with SL/TP
    res = broker.market_order("USDJPY", "BUY", 1.0, stop_loss=156.95432, take_profit=158.45678)
    assert res["ok"] is True
    assert len(captured_requests) == 1
    sent_order = captured_requests[0]["order"]
    assert sent_order["stopLossOnFill"]["price"] == "156.954"
    assert sent_order["takeProfitOnFill"]["price"] == "158.457"

    # Test EURUSD order with SL/TP
    broker.market_order("EURUSD", "BUY", 1.0, stop_loss=1.095432, take_profit=1.114567)
    assert len(captured_requests) == 2
    sent_eur = captured_requests[1]["order"]
    assert sent_eur["stopLossOnFill"]["price"] == "1.09543"
    assert sent_eur["takeProfitOnFill"]["price"] == "1.11457"


def test_oanda_close_position_already_closed_404(monkeypatch):
    monkeypatch.setenv("OANDA_API_KEY", "dummy")
    monkeypatch.setenv("OANDA_ACCOUNT_ID", "dummy")
    broker = OANDABroker()

    import io
    import urllib.error

    def mock_urlopen_404(req, **kwargs):
        fp = io.BytesIO(b'{"errorMessage": "The instrument does not have an open position."}')
        raise urllib.error.HTTPError("http://dummy", 404, "Not Found", {}, fp)

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen_404)
    res = broker.close_position("EURUSD")
    assert res["ok"] is True
    assert res["reason"] == "already_closed"
    assert res["closed"] == 0


def test_multi_pair_wrap_isolation():
    """Verify that multiple engines maintain isolated observation buffers and feature matching."""
    mock_fast = MagicMock()
    mock_fast.select_action.return_value = 1
    mock_fast.seq_len = 10
    mock_fast.n_features = 146

    mock_slow = MagicMock()
    mock_slow.select_action.return_value = 1
    mock_slow.seq_len = 10
    mock_slow.n_features = 584  # MultiPair model

    broker = PaperBroker(initial_equity=10_000, synthetic=True)

    engine_eur = LiveTradingEngine(
        broker=broker,
        fast_agent=mock_fast,
        slow_model=mock_slow,
        pair="EURUSD",
        equity=10_000,
        max_lots=0.1,
        sentiment_mode="off",
        cross_asset={},
        prometheus_enabled=False,
        db_enabled=False,
        inference_meta={"seq_len": 10, "n_features": 584},
    )

    engine_jpy = LiveTradingEngine(
        broker=broker,
        fast_agent=mock_fast,
        slow_model=mock_slow,
        pair="USDJPY",
        equity=10_000,
        max_lots=0.1,
        sentiment_mode="off",
        cross_asset={},
        prometheus_enabled=False,
        db_enabled=False,
        inference_meta={"seq_len": 10, "n_features": 584},
    )

    # Confirm distinct observation buffer deques
    assert engine_eur.slow._obs_buffer is not engine_jpy.slow._obs_buffer

    # Push 146-dim observations to EUR (slot 0) and JPY (slot 1 in canonical schema)
    obs_eur = np.ones(146, dtype=np.float32)
    obs_jpy = np.full(146, 2.0, dtype=np.float32)

    for _ in range(10):
        engine_eur.slow.select_action(obs_eur)
        engine_jpy.slow.select_action(obs_jpy)

    # In EUR buffer, slot 0 (0:146) should have 1.0, rest 0.0
    eur_top = engine_eur.slow._obs_buffer[-1]
    assert eur_top.shape == (584,)
    assert eur_top[0] == 1.0
    assert eur_top[146] == 0.0

    # In JPY buffer, slot 1 (146:292 in canonical schema) should have 2.0, rest 0.0
    jpy_top = engine_jpy.slow._obs_buffer[-1]
    assert jpy_top.shape == (584,)
    assert jpy_top[0] == 0.0
    assert jpy_top[146] == 2.0


def test_position_reconciliation_external_close():
    """Verify engine detects when broker is flat and resets internal position."""
    broker = PaperBroker(initial_equity=10_000, synthetic=True)
    engine = LiveTradingEngine(
        broker=broker,
        fast_agent=MagicMock(),
        slow_model=MagicMock(),
        pair="EURUSD",
        equity=10_000,
        max_lots=0.1,
        sentiment_mode="off",
        cross_asset={},
        prometheus_enabled=False,
        db_enabled=False,
    )

    # Engine has an internal position
    engine._position = 0.5
    engine._entry_price = 1.1000

    # Broker has NO open positions
    broker.get_positions = MagicMock(return_value={})
    broker.get_bid_ask = MagicMock(return_value=(1.1020, 1.1022))

    engine._reconcile_positions()

    # Reconciled to flat
    assert engine._position == 0.0
    assert engine._entry_price == 0.0


def test_usdjpy_pnl_usd_conversion():
    """Verify non-USD quote currency PnL is properly normalized to USD."""
    broker = PaperBroker(initial_equity=10_000, synthetic=True)
    mock_risk = MagicMock()

    engine = LiveTradingEngine(
        broker=broker,
        fast_agent=MagicMock(),
        slow_model=MagicMock(),
        pair="USDJPY",
        equity=10_000,
        max_lots=0.1,
        sentiment_mode="off",
        cross_asset={},
        prometheus_enabled=False,
        db_enabled=False,
        risk_engine=mock_risk,
    )

    engine._position = 1.0  # 1 lot long
    engine._entry_price = 150.00
    mid = 151.00  # 100 pips move (1.00 JPY diff on 100 pip_size=0.01)

    engine._risk_trade_closed(mid, "test_close")

    # In USDJPY, 100 pips on 1 lot without conversion is $100.
    # Converted to USD, it should be divided by mid: 100 / 151.00 ≈ 0.662
    call_args = mock_risk.on_trade_closed.call_args[1]
    reported_pnl = call_args["pnl"]
    assert 0.60 < reported_pnl < 0.70


def test_instant_buffer_warmup_from_candles():
    """Verify that observation buffers are instantly seeded from historical candle features."""
    mock_fast = MagicMock()
    mock_fast.select_action.return_value = 1
    mock_fast.seq_len = 10
    mock_fast.n_features = 146

    mock_slow = MagicMock()
    mock_slow.select_action.return_value = 1
    mock_slow.seq_len = 10
    mock_slow.n_features = 146

    broker = PaperBroker(initial_equity=10_000, synthetic=True)
    engine = LiveTradingEngine(
        broker=broker,
        fast_agent=mock_fast,
        slow_model=mock_slow,
        pair="EURUSD",
        equity=10_000,
        max_lots=0.1,
        sentiment_mode="off",
        cross_asset={},
        prometheus_enabled=False,
        db_enabled=False,
        inference_meta={"seq_len": 10, "n_features": 146},
    )

    # Initial buffer is empty
    assert len(engine.slow._obs_buffer) == 0

    # Seed 15 bars into engine buffer
    dates = pd.date_range("2026-09-23", periods=15, freq="5min")
    df = pd.DataFrame(
        {
            "open": [1.10 + i * 0.0001 for i in range(15)],
            "high": [1.101 + i * 0.0001 for i in range(15)],
            "low": [1.099 + i * 0.0001 for i in range(15)],
            "close": [1.1005 + i * 0.0001 for i in range(15)],
            "volume": [100.0] * 15,
        },
        index=dates,
    )
    engine.buf.seed_bars(df)

    # Execute _on_new_bar with the 15 bars
    engine._on_new_bar(df, bar_idx=0)

    # Buffer should now be fully populated (10 items = seq_len for slow, 9 warmed up for fast)
    assert len(engine.slow._obs_buffer) == 10
    assert len(engine.fast._obs_buffer) >= 9


def test_online_hedge_weight_adaptation_in_live_engine():
    mock_fast = MagicMock()
    mock_fast.select_action.return_value = 1  # BUY
    mock_fast.seq_len = 10
    mock_fast.n_features = 146
    mock_slow = MagicMock()
    mock_slow.select_action.return_value = 1  # BUY
    mock_slow.seq_len = 10
    mock_slow.n_features = 146

    broker = PaperBroker(initial_equity=10_000, synthetic=True)
    engine = LiveTradingEngine(
        broker=broker,
        fast_agent=mock_fast,
        slow_model=mock_slow,
        pair="EURUSD",
        equity=10_000,
        max_lots=0.1,
        sentiment_mode="off",
        cross_asset={},
        prometheus_enabled=False,
        db_enabled=False,
        inference_meta={"seq_len": 10, "n_features": 146},
    )

    assert hasattr(engine, "hedge_ensemble")
    init_weights = engine.hedge_ensemble.get_weights()
    assert "slow_model" in init_weights
    assert "fast_agent" in init_weights

    # Bar 0
    dates = pd.date_range("2026-09-23 10:00", periods=5, freq="5min")
    df0 = pd.DataFrame(
        {"open": [1.10] * 5, "high": [1.102] * 5, "low": [1.098] * 5, "close": [1.101] * 5, "volume": [100.0] * 5},
        index=dates,
    )
    engine._on_new_bar(df0, bar_idx=0)
    assert "slow_model" in engine._last_bar_preds

    # Bar 1 with higher close price (positive return)
    dates1 = pd.date_range("2026-09-23 10:05", periods=5, freq="5min")
    df1 = pd.DataFrame(
        {"open": [1.101] * 5, "high": [1.105] * 5, "low": [1.100] * 5, "close": [1.104] * 5, "volume": [100.0] * 5},
        index=dates1,
    )
    engine._on_new_bar(df1, bar_idx=1)

    # Verify that hedge weights were logged in engine._bar_log
    last_log = engine._bar_log[-1]
    assert "weights" in last_log
    assert "slow_model" in last_log["weights"]
    assert np.isclose(sum(last_log["weights"].values()), 1.0)


def test_cross_pair_pnl_usd_conversion():
    """Verify non-USD cross pair (e.g. EURGBP) PnL converts to USD via quote/USD rate."""
    broker = PaperBroker(initial_equity=10_000, synthetic=True)
    mock_risk = MagicMock()

    engine = LiveTradingEngine(
        broker=broker,
        fast_agent=MagicMock(),
        slow_model=MagicMock(),
        pair="EURGBP",
        equity=10_000,
        max_lots=0.1,
        sentiment_mode="off",
        cross_asset={},
        prometheus_enabled=False,
        db_enabled=False,
        risk_engine=mock_risk,
    )

    # Mock GBPUSD rate on broker
    broker.get_bid_ask = MagicMock(return_value=(1.3000, 1.3000))

    engine._position = 1.0  # 1 lot long
    engine._entry_price = 0.8500
    mid = 0.8510  # 10 pips in GBP (0.0010 / 0.0001 = 10 pips, 1 lot -> £10)

    engine._risk_trade_closed(mid, "test_cross_close")

    # In EURGBP, 10 pips on 1 lot is £10. Converted to USD via GBPUSD 1.30 -> $13.0
    call_args = mock_risk.on_trade_closed.call_args[1]
    reported_pnl = call_args["pnl"]
    assert np.isclose(reported_pnl, 13.0, atol=0.1)



