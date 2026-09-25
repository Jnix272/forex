"""Paper / live trading contract tests.

Guards the money and observation contracts the paper-trading paths depend on:

* ``PaperBroker`` must price in the same lot/unit convention as the live broker
  (``OANDA_UNITS_PER_LOT`` = 10,000 units per lot) and convert quote-currency
  P&L (USDJPY, USDCAD) into USD.
* ``--broker paper`` must be able to run against a *moving* market, so the
  broker can synthesise quotes and warmup candles.
* The RL agent-state block and action mask must match what the policy was
  trained on (``ForexTradingEnv``), otherwise the policy silently receives an
  out-of-contract observation.
"""

from __future__ import annotations

import numpy as np
import pytest

from models.rl_agents import (
    RL_ACTION_COUNT,
    RL_AGENT_STATE_DIM,
    build_action_mask,
    build_agent_state,
)
from trading.live_engine import OANDABroker, PaperBroker


# ─────────────────────────────────────────────────────────────────────────────
# PaperBroker money conventions
# ─────────────────────────────────────────────────────────────────────────────


def test_paper_broker_units_per_lot_matches_live_convention(monkeypatch):
    """Paper sizes must match OANDABroker/OANDA_UNITS_PER_LOT, not 100k units."""
    monkeypatch.setenv("OANDA_UNITS_PER_LOT", "10000")
    oanda_units = OANDABroker().units_per_lot
    broker = PaperBroker(initial_equity=10_000.0)

    for pair in ("EURUSD", "GBPUSD", "USDCAD", "USDJPY"):
        assert broker._units_per_lot(pair) == pytest.approx(oanda_units)
        # Back-compat alias must agree with the new name.
        assert broker._get_multiplier(pair) == pytest.approx(oanda_units)


def test_paper_broker_eurusd_pnl_uses_units_per_lot():
    """0.20 lots over +10 pips at 10k units/lot is ~$1.90 (marked at bid)."""
    broker = PaperBroker(initial_equity=10_000.0)
    broker.update_quote(1.10000, 1.10005, pair="EURUSD")
    broker.market_order("EURUSD", "buy", 0.20)
    broker.update_quote(1.10100, 1.10105, pair="EURUSD")  # +10 pips

    pnl = broker.get_account()["equity"] - 10_000.0
    # (1.10100 - 1.10005) * 10_000 units/lot * 0.20 lots = 1.90
    assert pnl == pytest.approx(1.90, abs=0.02)


def test_paper_broker_usdjpy_pnl_converts_quote_currency():
    """USDJPY P&L is quoted in JPY and must be divided back into USD."""
    broker = PaperBroker(initial_equity=10_000.0)
    broker.update_quote(150.000, 150.002, pair="USDJPY")
    broker.market_order("USDJPY", "buy", 0.20)
    broker.update_quote(150.100, 150.102, pair="USDJPY")  # +10 JPY pips

    pnl = broker.get_account()["equity"] - 10_000.0
    # (150.100 - 150.002) * 10_000 * 0.20 / 150.100 = 1.3058
    assert pnl == pytest.approx(1.31, abs=0.02)
    # The old 1000x "multiplier" reported ~19.60 for this move.
    assert pnl < 5.0


def test_paper_broker_close_position_realises_converted_pnl():
    broker = PaperBroker(initial_equity=10_000.0)
    broker.update_quote(150.000, 150.002, pair="USDJPY")
    broker.market_order("USDJPY", "sell", 0.20)
    broker.update_quote(149.900, 149.902, pair="USDJPY")  # +10 pips in favour

    before = broker.balance
    broker.close_position("USDJPY")
    realized = broker.balance - before
    # Entry at bid 150.000, exit at ask 149.902.
    assert realized == pytest.approx((150.000 - 149.902) * 10_000 * 0.20 / 149.902, rel=1e-6)
    assert broker.get_positions() == {}


def test_paper_broker_default_book_is_passive():
    """Without synthetic mode the legacy static-quote behaviour is preserved."""
    broker = PaperBroker(initial_equity=10_000.0)

    assert broker.synthetic is False
    assert broker.get_candles("EURUSD") is None
    assert broker.get_bid_ask("USDJPY") == (1.10000, 1.10005)

    broker.update_quote(1.23450, 1.23460)
    assert broker.get_bid_ask("EURUSD") == (1.23450, 1.23460)



# ─────────────────────────────────────────────────────────────────────────────
# Synthetic market feed for --broker paper
# ─────────────────────────────────────────────────────────────────────────────


def test_paper_broker_synthetic_feed_produces_warmup_candles():
    broker = PaperBroker(initial_equity=10_000.0, synthetic=True, seed=7)
    candles = broker.get_candles("USDCAD", count=120, granularity="5min")

    assert candles is not None
    assert len(candles) == 120
    assert candles.index.name == "timestamp"
    assert list(candles.columns) == [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "bid_close",
        "ask_close",
    ]
    assert (candles["high"] >= candles["low"]).all()
    assert 1.0 < float(candles["close"].iloc[-1]) < 2.0

    jpy = broker.get_candles("USDJPY", count=120, granularity="5min")
    assert 100.0 < float(jpy["close"].iloc[-1]) < 200.0


def test_paper_broker_synthetic_feed_moves_and_is_per_pair():
    broker = PaperBroker(initial_equity=10_000.0, synthetic=True, seed=3, tick_interval_s=0.0)
    seen = {round(broker.get_bid_ask("EURUSD")[0], 8) for _ in range(25)}

    assert len(seen) > 1, "synthetic feed must move the market"
    for pair, lo, hi in (("EURUSD", 1.0, 1.2), ("GBPUSD", 1.2, 1.4), ("USDJPY", 130.0, 170.0)):
        bid, ask = broker.get_bid_ask(pair)
        assert lo < bid < hi
        assert ask > bid  # positive spread


def test_paper_broker_external_quote_overrides_synthetic():
    broker = PaperBroker(initial_equity=10_000.0, synthetic=True, seed=11)
    broker.get_bid_ask("EURUSD")  # start the walk
    broker.update_quote(1.55550, 1.55560, pair="EURUSD")

    assert broker.get_bid_ask("EURUSD") == (1.55550, 1.55560)


def test_paper_broker_synthetic_quotes_drive_equity():
    """A synthetic-mode position must mark-to-market instead of staying frozen."""
    broker = PaperBroker(
        initial_equity=10_000.0, synthetic=True, seed=5, tick_interval_s=0.0, tick_vol_pips=50.0
    )
    broker.market_order("EURUSD", "buy", 0.20)
    equities = {broker.get_account()["equity"] for _ in range(40)}

    assert len(equities) > 1


# ─────────────────────────────────────────────────────────────────────────────
# RL observation / action-mask contract
# ─────────────────────────────────────────────────────────────────────────────


def _make_env():
    from models.rl_agents import ForexTradingEnv

    return ForexTradingEnv(
        features=np.zeros((50, 4), dtype=np.float32),
        prices=np.full(50, 1.1000, dtype=np.float64),
        atr=np.full(50, 0.001, dtype=np.float64),
        spreads=np.full(50, 0.0001, dtype=np.float64),
        initial_equity=10_000.0,
        max_lots=3.0,
        random_reset=False,
        episode_len=None,
    )


def test_agent_state_shape_and_flat_flag():
    state = build_agent_state(
        position_lots=0.0,
        max_lots=3.0,
        current_price=1.1000,
        entry_price=0.0,
        lot_size=10_000.0,
        holding_bars=0,
        equity=10_000.0,
        initial_equity=10_000.0,
    )

    assert state.shape == (RL_AGENT_STATE_DIM,)
    assert state.dtype == np.float32
    # Key regression: a flat position must report "not open" (the old demo sent
    # a hard-coded 1.0 here, telling the policy a position was always open).
    assert state[4] == 0.0


def test_agent_state_values_match_training_env_bounds():
    state = build_agent_state(
        position_lots=-3.0,
        max_lots=3.0,
        current_price=1.3000,
        entry_price=1.1000,
        lot_size=10_000.0,
        holding_bars=250,
        equity=12_000.0,
        initial_equity=10_000.0,
    )

    assert state[0] == pytest.approx(-1.0)  # position / max_lots, clipped
    # Short above entry => losing. upnl = (1.30 - 1.10) * -3.0 * 10_000 = -6_000,
    # i.e. -0.60 of initial equity, clipped to the trained floor of -0.5.
    assert state[1] == pytest.approx(-0.5)
    assert state[2] == pytest.approx(1.0)  # holding bars capped at 100
    assert state[3] == pytest.approx(0.2)
    assert state[4] == 1.0


def test_agent_state_loss_within_bounds_is_not_clipped():
    """A -0.30 equity loss must pass through unclipped."""
    state = build_agent_state(
        position_lots=-3.0,
        max_lots=3.0,
        current_price=1.2000,
        entry_price=1.1000,
        lot_size=10_000.0,
        holding_bars=0,
        equity=10_000.0,
        initial_equity=10_000.0,
    )

    assert state[1] == pytest.approx(-0.3)


def test_agent_state_short_profit_is_positive():
    """P&L sign must follow direction: a short below entry is in profit."""
    state = build_agent_state(
        position_lots=-0.20,
        max_lots=3.0,
        current_price=1.0900,
        entry_price=1.1000,
        lot_size=10_000.0,
        holding_bars=1,
        equity=10_200.0,
        initial_equity=10_000.0,
    )

    assert state[1] > 0.0


def test_agent_state_matches_env_obs_tail():
    """``ForexTradingEnv._obs()`` must use the shared agent-state block."""
    env = _make_env()
    env.idx = 10
    env.position = -0.5
    env.entry_price = 1.1050
    env.holding = 7
    env.equity = 9_800.0
    env.prices[10] = 1.1020

    obs = env._obs()
    expected = build_agent_state(
        position_lots=-0.5,
        max_lots=3.0,
        current_price=1.1020,
        entry_price=1.1050,
        lot_size=env.lot_size,
        holding_bars=7,
        equity=9_800.0,
        initial_equity=10_000.0,
    )

    assert obs.shape == (4 + RL_AGENT_STATE_DIM,)
    assert np.allclose(obs[-RL_AGENT_STATE_DIM:], expected)


def test_action_mask_matches_env_and_blocks_invalid_actions():
    env = _make_env()

    # Flat: only HOLD/OPEN_LONG/OPEN_SHORT are legal.
    env.position = 0.0
    flat = build_action_mask(env.position, env.max_lots)
    assert flat[:3].all()
    assert not flat[3:].any()
    assert np.array_equal(flat, env.action_mask())

    # Long 1.0 lot: cannot open another long, can still scale in.
    env.position = 1.0
    long_mask = build_action_mask(env.position, env.max_lots)
    assert not long_mask[1]
    assert long_mask[2]
    assert long_mask[3:6].all()
    assert np.array_equal(long_mask, env.action_mask())

    # Short at the cap: cannot open a short or scale in further.
    env.position = -3.0
    capped = build_action_mask(env.position, env.max_lots)
    assert not capped[2]
    assert capped[1]
    assert not capped[3:6].any()
    assert np.array_equal(capped, env.action_mask())


def test_action_mask_length_matches_scaling_action_space():
    from backtesting.backtest import ScalingAction

    assert RL_ACTION_COUNT == int(len(ScalingAction)) == 10
    assert len(build_action_mask(0.0, 3.0)) == 10


# ─────────────────────────────────────────────────────────────────────────────
# Harness wiring
# ─────────────────────────────────────────────────────────────────────────────


def test_demo_harness_uses_broker_lot_convention():
    """The demo must not re-declare its own contract size."""
    from scripts.run_paper_trading_demo import LivePaperTradingSession

    assert LivePaperTradingSession.LOT_SIZE == PaperBroker.UNITS_PER_LOT


def test_demo_observation_size_matches_rl_onnx_contract():
    """2 signal columns + 584 market features + 5 agent-state values = 591."""
    from scripts.run_paper_trading_demo import LivePaperTradingSession

    assert 2 + 584 + RL_AGENT_STATE_DIM == 591
    assert LivePaperTradingSession.MAX_LOTS == 3.0

