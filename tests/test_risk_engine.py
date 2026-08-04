"""
Tests for risk.risk_engine (Improvement #16): pre-trade checks, post-trade
monitoring, VaR/CVaR, circuit breakers, and audit log.
"""
from __future__ import annotations

import numpy as np
import pytest

from risk.risk_engine import RiskConfig, RiskDecision, RiskEngine


@pytest.fixture
def engine():
    return RiskEngine(equity=10_000.0)


# ═════════════════════════════════════════════════════════════════════════════
# Pre-trade checks
# ═════════════════════════════════════════════════════════════════════════════

def test_check_order_allowed(engine):
    d = engine.check_order(pair="EURUSD", lots=0.5, price=1.10, position_size_pct=0.01)
    assert d.allowed is True
    assert d.action == "ok"


def test_check_order_rejects_position_size(engine):
    d = engine.check_order(pair="EURUSD", lots=5.0, price=1.10, position_size_pct=0.5)
    assert d.allowed is False
    assert d.rule == "max_position_pct"
    assert d.action == "reject"


def test_check_order_rejects_lots(engine):
    d = engine.check_order(pair="EURUSD", lots=50.0, price=1.10, position_size_pct=0.01)
    assert d.allowed is False
    assert d.rule == "max_total_lots"


def test_check_order_rejects_notional(engine):
    cfg = RiskConfig(max_notional_usd=1_000.0)
    eng = RiskEngine(equity=10_000.0, cfg=cfg)
    d = eng.check_order(pair="EURUSD", lots=0.5, price=1.10, position_size_pct=0.01)
    assert d.allowed is False
    assert d.rule == "max_notional_usd"


def test_check_order_rejects_concentration(engine):
    cfg = RiskConfig(max_instrument_concentration=0.30)
    eng = RiskEngine(equity=10_000.0, cfg=cfg)
    eng.open_position("EURUSD", lots=0.5, entry_price=1.10)
    d = eng.check_order(pair="EURUSD", lots=0.5, price=1.10, position_size_pct=0.01)
    assert d.allowed is False
    assert d.rule == "concentration"


def test_order_frequency_cap(engine):
    cfg = RiskConfig(max_order_freq_per_min=3)
    eng = RiskEngine(equity=10_000.0, cfg=cfg)
    for _ in range(3):
        assert eng.check_order(pair="EURUSD", lots=0.5, price=1.10, position_size_pct=0.01).allowed
    d = eng.check_order(pair="EURUSD", lots=0.5, price=1.10, position_size_pct=0.01)
    assert d.allowed is False
    assert d.rule == "max_order_freq"


# ═════════════════════════════════════════════════════════════════════════════
# Post-trade monitoring + circuit breakers
# ═════════════════════════════════════════════════════════════════════════════

def test_consecutive_losses_trigger(engine):
    engine._returns.setdefault("EURUSD", __import__("collections").deque())
    for i in range(engine.cfg.max_consecutive_losses):
        d = engine.on_trade_closed(pnl=-20.0, equity=10_000.0 - (i + 1) * 20.0)
    assert d is not None
    assert d.rule == "max_consecutive_losses"


def test_drawdown_circuit_breaker(engine):
    engine.peak_equity = 10_000.0
    mon = engine.update_equity(equity=8_900.0)  # 11 % drawdown
    assert mon["circuit_breaker"] is True
    assert "max_drawdown_halt" in mon["breach_reasons"]
    assert mon["decision"]["action"] == "flatten"


def test_daily_loss_circuit_breaker(engine):
    engine.day_realized_pnl = -800.0  # 8 % of 10k
    mon = engine.update_equity(equity=9_200.0)
    assert mon["circuit_breaker"] is True
    assert "daily_loss_limit" in mon["breach_reasons"]


def test_halted_blocks_new_orders(engine):
    engine.update_equity(equity=8_900.0)
    d = engine.check_order(pair="EURUSD", lots=0.5, price=1.10, position_size_pct=0.01)
    assert d.allowed is False
    assert d.rule == "circuit_breaker"
    assert d.action == "standby"


def test_soft_drawdown_reduce(engine):
    engine.update_equity(equity=9_480.0)  # 5.2 % drawdown
    mon = engine.update_equity(equity=9_480.0)
    assert mon["circuit_breaker"] is False
    assert mon["soft_reduce"] is True


def test_flatten(engine):
    engine.open_position("EURUSD", lots=0.5, entry_price=1.10)
    engine.open_position("USDJPY", lots=0.3, entry_price=150.0)
    decision = engine.flatten()
    assert decision["action"] == "flatten"
    assert decision["details"]["flattened_pairs"] == ["EURUSD", "USDJPY"]
    assert engine.positions == {}


def test_resume_clears_state(engine):
    engine.update_equity(equity=8_900.0)
    assert engine._halted
    engine.resume()
    assert not engine._halted
    assert engine.day_realized_pnl == 0.0
    assert engine.consecutive_losses == 0


# ═════════════════════════════════════════════════════════════════════════════
# VaR / CVaR / exposure
# ═════════════════════════════════════════════════════════════════════════════

def test_historical_var(engine):
    rng = np.random.default_rng(0)
    for _ in range(300):
        engine._portfolio_returns.append(float(rng.normal(0.0001, 0.005)))
    res = engine.historical_var()
    assert 0.0 <= res["var_pct"] < 0.1
    assert res["n_obs"] == 300


def test_parametric_var_known(engine):
    # known: z(0.99)=2.326, sd=0.01, mu=0 -> var=2.326%
    for _ in range(500):
        engine._portfolio_returns.append(float(np.random.normal(0.0, 0.01)))
    res = engine.parametric_var(confidence=0.99)
    assert abs(res["var_pct"] - 0.02326) < 0.004


def test_portfolio_var_usd(engine):
    rng = np.random.default_rng(1)
    for _ in range(300):
        engine._portfolio_returns.append(float(rng.normal(0.0001, 0.005)))
    res = engine.portfolio_var()
    assert "var_usd" in res and "cvar_usd" in res
    assert res["var_usd"] > 0


def test_exposure_by_currency(engine):
    engine.open_position("EURUSD", lots=1.0, entry_price=1.10)
    engine.open_position("USDJPY", lots=1.0, entry_price=150.0)
    ex = engine.exposure_by_currency()
    assert ex["EUR"] == pytest.approx(100_000.0)
    assert ex["JPY"] == pytest.approx(-100_000.0)


def test_gap_flag(engine):
    engine._returns["EURUSD"] = __import__("collections").deque([0.001, 0.05], maxlen=500)
    gap = engine._check_gaps()
    assert gap["EURUSD"] is True


# ═════════════════════════════════════════════════════════════════════════════
# Audit log
# ═════════════════════════════════════════════════════════════════════════════

def test_audit_log_records_decisions(engine):
    engine.check_order(pair="EURUSD", lots=0.5, price=1.10, position_size_pct=0.01)
    engine.check_order(pair="EURUSD", lots=50.0, price=1.10, position_size_pct=0.5)
    log = engine.get_audit()
    assert len(log) >= 2
    rules = {e["rule"] for e in log}
    assert "max_total_lots" in rules
    assert all("ts" in e and "rule" in e and "action" in e for e in log)


def test_audit_filter_by_rule(engine):
    engine.check_order(pair="EURUSD", lots=50.0, price=1.10, position_size_pct=0.5)
    engine.check_order(pair="EURUSD", lots=0.5, price=1.10, position_size_pct=0.01)
    filtered = engine.get_audit(rule="max_total_lots")
    assert len(filtered) == 1
    assert filtered[0]["rule"] == "max_total_lots"


def test_risk_engine_new_day_resets_daily_pnl(engine):
    engine.on_trade_closed(pnl=-50.0, equity=9_950.0, pair="EURUSD", lots=0.1)
    assert engine.day_realized_pnl < 0
    engine.new_day(equity=9_950.0)
    assert engine.day_realized_pnl == 0.0
    assert engine.consecutive_losses == 0


def test_risk_decision_to_audit_shape():
    d = RiskDecision(False, "test_rule", 1.5, 1.0, action="reject", reason="nope")
    entry = d.to_audit(ts="2026-01-01T00:00:00Z")
    assert entry["ts"] == "2026-01-01T00:00:00Z"
    assert entry["value"] == 1.5
    assert entry["limit"] == 1.0
    assert entry["allowed"] is False
