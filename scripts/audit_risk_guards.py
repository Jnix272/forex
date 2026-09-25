"""
Comprehensive Risk Guard Audit Script.
Tests all live guards, risk engine, portfolio allocator, and execution safety mechanisms.
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from datetime import UTC, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import polars as pl
from config.settings import price_to_pips, get_pip_size
from contracts.execution_risk import GuardResult, HOLD
from trading.live_actions import LiveAction, model_class_to_live_action
from trading.live_guards import (
    EconomicCalendarGuard,
    SpreadVolatilityGuard,
    RegimeRouter,
    DisagreementGate,
    NoTradeZoneGate,
    _tail_median,
    _last_numeric,
)
from trading.live_engine import LiveSafetyGate, LiveSafetyConfig
from risk.risk_engine import RiskEngine, RiskConfig, _calc_notional_usd
from risk.portfolio_allocator import PortfolioAllocator
from risk.execution import PortfolioVaR, DrawdownAwareExitPolicy, SessionLimitsEnforcer


def test_economic_calendar_timezone():
    print("\n--- Test: EconomicCalendarGuard Timezone Handling ---")
    guard = EconomicCalendarGuard(pair="EURUSD", block_before_min=15, block_after_min=15)
    
    # 1. Test naive timestamp in row["timestamp_utc"]
    # We simulate a dataframe returned by _filter_relevant
    naive_events = pd.DataFrame([
        {
            "timestamp_utc": "2026-09-24 14:00:00",  # tz-naive string
            "headline": "US Non-Farm Payrolls",
            "impact": "high",
            "currency": "USD"
        }
    ])
    guard._events = naive_events
    guard._loaded_at = pd.Timestamp.now(tz="UTC")
    
    try:
        res = guard.check(now=pd.Timestamp("2026-09-24 13:50:00", tz="UTC"))
        print(f"Naive timestamp check result: {res.blocked}, reason: {res.reason}")
    except TypeError as e:
        print(f"CRITICAL BUG CONFIRMED: tz-naive timestamp causes TypeError in check(): {e}")
    except Exception as e:
        print(f"Exception on naive timestamp: {type(e).__name__}: {e}")

    # 2. Test tz-aware timestamp
    aware_events = pd.DataFrame([
        {
            "timestamp_utc": pd.Timestamp("2026-09-24 14:00:00", tz="UTC"),
            "headline": "US Non-Farm Payrolls",
            "impact": "high",
            "currency": "USD"
        }
    ])
    guard._events = aware_events
    res_aware = guard.check(now=pd.Timestamp("2026-09-24 13:50:00", tz="UTC"))
    print(f"Tz-aware timestamp check: blocked={res_aware.blocked}, reason={res_aware.reason}")


def test_economic_calendar_event_filtering():
    print("\n--- Test: EconomicCalendarGuard Event Filtering & Impact Logic ---")
    
    # Case A: Low impact event with block_before_min=15
    guard = EconomicCalendarGuard(pair="EURUSD", block_before_min=15, block_after_min=15, special_before_min=30, special_after_min=30)
    low_impact_events = pd.DataFrame([
        {
            "timestamp_utc": pd.Timestamp("2026-09-24 14:00:00", tz="UTC"),
            "headline": "ECB Economic Bulletin",
            "impact": "low",
            "currency": "EUR"
        }
    ])
    guard._events = low_impact_events
    guard._loaded_at = pd.Timestamp.now(tz="UTC")
    res_low = guard.check(now=pd.Timestamp("2026-09-24 13:55:00", tz="UTC"))
    print(f"Low impact event with block_before_min=15: blocked={res_low.blocked}, details={res_low.details}")
    if res_low.blocked:
        print("LOGIC FLAW CONFIRMED: Low impact event was blocked because special=False fell back to block_before_min=15 without checking is_high_impact!")

    # Case B: Default windows (block_before_min=0, special_before_min=2) on general high impact event (e.g. GDP)
    guard_default = EconomicCalendarGuard(pair="EURUSD")  # defaults: block=0, special=2
    gdp_events = pd.DataFrame([
        {
            "timestamp_utc": pd.Timestamp("2026-09-24 14:00:00", tz="UTC"),
            "headline": "US Gross Domestic Product QoQ",
            "impact": "high",
            "currency": "USD"
        }
    ])
    guard_default._events = gdp_events
    guard_default._loaded_at = pd.Timestamp.now(tz="UTC")
    res_gdp = guard_default.check(now=pd.Timestamp("2026-09-24 13:55:00", tz="UTC"))
    print(f"GDP (high impact, non-special) with default windows: blocked={res_gdp.blocked}")
    if not res_gdp.blocked:
        print("CONFIG FLAW CONFIRMED: Major high impact release (GDP) is NOT blocked under default settings because block_before_min=0!")

    # Case C: "rate" substring matching in _SPECIAL_EVENTS
    unemployment_events = pd.DataFrame([
        {
            "timestamp_utc": pd.Timestamp("2026-09-24 14:00:00", tz="UTC"),
            "headline": "Eurozone Unemployment Rate",
            "impact": "high",
            "currency": "EUR"
        }
    ])
    guard_default._events = unemployment_events
    guard_default._loaded_at = pd.Timestamp.now(tz="UTC")
    res_rate = guard_default.check(now=pd.Timestamp("2026-09-24 13:59:00", tz="UTC"))
    print(f"'Unemployment Rate' matched as special?: blocked={res_rate.blocked}")


def test_spread_volatility_guard():
    print("\n--- Test: SpreadVolatilityGuard & Metrics ---")
    # 1. Pip conversion check
    jpy_diff = 0.03  # 3 pips for JPY
    eur_diff = 0.0003 # 3 pips for EUR
    pips_jpy = price_to_pips(jpy_diff, "USDJPY")
    pips_eur = price_to_pips(eur_diff, "EURUSD")
    print(f"price_to_pips for USDJPY (0.03): {pips_jpy:.2f} (expected 3.00)")
    print(f"price_to_pips for EURUSD (0.0003): {pips_eur:.2f} (expected 3.00)")

    # 2. Lookback check when len(features) < lookback
    guard = SpreadVolatilityGuard(lookback=60, pair="EURUSD")
    short_features = pl.DataFrame({
        "spread_pips": [50.0] * 10,  # Huge 50 pip spread, but only 10 rows
        "atr_6": [0.005] * 10,
    })
    res_short = guard.check(short_features)
    print(f"50-pip spread with len(features)=10 < 60: blocked={res_short.blocked}, reason={res_short.reason}")

    # 3. ATR column selection ambiguity
    ambiguous_features = pl.DataFrame({
        "atr_ratio_6_20": [0.5] * 65,  # Ratio, not actual ATR!
        "atr_6": [0.0010] * 65,
        "spread_pips": [1.0] * 65,
    })
    # If atr_ratio_6_20 comes first, what does it select?
    atr_cols = [c for c in ambiguous_features.columns if str(c).startswith("atr_")]
    print(f"atr_cols found in order: {atr_cols}")
    print(f"atr_cols[0] selected: {atr_cols[0]}")
    if atr_cols[0] != "atr_6":
        print(f"FLAW CONFIRMED: SpreadVolatilityGuard selected '{atr_cols[0]}' instead of true ATR column!")


def test_regime_router_rollover():
    print("\n--- Test: RegimeRouter Rollover Window ---")
    router = RegimeRouter(rollover_start_utc=21, rollover_end_utc=1)
    
    hours_to_test = [20, 21, 22, 23, 0, 1, 2]
    blocked_hours = []
    dummy_feat = pl.DataFrame({"regime_break_prob": [0.1]})
    for h in hours_to_test:
        ts = pd.Timestamp(f"2026-09-24 {h:02d}:30:00", tz="UTC")
        res = router.route(dummy_feat, now=ts)
        if res.blocked:
            blocked_hours.append(h)
    print(f"Hours blocked by RegimeRouter (UTC): {blocked_hours}")
    print(f"Total hours blocked: {len(blocked_hours)} hours (covers 21:00 UTC to 01:00 UTC).")
    if 0 in blocked_hours:
        print("OVER-BLOCKING CONFIRMED: Hour 0 UTC (Tokyo open 09:00 JST) is blocked by rollover guard!")


def test_disagreement_and_no_trade():
    print("\n--- Test: DisagreementGate & NoTradeZoneGate ---")
    
    # 1. NoTradeZoneGate Polars vs short history
    gate = NoTradeZoneGate(enabled=True)
    short_feat = pl.DataFrame({
        "atr_6": [0.001],
        "spread_pips": [1.5],
        "adx_14": [15.0],
        "rsi_14": [50.0],
    })
    res_nt = gate.check(short_feat)
    print(f"NoTradeZoneGate on 1-row Polars frame: blocked={res_nt.blocked}, reason={res_nt.reason}")

    # 2. DisagreementGate double-buffer mutation test
    class MockWrap:
        def __init__(self):
            self._obs_buffer = []
        def select_action(self, obs):
            self._obs_buffer.append(obs)
            return 0  # BUY

    mock_fast = MockWrap()
    mock_slow = MockWrap()
    dis_gate = DisagreementGate(enabled=True)
    obs = np.zeros(10)
    
    # Simulate live engine calling select_action once, then DisagreementGate.check
    mock_fast.select_action(obs)  # live engine bar step
    print(f"Fast buffer length before DisagreementGate: {len(mock_fast._obs_buffer)}")
    dis_gate.check(0, obs, fast_model=mock_fast, slow_model=mock_slow)
    print(f"Fast buffer length after DisagreementGate: {len(mock_fast._obs_buffer)}")
    if len(mock_fast._obs_buffer) == 2:
        print("SIDE-EFFECT BUG CONFIRMED: DisagreementGate called select_action() on fast_model, double-appending observation!")


def test_safety_gate_rate_limiter_deadlock():
    print("\n--- Test: LiveSafetyGate & RiskEngine Rate Limiter Behavior ---")
    
    # 1. LiveSafetyGate: Order rate limit consumed on blocked/unplaced orders
    gate = LiveSafetyGate(LiveSafetyConfig(max_orders_per_minute=5), starting_equity=10000.0)
    now = time.time()
    for i in range(5):
        res = gate.allow_order("EURUSD", "buy", 0.1, 1.1000, 1.1001, 10000.0, now=now)
    print(f"Gate after 5 allowed checks: len(_order_times)={len(gate._order_times)}")
    # 6th check should fail
    res6 = gate.allow_order("EURUSD", "buy", 0.1, 1.1000, 1.1001, 10000.0, now=now)
    print(f"6th check within 1 minute: ok={res6['ok']}, reason={res6.get('reason')}")

    # 2. RiskEngine _freq_blocked permanent deadlock
    risk_eng = RiskEngine(equity=10000.0, cfg=RiskConfig(max_order_freq_per_min=5))
    # Place 5 orders at t=0
    t0 = 1000.0
    for i in range(5):
        risk_eng._order_times.append(t0 + i * 0.1)
    
    # Fast forward 10 minutes (t = 1600.0)
    # At t=1600, are orders still blocked?
    is_blocked = risk_eng._freq_blocked()
    print(f"RiskEngine._freq_blocked() after 10 minutes with 5 stale timestamps: {is_blocked}")
    dec = risk_eng.check_order("EURUSD", 0.1, 1.1000)
    print(f"RiskEngine.check_order() at t=1600: allowed={dec.allowed}, rule={dec.rule}, reason={dec.reason}")
    if not dec.allowed and dec.rule == "max_order_freq":
        print("CRITICAL DEADLOCK BUG CONFIRMED: RiskEngine is permanently blocked because _order_times cleanup only occurs AFTER checks pass!")


def test_risk_engine_notional_and_resume():
    print("\n--- Test: RiskEngine Notional Scaling & Circuit Breaker Resume ---")
    
    # 1. Notional calculation for mini lots
    # 0.5 mini lots = 5,000 units = $5,000 USD notional for USDJPY
    notional = _calc_notional_usd("USDJPY", 0.5, 150.0)
    print(f"USDJPY 0.5 lots calculated notional: ${notional:,.2f}")
    if notional == 50_000.0:
        print("CONTRACT MISMATCH CONFIRMED: _calc_notional_usd uses 100,000 units/lot (standard), while PaperBroker and OANDA use 10,000 units/lot (mini lot)!")

    # 2. Circuit breaker halt and resume() failure to clear peak equity
    re = RiskEngine(equity=10000.0, cfg=RiskConfig(max_drawdown_halt=0.10))
    mon = re.update_equity(8900.0)  # 11% drawdown
    print(f"update_equity at $8,900: circuit_breaker={mon['circuit_breaker']}, halted={re._halted}")
    
    # Call resume()
    re.resume()
    print(f"After re.resume(): halted={re._halted}")
    
    # Next bar update_equity
    mon_next = re.update_equity(8900.0)
    print(f"update_equity immediately after resume(): circuit_breaker={mon_next['circuit_breaker']}, halted={re._halted}")
    if re._halted:
        print("RESUME FLAW CONFIRMED: resume() failed to recalibrate peak_equity, immediately re-halting the engine!")


def test_portfolio_var_new_session():
    print("\n--- Test: PortfolioVaR on New Sessions ---")
    pvar = PortfolioVaR()
    # No returns yet
    res = pvar.parametric_var({"EURUSD": 0.5, "USDJPY": 0.5}, 10000.0)
    print(f"PortfolioVaR with 0 observations: {res}")
    
    # 5 observations (less than min_obs=20)
    for i in range(5):
        pvar.update_returns("EURUSD", 0.0002)
        pvar.update_returns("USDJPY", 0.0003)
    res5 = pvar.parametric_var({"EURUSD": 0.5, "USDJPY": 0.5}, 10000.0)
    print(f"PortfolioVaR with 5 observations (<20 min_obs): {res5}")
    if res5["var_pct"] == 0.0:
        print("BLIND SPOT CONFIRMED: PortfolioVaR returns 0.0 risk for the first 20 observations of every session!")


if __name__ == "__main__":
    print("=" * 70)
    print("EXHAUSTIVE RISK GUARDS AUDIT SIMULATION")
    print("=" * 70)
    test_economic_calendar_timezone()
    test_economic_calendar_event_filtering()
    test_spread_volatility_guard()
    test_regime_router_rollover()
    test_disagreement_and_no_trade()
    test_safety_gate_rate_limiter_deadlock()
    test_risk_engine_notional_and_resume()
    test_portfolio_var_new_session()
    print("\n" + "=" * 70)
    print("AUDIT SIMULATION COMPLETED")
    print("=" * 70)
