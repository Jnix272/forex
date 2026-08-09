"""Focused tests for session SoT (P1), live enforcer wiring (P3), spread/slip names (P4)."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from risk.execution import SessionLimitsEnforcer
from trading.session_utils import (
    classify_session,
    default_session_slip_factors,
    normalize_session_name,
    session_spread_mult,
)


class TestSessionSoT:
    def test_normalize_legacy_aliases(self):
        assert normalize_session_name("overlap") == "london_ny"
        assert normalize_session_name("tokyo") == "asia"
        assert normalize_session_name("sydney") == "asia"
        assert normalize_session_name("overnight") == "off"

    def test_classify_london_ny_summer(self):
        # 14:00 UTC on 2026-08-06 — London BST + NY EDT overlap
        info = classify_session(datetime(2026, 8, 6, 14, 0, tzinfo=UTC))
        assert info.london_ny is True
        assert info.policy_key == "london_ny"

    def test_classify_asia_winter_morning(self):
        info = classify_session(datetime(2026, 1, 15, 3, 0, tzinfo=UTC))
        assert info.primary == "asia"
        assert info.policy_key == "asia"

    def test_session_spread_mult_ordering(self):
        assert session_spread_mult("london_ny") < session_spread_mult("london")
        assert session_spread_mult("off") > session_spread_mult("asia")

    def test_slip_factors_are_production_keys(self):
        factors = default_session_slip_factors()
        assert "london_ny" in factors
        assert factors["london_ny"] == pytest.approx(1.0)
        assert "tokyo" not in factors
        assert "overnight" not in factors


class TestSessionLimitsLiveContract:
    def test_enforcer_uses_london_ny_not_overlap(self):
        enf = SessionLimitsEnforcer(session_limits={
            "london": {"max_lots": 3.0, "max_open_trades": 5},
            "london_ny": {"max_lots": 2.0, "max_open_trades": 4},
            "off": {"max_lots": 0.0, "max_open_trades": 0},
        })
        r = enf.check(now=datetime(2026, 8, 6, 14, 0, tzinfo=UTC), open_lots=0.0, open_trades=0)
        assert r["session"] == "london_ny"
        assert r["max_lots"] == 2.0

    def test_live_engine_wires_session_limits(self):
        import inspect

        import trading.live_engine as le

        src = inspect.getsource(le.LiveTradingEngine)
        assert "SessionLimitsEnforcer" in inspect.getsource(le)
        assert "session_limits" in src
        assert "session_limits.check" in src or "self.session_limits.check" in src


class TestSlippageNameUnify:
    def test_calibrator_maps_tokyo_to_asia(self):
        from backtesting.improvements import SlippageCalibrator

        cal = SlippageCalibrator()
        assert "asia" in cal.session_factors
        assert "tokyo" not in cal.session_factors
        assert cal.predict(1.0, session="tokyo") == cal.predict(1.0, session="asia")
        assert cal.predict(1.0, session="overnight") == cal.predict(1.0, session="off")
