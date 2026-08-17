"""
Tests for risk management: session limits, regime sizing, drawdown exit,
Almgren-Chriss executor, portfolio VaR, and Kelly criterion.
"""

from __future__ import annotations

import numpy as np
import pytest

from risk.execution import (
    AlmgrenChrissExecutor,
    DrawdownAwareExitManager,
    PortfolioVaR,
    RegimePositionSizer,
    SessionLimitsEnforcer,
)
from sizing.kelly_criterion import (
    PositionSizer,
    fractional_kelly,
    kelly_binary,
    square_root_impact,
    vol_target_scalar,
)

# ---------------------------------------------------------------------------
# 1. SessionLimitsEnforcer
# ---------------------------------------------------------------------------


class TestSessionLimitsEnforcer:
    @pytest.fixture
    def enforcer(self):
        limits = {
            "asia": {"max_lots": 1.0, "max_open_trades": 3, "hours_utc": (0, 9)},
            "london": {"max_lots": 3.0, "max_open_trades": 5, "hours_utc": (7, 16)},
            "ny": {"max_lots": 3.0, "max_open_trades": 5, "hours_utc": (12, 21)},
            "off": {"max_lots": 0.5, "max_open_trades": 1, "hours_utc": (21, 24)},
        }
        return SessionLimitsEnforcer(session_limits=limits)

    def test_asia_session_classification(self, enforcer):
        # Hours 0-6 are purely asia (before london opens at 7)
        for hour in (0, 3, 6):
            result = enforcer.check(hour, 0.0, 0)
            assert result["session"] == "asia", f"Hour {hour} should be asia"

    def test_london_session_classification(self, enforcer):
        # Pure London after Asia close (09 UTC JST end) and before NY open
        for hour in (10, 11):
            result = enforcer.check(hour, 0.0, 0)
            assert result["session"] == "london", f"Hour {hour} should be london"

    def test_asia_london_overlap(self, enforcer):
        # Summer: Asia still open + London BST open around 07–09 UTC  # noqa: RUF003
        result = enforcer.check(7, 0.0, 0)
        assert result["session"] == "asia_london"

    def test_ny_session_classification(self, enforcer):
        # NY: 12-20, but london is checked first so 12-15 are london
        # Pure NY hours are 16-20
        for hour in (16, 17, 20):
            result = enforcer.check(hour, 0.0, 0)
            assert result["session"] == "ny", f"Hour {hour} should be ny"

    def test_off_session_classification(self, enforcer):
        for hour in (21, 22, 23):
            result = enforcer.check(hour, 0.0, 0)
            assert result["session"] == "off", f"Hour {hour} should be off"

    def test_london_ny_overlap(self, enforcer):
        # Mid-summer DST: London/NY overlap ~13:30–15:30 UTC → policy key london_ny  # noqa: RUF003
        result = enforcer.check(14, 0.0, 0)
        assert result["session"] == "london_ny"

    def test_overlap_alias_maps_to_london_ny(self, enforcer):
        result = enforcer.check(14, 0.0, 0, session="overlap")
        assert result["session"] == "london_ny"
        assert result["max_lots"] == 3.0  # falls back to london limits in fixture

    def test_london_ny_limits_when_configured(self):
        limits = {
            "asia": {"max_lots": 1.0, "max_open_trades": 3},
            "london": {"max_lots": 3.0, "max_open_trades": 5},
            "ny": {"max_lots": 3.0, "max_open_trades": 5},
            "london_ny": {"max_lots": 2.5, "max_open_trades": 4},
            "off": {"max_lots": 0.5, "max_open_trades": 1},
        }
        enf = SessionLimitsEnforcer(session_limits=limits)
        result = enf.check(14, 0.0, 0)
        assert result["session"] == "london_ny"
        assert result["max_lots"] == 2.5
        assert result["max_trades"] == 4

    def test_off_session_no_999_bypass(self):
        """Missing keys must not silently allow 999-lot trading."""
        enf = SessionLimitsEnforcer(session_limits={"off": {"max_lots": 0.0, "max_open_trades": 0}})
        result = enf.check(22, 0.0, 0)
        assert result["allowed"] is False
        assert result["max_lots"] == 0.0

    def test_allows_when_under_limits(self, enforcer):
        result = enforcer.check(10, 1.0, 2)
        assert result["allowed"] is True

    def test_blocks_when_lots_exceeded(self, enforcer):
        result = enforcer.check(10, 3.5, 1)
        assert result["allowed"] is False

    def test_blocks_when_trades_exceeded(self, enforcer):
        result = enforcer.check(10, 0.5, 6)
        assert result["allowed"] is False

    def test_off_session_strict_limits(self, enforcer):
        result = enforcer.check(22, 0.6, 0)
        assert result["allowed"] is False
        assert result["max_lots"] == 0.5

    def test_returns_all_expected_keys(self, enforcer):
        result = enforcer.check(10, 1.0, 2)
        expected = {"allowed", "session", "max_lots", "max_trades", "open_lots", "open_trades"}
        assert set(result.keys()) == expected


# ---------------------------------------------------------------------------
# 2. RegimePositionSizer
# ---------------------------------------------------------------------------


class TestRegimePositionSizer:
    @pytest.fixture
    def sizer(self):
        return RegimePositionSizer(
            base_kelly=0.25,
            max_kelly=0.40,
            min_kelly=0.05,
            corr_crisis_thresh=0.70,
            corr_crisis_scale=0.50,
            hurst_trending=0.60,
            hurst_mean_rev=0.40,
            trending_bonus=1.20,
            mean_rev_penalty=0.75,
        )

    def test_normal_regime(self, sizer):
        ret = np.random.default_rng(1).normal(0, 0.003, 100)
        result = sizer.size(10000, 0.55, 1.8, ret, 0.0005)
        assert result["regime"] == "normal"
        assert result["lots"] > 0

    def test_crisis_regime_scales_down(self, sizer):
        ret = np.random.default_rng(1).normal(0, 0.003, 100)
        normal = sizer.size(10000, 0.55, 1.8, ret, 0.0005, corr_avg=0.3)
        crisis = sizer.size(10000, 0.55, 1.8, ret, 0.0005, corr_avg=0.80)
        assert crisis["regime"] == "crisis"
        assert crisis["lots"] <= normal["lots"]

    def test_trending_regime_scales_up(self, sizer):
        ret = np.random.default_rng(1).normal(0, 0.003, 100)
        normal = sizer.size(10000, 0.55, 1.8, ret, 0.0005, hurst=0.50)
        trending = sizer.size(10000, 0.55, 1.8, ret, 0.0005, hurst=0.70)
        assert trending["regime"] == "trending"
        assert trending["lots"] >= normal["lots"]

    def test_mean_reversion_regime_scales_down(self, sizer):
        ret = np.random.default_rng(1).normal(0, 0.003, 100)
        normal = sizer.size(10000, 0.55, 1.8, ret, 0.0005, hurst=0.50)
        mr = sizer.size(10000, 0.55, 1.8, ret, 0.0005, hurst=0.30)
        assert mr["regime"] == "mean_rev"
        assert mr["lots"] <= normal["lots"]

    def test_kelly_bounded(self, sizer):
        ret = np.random.default_rng(1).normal(0, 0.003, 100)
        result = sizer.size(10000, 0.55, 1.8, ret, 0.0005)
        assert sizer.min_k <= result["kelly"] <= sizer.max_k

    def test_lots_positive_for_positive_edge(self, sizer):
        ret = np.random.default_rng(1).normal(0, 0.003, 100)
        result = sizer.size(10000, 0.60, 2.0, ret, 0.0005)
        assert result["lots"] > 0

    def test_returns_expected_keys(self, sizer):
        ret = np.random.default_rng(1).normal(0, 0.003, 100)
        result = sizer.size(10000, 0.55, 1.8, ret, 0.0005)
        expected = {"lots", "kelly", "regime_scale", "vol_scalar", "regime", "risk_usd"}
        assert set(result.keys()) == expected

    def test_negative_expected_return_not_clamped_to_min_k(self, sizer):
        ret = np.random.default_rng(1).normal(0, 0.003, 100)
        result = sizer.size(10000, 0.30, 1.0, ret, 0.0005)
        assert result["kelly"] == 0.0
        assert result["lots"] == 0.0


# ---------------------------------------------------------------------------
# 3. DrawdownAwareExitManager
# ---------------------------------------------------------------------------


class TestDrawdownAwareExitManager:
    def test_continue_under_soft_threshold(self):
        dm = DrawdownAwareExitManager(soft_dd=0.05, hard_dd=0.10, daily_limit=0.03)
        result = dm.update(10050, 50)
        assert result["action"] == "continue"
        assert result["size_multiplier"] == 1.0

    def test_reduce_at_soft_drawdown(self):
        dm = DrawdownAwareExitManager(soft_dd=0.05, hard_dd=0.10, daily_limit=0.03)
        dm.update(10000, 0)
        dm.update(10100, 100)
        result = dm.update(9500, -600)
        assert result["action"] == "reduce_50"
        assert result["size_multiplier"] == 0.5

    def test_close_all_at_hard_drawdown(self):
        dm = DrawdownAwareExitManager(soft_dd=0.05, hard_dd=0.10, daily_limit=0.03)
        dm.update(10000, 0)
        dm.update(10100, 100)
        result = dm.update(9000, -1100)
        assert result["action"] == "close_all"
        assert result["size_multiplier"] == 0.0

    def test_halt_after_hard_drawdown(self):
        dm = DrawdownAwareExitManager(
            soft_dd=0.05,
            hard_dd=0.10,
            daily_limit=0.03,
            rec_bars=5,
        )
        dm.update(10000, 0)
        dm.update(10100, 100)
        dm.update(9000, -1100)
        result = dm.update(9000, 0)
        assert result["action"] == "halt"
        assert result["size_multiplier"] == 0.0

    def test_consecutive_losses_reduce_size(self):
        dm = DrawdownAwareExitManager(
            soft_dd=0.50,
            hard_dd=0.90,
            daily_limit=0.90,
            max_cons=3,
        )
        eq = 10000
        for _ in range(4):
            eq -= 10
            dm.update(eq, -10)
        result = dm.update(eq - 10, -10)
        assert result["consec_losses"] >= 5
        assert result["size_multiplier"] < 1.0

    def test_new_day_resets_counters(self):
        dm = DrawdownAwareExitManager(soft_dd=0.05, hard_dd=0.10, daily_limit=0.03)
        dm.update(10000, 0)
        dm.update(9800, -200)
        dm.new_day()
        result = dm.update(9850, 50)
        assert result["consec_losses"] == 0

    def test_status_reports_correctly(self):
        dm = DrawdownAwareExitManager()
        dm.update(10000, 0)
        dm.update(9500, -500)
        status = dm.status()
        assert "equity" in status
        assert "drawdown" in status
        assert status["drawdown"] > 0


# ---------------------------------------------------------------------------
# 4. AlmgrenChrissExecutor
# ---------------------------------------------------------------------------


class TestAlmgrenChrissExecutor:
    def test_schedule_sums_to_total(self):
        ac = AlmgrenChrissExecutor()
        sched = ac.optimal_schedule(3.0, n_slices=10)
        assert len(sched) == 10
        assert abs(sched.sum() - 3.0) < 0.01

    def test_single_slice_equals_total(self):
        ac = AlmgrenChrissExecutor()
        sched = ac.optimal_schedule(2.5, n_slices=1)
        assert len(sched) == 1
        assert abs(sched[0] - 2.5) < 0.01

    def test_impact_cost_increases_with_size(self):
        ac = AlmgrenChrissExecutor()
        small = ac.estimate_impact_cost(0.5)
        large = ac.estimate_impact_cost(5.0)
        assert large["impact_pips"] > small["impact_pips"]

    def test_should_split_large_order(self):
        ac = AlmgrenChrissExecutor()
        should, n = ac.should_split(3.0)
        assert should is True
        assert n > 1

    def test_should_not_split_tiny_normal(self):
        ac = AlmgrenChrissExecutor()
        should, _ = ac.should_split(0.3, urgency="normal")
        assert should is False


# ---------------------------------------------------------------------------
# 5. PortfolioVaR
# ---------------------------------------------------------------------------


class TestPortfolioVaR:
    @pytest.fixture
    def var_calc(self):
        pv = PortfolioVaR(confidence=0.99)
        rng = np.random.default_rng(42)
        for _ in range(200):
            pv.update_returns("EURUSD", rng.normal(0, 0.0003))
            pv.update_returns("GBPUSD", rng.normal(0, 0.0004))
        return pv

    def test_var_positive_with_positions(self, var_calc):
        result = var_calc.parametric_var(
            {"EURUSD": 1.0, "GBPUSD": 0.8},
            10000,
        )
        assert result["var_pct"] > 0
        assert result["var_usd"] > 0

    def test_var_zero_with_no_positions(self, var_calc):
        result = var_calc.parametric_var({}, 10000)
        assert result["var_pct"] == 0.0

    def test_cvar_exceeds_var(self, var_calc):
        result = var_calc.parametric_var(
            {"EURUSD": 1.0, "GBPUSD": 0.8},
            10000,
        )
        assert result["cvar_usd"] >= result["var_usd"]

    def test_max_allowed_lots_positive(self, var_calc):
        lots = var_calc.max_allowed_lots("EURUSD", 10000, {"EURUSD": 0.0})
        assert lots > 0

    def test_correlation_in_range(self, var_calc):
        result = var_calc.parametric_var(
            {"EURUSD": 1.0, "GBPUSD": 0.8},
            10000,
        )
        assert -1.0 <= result["correlation_avg"] <= 1.0


# ---------------------------------------------------------------------------
# 6. Kelly Criterion functions
# ---------------------------------------------------------------------------


class TestKellyCriterion:
    def test_kelly_binary_positive_edge(self):
        k = kelly_binary(0.55, 1.5)
        assert k > 0

    def test_kelly_binary_negative_edge(self):
        k = kelly_binary(0.30, 1.0)
        assert k < 0

    def test_kelly_binary_fair_game(self):
        k = kelly_binary(0.50, 1.0)
        assert abs(k) < 1e-10

    def test_fractional_kelly_clips(self):
        full = kelly_binary(0.90, 10.0)
        frac = fractional_kelly(full, 0.25)
        assert 0 <= frac <= 1

    def test_fractional_kelly_quarter(self):
        frac = fractional_kelly(0.20, 0.25)
        assert abs(frac - 0.05) < 1e-6

    def test_vol_target_scalar_neutral_at_target(self):
        returns = np.random.default_rng(1).normal(0, 0.10 / np.sqrt(252), 100)
        scalar = vol_target_scalar(returns, target_vol=0.10)
        assert 0.5 < scalar < 2.0

    def test_vol_target_scalar_scales_down_high_vol(self):
        high_vol = np.random.default_rng(1).normal(0, 0.005, 100)
        low_vol = np.random.default_rng(1).normal(0, 0.0001, 100)
        s_high = vol_target_scalar(high_vol)
        s_low = vol_target_scalar(low_vol)
        assert s_high < s_low

    def test_vol_target_scalar_short_series(self):
        scalar = vol_target_scalar(np.array([0.01]))
        assert scalar == 1.0

    def test_square_root_impact_increases_with_size(self):
        small = square_root_impact(0.1)
        large = square_root_impact(5.0)
        assert large > small

    def test_square_root_impact_non_negative(self):
        assert square_root_impact(1.0) >= 0


# ---------------------------------------------------------------------------
# 7. PositionSizer (legacy adapter)
# ---------------------------------------------------------------------------


class TestPositionSizer:
    def test_size_position_returns_expected_keys(self):
        ps = PositionSizer(equity=10000)
        ret = np.random.default_rng(1).normal(0, 0.003, 100).tolist()
        result = ps.size_position(0.55, 1.5, ret, 1.1000, 0.0005)
        expected = {"lots", "full_kelly", "frac_kelly", "vol_scalar", "risk_usd", "impact_usd"}
        assert set(result.keys()) == expected

    def test_lots_positive_with_edge(self):
        ps = PositionSizer(equity=10000)
        ret = np.random.default_rng(1).normal(0, 0.003, 100).tolist()
        result = ps.size_position(0.55, 1.5, ret, 1.1000, 0.0005)
        assert result["lots"] > 0

    def test_impact_non_negative(self):
        ps = PositionSizer(equity=10000)
        ret = np.random.default_rng(1).normal(0, 0.003, 100).tolist()
        result = ps.size_position(0.55, 1.5, ret, 1.1000, 0.0005)
        assert result["impact_usd"] >= 0
