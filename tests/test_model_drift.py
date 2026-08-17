"""
Tests for drift.model_drift (Improvement #5): champion-challenger harness,
canary rollout, automated rollback, and the orchestrator.
"""

from __future__ import annotations

import numpy as np
import pytest

from drift.model_drift import (
    AutomatedRollbackMonitor,
    CanaryRollout,
    ChampionChallengerHarness,
    ModelStats,
    RollbackConfig,
    run_model_drift_check,
)


@pytest.fixture
def good_trades():
    rng = np.random.default_rng(0)
    return rng.normal(50.0, 30.0, 200)


@pytest.fixture
def bad_trades():
    rng = np.random.default_rng(1)
    return rng.normal(-30.0, 30.0, 200)


# ═════════════════════════════════════════════════════════════════════════════
# ModelStats
# ═════════════════════════════════════════════════════════════════════════════


def test_model_stats_rolls_window():
    s = ModelStats("m", maxlen=10)
    for i in range(30):
        s.record_trade(1.0, equity=100.0 + i)
    assert len(s.pnls) == 10
    assert s.n_trades == 30


def test_model_stats_summary_shape():
    s = ModelStats("m", maxlen=100)
    for i in range(50):
        s.record_trade(float(i % 2 * 10 - 5), equity=1000.0 + i)
        s.record_error(i % 3 == 0)
    summ = s.summary()
    assert set(summ) == {"n_trades", "sharpe", "win_rate", "max_drawdown", "error_rate"}
    assert summ["n_trades"] == 50
    assert 0.0 <= summ["error_rate"] <= 1.0


def test_model_stats_error_rate():
    s = ModelStats("m")
    for i in range(100):
        s.record_error(i < 25)
    assert s.error_rate == pytest.approx(0.25)


def test_model_stats_psr_bounds():
    s = ModelStats("m")
    s.record_trade(1.0)
    s.record_trade(1.0)
    s.record_trade(1.0)
    s.record_trade(1.0)
    s.record_trade(1.0)
    assert 0.0 <= s.psr() <= 1.0


# ═════════════════════════════════════════════════════════════════════════════
# Champion-challenger harness
# ═════════════════════════════════════════════════════════════════════════════


def test_harness_challenger_wins(good_trades, bad_trades):
    h = ChampionChallengerHarness("champ", ["chal"])
    for pnl in good_trades:
        h.record_trade("champ", pnl)
    for pnl in bad_trades:
        h.record_trade("chal", pnl)
    cmp = h.compare("chal")
    assert cmp["ready"] is True
    assert cmp["beats_champion"] is False  # challenger is worse


def test_harness_challenger_loses(good_trades, bad_trades):
    h = ChampionChallengerHarness("champ", ["chal"])
    for pnl in bad_trades:
        h.record_trade("champ", pnl)
    for pnl in good_trades:
        h.record_trade("chal", pnl)
    cmp = h.compare("chal")
    assert cmp["ready"] is True
    assert cmp["beats_champion"] is True


def test_harness_not_ready():
    h = ChampionChallengerHarness("champ", ["chal"])
    h.record_trade("champ", 1.0)
    h.record_trade("chal", 1.0)
    cmp = h.compare("chal")
    assert cmp["ready"] is False
    assert cmp["gate"] is None


def test_harness_add_challenger_dynamic(good_trades):
    h = ChampionChallengerHarness("champ")
    h.add_challenger("new")
    for pnl in good_trades:
        h.record_trade("champ", pnl)
        h.record_trade("new", pnl)
    assert "new" in h.challengers
    assert h.stats("new")["new"]["n_trades"] == 200


def test_harness_errors_tracked():
    h = ChampionChallengerHarness("champ", ["chal"])
    for i in range(100):
        h.record_error("champ", i < 10)
        h.record_error("chal", i >= 90)
    stats = h.stats()
    assert stats["champ"]["error_rate"] == pytest.approx(0.10)
    assert stats["chal"]["error_rate"] == pytest.approx(0.10)


# ═════════════════════════════════════════════════════════════════════════════
# Canary rollout
# ═════════════════════════════════════════════════════════════════════════════


def test_canary_starts_at_min_fraction():
    c = CanaryRollout("champ", "chal")
    assert c.fraction == pytest.approx(0.05)


def test_canary_route_returns_valid_model():
    c = CanaryRollout("champ", "chal")
    ids = {c.route() for _ in range(200)}
    assert ids <= {"champ", "chal"}
    assert "chal" in ids  # 5% of 200 >> 1 in expectation


def test_canary_escalate_and_deescalate():
    c = CanaryRollout("champ", "chal", min_fraction=0.1, max_fraction=0.8, step=0.1)
    for _ in range(10):
        c.escalate()
    assert c.fraction == pytest.approx(0.8)  # clamped at max
    for _ in range(10):
        c.deescalate()
    assert c.fraction == pytest.approx(0.1)  # clamped at min


def test_canary_full_route():
    c = CanaryRollout("champ", "chal", min_fraction=1.0)
    for _ in range(50):
        assert c.route() == "chal"


def test_canary_status_shape():
    c = CanaryRollout("champ", "chal")
    for _ in range(100):
        c.route()
    c.escalate()
    st = c.status()
    assert st["champion"] == "champ"
    assert st["challenger"] == "chal"
    assert st["n_signals"] == 100
    assert st["escalations"] == 1


# ═════════════════════════════════════════════════════════════════════════════
# Automated rollback
# ═════════════════════════════════════════════════════════════════════════════


def test_rollback_fires_on_drawdown():
    mon = AutomatedRollbackMonitor("model", config=RollbackConfig(max_drawdown_pct=0.05, min_trades=10))
    mon.set_baseline(error_rate=0.1)
    alert = None
    eq = 1000.0
    for _i in range(30):
        pnl = -50.0  # steady losses -> deep drawdown
        eq = max(eq + pnl, 1.0)
        alert = mon.on_trade_closed(pnl, equity=eq) or alert
    assert alert is not None
    assert alert["rolled_back"] is True
    assert any("MaxDD" in t for t in alert["triggers"])


def test_rollback_fires_once():
    mon = AutomatedRollbackMonitor("model", config=RollbackConfig(max_drawdown_pct=0.01, min_trades=5))
    eq = 1000.0
    alerts = []
    for _i in range(50):
        eq = max(eq - 50, 1.0)
        a = mon.on_trade_closed(-50.0, equity=eq)
        if a:
            alerts.append(a)
    assert len(alerts) == 1


def test_rollback_not_enough_trades():
    mon = AutomatedRollbackMonitor("model", config=RollbackConfig(min_trades=100))
    for _i in range(10):
        assert mon.on_trade_closed(-1.0, equity=1000.0) is None


def test_rollback_error_spike():
    mon = AutomatedRollbackMonitor(
        "model",
        config=RollbackConfig(
            max_drawdown_pct=0.99,
            min_psr=0.0,
            min_sharpe=-99.0,
            error_spike_ratio=2.0,
            min_baseline_errors=5,
            min_trades=10,
        ),
    )
    mon.set_baseline(error_rate=0.10)
    # good trades to keep drawdown/psr/sharpe quiet
    for i in range(10):  # noqa: B007
        mon.on_trade_closed(50.0, equity=5000.0)
    for i in range(20):
        mon.on_prediction_error(i % 5 == 0)  # 20% error rate = 2x baseline
    alert = mon._check() if len(mon._live.errors) >= 5 else None
    # 20% vs 10% baseline exactly at ratio → boundary; force clearly above
    for i in range(10):  # noqa: B007
        mon.on_prediction_error(True)
    alert = mon._check()
    assert alert is not None
    assert any("Error spike" in t for t in alert["triggers"])


def test_rollback_callback_invoked():
    calls = []
    mon = AutomatedRollbackMonitor(
        "model",
        config=RollbackConfig(max_drawdown_pct=0.01, min_trades=5),
        rollback_callback=lambda a: calls.append(a["model_id"]),
    )
    eq = 1000.0
    for _i in range(20):
        eq = max(eq - 40, 1.0)
        mon.on_trade_closed(-40.0, equity=eq)
    assert calls == ["model"]


def test_rollback_healthy_no_fire():
    mon = AutomatedRollbackMonitor(
        "model",
        config=RollbackConfig(max_drawdown_pct=0.2, min_psr=0.0, min_sharpe=-99.0),
    )
    mon.set_baseline(error_rate=0.1)
    eq = 1000.0
    for _i in range(50):
        eq += 20.0
        mon.on_prediction_error(False)
        assert mon.on_trade_closed(20.0, equity=eq) is None


# ═════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═════════════════════════════════════════════════════════════════════════════


def test_run_model_drift_check_alert(good_trades, bad_trades):
    res = run_model_drift_check(
        champion_id="champ",
        challenger_ids=["chal"],
        champion_pnls=good_trades.tolist(),
        challenger_pnls={"chal": bad_trades.tolist()},
    )
    assert res["alert"] is True
    assert res["events"][0]["type"] == "model_drift"
    assert res["events"][0]["event"] == "challenger_losing"


def test_run_model_drift_check_clean(good_trades):
    res = run_model_drift_check(
        champion_id="champ",
        challenger_ids=["chal"],
        champion_pnls=good_trades.tolist(),
        challenger_pnls={"chal": good_trades.tolist()},
        window=300,
    )
    # equal trades → challenger not strictly better than champion by margin
    assert isinstance(res["comparisons"], list)
    assert len(res["comparisons"]) == 1
    assert "stats" in res
