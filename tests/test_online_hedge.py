"""
tests/test_online_hedge.py
==========================
Unit tests for OnlineHedgeEnsemble (Exp3 / Hedge algorithm).
"""

from pathlib import Path
import pytest
import numpy as np

from models.online_hedge import OnlineHedgeEnsemble


def test_initialization():
    models = ["haelt", "tft", "mamba"]
    hedge = OnlineHedgeEnsemble(model_names=models)

    assert hedge.k == 3
    weights = hedge.get_weights()
    assert len(weights) == 3
    for m in models:
        assert np.isclose(weights[m], 1.0 / 3.0)

    with pytest.raises(ValueError):
        OnlineHedgeEnsemble(model_names=[])


def test_weight_convergence_on_winning_model():
    models = ["model_bull", "model_bear", "model_neutral"]
    hedge = OnlineHedgeEnsemble(model_names=models, learning_rate=0.5, discount_factor=0.99)

    # In a rising market, model_bull predicts +1, model_bear predicts -1, neutral predicts 0
    predictions = {
        "model_bull": 1.0,
        "model_bear": -1.0,
        "model_neutral": 0.0,
    }

    for _ in range(20):
        hedge.update(predictions, realized_return=0.002, current_atr=0.001)

    weights = hedge.get_weights()
    # model_bull should have by far the highest weight
    assert weights["model_bull"] > weights["model_neutral"]
    assert weights["model_neutral"] > weights["model_bear"]
    assert weights["model_bull"] > 0.5
    assert np.isclose(sum(weights.values()), 1.0)


def test_regime_shift_adaptation():
    models = ["trend_follower", "mean_reverter"]
    hedge = OnlineHedgeEnsemble(model_names=models, learning_rate=0.4, discount_factor=0.95, min_weight_floor=0.1)

    # Phase 1: Trend regime where trend_follower wins
    for _ in range(25):
        hedge.update(
            {"trend_follower": 1.0, "mean_reverter": -1.0},
            realized_return=0.002,
            current_atr=0.001,
        )
    w_phase1 = hedge.get_weights()
    assert w_phase1["trend_follower"] > 0.70

    # Phase 2: Regime reverses, mean_reverter is right every bar
    for _ in range(50):
        hedge.update(
            {"trend_follower": 1.0, "mean_reverter": -1.0},
            realized_return=-0.002,
            current_atr=0.001,
        )
    w_phase2 = hedge.get_weights()
    # Mean reverter should have successfully adapted and overtaken trend follower
    assert w_phase2["mean_reverter"] > w_phase2["trend_follower"]
    assert w_phase2["mean_reverter"] > 0.70


def test_exploration_floor_and_normalization():
    models = ["good_model", "bad_model"]
    eps = 0.08
    hedge = OnlineHedgeEnsemble(model_names=models, learning_rate=1.0, discount_factor=1.0, min_weight_floor=eps)

    # bad_model suffers 100 consecutive massive losses
    for _ in range(100):
        hedge.update(
            {"good_model": 1.0, "bad_model": -1.0},
            realized_return=0.01,
            current_atr=0.001,
        )

    weights = hedge.get_weights()
    # Bad model must still retain at least its exploration floor eps / K
    expected_min = (eps / 2.0) * 0.99  # allow slight floating tolerance
    assert weights["bad_model"] >= expected_min
    assert np.isclose(sum(weights.values()), 1.0)


def test_consensus_prediction():
    models = ["m1", "m2"]
    hedge = OnlineHedgeEnsemble(model_names=models)
    # Start uniform 0.5, 0.5
    pred, w = hedge.predict({"m1": 0.8, "m2": -0.2})
    assert np.isclose(pred, 0.3)


def test_persistence_save_load(tmp_path: Path):
    state_file = tmp_path / "hedge_state.json"
    models = ["alpha", "beta", "gamma"]
    hedge1 = OnlineHedgeEnsemble(model_names=models, state_path=state_file, learning_rate=0.2)

    hedge1.update({"alpha": 1.0, "beta": -0.5, "gamma": 0.0}, realized_return=0.003, current_atr=0.001)
    w1 = hedge1.get_weights()
    s1 = hedge1.get_scores()

    # Create fresh instance pointing to same file
    hedge2 = OnlineHedgeEnsemble(model_names=models, state_path=state_file)
    w2 = hedge2.get_weights()
    s2 = hedge2.get_scores()

    for m in models:
        assert np.isclose(w1[m], w2[m])
        assert np.isclose(s1[m], s2[m])
    assert hedge2.update_count == 1


def test_zero_atr_and_edge_cases():
    models = ["m1", "m2"]
    hedge = OnlineHedgeEnsemble(model_names=models)

    # ATR = 0.0 should not cause ZeroDivisionError
    w = hedge.update({"m1": 1.0, "m2": 0.0}, realized_return=0.001, current_atr=0.0)
    assert not np.isnan(w["m1"])
    assert not np.isnan(w["m2"])
    assert np.isclose(sum(w.values()), 1.0)

    # Empty predictions should return current weights without crashing
    w_empty = hedge.update({}, realized_return=0.001)
    assert w_empty == w


def test_cold_start_seeding():
    """Verify that initial_sharpes biases weights towards higher-performing models from bar 1."""
    models = ["slow_model", "fast_agent"]
    sharpes = {"slow_model": 1.5, "fast_agent": 0.5}

    hedge = OnlineHedgeEnsemble(model_names=models, initial_sharpes=sharpes, learning_rate=0.5)
    weights = hedge.get_weights()

    # slow_model with Sharpe 1.5 must have strictly higher initial weight than fast_agent with Sharpe 0.5
    assert weights["slow_model"] > weights["fast_agent"]
    assert weights["slow_model"] > 0.55
    assert np.isclose(sum(weights.values()), 1.0)

    # Calling reset should restore the seeded Sharpe distribution
    hedge.update({"slow_model": -1.0, "fast_agent": 1.0}, realized_return=-0.01)
    hedge.reset()
    w_reset = hedge.get_weights()
    assert np.isclose(w_reset["slow_model"], weights["slow_model"])
    assert np.isclose(w_reset["fast_agent"], weights["fast_agent"])

