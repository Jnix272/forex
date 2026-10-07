"""Unit tests for StationaryEnsembleInferenceEngine and production 4h models."""
import numpy as np
import pytest

from trading.inference_engines import StationaryEnsembleInferenceEngine


def test_stationary_inference_engine_init():
    for pair in ["EURUSD", "USDJPY", "USDCAD"]:
        engine = StationaryEnsembleInferenceEngine(pair=pair)
        assert engine.pair == pair
        assert engine.threshold_bps > 0.0
        assert len(engine.curated_indices) == 192


def test_stationary_inference_engine_prediction_full_obs():
    engine = StationaryEnsembleInferenceEngine(pair="USDJPY")
    # 584-feature dummy observation
    dummy_obs_584 = np.random.randn(584).astype(np.float32)

    pred_bps = engine.predict_return(dummy_obs_584)
    assert isinstance(pred_bps, float)
    assert np.isfinite(pred_bps)

    action = engine.select_action(dummy_obs_584)
    assert action in (0, 1, 2)  # 0=Buy, 1=Hold, 2=Sell


def test_stationary_inference_engine_curated_obs():
    engine = StationaryEnsembleInferenceEngine(pair="EURUSD")
    # 192-feature dummy observation (already filtered)
    dummy_obs_192 = np.zeros(192, dtype=np.float32)

    pred_bps = engine.predict_return(dummy_obs_192)
    assert isinstance(pred_bps, float)

    action = engine.select_action(dummy_obs_192)
    assert action in (0, 1, 2)


def test_stationary_inference_engine_threshold_override():
    # If threshold is huge, it should always Hold (1)
    engine_conservative = StationaryEnsembleInferenceEngine(pair="USDCAD", override_threshold=1000.0)
    dummy_obs = np.random.randn(584).astype(np.float32)
    assert engine_conservative.select_action(dummy_obs) == 1

    # Reset buffer works cleanly
    engine_conservative.reset_buffer()


def test_stationary_inference_engine_live_attributes():
    from trading.live_actions import LiveAction

    engine = StationaryEnsembleInferenceEngine(pair="USDJPY")
    assert getattr(engine, "returns_live_actions", False) is True
    assert getattr(engine, "seq_len", 0) == 1
    assert getattr(engine, "n_features", 0) == 584

    # Verification of action enum consistency
    assert int(LiveAction.BUY) == 0
    assert int(LiveAction.HOLD) == 1
    assert int(LiveAction.SELL) == 2


def test_stationary_inference_engine_for_pair():
    engine_eur = StationaryEnsembleInferenceEngine(pair="EURUSD")
    engine_jpy = engine_eur.for_pair("USDJPY")
    assert engine_jpy.pair == "USDJPY"
    assert engine_jpy is not engine_eur

    # Same pair returns self
    assert engine_eur.for_pair("EURUSD") is engine_eur

