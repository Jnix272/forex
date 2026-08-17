"""
Tests for drift.data_drift (Improvement #4): feature distribution drift, SHAP
attribution drift, concept-drift detectors, adversarial validation, and the
orchestrator.
"""

from __future__ import annotations

import numpy as np
import pytest

from drift.data_drift import (
    ADWIN,
    DDM,
    EDDM,
    ConceptDriftTracker,
    PageHinkley,
    _psi,
    adversarial_validation,
    check_feature_distribution_drift,
    check_shap_attribution_drift,
    run_data_drift_check,
)


@pytest.fixture
def ref_features():
    rng = np.random.default_rng(0)
    return {
        "f1": rng.normal(0.0, 1.0, 2000),
        "f2": rng.exponential(1.0, 2000),
    }


@pytest.fixture
def live_features_same(ref_features):
    rng = np.random.default_rng(1)
    return {k: v + rng.normal(0.0, 0.01, len(v)) for k, v in ref_features.items()}


@pytest.fixture
def live_features_drifted(ref_features):
    np.random.default_rng(2)
    return {
        "f1": ref_features["f1"] + 3.0,  # big location shift
        "f2": ref_features["f2"] * 5.0,  # big scale shift
    }


# ═════════════════════════════════════════════════════════════════════════════
# Feature distribution drift
# ═════════════════════════════════════════════════════════════════════════════


def test_psi_identical():
    a = np.random.default_rng(0).normal(0.0, 1.0, 5000)
    assert _psi(a, a + 0.001) < 0.05


def test_psi_drifted():
    a = np.random.default_rng(0).normal(0.0, 1.0, 5000)
    b = np.random.default_rng(1).normal(3.0, 1.0, 5000)
    assert _psi(a, b) > 0.2


def test_feature_drift_no_drift(ref_features, live_features_same):
    res = check_feature_distribution_drift(ref_features, live_features_same)
    assert res["drifted_count"] == 0
    assert res["feature_count"] == 2


def test_feature_drift_detected(ref_features, live_features_drifted):
    res = check_feature_distribution_drift(ref_features, live_features_drifted)
    assert res["drifted_count"] >= 1
    assert "f1" in res["drifted_features"]
    assert res["max_psi"] > 0.2


def test_feature_drift_to_event_shape(ref_features, live_features_drifted):
    res = check_feature_distribution_drift(ref_features, live_features_drifted)
    ev = res["results"][0]
    assert ev["type"] == "feature_drift"
    for k in ["feature", "psi", "ks_stat", "ks_pvalue", "wasserstein", "drift"]:
        assert k in ev


# ═════════════════════════════════════════════════════════════════════════════
# SHAP attribution drift
# ═════════════════════════════════════════════════════════════════════════════


def test_shap_drift_same_importance():
    ref = {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}
    live = {"a": 0.41, "b": 0.30, "c": 0.20, "d": 0.09}
    res = check_shap_attribution_drift(ref, live, shift_threshold=0.05)
    assert res["drifted_count"] == 0


def test_shap_drift_detected():
    ref = {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}
    live = {"a": 0.05, "b": 0.05, "c": 0.4, "d": 0.5}
    res = check_shap_attribution_drift(ref, live, shift_threshold=0.05)
    assert res["drifted_count"] >= 1
    assert "a" in res["drifted_features"]


def test_shap_drift_normalised_weights():
    ref = {"x": 1.0, "y": 3.0}
    live = {"x": 0.25, "y": 0.75}
    # relative importance identical after normalisation
    res = check_shap_attribution_drift(ref, live, shift_threshold=0.01)
    assert res["drifted_count"] == 0


# ═════════════════════════════════════════════════════════════════════════════
# Concept-drift detectors
# ═════════════════════════════════════════════════════════════════════════════


def test_adwin_detects_shift():
    adwin = ADWIN(min_window=20)
    fired = False
    for _ in range(100):
        fired = adwin.add(0.1) or fired
    for _ in range(100):
        fired = adwin.add(0.9) or fired
    assert fired


def test_adwin_steady_state():
    adwin = ADWIN(min_window=20)
    for _ in range(300):
        adwin.add(0.5)
    assert adwin._changed is False


def test_page_hinkley_detects_drift():
    pht = PageHinkley(delta=0.001, lambda_=20.0)
    fired = False
    for _ in range(50):
        fired = pht.add(0.1) or fired
    for _ in range(100):
        fired = pht.add(2.0) or fired
    assert fired


def test_ddm_warning_then_drift():
    ddm = DDM(min_instances=10)
    # first a clean run
    for _ in range(30):
        ddm.add(False)
    state = None
    for _ in range(60):
        state = ddm.add(True)  # suddenly everything errors
    assert state in ("warning", "drift")
    assert ddm.state in ("warning", "drift")


def test_eddm_steady_state():
    eddm = EDDM(min_instances=10)
    for _ in range(50):
        eddm.add(False)
    assert eddm.state == "in_concept"


def test_concept_tracker_score_and_events():
    tracker = ConceptDriftTracker()
    for _ in range(30):
        tracker.add_error(False)
    # inject a burst of errors
    for _ in range(60):
        tracker.add_error(True)
    assert 0.0 <= tracker.streaming_score() <= 1.0
    assert isinstance(tracker.events(), list)
    assert tracker.events()[0]["type"] == "concept_drift"


# ═════════════════════════════════════════════════════════════════════════════
# Adversarial validation
# ═════════════════════════════════════════════════════════════════════════════


def test_adversarial_validation_identical():
    rng = np.random.default_rng(3)
    X = rng.normal(0.0, 1.0, (300, 5))
    res = adversarial_validation(X, X + rng.normal(0.0, 0.001, X.shape))
    assert res["auc"] < 0.7
    assert res["drift"] is False


def test_adversarial_validation_drift():
    rng = np.random.default_rng(4)
    train = rng.normal(0.0, 1.0, (300, 5))
    live = rng.normal(4.0, 1.0, (300, 5))  # clearly separable
    res = adversarial_validation(train, live)
    assert res["auc"] > 0.8
    assert res["drift"] is True


def test_adversarial_validation_insufficient():
    res = adversarial_validation(np.zeros((5, 2)), np.zeros((5, 2)))
    assert res["drift"] is False
    assert res["method"] == "insufficient_data"


# ═════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═════════════════════════════════════════════════════════════════════════════


def test_run_data_drift_check_clean(ref_features, live_features_same):
    res = run_data_drift_check(ref_features, live_features_same)
    assert res["alert"] is False
    assert isinstance(res["events"], list)
    assert "feature_distribution" in res


def test_run_data_drift_check_alert(ref_features, live_features_drifted):
    res = run_data_drift_check(
        ref_features,
        live_features_drifted,
        ref_shap_importance={"f1": 0.5, "f2": 0.5},
        live_shap_importance={"f1": 0.1, "f2": 0.9},
    )
    assert res["alert"] is True
    assert res["drifted_count"] >= 1
    assert res["shap_attribution"] is not None


def test_run_data_drift_check_with_concept():
    errors = [False] * 20 + [True] * 40
    res = run_data_drift_check(
        {"f": np.random.default_rng(5).normal(0, 1, 500)},
        {"f": np.random.default_rng(6).normal(0, 1, 500)},
        error_stream=errors,
    )
    assert res["concept_drift"] is not None
    assert 0.0 <= res["concept_drift"]["streaming_score"] <= 1.0
