from __future__ import annotations

from validation.model_diagnostics import (
    ARCHITECTURE_PRIORITY,
    classification_calibration_metrics,
    ensemble_safety_decision,
    model_leaderboard,
)


def test_calibration_metrics_detect_overconfidence():
    y_true = [0, 1, 1, 0]
    probs = [
        [0.90, 0.10],
        [0.90, 0.10],
        [0.20, 0.80],
        [0.10, 0.90],
    ]

    metrics = classification_calibration_metrics(y_true, probs, n_bins=5)

    assert metrics["n"] == 4
    assert metrics["accuracy"] == 0.5
    assert metrics["ece"] > 0.0
    assert metrics["brier"] > 0.0
    assert metrics["nll"] > 0.0


def test_logits_are_converted_to_probabilities_for_calibration():
    y_true = [0, 1]
    logits = [[3.0, 0.0], [0.0, 3.0]]

    metrics = classification_calibration_metrics(y_true, logits, n_bins=2)

    assert metrics["accuracy"] == 1.0
    assert 0.0 <= metrics["ece"] <= 1.0


def test_ensemble_safety_blocks_high_or_invalid_disagreement():
    high = ensemble_safety_decision(1.2, low_threshold=0.5, high_threshold=1.0)
    invalid = ensemble_safety_decision(float("nan"))
    moderate = ensemble_safety_decision(0.75, low_threshold=0.5, high_threshold=1.0)

    assert high["decision"] == "block"
    assert high["size_multiplier"] == 0.0
    assert invalid["decision"] == "block"
    assert moderate["decision"] == "half_size"
    assert moderate["size_multiplier"] == 0.5


def test_model_leaderboard_prefers_live_use_metrics():
    rows = [
        {"model": "transformer", "net_sharpe": 1.6, "max_drawdown": 0.20, "ece": 0.20},
        {"model": "haelt", "net_sharpe": 1.7, "max_drawdown": 0.10, "ece": 0.05, "promoted": True},
        {"model": "mamba", "net_sharpe": 1.2, "max_drawdown": 0.05, "latency_ms": 2.0, "ece": 0.04},
    ]

    leaderboard = model_leaderboard(rows)

    assert leaderboard[0]["model"] == "haelt"
    assert leaderboard[0]["rank"] == 1
    assert all("live_use_score" in row for row in leaderboard)


def test_architecture_priority_is_explicit():
    assert [row["model"] for row in ARCHITECTURE_PRIORITY[:4]] == [
        "haelt",
        "mamba",
        "expert",
        "ensemble",
    ]
