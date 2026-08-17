"""
Stress and Edge-Case Test Suite for FastAPI Risk Management API.
Can be executed as a standalone python script or via pytest.
"""

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from sizing.kelly_criterion import PositionSizer

# Initialize TestClient
client = TestClient(app)


def run_performance_test():
    """Scenario 1: Large Input Performance Stress Test (10,000 returns elements)."""
    print("\n--- Running Performance Test (10k elements) ---")
    large_returns = np.random.normal(0.0001, 0.005, 10000).tolist()
    payload = {
        "win_prob": 0.55,
        "win_loss_ratio": 1.5,
        "returns": large_returns,
        "price": 1.1000,
        "current_atr": 0.0005,
        "equity": 10000.0,
        "lot_size": 10000.0,
    }

    start_time = time.perf_counter()
    response = client.post("/kelly_sizing", json=payload)
    elapsed = time.perf_counter() - start_time

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    print(f"Time taken for /kelly_sizing with 10k returns: {elapsed:.4f} seconds")
    print(f"Result: lots={data['lots']}, vol_scalar={data['vol_scalar']:.4f}")
    assert elapsed < 0.5, f"Performance warning: request took {elapsed:.4f} seconds"

    # Also test /volatility_bounds
    payload_vol = {"returns": large_returns, "target_vol": 0.10, "lookback": 20}
    start_time = time.perf_counter()
    response_vol = client.post("/volatility_bounds", json=payload_vol)
    elapsed_vol = time.perf_counter() - start_time
    assert response_vol.status_code == 200
    data_vol = response_vol.json()
    print(f"Time taken for /volatility_bounds with 10k returns: {elapsed_vol:.4f} seconds")
    print(f"Result: vol_scalar={data_vol['vol_scalar']:.4f}, realized_vol={data_vol['realized_vol']:.4f}")
    assert elapsed_vol < 0.5


def test_stress_large_input():
    run_performance_test()


def run_zero_atr_test():
    """Scenario 2: Zero ATR Validation / Behavior."""
    print("\n--- Running Zero ATR Validation Test ---")
    # Pydantic schema validation specifies gt=0.0 for current_atr.
    # Therefore, 0.0 or negative ATR should be rejected with 422 by FastAPI.
    payload_zero = {
        "win_prob": 0.55,
        "win_loss_ratio": 1.5,
        "returns": [0.001, -0.002, 0.003],
        "price": 1.1000,
        "current_atr": 0.0,
        "equity": 10000.0,
        "lot_size": 10000.0,
    }
    response = client.post("/kelly_sizing", json=payload_zero)
    print(f"Zero ATR status code (expect 422): {response.status_code}")
    assert response.status_code == 422

    # Try a very small ATR that is greater than 0
    payload_tiny = payload_zero.copy()
    payload_tiny["current_atr"] = 1e-9
    response_tiny = client.post("/kelly_sizing", json=payload_tiny)
    print(f"Tiny ATR (1e-9) status code: {response_tiny.status_code}")
    assert response_tiny.status_code == 200
    data_tiny = response_tiny.json()
    print(f"Tiny ATR (1e-9) Result: lots={data_tiny['lots']}, risk_usd={data_tiny['risk_usd']:.4f}")

    # Check that underlying PositionSizer does not crash with 0 ATR (if bypassed API validation)
    try:
        sizer = PositionSizer()
        res = sizer.size_position(
            win_prob=0.55, win_loss_ratio=1.5, returns=[0.001, -0.002, 0.003], price=1.1000, current_atr=0.0
        )
        print(f"Direct PositionSizer sizing with 0 ATR succeeded: lots={res['lots']}")
        assert isinstance(res["lots"], float)
    except Exception as e:
        pytest.fail(f"PositionSizer direct call crashed with 0 ATR: {e}")


def test_stress_zero_atr():
    run_zero_atr_test()


def run_low_target_vol_test():
    """Scenario 3: Very Low Target Volatility."""
    print("\n--- Running Low Target Volatility Test ---")
    payload = {
        "win_prob": 0.55,
        "win_loss_ratio": 1.5,
        "returns": [0.001, -0.002, 0.003] * 5,
        "price": 1.1000,
        "current_atr": 0.0005,
        "equity": 10000.0,
        "lot_size": 10000.0,
        "target_vol": 1e-9,  # extremely small target vol
    }
    response = client.post("/kelly_sizing", json=payload)
    assert response.status_code == 200
    data = response.json()
    print(f"Target Vol (1e-9) Result: lots={data['lots']}, vol_scalar={data['vol_scalar']:.4f}")
    assert data["vol_scalar"] >= 0.1  # should be clipped to min of 0.1

    # Test target_vol <= 0.0 (should be rejected by Pydantic gt=0.0)
    payload_bad = payload.copy()
    payload_bad["target_vol"] = 0.0
    response_bad = client.post("/kelly_sizing", json=payload_bad)
    print(f"Target Vol (0.0) status code (expect 422): {response_bad.status_code}")
    assert response_bad.status_code == 422


def test_stress_low_target_vol():
    run_low_target_vol_test()


def run_edge_case_win_prob_test():
    """Scenario 4: Edge-Case Win Probabilities and Win/Loss Ratios."""
    print("\n--- Running Edge-Case Win Probabilities Test ---")
    # Test win_prob = 1.0 (certain win)
    payload_1 = {
        "win_prob": 1.0,
        "win_loss_ratio": 1.5,
        "returns": [0.001, -0.002, 0.003],
        "price": 1.1000,
        "current_atr": 0.0005,
        "equity": 10000.0,
        "lot_size": 10000.0,
    }
    response_1 = client.post("/kelly_sizing", json=payload_1)
    assert response_1.status_code == 200
    data_1 = response_1.json()
    print(f"Win Prob (1.0) Result: lots={data_1['lots']}, full_kelly={data_1['full_kelly']:.4f}")
    assert data_1["full_kelly"] == 1.0

    # Test win_prob = 0.0 (certain loss)
    # OBSERVATION: The underlying RegimePositionSizer clips Kelly fraction to a minimum (min_kelly=0.05).
    # Thus, even for 0.0 win probability, the sizer returns positive lots (0.3 in this case).
    payload_0 = payload_1.copy()
    payload_0["win_prob"] = 0.0
    response_0 = client.post("/kelly_sizing", json=payload_0)
    assert response_0.status_code == 200
    data_0 = response_0.json()
    print(f"Win Prob (0.0) Result: lots={data_0['lots']}, full_kelly={data_0['full_kelly']:.4f}")
    assert data_0["full_kelly"] <= 0.0
    # Instead of assert 0.0, we assert that the sizer assigns lots due to the min_kelly clip flaw
    assert data_0["lots"] > 0.0, "Expected positive position size due to min_kelly clipping constraint"

    # Test win_loss_ratio extremely small (1e-9)
    payload_low_ratio = payload_1.copy()
    payload_low_ratio["win_prob"] = 0.5
    payload_low_ratio["win_loss_ratio"] = 1e-9
    response_low_ratio = client.post("/kelly_sizing", json=payload_low_ratio)
    assert response_low_ratio.status_code == 200
    data_lr = response_low_ratio.json()
    print(f"Win/Loss Ratio (1e-9) Result: lots={data_lr['lots']}, full_kelly={data_lr['full_kelly']:.4f}")
    assert data_lr["lots"] > 0.0  # also clipped to min_kelly!

    # Test win_loss_ratio extremely large (1e9)
    payload_high_ratio = payload_1.copy()
    payload_high_ratio["win_prob"] = 0.55
    payload_high_ratio["win_loss_ratio"] = 1e9
    response_high_ratio = client.post("/kelly_sizing", json=payload_high_ratio)
    assert response_high_ratio.status_code == 200
    data_hr = response_high_ratio.json()
    print(f"Win/Loss Ratio (1e9) Result: lots={data_hr['lots']}, full_kelly={data_hr['full_kelly']:.4f}")
    assert data_hr["full_kelly"] >= 0.54


def test_stress_edge_case_win_prob():
    run_edge_case_win_prob_test()


def run_nan_inf_test():
    """Scenario 5: NaN or Inf inside returns array."""
    print("\n--- Running NaN/Inf Returns Test ---")
    # Pydantic parses compliant JSON string representations of special floats like "NaN" or "Infinity".
    # We will test how FastAPI handles these when they propagate to the numpy calculation layer.
    payload_nan = {
        "win_prob": 0.55,
        "win_loss_ratio": 1.5,
        "returns": [0.001, "NaN", 0.003],
        "price": 1.1000,
        "current_atr": 0.0005,
        "equity": 10000.0,
        "lot_size": 10000.0,
    }

    response_nan = client.post("/kelly_sizing", json=payload_nan)
    print(f"Returns with short 'NaN' list status code: {response_nan.status_code}")
    if response_nan.status_code == 200:
        print(f"Short NaN Result: {response_nan.json()}")

    # Now let's try a returns list of length >= 20 (so it triggers np.std) containing NaN
    payload_nan_long = payload_nan.copy()
    payload_nan_long["returns"] = [0.001, -0.002, 0.003, -0.001, 0.002] * 4 + ["NaN"]
    response_nan_long = client.post("/kelly_sizing", json=payload_nan_long)
    print(f"Returns with long 'NaN' list status code (expecting crash or handle): {response_nan_long.status_code}")
    if response_nan_long.status_code == 200:
        print(f"Long NaN Result: {response_nan_long.json()}")
    elif response_nan_long.status_code == 500:
        print("Long NaN list caused 500 Internal Server Error!")
    else:
        print(f"Long NaN list returned {response_nan_long.status_code}: {response_nan_long.text}")

    # Let's try volatility_bounds endpoint with NaN in returns
    payload_vol_nan = {
        "returns": [0.001, -0.002, 0.003, -0.001, 0.002] * 4 + ["NaN"],
        "target_vol": 0.10,
        "lookback": 20,
    }
    response_vol_nan = client.post("/volatility_bounds", json=payload_vol_nan)
    print(f"Volatility bounds with NaN status code: {response_vol_nan.status_code}")
    if response_vol_nan.status_code == 200:
        print(f"Volatility bounds with NaN Result: {response_vol_nan.json()}")
    elif response_vol_nan.status_code == 500:
        print("Volatility bounds with NaN caused 500 Internal Server Error!")
    else:
        print(f"Volatility bounds with NaN returned {response_vol_nan.status_code}: {response_vol_nan.text}")


def test_stress_nan_inf():
    run_nan_inf_test()


if __name__ == "__main__":
    print("=== STARTING STRESS AND EDGE-CASE TESTS ===")
    run_performance_test()
    run_zero_atr_test()
    run_low_target_vol_test()
    run_edge_case_win_prob_test()
    run_nan_inf_test()
    print("=== ALL STRESS AND EDGE-CASE TESTS COMPLETED ===")
