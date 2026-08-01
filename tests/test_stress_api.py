"""
Stress tests for the Risk Management API.
Verifies behavior under extreme and boundary inputs.
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient

try:
    from api.main import app
    HAS_API = True
except ImportError:
    HAS_API = False
    app = None

@pytest.fixture
def client():
    if not HAS_API:
        pytest.skip("API application is not available")
    return TestClient(app)

def test_stress_large_returns_array(client):
    """Verify API handles extremely large returns array (10,000 elements) without performance degradation."""
    import time
    returns = np.random.normal(0.0001, 0.005, 10000).tolist()
    payload = {
        "win_prob": 0.55,
        "win_loss_ratio": 1.5,
        "returns": returns,
        "price": 1.1000,
        "current_atr": 0.0005,
        "equity": 10000.0,
        "lot_size": 10000.0
    }
    
    start_time = time.time()
    response = client.post("/kelly_sizing", json=payload)
    elapsed = time.time() - start_time
    
    assert response.status_code == 200
    data = response.json()
    assert "lots" in data
    assert elapsed < 0.2  # Should be very fast (< 200ms)

def test_stress_volatility_bounds_large_returns(client):
    """Verify /volatility_bounds handles 10,000 elements and lookback <= returns length."""
    returns = np.random.normal(0.0001, 0.005, 10000).tolist()
    payload = {
        "returns": returns,
        "target_vol": 0.10,
        "lookback": 5000
    }
    response = client.post("/volatility_bounds", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "vol_scalar" in data
    assert "realized_vol" in data

def test_zero_atr_pydantic_validation(client):
    """Verify that zero ATR is rejected by Pydantic API validation (gt=0.0)."""
    payload = {
        "win_prob": 0.55,
        "win_loss_ratio": 1.5,
        "returns": [0.001, -0.002, 0.003],
        "price": 1.1000,
        "current_atr": 0.0,
        "equity": 10000.0,
        "lot_size": 10000.0
    }
    response = client.post("/kelly_sizing", json=payload)
    assert response.status_code == 422  # Validation Error

def test_extremely_small_atr(client):
    """Verify extremely small ATR does not cause division-by-zero or infinite lots."""
    payload = {
        "win_prob": 0.55,
        "win_loss_ratio": 1.5,
        "returns": [0.001, -0.002, 0.003],
        "price": 1.1000,
        "current_atr": 1e-15,
        "equity": 10000.0,
        "lot_size": 10000.0
    }
    response = client.post("/kelly_sizing", json=payload)
    assert response.status_code == 200
    data = response.json()
    # Lots should be bounded by max_position_pct or min_stop_pips, not inf/nan
    assert data["lots"] >= 0.0
    assert not np.isnan(data["lots"])
    assert not np.isinf(data["lots"])

def test_extremely_low_target_vol(client):
    """Verify that very low target vol scales down position size safely but doesn't crash."""
    payload = {
        "win_prob": 0.55,
        "win_loss_ratio": 1.5,
        "returns": [0.001, -0.002, 0.003] * 10,
        "price": 1.1000,
        "current_atr": 0.0005,
        "equity": 10000.0,
        "lot_size": 10000.0,
        "target_vol": 1e-15
    }
    response = client.post("/kelly_sizing", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["lots"] >= 0.0
    assert data["vol_scalar"] >= 0.1  # vol_target_scalar clips to 0.1 minimum

def test_extremely_high_target_vol(client):
    """Verify that very high target vol scales up position size safely up to clip limit (3.0)."""
    payload = {
        "win_prob": 0.55,
        "win_loss_ratio": 1.5,
        "returns": [0.001, -0.002, 0.003] * 10,
        "price": 1.1000,
        "current_atr": 0.0005,
        "equity": 10000.0,
        "lot_size": 10000.0,
        "target_vol": 1e9
    }
    response = client.post("/kelly_sizing", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["lots"] >= 0.0
    assert data["vol_scalar"] <= 3.0  # vol_target_scalar clips to 3.0 maximum

def test_win_prob_zero_kelly_clamping(client):
    """Verify behavior when win probability is 0.0."""
    payload = {
        "win_prob": 0.0,
        "win_loss_ratio": 1.5,
        "returns": [0.001, -0.002, 0.003],
        "price": 1.1000,
        "current_atr": 0.0005,
        "equity": 10000.0,
        "lot_size": 10000.0
    }
    response = client.post("/kelly_sizing", json=payload)
    assert response.status_code == 200
    data = response.json()
    # Zero win probability must produce zero size (no min-lot clamp on dead edge).
    assert data["full_kelly"] <= 0.0
    assert data["frac_kelly"] == 0.0
    assert data["lots"] == 0.0

def test_win_prob_one(client):
    """Verify behavior when win probability is 1.0."""
    payload = {
        "win_prob": 1.0,
        "win_loss_ratio": 1.5,
        "returns": [0.001, -0.002, 0.003],
        "price": 1.1000,
        "current_atr": 0.0005,
        "equity": 10000.0,
        "lot_size": 10000.0
    }
    response = client.post("/kelly_sizing", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["full_kelly"] == 1.0
    assert data["lots"] > 0.0

def test_extreme_win_loss_ratio_low(client):
    """Verify behavior when win_loss_ratio is extremely low (but valid gt=0.0)."""
    payload = {
        "win_prob": 0.5,
        "win_loss_ratio": 1e-9,
        "returns": [0.001, -0.002, 0.003],
        "price": 1.1000,
        "current_atr": 0.0005,
        "equity": 10000.0,
        "lot_size": 10000.0
    }
    response = client.post("/kelly_sizing", json=payload)
    assert response.status_code == 200
    data = response.json()
    # Expect full_kelly <= 0 with near-zero win/loss ratio; size must stay flat.
    assert data["lots"] == 0.0

def test_nan_string_returns_behavior(client):
    """Check how the API behaves if returns contain 'NaN' string (parsed as float NaN by Pydantic)."""
    payload = {
        "win_prob": 0.55,
        "win_loss_ratio": 1.5,
        "returns": [0.001, "NaN", 0.003] * 10,
        "price": 1.1000,
        "current_atr": 0.0005,
        "equity": 10000.0,
        "lot_size": 10000.0
    }
    response = client.post("/kelly_sizing", json=payload)
    # The server might raise 400 because np.std contains NaN, leading to exception or NaN output.
    # If it returns 200, check the output structure.
    if response.status_code == 200:
        data = response.json()
        print("Received 200 for NaN string returns:", data)
    else:
        assert response.status_code in (400, 422)

def test_inf_string_returns_behavior(client):
    """Check how the API behaves if returns contain 'Infinity' string (parsed as float inf by Pydantic)."""
    payload = {
        "win_prob": 0.55,
        "win_loss_ratio": 1.5,
        "returns": [0.001, "Infinity", 0.003] * 10,
        "price": 1.1000,
        "current_atr": 0.0005,
        "equity": 10000.0,
        "lot_size": 10000.0
    }
    response = client.post("/kelly_sizing", json=payload)
    if response.status_code == 200:
        data = response.json()
        print("Received 200 for Infinity string returns:", data)
    else:
        assert response.status_code in (400, 422)

def test_null_returns_behavior(client):
    """Verify that null (None) in returns array is rejected by Pydantic (not a valid float)."""
    payload = {
        "win_prob": 0.55,
        "win_loss_ratio": 1.5,
        "returns": [0.001, None, 0.003],
        "price": 1.1000,
        "current_atr": 0.0005,
        "equity": 10000.0,
        "lot_size": 10000.0
    }
    response = client.post("/kelly_sizing", json=payload)
    assert response.status_code == 422

def test_api_performance_benchmark(client):
    """Benchmark 100 serial requests to /kelly_sizing and calculate mean response time."""
    import time
    returns = np.random.normal(0.0001, 0.005, 500).tolist()
    payload = {
        "win_prob": 0.55,
        "win_loss_ratio": 1.5,
        "returns": returns,
        "price": 1.1000,
        "current_atr": 0.0005,
        "equity": 10000.0,
        "lot_size": 10000.0
    }
    
    times = []
    for _ in range(100):
        start = time.time()
        response = client.post("/kelly_sizing", json=payload)
        times.append(time.time() - start)
        assert response.status_code == 200
        
    mean_time = np.mean(times)
    p95_time = np.percentile(times, 95)
    print(f"\nAPI Performance Benchmark (500 return elements): Mean = {mean_time*1000:.2f}ms, 95th Percentile = {p95_time*1000:.2f}ms")
    assert mean_time < 0.05  # Mean response time should be less than 50ms


