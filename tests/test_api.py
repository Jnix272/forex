"""
End-to-End Test Suite for the Risk Management API.
Following the 4-tier testing pattern:
- Tier 1: Feature Coverage (Basic endpoint validation)
- Tier 2: Boundary & Corner Cases (Extreme inputs, division by zero, empty list/dict, error paths)
- Tier 3: Cross-Feature Interaction (Stateful flow of drawdown awareness and position sizer)
- Tier 4: Real-World Scenarios (Simulated trading day sequence)

MANDATORY INTEGRITY WARNING: DO NOT CHEAT. All implementations must be genuine.
Do not hardcode test results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Attempt to import the application and FastAPI TestClient
try:
    from fastapi.testclient import TestClient

    HAS_TESTCLIENT = True
except ImportError:
    HAS_TESTCLIENT = False

try:
    from api.main import app

    HAS_API = True
except ImportError:
    HAS_API = False
    app = None


@pytest.fixture
def client():
    """Fixture providing a TestClient for the API if available."""
    if not HAS_API:
        pytest.skip("api.main.app is missing: Implementation code does not exist yet.")
    if not HAS_TESTCLIENT:
        pytest.skip("fastapi.testclient is missing: fastapi/uvicorn not installed in this environment.")
    return TestClient(app)


def test_api_import_gating():
    """
    Gating test that verifies the API package is implemented.
    Fails with a clear message if missing.
    """
    if not HAS_API:
        pytest.fail(
            "Gating check failed: api.main.app could not be imported. "
            "The Risk Management API implementation (api/main.py) is missing."
        )


# ===========================================================================
# Tier 1: Feature Coverage (5 tests)
# ===========================================================================


def test_api_health_check(client):
    """1. Get /docs or root / health check."""
    # Try docs first, fallback to root health check
    response = client.get("/docs")
    if response.status_code != 200:
        response = client.get("/")
    assert response.status_code == 200


def test_kelly_sizing_success(client):
    """2. POST /kelly_sizing returns 200 OK."""
    payload = {
        "win_prob": 0.55,
        "win_loss_ratio": 1.5,
        "returns": [0.001, -0.002, 0.003, -0.001, 0.002] * 5,
        "price": 1.1000,
        "current_atr": 0.0005,
        "equity": 10000.0,
        "lot_size": 10000.0,
    }
    response = client.post("/kelly_sizing", json=payload)
    assert response.status_code == 200


def test_kelly_sizing_response_schema(client):
    """3. Response of /kelly_sizing has exact keys with valid float/int values."""
    payload = {
        "win_prob": 0.55,
        "win_loss_ratio": 1.5,
        "returns": [0.001, -0.002, 0.003, -0.001, 0.002] * 5,
        "price": 1.1000,
        "current_atr": 0.0005,
        "equity": 10000.0,
        "lot_size": 10000.0,
    }
    response = client.post("/kelly_sizing", json=payload)
    assert response.status_code == 200
    data = response.json()
    expected_keys = {"lots", "full_kelly", "frac_kelly", "vol_scalar", "risk_usd", "impact_usd"}
    assert set(data.keys()) == expected_keys
    for k in expected_keys:
        assert isinstance(data[k], (int, float))


def test_volatility_bounds_success(client):
    """4. POST /volatility_bounds returns 200 OK."""
    payload = {"returns": [0.001, -0.002, 0.003, -0.001, 0.002] * 5, "target_vol": 0.10, "lookback": 10}
    response = client.post("/volatility_bounds", json=payload)
    assert response.status_code == 200


def test_volatility_bounds_response_schema(client):
    """5. Response of /volatility_bounds has exact keys with valid float/int values."""
    payload = {"returns": [0.001, -0.002, 0.003, -0.001, 0.002] * 5, "target_vol": 0.10, "lookback": 10}
    response = client.post("/volatility_bounds", json=payload)
    assert response.status_code == 200
    data = response.json()
    expected_keys = {"vol_scalar", "realized_vol"}
    assert set(data.keys()) == expected_keys
    for k in expected_keys:
        assert isinstance(data[k], (int, float))


# ===========================================================================
# Tier 2: Boundary & Corner Cases (5 tests)
# ===========================================================================


def test_kelly_sizing_invalid_win_prob(client):
    """1. win_prob < 0.0 or > 1.0 (expect error response like 422 or 400)."""
    payload_low = {
        "win_prob": -0.1,
        "win_loss_ratio": 1.5,
        "returns": [0.001, -0.002, 0.003],
        "price": 1.1000,
        "current_atr": 0.0005,
        "equity": 10000.0,
        "lot_size": 10000.0,
    }
    payload_high = payload_low.copy()
    payload_high["win_prob"] = 1.1

    res_low = client.post("/kelly_sizing", json=payload_low)
    res_high = client.post("/kelly_sizing", json=payload_high)
    assert res_low.status_code in (400, 422)
    assert res_high.status_code in (400, 422)


def test_kelly_sizing_empty_returns(client):
    """2. returns array is empty."""
    payload = {
        "win_prob": 0.55,
        "win_loss_ratio": 1.5,
        "returns": [],
        "price": 1.1000,
        "current_atr": 0.0005,
        "equity": 10000.0,
        "lot_size": 10000.0,
    }
    response = client.post("/kelly_sizing", json=payload)
    assert response.status_code in (200, 400, 422)
    if response.status_code == 200:
        data = response.json()
        assert "lots" in data
        assert isinstance(data["lots"], (int, float))


def test_kelly_sizing_negative_values(client):
    """3. negative values for price, current_atr, equity, or lot_size."""
    base_payload = {
        "win_prob": 0.55,
        "win_loss_ratio": 1.5,
        "returns": [0.001, -0.002, 0.003],
        "price": 1.1000,
        "current_atr": 0.0005,
        "equity": 10000.0,
        "lot_size": 10000.0,
    }
    for field in ["price", "current_atr", "equity", "lot_size"]:
        bad_payload = base_payload.copy()
        bad_payload[field] = -10.0
        response = client.post("/kelly_sizing", json=bad_payload)
        assert response.status_code in (400, 422)


def test_volatility_bounds_lookback_bounds(client):
    """4. lookback <= 0."""
    payload = {"returns": [0.001, -0.002, 0.003], "target_vol": 0.10, "lookback": 0}
    response = client.post("/volatility_bounds", json=payload)
    assert response.status_code in (400, 422)


def test_volatility_bounds_empty_returns(client):
    """5. empty returns array."""
    payload = {"returns": [], "target_vol": 0.10, "lookback": 20}
    response = client.post("/volatility_bounds", json=payload)
    assert response.status_code in (200, 400, 422)
    if response.status_code == 200:
        data = response.json()
        assert data["vol_scalar"] == 1.0
        assert data["realized_vol"] == 0.0


# ===========================================================================
# Tier 3: Cross-Feature Interaction (3 tests)
# ===========================================================================


def test_api_inputs_from_zarr(client):
    """1. Load historical returns from test_rl.zarr and call /volatility_bounds."""
    import zarr

    zarr_path = Path(__file__).resolve().parent.parent / "test_rl.zarr"
    if not zarr_path.exists():
        pytest.skip(f"test_rl.zarr not found at {zarr_path}")

    store = zarr.open(str(zarr_path), mode="r")
    close = np.array(store["close"])
    # Calculate returns
    returns = np.diff(close) / (close[:-1] + 1e-9)
    returns_list = returns.tolist()

    payload = {"returns": returns_list, "target_vol": 0.10, "lookback": 20}
    response = client.post("/volatility_bounds", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["vol_scalar"] > 0.0
    assert data["realized_vol"] >= 0.0


def test_api_volatility_bounds_to_kelly(client):
    """2. Call /volatility_bounds first, scale inputs using the returned vol_scalar, and call /kelly_sizing."""
    returns = [0.001, -0.002, 0.003, -0.001, 0.002] * 10
    payload_vol = {"returns": returns, "target_vol": 0.10, "lookback": 20}
    response_vol = client.post("/volatility_bounds", json=payload_vol)
    assert response_vol.status_code == 200
    vol_scalar = response_vol.json()["vol_scalar"]

    payload_sizing = {
        "win_prob": 0.55,
        "win_loss_ratio": 1.5,
        "returns": returns,
        "price": 1.1000,
        "current_atr": 0.0005,
        "equity": 10000.0,
        "lot_size": 10000.0,
        "target_vol": 0.10 * vol_scalar,
    }
    response_sizing = client.post("/kelly_sizing", json=payload_sizing)
    assert response_sizing.status_code == 200
    assert response_sizing.json()["lots"] >= 0.0


def test_api_sizing_under_drawdown(client):
    """3. Call /kelly_sizing with varying equity to simulate drawdown scaling."""
    base_payload = {
        "win_prob": 0.55,
        "win_loss_ratio": 1.5,
        "returns": [0.001, -0.002, 0.003, -0.001, 0.002] * 5,
        "price": 1.1000,
        "current_atr": 0.0005,
        "equity": 10000.0,
        "lot_size": 10000.0,
    }
    response_normal = client.post("/kelly_sizing", json=base_payload)
    assert response_normal.status_code == 200
    lots_normal = response_normal.json()["lots"]

    drawdown_payload = base_payload.copy()
    drawdown_payload["equity"] = 5000.0
    response_drawdown = client.post("/kelly_sizing", json=drawdown_payload)
    assert response_drawdown.status_code == 200
    lots_drawdown = response_drawdown.json()["lots"]

    assert lots_drawdown < lots_normal


# ===========================================================================
# Tier 4: Real-World Scenarios (5 tests)
# ===========================================================================


def test_scenario_volatility_spike(client):
    """1. Simulate a volatility spike, verify vol_scalar goes down and realized vol goes up, and verify kelly scales down."""
    low_vol_returns = [0.001, -0.001, 0.0015, -0.0012, 0.0008] * 10
    high_vol_returns = [r * 10.0 for r in low_vol_returns]

    res_low = client.post("/volatility_bounds", json={"returns": low_vol_returns, "target_vol": 0.10, "lookback": 20})
    assert res_low.status_code == 200
    vol_low = res_low.json()

    res_high = client.post("/volatility_bounds", json={"returns": high_vol_returns, "target_vol": 0.10, "lookback": 20})
    assert res_high.status_code == 200
    vol_high = res_high.json()

    assert vol_high["realized_vol"] > vol_low["realized_vol"]
    assert vol_high["vol_scalar"] < vol_low["vol_scalar"]

    payload_low = {
        "win_prob": 0.55,
        "win_loss_ratio": 1.5,
        "returns": low_vol_returns,
        "price": 1.1000,
        "current_atr": 0.1500,
        "equity": 10000.0,
        "lot_size": 10000.0,
        "max_position_pct": 1.0,
        "pip_risk": 2000.0,
    }
    payload_high = payload_low.copy()
    payload_high["returns"] = high_vol_returns

    size_low = client.post("/kelly_sizing", json=payload_low).json()
    size_high = client.post("/kelly_sizing", json=payload_high).json()

    assert size_high["lots"] < size_low["lots"]


def test_scenario_drawdown_halt(client):
    """2. Simulate drawdown triggering sizing block."""
    from risk.execution import DrawdownAwareExitManager

    dm = DrawdownAwareExitManager(soft_dd=0.05, hard_dd=0.10, daily_limit=0.03)
    equity = 10000.0
    dm.update(equity, 0.0)

    equity = 9800.0
    status = dm.update(equity, -200.0)
    assert status["action"] == "continue"

    equity = 9450.0
    status = dm.update(equity, -350.0)
    assert status["action"] == "reduce_50"
    assert status["size_multiplier"] == 0.5

    equity = 8900.0
    status = dm.update(equity, -550.0)
    assert status["action"] == "close_all"
    assert status["size_multiplier"] == 0.0
    assert dm.status()["halted"] is True


def test_scenario_walk_forward(client):
    """3. Simulate walk-forward scaling."""
    np.random.seed(42)
    sim_returns = np.random.normal(0.0001, 0.005, 100).tolist()
    lookback = 20
    equity = 10000.0

    for i in range(5):
        window_returns = sim_returns[i : i + lookback]
        res_vol = client.post(
            "/volatility_bounds", json={"returns": window_returns, "target_vol": 0.10, "lookback": lookback}
        )
        assert res_vol.status_code == 200
        vol_scalar = res_vol.json()["vol_scalar"]

        res_size = client.post(
            "/kelly_sizing",
            json={
                "win_prob": 0.54,
                "win_loss_ratio": 1.4,
                "returns": window_returns,
                "price": 1.1000 + i * 0.01,
                "current_atr": 0.0005,
                "equity": equity,
                "target_vol": 0.10 * vol_scalar,
            },
        )
        assert res_size.status_code == 200
        lots = res_size.json()["lots"]
        assert lots >= 0.0
        equity += lots * 10


def test_scenario_rl_vs_xgboost(client):
    """4. Check performance comparison between RL and XGBoost models."""
    returns = [0.001, -0.002, 0.003, -0.001, 0.002] * 5

    payload_rl = {
        "win_prob": 0.58,
        "win_loss_ratio": 1.5,
        "returns": returns,
        "price": 1.1000,
        "current_atr": 0.1500,
        "equity": 10000.0,
        "target_vol": 0.02,
        "max_position_pct": 1.0,
        "pip_risk": 10000.0,
    }

    payload_xgb = payload_rl.copy()
    payload_xgb["win_prob"] = 0.52

    res_rl = client.post("/kelly_sizing", json=payload_rl)
    res_xgb = client.post("/kelly_sizing", json=payload_xgb)

    assert res_rl.status_code == 200
    assert res_xgb.status_code == 200

    lots_rl = res_rl.json()["lots"]
    lots_xgb = res_xgb.json()["lots"]

    assert lots_rl > lots_xgb


def test_scenario_live_sizing_regimes(client):
    """5. Check normal, trending, and mean-reversion regimes."""
    from risk.execution import RegimePositionSizer

    sizer_trend = RegimePositionSizer(base_kelly=0.25, hurst_trending=0.60, trending_bonus=1.20)
    res_trend = sizer_trend.size(
        equity=10000.0, win_prob=0.55, win_loss_r=1.5, returns=np.array([0.001, -0.002, 0.003]), atr=0.0005, hurst=0.65
    )
    assert res_trend["regime"] == "trending"

    sizer_mr = RegimePositionSizer(base_kelly=0.25, hurst_mean_rev=0.40, mean_rev_penalty=0.75)
    res_mr = sizer_mr.size(
        equity=10000.0, win_prob=0.55, win_loss_r=1.5, returns=np.array([0.001, -0.002, 0.003]), atr=0.0005, hurst=0.35
    )
    assert res_mr["regime"] == "mean_rev"

    sizer_crisis = RegimePositionSizer(base_kelly=0.25, corr_crisis_thresh=0.70, corr_crisis_scale=0.50)
    res_crisis = sizer_crisis.size(
        equity=10000.0,
        win_prob=0.55,
        win_loss_r=1.5,
        returns=np.array([0.001, -0.002, 0.003]),
        atr=0.0005,
        corr_avg=0.85,
    )
    assert res_crisis["regime"] == "crisis"


def test_non_finite_returns_filtering(client):
    """Verify that returns containing NaN/Inf trigger a 400 Bad Request."""
    import json

    payload_vol = {
        "returns": [0.001, float("nan"), -0.002, float("inf"), 0.003, float("-inf"), -0.001, 0.002] * 5,
        "target_vol": 0.10,
        "lookback": 10,
    }
    response_vol = client.post(
        "/volatility_bounds", content=json.dumps(payload_vol), headers={"Content-Type": "application/json"}
    )
    assert response_vol.status_code == 400

    payload_sizing = {
        "win_prob": 0.55,
        "win_loss_ratio": 1.5,
        "returns": [0.001, float("nan"), -0.002, float("inf"), 0.003, float("-inf"), -0.001, 0.002] * 5,
        "price": 1.1000,
        "current_atr": 0.0005,
        "equity": 10000.0,
        "lot_size": 10000.0,
    }
    response_sizing = client.post(
        "/kelly_sizing", content=json.dumps(payload_sizing), headers={"Content-Type": "application/json"}
    )
    assert response_sizing.status_code == 400


def test_kelly_sizing_win_prob_zero_or_negative(client):
    """Verify that win_prob <= 0.0 ensures lots = 0.0 and other attributes are zeroed."""
    import json

    payload_zero = {
        "win_prob": 0.0,
        "win_loss_ratio": 1.5,
        "returns": [0.001, -0.002, 0.003, -0.001, 0.002] * 5,
        "price": 1.1000,
        "current_atr": 0.0005,
        "equity": 10000.0,
        "lot_size": 10000.0,
    }
    response_zero = client.post(
        "/kelly_sizing", content=json.dumps(payload_zero), headers={"Content-Type": "application/json"}
    )
    assert response_zero.status_code == 200
    data = response_zero.json()
    assert data["lots"] == 0.0
    assert data["full_kelly"] == 0.0
    assert data["frac_kelly"] == 0.0
    assert data["vol_scalar"] == 0.0
    assert data["risk_usd"] == 0.0
    assert data["impact_usd"] == 0.0
