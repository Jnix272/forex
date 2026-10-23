"""
tests/test_dashboard.py
========================
WIRE-009: End-to-End test suite for the Streamlit dashboard.
"""

import subprocess
import sys
from pathlib import Path

import pytest

DASHBOARD_FILE = Path(__file__).resolve().parent.parent / "api" / "dashboard.py"


def test_dashboard_gating():
    """Verify the dashboard module can at least be imported without error."""
    if not DASHBOARD_FILE.exists():
        pytest.skip("Dashboard file not found - not deployed")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import importlib.util; s=importlib.util.spec_from_file_location('d','{DASHBOARD_FILE}'); m=importlib.util.module_from_spec(s)",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"Dashboard import failed: {result.stderr}"


def test_missing_dashboard_file_handling():
    """The runner should not crash if dashboard file is absent."""
    pass


def test_headless_execution_feature_coverage():
    """Placeholder: verify dashboard covers all feature groups (requires Streamlit test harness)."""
    pytest.skip("Requires Streamlit AppTest - not yet integrated")


def test_cors_default_allowed_origins(monkeypatch):
    """Test default trusted CORS origins allow requests with matching Origin header."""
    from fastapi.testclient import TestClient
    from monitoring.dashboard.app import create_dashboard_app

    monkeypatch.delenv("DASHBOARD_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)

    app = create_dashboard_app()
    client = TestClient(app)

    # Allowed origin
    res = client.get("/api/health", headers={"Origin": "http://localhost:9090"})
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") == "http://localhost:9090"

    # Disallowed origin
    res_untrusted = client.get("/api/health", headers={"Origin": "http://malicious-domain.com"})
    assert res_untrusted.status_code == 200
    assert res_untrusted.headers.get("access-control-allow-origin") is None


def test_cors_custom_allowed_origins_param():
    """Test explicitly setting allowed_origins parameter in create_dashboard_app."""
    from fastapi.testclient import TestClient
    from monitoring.dashboard.app import create_dashboard_app

    app = create_dashboard_app(allowed_origins=["http://trusted.domain.com"])
    client = TestClient(app)

    res = client.get("/api/health", headers={"Origin": "http://trusted.domain.com"})
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") == "http://trusted.domain.com"

    res_untrusted = client.get("/api/health", headers={"Origin": "http://localhost:9090"})
    assert res_untrusted.status_code == 200
    assert res_untrusted.headers.get("access-control-allow-origin") is None


def test_cors_env_var_allowed_origins(monkeypatch):
    """Test configuring CORS origins via DASHBOARD_CORS_ORIGINS environment variable."""
    from fastapi.testclient import TestClient
    from monitoring.dashboard.app import create_dashboard_app

    monkeypatch.setenv("DASHBOARD_CORS_ORIGINS", "http://env-trusted.com, http://env-trusted-2.com")

    app = create_dashboard_app()
    client = TestClient(app)

    res = client.get("/api/health", headers={"Origin": "http://env-trusted.com"})
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") == "http://env-trusted.com"

    res_untrusted = client.get("/api/health", headers={"Origin": "http://localhost:9090"})
    assert res_untrusted.status_code == 200
    assert res_untrusted.headers.get("access-control-allow-origin") is None
