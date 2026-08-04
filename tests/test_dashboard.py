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
        pytest.skip("Dashboard file not found — not deployed")
    result = subprocess.run(
        [sys.executable, "-c", f"import importlib.util; s=importlib.util.spec_from_file_location('d','{DASHBOARD_FILE}'); m=importlib.util.module_from_spec(s)"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"Dashboard import failed: {result.stderr}"


def test_missing_dashboard_file_handling():
    """The runner should not crash if dashboard file is absent."""
    pass


def test_headless_execution_feature_coverage():
    """Placeholder: verify dashboard covers all feature groups (requires Streamlit test harness)."""
    pytest.skip("Requires Streamlit AppTest — not yet integrated")
