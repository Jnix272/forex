"""Static guard against NameError-class crashes in library code.

2026-09-25 audit found 8 undefined names that crash at runtime (incl. a
missing ``import os`` on the live ONNX inference path and a Cyrillic-lettered
identifier). These ruff rules are clean now; keep them that way.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = [
    "data", "features", "labeling", "training", "models", "backtesting", "evaluation",
    "inference", "trading", "execution", "risk", "sizing", "monitoring", "pipeline",
    "retraining", "config", "common", "feature_store", "drift", "validation",
    "contracts", "api", "lineage", "audit",
]
# F821 undefined name, F823 local used before assignment,
# B023 loop variable captured by closure, E722 bare except
RULES = "F821,F823,B023,E722"


def _ruff_cmd():
    try:
        import ruff  # noqa: F401

        return [sys.executable, "-m", "ruff"]
    except ImportError:
        exe = shutil.which("ruff")
        return [exe] if exe else None


def test_no_undefined_names_in_library_code():
    cmd = _ruff_cmd()
    if cmd is None:
        pytest.skip("ruff not installed (pip install -r requirements-dev.txt)")
    pkgs = [p for p in PACKAGES if (ROOT / p).is_dir()]
    res = subprocess.run(
        [*cmd, "check", *pkgs, "--select", RULES, "--output-format", "concise", "--no-cache", "-q"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert res.returncode == 0, f"ruff {RULES} findings:\n{res.stdout}{res.stderr}"
