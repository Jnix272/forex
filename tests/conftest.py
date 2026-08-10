"""Pytest: ensure repo root is on sys.path (same layout as training scripts)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# Standalone integration harness (custom `test()` decorator, runs under
# `python tests/test_all_new_components.py`). Not a pytest module — exclude
# from collection so pytest doesn't interpret `def test(name)` as a fixture.
collect_ignore = ["test_all_new_components.py"]


def pytest_addoption(parser):
    parser.addoption(
        "--quick-mode",
        action="store_true",
        default=False,
        help="Pass --quick-mode to train_gpu.py: fewer folds/epochs, no ensemble or RL.",
    )
