"""Fast import-level smoke tests for core project wiring.

These tests intentionally skip when a third-party runtime dependency is absent
in a minimal shell, but they fail if one of our internal project imports is
broken. Pytest can discover this file, and it can also run with unittest alone.
"""

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CORE_MODULES = (
    "validation.model_diagnostics",
    "models.architectures",
    "models.ensemble",
    "trading.live_engine",
    "training.train_gpu",
)

PROJECT_ROOTS = {
    "backtesting",
    "config",
    "data",
    "features",
    "inference",
    "infrastructure",
    "labeling",
    "models",
    "monitoring",
    "pretrain",
    "risk",
    "sizing",
    "tests",
    "trading",
    "training",
    "validation",
}


def _safe_test_name(module_name: str) -> str:
    return module_name.replace(".", "_")


def _make_import_test(module_name: str):
    def test_import(self: unittest.TestCase) -> None:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            missing_root = (exc.name or "").split(".", maxsplit=1)[0]
            if missing_root and missing_root not in PROJECT_ROOTS:
                self.skipTest(f"optional/runtime dependency missing while importing {module_name}: {exc.name}")
            raise
        self.assertIsNotNone(module)

    test_import.__name__ = f"test_import_{_safe_test_name(module_name)}"
    return test_import


class TestCoreImportSmoke(unittest.TestCase):
    pass


for _module_name in CORE_MODULES:
    setattr(
        TestCoreImportSmoke,
        f"test_import_{_safe_test_name(_module_name)}",
        _make_import_test(_module_name),
    )


if __name__ == "__main__":
    unittest.main()
