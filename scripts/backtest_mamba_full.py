"""
Backward-compatible Mamba backtest entrypoint.

The generic backtester now handles checkpoint/config discovery for all models,
including Mamba. Keep this wrapper so older commands still work.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.backtest_model import run_backtest

if __name__ == "__main__":
    if "--model" not in sys.argv:
        sys.argv.extend(["--model", "mamba"])
    run_backtest()
