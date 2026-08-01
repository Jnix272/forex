"""Cross-platform Python interpreter discovery for wrapper scripts.

Zero-dependency helper so scripts/train.py, scripts/download_all.py,
scripts/run_pipeline.py, scripts/continuous_finetune.py and the e2e test
all resolve the project venv the same way on Windows and Linux.

Windows venv layout:  .venv\\Scripts\\python.exe
Linux/macOS layout:   .venv/bin/python
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _candidates() -> list[Path]:
    if platform.system() == "Windows":
        return [
            _ROOT / ".venv-gpu" / "Scripts" / "python.exe",
            _ROOT / ".venv" / "Scripts" / "python.exe",
        ]
    home = Path(os.path.expanduser("~"))
    return [
        _ROOT / ".venv" / "bin" / "python",
        _ROOT / ".venv-gpu" / "bin" / "python",
        home / "forex_venv" / "bin" / "python",
    ]


def python_exe() -> str:
    """Return a project venv interpreter if one exists, else sys.executable."""
    for venv in _candidates():
        if venv.is_file():
            return str(venv)
    return sys.executable


def run_command(script: str, passthrough: list[str]) -> int:
    """Run a project script under the resolved interpreter."""
    import subprocess

    cmd = [python_exe(), str(_ROOT / script), *passthrough]
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    return int(subprocess.run(cmd, cwd=str(_ROOT)).returncode)
