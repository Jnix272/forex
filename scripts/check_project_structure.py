"""Check for files that should not live at the repository root."""

from __future__ import annotations

import fnmatch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ALLOWED_ROOT_FILES = {
    ".dockerignore",
    ".env",
    ".gitignore",
    "__init__.py",
    "CHANGELOG.md",
    "main.py",
    "pyproject.toml",
    "README.md",
    "requirements.txt",
    "requirements_gpu.txt",
    "requirements-base.txt",
    "requirements-dev.txt",
    "run.ps1",
    "SETUP.md",
    "setup_ubuntu.sh",
}

ALLOWED_ROOT_DIRS = {
    ".claude",
    ".codex",
    ".venv-gpu",
    ".vs",
    ".vscode",
    "artifacts",
    "backtesting",
    "checkpoints",
    "config",
    "configs",
    "data",
    "docs",
    "features",
    "inference",
    "infrastructure",
    "labeling",
    "logs",
    "models",
    "monitoring",
    "pretrain",
    "risk",
    "scripts",
    "sizing",
    "tests",
    "trading",
    "training",
    "validation",
    "vcpkg",
    "wandb",
}

ROOT_CLUTTER_PATTERNS = (
    "patch*.py",
    "fix_*.py",
    "find_terms.py",
    "test.py",
    "promotion_*.json",
    "parity_*.csv",
)


def _matches_any(name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def main() -> int:
    problems: list[str] = []

    for path in sorted(ROOT.iterdir(), key=lambda item: item.name.lower()):
        name = path.name
        if path.is_file():
            if name not in ALLOWED_ROOT_FILES or _matches_any(name, ROOT_CLUTTER_PATTERNS):
                problems.append(f"root file should move: {name}")
        elif path.is_dir() and name not in ALLOWED_ROOT_DIRS:
            problems.append(f"root directory should move or be documented: {name}/")

    if not problems:
        print("Project structure check passed.")
        return 0

    print("Project structure check found root clutter:")
    for problem in problems:
        print(f"- {problem}")
    print("\nSee docs/PROJECT_STRUCTURE_IMPROVEMENT_PLAN.md for target locations.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
