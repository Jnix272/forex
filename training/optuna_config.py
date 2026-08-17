"""Resolve and apply Optuna best-run YAML overlays for train_gpu."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:
    import yaml as _yaml

    _YAML = True
except ImportError:
    _YAML = False

OPTUNA_CONFIG_DIR = Path("config/optuna")
DEFAULT_METRICS = ("val_loss", "val_sharpe")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", str(text).lower()).strip("-") or "run"


def find_best_optuna_config(model_name: str, metric: str | None = None) -> Path | None:
    """Return archived best Optuna YAML for *model_name* (and optional *metric*)."""
    model = str(model_name).lower().strip()
    if metric:
        path = OPTUNA_CONFIG_DIR / f"run_optuna_best_{_slug(model)}_{_slug(metric)}.yaml"
        return path if path.is_file() else None
    candidates = sorted(
        OPTUNA_CONFIG_DIR.glob(f"run_optuna_best_{_slug(model)}_*.yaml"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def resolve_optuna_config_path(model_name: str, base_cfg: dict[str, Any]) -> Path | None:
    """Pick the best archived Optuna config for *model_name* using yaml hints."""
    optuna_sec = base_cfg.get("optuna") or {}
    metric = optuna_sec.get("metric")
    if metric:
        found = find_best_optuna_config(model_name, str(metric))
        if found:
            return found
    for fallback in DEFAULT_METRICS:
        found = find_best_optuna_config(model_name, fallback)
        if found:
            return found
    return find_best_optuna_config(model_name, None)


def should_apply_optuna_overlay(
    base_cfg: dict[str, Any],
    optuna_path: Path,
    base_path: Path,
    cli_auto: bool | None,
) -> bool:
    optuna_sec = base_cfg.get("optuna") or {}
    if cli_auto is False:
        return False
    if cli_auto is None and not optuna_sec.get("auto_load", True):
        return False
    if not optuna_path.is_file():
        return False
    if optuna_path.resolve() == base_path.resolve():
        return False
    if optuna_sec.get("applied") and cli_auto is not True:
        return False
    if not optuna_sec.get("applied"):
        return True
    if optuna_path.stat().st_mtime > base_path.stat().st_mtime:
        return True
    return cli_auto is True


def apply_optuna_overlay_if_needed(
    parser: Any,
    config_path: str | None,
    cli_auto: bool | None,
    apply_yaml_fn: Callable[[Any, str], None],
) -> None:
    """Merge Optuna best YAML into argparse defaults when auto-load is enabled."""
    if not _YAML:
        return

    base_path = Path(config_path or "config/run.yaml")
    if not base_path.is_file():
        return

    defaults = getattr(parser, "_defaults", {}) or {}
    if bool(defaults.get("all_models")):
        return

    try:
        with base_path.open("r", encoding="utf-8") as fh:
            base_cfg = _yaml.safe_load(fh) or {}
    except Exception:
        return

    model = str(defaults.get("model") or (base_cfg.get("model") or {}).get("name") or "haelt").lower()
    optuna_path = resolve_optuna_config_path(model, base_cfg)
    if optuna_path is None:
        return
    if not should_apply_optuna_overlay(base_cfg, optuna_path, base_path, cli_auto):
        return

    apply_yaml_fn(parser, str(optuna_path))
    print(f"[Optuna] Auto-loaded best config overlay from {optuna_path}")


def read_run_yaml_optuna_section(config_path: Path | None = None) -> dict[str, Any]:
    if not _YAML:
        return {}
    path = config_path or Path("config/run.yaml")
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            cfg = _yaml.safe_load(fh) or {}
    except Exception:
        return {}
    sec = cfg.get("optuna")
    return sec if isinstance(sec, dict) else {}
