"""
settings.py ↔ run.yaml mismatch audits (section-by-section).

YAML is authoritative when loaded via --config. Shared-key value drift is
normally a warning (documentation / fallback smell). A small critical set
fails closed when settings stubs are meant to mirror the active YAML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# YAML top-level key → settings.py attribute name
SECTION_MAP: dict[str, str] = {
    "training": "TRAINING",
    "execution": "EXECUTION",
    "risk": "LIVE_RISK",
    "backtest": "BACKTEST",
    "data": "DATA",
    "distillation": "DISTILLATION",
    "rl": "RL",
    "monitoring": "MONITORING",
    "curriculum": "CURRICULUM",
    "validation": "VALIDATION",
    "pretrain": "PRETRAIN",
}

# Shared keys that must stay synced (settings used as no-YAML fallback / docs).
# Curriculum schedule fields are owned by audit_settings_yaml_curriculum_drift.
CRITICAL_SHARED_KEYS: frozenset[str] = frozenset(
    {
        "backtest.atr_stop_mult",
        "rl.reward.overtrade",
        "training.seq_len",
        "training.loss",
        "training.sharpe_annualization_factor",
        "training.lr_warmup_epochs",
        "validation.embargo_bars",
        "validation.purge_bars",
        "pretrain.epochs",
        "distillation.temperature",
    }
)

# Hardware / duration knobs: intentional drift on profile YAMLs (e.g. run_ubuntu.yaml).
# Still fail-closed when auditing the canonical run.yaml against settings stubs.
PROFILE_SCALE_KEYS: frozenset[str] = frozenset(
    {
        "pretrain.epochs",
        "pretrain.min_epochs",
        "pretrain.pretrain_epochs",
        "training.epochs",
        "training.batch_size",
        "training.patience",
        "data.chunk_size",
    }
)

_CANONICAL_YAML_NAMES: frozenset[str] = frozenset({"run.yaml"})

# YAML strategy.* ↔ settings.LABELING (label/target contract).
_STRATEGY_LABELING_MAP: dict[str, str] = {
    "lookahead_bars": "lookahead_bars",
    "profit_target_atr": "profit_target_atr",
    "stop_loss_atr": "stop_loss_atr",
}

# Path / machine-local / order-only keys - never fail, optionally skip entirely.
IGNORE_KEYS: frozenset[str] = frozenset(
    {
        "training.checkpoint_dir",
        "distillation.teacher_ckpt",
        "pretrain.checkpoint",
        "data.pairs",  # order-only; same membership is fine
    }
)

# Nested curriculum feature lists are audited elsewhere.
_SKIP_PREFIXES: tuple[str, ...] = (
    "curriculum.feature_groups.",
    "curriculum.adaptation.",
    "curriculum.calibration.",
)


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not isinstance(obj, dict):
        return out
    for key, value in obj.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten(value, path))
        else:
            out[path] = value
    return out


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    if isinstance(left, list) and isinstance(right, list):
        if left == right:
            return True
        # Order-insensitive string lists (e.g. pairs)
        if left and right and all(isinstance(x, str) for x in (*left, *right)):
            return set(left) == set(right)
        return False
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left) == bool(right)
    return left == right


def _should_skip(path: str) -> bool:
    if path in IGNORE_KEYS:
        return True
    return any(path.startswith(p) for p in _SKIP_PREFIXES)


def _is_canonical_yaml(yaml_path: str) -> bool:
    return Path(yaml_path).name in _CANONICAL_YAML_NAMES


def _key_is_critical(key: str, yaml_path: str, critical: frozenset[str]) -> bool:
    if key not in critical:
        return False
    # Profile YAMLs may scale epochs/batch/chunk without updating settings stubs.
    return not (key in PROFILE_SCALE_KEYS and not _is_canonical_yaml(yaml_path))


def audit_strategy_vs_labeling(
    yaml_cfg: dict | None,
    *,
    yaml_path: str = "config/run.yaml",
) -> dict[str, Any]:
    """Fail closed when YAML strategy barriers diverge from settings.LABELING."""
    errors: list[str] = []
    warnings: list[str] = []
    mismatches: list[dict[str, Any]] = []
    raw = yaml_cfg if isinstance(yaml_cfg, dict) else {}
    strat = raw.get("strategy") if isinstance(raw.get("strategy"), dict) else {}
    lbl = _load_settings_section("LABELING") or {}
    if not strat:
        warnings.append(f"{yaml_path}: strategy block missing - skip LABELING cross-check")
        return {
            "yaml_path": yaml_path,
            "mismatches": mismatches,
            "errors": errors,
            "warnings": warnings,
        }
    for y_key, s_key in _STRATEGY_LABELING_MAP.items():
        if y_key not in strat or s_key not in lbl:
            continue
        if not _values_equal(strat[y_key], lbl[s_key]):
            msg = f"strategy.{y_key}={strat[y_key]!r} != LABELING.{s_key}={lbl[s_key]!r} ({yaml_path})"
            mismatches.append(
                {
                    "key": f"strategy.{y_key}",
                    "settings": lbl[s_key],
                    "yaml": strat[y_key],
                    "critical": True,
                }
            )
            errors.append(msg)
    return {
        "yaml_path": yaml_path,
        "mismatches": mismatches,
        "errors": errors,
        "warnings": warnings,
    }


def _load_settings_section(attr: str) -> dict | None:
    try:
        from config import settings as S
    except Exception:
        return None
    obj = getattr(S, attr, None)
    return obj if isinstance(obj, dict) else None


def audit_settings_yaml_section_mismatches(
    yaml_cfg: dict | None,
    *,
    yaml_path: str = "config/run.yaml",
    sections: dict[str, str] | None = None,
    critical_keys: frozenset[str] | None = None,
) -> dict[str, Any]:
    """
    Compare shared keys between settings.py dicts and YAML sections.

    Returns per-section reports plus aggregated errors/warnings.
    """
    raw = yaml_cfg if isinstance(yaml_cfg, dict) else {}
    section_map = sections or SECTION_MAP
    critical = critical_keys if critical_keys is not None else CRITICAL_SHARED_KEYS

    errors: list[str] = []
    warnings: list[str] = []
    parts: dict[str, Any] = {}
    mismatches: list[dict[str, Any]] = []

    for yaml_key, settings_attr in section_map.items():
        y_sec = raw.get(yaml_key)
        s_sec = _load_settings_section(settings_attr)
        part: dict[str, Any] = {
            "yaml_key": yaml_key,
            "settings_attr": settings_attr,
            "present_in_yaml": isinstance(y_sec, dict),
            "present_in_settings": isinstance(s_sec, dict),
            "mismatches": [],
            "only_settings": [],
            "only_yaml": [],
        }
        if not isinstance(y_sec, dict) or not isinstance(s_sec, dict):
            if isinstance(y_sec, dict) and not isinstance(s_sec, dict):
                warnings.append(f"{yaml_path}:{yaml_key} has no settings.{settings_attr} counterpart")
            parts[yaml_key] = part
            continue

        # Prefix flattened paths with section name for critical-key matching.
        sf = {f"{yaml_key}.{k}": v for k, v in _flatten(s_sec).items()}
        yf = {f"{yaml_key}.{k}": v for k, v in _flatten(y_sec).items()}
        sk = {k for k in sf if not _should_skip(k)}
        yk = {k for k in yf if not _should_skip(k)}

        # For curriculum, only compare non-feature-list leaves (schedules already
        # covered by curriculum drift audit) - still report scalar/list drift here.
        only_s = sorted(sk - yk)
        only_y = sorted(yk - sk)
        part["only_settings"] = only_s[:40]
        part["only_yaml"] = only_y[:40]

        for key in sorted(sk & yk):
            if not _values_equal(sf[key], yf[key]):
                is_crit = _key_is_critical(key, yaml_path, critical)
                entry = {
                    "key": key,
                    "settings": sf[key],
                    "yaml": yf[key],
                    "critical": is_crit,
                }
                part["mismatches"].append(entry)
                mismatches.append(entry)
                msg = f"{key}: settings={sf[key]!r} != {yaml_path}={yf[key]!r}"
                if is_crit:
                    errors.append(msg)
                else:
                    warnings.append(msg)

        parts[yaml_key] = part

    # Cross-check strategy barriers vs LABELING (label/target contract).
    strat_rep = audit_strategy_vs_labeling(raw, yaml_path=yaml_path)
    errors.extend(strat_rep.get("errors") or [])
    warnings.extend(strat_rep.get("warnings") or [])
    mismatches.extend(strat_rep.get("mismatches") or [])
    parts["strategy_labeling"] = strat_rep

    return {
        "yaml_path": yaml_path,
        "parts": parts,
        "mismatches": mismatches,
        "errors": errors,
        "warnings": warnings,
    }


def audit_args_vs_yaml_mismatches(
    args: Any,
    yaml_cfg: dict | None,
    *,
    yaml_path: str = "config/run.yaml",
    arg_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Compare resolved CLI/args values against YAML for keys that should have loaded.

    Catches silent YAML-load failures (wrong indent, missing mapping).
    """
    # dest on args → dotted YAML path
    mapping = arg_map or {
        "batch_size": "training.batch_size",
        "epochs": "training.epochs",
        "seq_len": "training.seq_len",
        "loss": "training.loss",
        "grad_clip": "training.grad_clip",
        "patience": "training.patience",
        "weight_decay": "training.weight_decay",
        "sharpe_annualization_factor": "training.sharpe_annualization_factor",
        "bar_freq": "strategy.bar_freq",
        "strategy_mode": "strategy.mode",
        "profit_target_atr": "strategy.profit_target_atr",
        "stop_loss_atr": "strategy.stop_loss_atr",
        "lookahead_bars": "strategy.lookahead_bars",
    }
    raw = yaml_cfg if isinstance(yaml_cfg, dict) else {}
    flat = _flatten(raw)
    errors: list[str] = []
    warnings: list[str] = []
    mismatches: list[dict[str, Any]] = []

    for dest, ypath in mapping.items():
        if not hasattr(args, dest):
            continue
        if ypath not in flat:
            continue
        arg_val = getattr(args, dest)
        y_val = flat[ypath]
        if not _values_equal(arg_val, y_val):
            # Special case for time-anchored seq_len: args.seq_len is an int (resolved), y_val is a str (unresolved)
            if dest == "seq_len" and isinstance(arg_val, int) and isinstance(y_val, str):
                try:
                    # we don't have bar_freq readily available here, but we can assume if it's seq_len, it was intentionally resolved
                    continue
                except Exception:
                    pass

            entry = {
                "key": ypath,
                "args": arg_val,
                "yaml": y_val,
                "dest": dest,
            }
            mismatches.append(entry)
            msg = (
                f"args.{dest}={arg_val!r} != {yaml_path}:{ypath}={y_val!r} "
                "(CLI/strategy override or YAML may not have applied)"
            )
            # Soft: CLI flags and strategy profiles intentionally override YAML.
            # Hard errors for silent load failure belong in settings↔YAML critical
            # keys and the strategy-block presence check - not every args drift.
            warnings.append(msg)

    return {
        "yaml_path": yaml_path,
        "mismatches": mismatches,
        "errors": errors,
        "warnings": warnings,
    }


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config mapping (empty dict on missing file)."""
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    raw = yaml.safe_load(p.read_text(encoding="utf-8-sig"))
    return raw if isinstance(raw, dict) else {}


def format_part_warnings(report: dict[str, Any], *, prefix: str = "[ConfigMismatch]") -> list[str]:
    lines = [f"{prefix} {w}" for w in (report.get("warnings") or [])]
    lines.extend(f"{prefix} ERROR: {e}" for e in (report.get("errors") or []))
    return lines
