"""
risk/yaml_risk.py - Apply the run YAML ``risk:`` block to ``settings.LIVE_RISK``.

The live engine is launched without the training CLI, so without this module it
traded on the hardcoded ``LIVE_RISK`` defaults while ``config/run.yaml`` held
different (tighter) session / regime limits. The YAML is the source of truth;
``LIVE_RISK`` is mutated in place so every module that captured a reference to
it (risk.execution, risk.risk_engine) sees the effective values.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_RUN_CONFIG = "config/run.yaml"

# YAML regime names -> LIVE_RISK["regime_scale"] keys read by RegimePositionSizer.
REGIME_SCALE_ALIASES = {
    "volatile": "crisis",
    "ranging": "mean_rev",
}

_SCALAR_KEYS = (
    "kelly_fraction",
    "max_position_pct",
    "max_total_lots",
    "target_annual_vol",
    "pip_risk_default",
    "max_drawdown_halt",
    "soft_drawdown_reduce",
    "daily_loss_limit",
    "max_consecutive_losses",
    "recovery_bars",
    "atr_multiplier",
    "trail_activation_r",
    "breakeven_at_r",
    "corr_crisis_threshold",
    "hurst_trending",
    "hurst_mean_rev",
    "var_confidence",
    "var_max_pct",
    "pip_value",
    "max_notional_usd",
    "max_order_freq_per_min",
    "max_instrument_concentration",
    "max_leverage",
)


def resolve_run_config_path(explicit: str | os.PathLike | None = None) -> Path:
    """Explicit (non-default) path > FOREX_RUN_CONFIG/FOREX_CONFIG env > config/run.yaml."""
    if explicit is not None and str(explicit) and str(explicit) != DEFAULT_RUN_CONFIG:
        return Path(explicit)
    env = os.environ.get("FOREX_RUN_CONFIG") or os.environ.get("FOREX_CONFIG")
    if env:
        return Path(env)
    return Path(explicit or DEFAULT_RUN_CONFIG)


def load_yaml_risk(path: str | os.PathLike) -> dict | None:
    """Return the ``risk:`` mapping from a run YAML, or None if the file is absent.

    A present-but-unparseable file raises: silently trading on defaults when the
    operator's limits exist but cannot be read is not acceptable.
    """
    p = Path(path)
    if not p.is_file():
        return None
    import yaml

    with open(p, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{p}: top-level YAML is not a mapping")
    risk = cfg.get("risk")
    if risk is None:
        return None
    if not isinstance(risk, dict):
        raise ValueError(f"{p}: risk: block is not a mapping")
    return risk


def _num(key: str, value: Any) -> float:
    v = float(value)
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"risk.{key}={value!r} must be a finite non-negative number")
    return v


def apply_yaml_risk(risk: dict, live_risk: dict | None = None) -> dict:
    """Deep-merge a YAML ``risk:`` mapping into ``live_risk`` (default settings.LIVE_RISK).

    Returns a dict of ``{dotted_key: (old, new)}`` for every value that changed.
    """
    if live_risk is None:
        from config import settings as _settings

        live_risk = _settings.LIVE_RISK
    changes: dict[str, tuple[Any, Any]] = {}

    for key in _SCALAR_KEYS:
        if key in risk and risk[key] is not None:
            new = _num(key, risk[key])
            if key in ("max_consecutive_losses", "recovery_bars", "max_order_freq_per_min"):
                new = int(new)
            old = live_risk.get(key)
            if old != new:
                changes[key] = (old, new)
            live_risk[key] = new

    rs = risk.get("regime_scale")
    if isinstance(rs, dict):
        dest = dict(live_risk.get("regime_scale") or {})
        for k, v in rs.items():
            canon = REGIME_SCALE_ALIASES.get(str(k), str(k))
            new = _num(f"regime_scale.{k}", v)
            if dest.get(canon) != new:
                changes[f"regime_scale.{canon}"] = (dest.get(canon), new)
            dest[canon] = new
        live_risk["regime_scale"] = dest

    sl = risk.get("session_limits")
    if isinstance(sl, dict):
        dest = dict(live_risk.get("session_limits") or {})
        for session, limits in sl.items():
            if not isinstance(limits, dict):
                continue
            base = dict(dest.get(str(session)) or {})
            for lk in ("max_lots", "max_open_trades"):
                if lk in limits and limits[lk] is not None:
                    new = _num(f"session_limits.{session}.{lk}", limits[lk])
                    if lk == "max_open_trades":
                        new = int(new)
                    if base.get(lk) != new:
                        changes[f"session_limits.{session}.{lk}"] = (base.get(lk), new)
                    base[lk] = new
            # hours_local / tz are preserved from LIVE_RISK defaults.
            dest[str(session)] = base
        live_risk["session_limits"] = dest
    return changes


def effective_limits_summary(live_risk: dict | None = None) -> dict:
    """Compact, loggable view of the limits that gate live orders."""
    if live_risk is None:
        from config import settings as _settings

        live_risk = _settings.LIVE_RISK
    sessions = {
        name: {"max_lots": lim.get("max_lots"), "max_open_trades": lim.get("max_open_trades")}
        for name, lim in (live_risk.get("session_limits") or {}).items()
        if isinstance(lim, dict)
    }
    return {
        "kelly_fraction": live_risk.get("kelly_fraction"),
        "max_position_pct": live_risk.get("max_position_pct"),
        "max_total_lots": live_risk.get("max_total_lots"),
        "daily_loss_limit": live_risk.get("daily_loss_limit"),
        "max_drawdown_halt": live_risk.get("max_drawdown_halt"),
        "var_confidence": live_risk.get("var_confidence"),
        "var_max_pct": live_risk.get("var_max_pct"),
        "regime_scale": dict(live_risk.get("regime_scale") or {}),
        "session_limits": sessions,
    }


def apply_run_config_risk(path: str | os.PathLike | None = None, live_risk: dict | None = None) -> dict:
    """Load ``risk:`` from the run YAML, apply it to LIVE_RISK and log the result.

    Returns ``{"source": str, "applied": bool, "changes": {...}, "effective": {...}}``.
    """
    cfg_path = resolve_run_config_path(path)
    risk = load_yaml_risk(cfg_path)
    changes: dict = {}
    if risk is None:
        source = f"settings.LIVE_RISK defaults (no risk: block in {cfg_path})"
        applied = False
    else:
        changes = apply_yaml_risk(risk, live_risk)
        source = str(cfg_path)
        applied = True
    effective = effective_limits_summary(live_risk)
    msg = f"[Risk] Effective LIVE_RISK source={source} changes={len(changes)}"
    print(msg)
    logger.info(msg)
    for key, (old, new) in sorted(changes.items()):
        print(f"[Risk]   {key}: {old} -> {new}")
    print(f"[Risk]   effective={effective}")
    return {"source": source, "applied": applied, "changes": changes, "effective": effective}
