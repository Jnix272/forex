"""YAML config loading and runtime-settings synchronisation helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import yaml as _yaml

    _YAML = True
except ImportError:
    _yaml = None  # type: ignore[assignment]
    _YAML = False

from config import settings as _settings
from config.settings import (
    EXECUTION as SETTINGS_EXECUTION,
)
from training.core import _GPU_CFG

from .yaml_map import _YAML_MAP

# Alias map for legacy YAML regime_scale keys -> LIVE_RISK / RegimeScale names.
_REGIME_SCALE_ALIASES = {
    "volatile": "crisis",
    "ranging": "mean_rev",
    "unknown": "normal",
}


def _apply_yaml_config(parser: argparse.ArgumentParser, config_path: str) -> None:
    """Load config/run.yaml and set argparse defaults from it.

    YAML parse failures are fatal: swallowing them silently drops the entire
    config (strategy ATR targets, bar_freq, curriculum, ...) onto hardcoded
    argparse defaults - see docs / mismatch B.
    """
    if not _YAML or _yaml is None:
        print("[Config] PyYAML not installed - ignoring --config. pip install pyyaml")
        return
    path = Path(config_path)
    if not path.exists():
        print(f"[Config] WARN: config file not found: {config_path}")
        return

    try:
        with open(path, encoding="utf-8-sig") as fh:  # utf-8-sig strips BOM; CRLF is fine
            cfg = _yaml.safe_load(fh)
    except Exception as e:
        raise RuntimeError(
            f"[Config] YAML parse failed for {config_path}: {e}\n"
            "[Config] Fix the YAML (check indentation under strategy:/curriculum:) "
            "before training - defaults would otherwise silently ignore the file."
        ) from e
    if cfg is None:
        cfg = {}

    defaults: dict = {}
    distillation_enabled = bool((cfg.get("distillation") or {}).get("enabled", False))
    for yaml_key, dest in _YAML_MAP.items():
        if yaml_key.startswith("distillation.") and not distillation_enabled:
            continue
        section, key = yaml_key.split(".", 1)
        val = (cfg.get(section) or {}).get(key)
        if val is None:
            continue
        if dest is None:
            # data.use_cache=false -> force_rebuild=true
            if yaml_key == "data.use_cache":
                defaults["force_rebuild"] = not bool(val)
            continue
        # Blank strings mean "use the hardcoded default"
        if isinstance(val, str) and val.strip() == "":
            continue
        defaults[dest] = val

    if isinstance(cfg.get("curriculum"), dict):
        defaults["curriculum"] = cfg["curriculum"]
    if isinstance(cfg.get("training"), dict):
        adv = cfg["training"].get("adversarial")
        if isinstance(adv, dict):
            defaults["training_adversarial"] = adv
    if isinstance(cfg.get("curriculum"), dict):
        miner_fb = cfg["curriculum"].get("miner_feedback")
        if isinstance(miner_fb, dict):
            defaults["curriculum_miner_feedback"] = bool(miner_fb.get("enabled", False))
        sp = cfg["curriculum"].get("self_paced")
        if isinstance(sp, dict):
            defaults["use_self_paced"] = bool(sp.get("enabled", False))
        lw = cfg["curriculum"].get("loss_weighting")
        if isinstance(lw, dict):
            defaults["use_loss_weighting"] = bool(lw.get("enabled", False))
    if isinstance(cfg.get("feature_ablation"), dict):
        defaults["feature_ablation"] = cfg["feature_ablation"]

    if isinstance(cfg.get("execution"), dict):
        defaults["execution"] = cfg["execution"]

    if isinstance(cfg.get("risk"), dict):
        defaults["risk"] = cfg["risk"]

    if isinstance(cfg.get("sidecar"), dict):
        defaults["sidecar"] = cfg["sidecar"]

    if isinstance(cfg.get("feature_cache"), dict):
        defaults["feature_cache"] = cfg["feature_cache"]

    if isinstance(cfg.get("maturity"), dict):
        stage = cfg["maturity"].get("stage")
        if stage is not None:
            defaults["maturity_stage"] = stage

    # Sync strategy ATR / lookahead into LABELING so workers that read
    # settings.LABELING (not argparse) match YAML strategy.*
    strategy = cfg.get("strategy") or {}
    if isinstance(strategy, dict):
        if strategy.get("profit_target_atr") is not None:
            _settings.LABELING["profit_target_atr"] = float(strategy["profit_target_atr"])
        if strategy.get("stop_loss_atr") is not None:
            _settings.LABELING["stop_loss_atr"] = float(strategy["stop_loss_atr"])
        if strategy.get("lookahead_bars") is not None:
            _settings.LABELING["lookahead_bars"] = int(strategy["lookahead_bars"])

    features_sec = cfg.get("features") or {}
    if isinstance(features_sec, dict):
        for k in (
            "atr_windows",
            "vol_windows",
            "ofi_windows",
            "momentum_windows",
            "vwap_window",
            "chop_window",
            "corr_window",
            "regime_window",
            "volatility_window",
        ):
            if features_sec.get(k) is not None:
                _settings.FEATURE_SCALES[k] = features_sec[k]

    fc = cfg.get("feature_cache")
    if isinstance(fc, dict):
        _settings.FEATURE_CACHE.update(fc)
        # Alias legacy slow_cols name
        slow = list(_settings.FEATURE_CACHE.get("slow_cols") or [])
        _settings.FEATURE_CACHE["slow_cols"] = ["hurst_exponent" if c == "hurst" else c for c in slow]

    maturity = cfg.get("maturity")
    if isinstance(maturity, dict) and maturity.get("stage") is not None:
        _settings.MATURITY["stage"] = str(maturity["stage"])

    pretrain_sec = cfg.get("pretrain") or {}
    if isinstance(pretrain_sec.get("augmentations"), dict):
        defaults["pretrain_augmentations"] = pretrain_sec["augmentations"]
    if pretrain_sec.get("read_windows") is not None:
        _settings.PRETRAIN["read_windows"] = int(pretrain_sec["read_windows"])
        defaults["pretrain_read_windows"] = int(pretrain_sec["read_windows"])

    rl_sec = cfg.get("rl") or {}
    if isinstance(rl_sec.get("reward"), dict):
        defaults["rl_reward_weights"] = rl_sec["reward"]
    rl_overrides = {}
    if isinstance(rl_sec.get("dqn"), dict):
        rl_overrides["dqn"] = rl_sec["dqn"]
    if isinstance(rl_sec.get("ppo"), dict):
        rl_overrides["ppo"] = rl_sec["ppo"]
    if rl_overrides:
        defaults["rl_algo_overrides"] = rl_overrides

    parser.set_defaults(**defaults)
    print(f"[Config] Loaded {config_path}")


def _apply_yaml_risk_to_live_risk(risk: dict) -> None:
    """Deep-merge YAML ``risk:`` into ``settings.LIVE_RISK`` (session/regime aware)."""
    lr = _settings.LIVE_RISK
    scalar_keys = (
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
    )
    for key in scalar_keys:
        if key in risk and risk[key] is not None:
            lr[key] = risk[key]

    # pip_value in YAML is documentation for notional; map if LIVE_RISK ever grows it.
    if "pip_value" in risk and "pip_risk_default" not in risk:
        pass  # intentionally unused by LIVE_RISK; keep YAML for docs/RiskEngine extras

    rs = risk.get("regime_scale")
    if isinstance(rs, dict):
        dest = dict(lr.get("regime_scale") or {})
        for k, v in rs.items():
            canon = _REGIME_SCALE_ALIASES.get(str(k), str(k))
            dest[canon] = float(v)
        lr["regime_scale"] = dest

    sl = risk.get("session_limits")
    if isinstance(sl, dict):
        dest = dict(lr.get("session_limits") or {})
        for session, limits in sl.items():
            if not isinstance(limits, dict):
                continue
            base = dict(dest.get(session) or {})
            for lk in ("max_lots", "max_open_trades"):
                if lk in limits:
                    base[lk] = limits[lk]
            # Preserve hours_local / tz from LIVE_RISK defaults when YAML omits them.
            dest[session] = base
        lr["session_limits"] = dest
    print(
        "[Config] LIVE_RISK synced from YAML risk "
        f"(kelly={lr.get('kelly_fraction')}, "
        f"london_lots={(lr.get('session_limits') or {}).get('london', {}).get('max_lots')})"
    )


def _sync_runtime_config(args) -> None:
    """Apply YAML-only nested config blocks to modules that read config.settings."""
    execution = getattr(args, "execution", None)
    if isinstance(execution, dict):
        _settings.EXECUTION.update(execution)
        SETTINGS_EXECUTION.update(execution)

    risk = getattr(args, "risk", None)
    if isinstance(risk, dict):
        _apply_yaml_risk_to_live_risk(risk)

    # Strategy ATR -> LABELING (CLI may override YAML after parse)
    if getattr(args, "profit_target_atr", None) is not None:
        _settings.LABELING["profit_target_atr"] = float(args.profit_target_atr)
    if getattr(args, "stop_loss_atr", None) is not None:
        _settings.LABELING["stop_loss_atr"] = float(args.stop_loss_atr)
    if getattr(args, "lookahead_bars", None) is not None:
        _settings.LABELING["lookahead_bars"] = int(args.lookahead_bars)

    if getattr(args, "pretrain_read_windows", None) is not None:
        _settings.PRETRAIN["read_windows"] = int(args.pretrain_read_windows)

    if getattr(args, "maturity_stage", None):
        _settings.MATURITY["stage"] = str(args.maturity_stage)

    fc = getattr(args, "feature_cache", None)
    if isinstance(fc, dict):
        _settings.FEATURE_CACHE.update(fc)

    # hardware.torch_compile -> settings.GPU (and bound _GPU_CFG if same dict)
    if getattr(args, "torch_compile", None) is not None:
        _settings.GPU["torch_compile"] = bool(args.torch_compile)
        try:
            _GPU_CFG["torch_compile"] = bool(args.torch_compile)
        except Exception:
            pass
    if getattr(args, "torch_compile_mode", None):
        mode = str(args.torch_compile_mode)
        _settings.GPU["torch_compile_mode"] = mode
        try:
            _GPU_CFG["torch_compile_mode"] = mode
        except Exception:
            pass

    if not getattr(args, "legacy_config_globals", False):
        try:
            from config.runtime import build_runtime_config
            args._runtime_config = build_runtime_config(args)
        except Exception as exc:
            import warnings
            warnings.warn(f"[Config] RuntimeConfig validation failed: {exc}. Use --legacy-config-globals to skip.", stacklevel=2)


def _resolve_seq_len(val, bar_freq: str) -> int:
    if isinstance(val, int):
        return val
    if isinstance(val, str) and val.isdigit():
        return int(val)
    if isinstance(val, str):
        try:
            import pandas as pd

            t_val = pd.Timedelta(val).total_seconds()
            t_freq = pd.Timedelta(bar_freq or "5m").total_seconds()
            if t_freq > 0:
                bars = max(1, int(t_val / t_freq))
                print(f"[Config] Resolved time-anchored seq_len '{val}' -> {bars} bars (at {bar_freq})")
                return bars
        except Exception as e:
            print(f"[Config] WARNING: Failed to parse time-anchored seq_len '{val}' ({e}). Defaulting to 60.")
            return 60
    return int(val) if val else 60
