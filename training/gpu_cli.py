"""CLI / argparse / YAML / run-dir helpers for GPU training.\n\nSee docs/CONTINUE.md."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from config import settings as _settings
from config.model_training_profile import (
    ModelTrainingProfile,
    get_training_profile,
)
from config.settings import (
    CURRICULUM as SETTINGS_CURRICULUM,
)
from config.settings import (
    DISTILLATION,
    HARDWARE_PROFILES,
    MONITORING,
    PATHS,
    PRETRAIN,
    RL,
    TRAINING,
)
from config.settings import (
    ENSEMBLE as SETTINGS_ENSEMBLE,
)
from config.settings import (
    EXECUTION as SETTINGS_EXECUTION,
)
from config.strategy_profiles import STRATEGY_PROFILES, strategy_profile
from training.cache_integrity import _get_pairs
from training.core import _GPU_CFG, _log_warn
from training.dataset_builder import _safe_save_json

try:
    import yaml as _yaml

    _YAML = True
except ImportError:
    _yaml = None  # type: ignore[assignment]
    _YAML = False

# -----------------------------------------------------------------------------
# ARGUMENT PARSING
# -----------------------------------------------------------------------------

# Maps YAML keys (section.key) -> argparse dest names
_YAML_MAP = {
    "strategy.mode": "strategy_mode",
    "strategy.bar_freq": "bar_freq",
    "strategy.lookahead_bars": "lookahead_bars",
    "strategy.profit_target_atr": "profit_target_atr",
    "strategy.stop_loss_atr": "stop_loss_atr",
    "data.source": "data_source",
    "data.pair": "pair",
    "data.pairs": "pairs",
    "data.pair_embed_dim": "pair_embed_dim",
    "data.pair_align": "pair_align",
    "data.corr_window": "corr_window",
    "data.corr_window_long": "corr_window_long",
    "data.momentum_window": "momentum_window",
    "data.start": "data_start",
    "data.end": "data_end",
    "data.full_day_data": "full_day_data",
    "data.n_ticks": "n_ticks",
    "data.chunk_size": "chunk_size",
    "data.real_data_window_days": "real_data_window_days",
    "data.window_batch_days": "window_batch_days",
    "data.dataset_build_workers": "dataset_build_workers",
    "data.parallel_window_workers": "parallel_window_workers",
    "data.use_cache": None,  # handled as force_rebuild inversion below
    "model.name": "model",
    "model.all_models": "all_models",
    "model.hidden_size": "hidden_size",
    "model.d_model": "d_model",
    "model.nhead": "nhead",
    "model.num_layers": "num_layers",
    "model.dropout": "dropout",
    "training.epochs": "epochs",
    "training.batch_size": "batch_size",
    "training.lr": "lr",
    "training.seq_len": "seq_len",
    "training.patience": "patience",
    "training.val_split": "val_split",
    "training.tune_split": "tune_split",
    "training.curriculum_gate_metric": "curriculum_gate_metric",
    "training.loss": "loss",
    "training.label_method": "label_method",
    "training.early_stop_metric": "early_stop_metric",
    "training.early_stop_min_delta": "early_stop_min_delta",
    "training.direction_weight": "direction_weight",
    "training.sharpe_weight": "sharpe_weight",
    "training.sharpe_annualization_factor": "sharpe_annualization_factor",
    "training.fx_full_day": "fx_full_day",
    "training.save_every": "save_every",
    "training.grad_clip": "grad_clip",
    "training.grad_accum_steps": "grad_accum_steps",
    "training.swa_enabled": "swa_enabled",
    "training.swa_start_frac": "swa_start_frac",
    "training.swa_lr": "swa_lr",
    "training.weight_decay": "weight_decay",
    "training.seed": "seed",
    "training.lr_schedule": "lr_schedule",
    "training.lr_warmup_epochs": "lr_warmup_epochs",
    "training.lr_warmup_pct": "lr_warmup_pct",
    "training.lr_min_ratio": "lr_min_ratio",
    "training.onecycle_pct_start": "onecycle_pct_start",
    "training.onecycle_max_lr_mult": "onecycle_max_lr_mult",
    "training.amp": "amp",
    "training.resume": "resume",
    "training.training_memory": "training_memory",
    "training.label_smoothing": "label_smoothing",
    "training.use_mixup": "use_mixup",
    "training.use_volatility_sampler": "use_volatility_sampler",
    "training.max_bad_frac": "max_bad_frac",
    "training.max_zero_frac": "max_zero_frac",
    "training.cross_asset_mode": "cross_asset_mode",
    "training.cross_asset_provider": "cross_asset_provider",
    "maturity.stage": "maturity_stage",
    "features.chop_window": "feat_chop_window",
    "features.corr_window": "feat_corr_window",
    "features.regime_window": "feat_regime_window",
    "features.volatility_window": "feat_volatility_window",
    "features.vwap_window": "feat_vwap_window",
    "rl.use_sharpe_reward": "rl_use_sharpe_reward",
    "rl.use_her": "rl_use_her",
    "news.historical_mode": "historical_news_mode",
    "news.historical_news_file": "historical_news_file",
    "news.economic_calendar_file": "economic_calendar_file",
    # Backward-compat aliases for older run.yaml layouts
    "data.historical_news_mode": "historical_news_mode",
    "data.historical_news_file": "historical_news_file",
    "data.economic_calendar_file": "economic_calendar_file",
    "backtest.execution_delay_bars": "execution_delay_bars",
    "walk_forward.enabled": "walk_forward_cv",
    "walk_forward.folds": "walk_forward_folds",
    "multitask.enabled": "multitask",
    "multitask.w_ret": "mt_w_ret",
    "multitask.w_conf": "mt_w_conf",
    "multitask.class_balance_weight": "mt_class_balance_weight",
    "multitask.entropy_weight": "mt_entropy_weight",
    "multitask.direction_weight_floor": "mt_direction_weight_floor",
    "multitask.focal_gamma": "mt_focal_gamma",
    "direction_training.probe": "direction_probe",
    "direction_training.probe_epochs": "direction_probe_epochs",
    "direction_training.probe_samples": "direction_probe_samples",
    "direction_training.warmup_epochs": "direction_warmup_epochs",
    "direction_training.min_true_class_share": "direction_min_true_class_share",
    "direction_training.min_pred_class_share": "direction_min_pred_class_share",
    "direction_training.max_pred_class_share": "direction_max_pred_class_share",
    "direction_training.min_recall": "direction_min_recall",
    "direction_training.use_mixup": "use_mixup",
    "direction_training.use_volatility_sampler": "use_volatility_sampler",
    "direction_training.label_smoothing": "label_smoothing",
    # Adversarial (new)
    "training.adversarial.enabled": "enable_adversarial",
    "training.adversarial.method": "adversarial_method",
    "training.adversarial.eps": "adversarial_eps",
    "training.adversarial.alpha": "adversarial_alpha",
    "training.adversarial.steps": "adversarial_steps",
    "training.adversarial.prob": "adversarial_prob",
    "training.adversarial.normalize_grad": "adversarial_normalize_grad",
    "training.adversarial.warmup_steps": "adversarial_warmup_steps",
    "training.adversarial.eps_curriculum_scale": "adversarial_eps_curriculum_scale",
    "training.adversarial.models": "adversarial_models",
    # Continuous learning (EWC / Synaptic Intelligence)
    "training.enable_ewc": "enable_ewc",
    "training.ewc_lambda": "ewc_lambda",
    "training.enable_si": "enable_si",
    "training.si_lambda": "si_lambda",
    # Framework selection (new)
    "training.training_framework": "training_framework",
    "training.pretrain_framework": "pretrain_framework",
    "training.rl_framework": "rl_framework",
    # Pretrain framework (new)
    "pretrain.enabled": "pretrain",
    "pretrain.framework": "pretrain_framework",
    # RL framework (new)
    "rl.framework": "rl_framework",
    # Curriculum miner feedback (new)
    "curriculum.miner_feedback.enabled": "curriculum_miner_feedback",
    "curriculum.miner_feedback.models": "curriculum_miner_models",
    "curriculum.miner_feedback.forgetting_threshold": "curriculum_forgetting_threshold",
    "curriculum.miner_feedback.easy_threshold": "curriculum_easy_threshold",
    "curriculum.miner_feedback.freeze_patience": "curriculum_freeze_patience",
    # Self-paced learning (new)
    "curriculum.self_paced.enabled": "use_self_paced",
    "curriculum.self_paced.pace": "self_paced_pace",
    "curriculum.self_paced.lambda_pace": "self_paced_lambda",
    "curriculum.self_paced.models": "self_paced_models",
    # Loss weighting (new)
    "curriculum.loss_weighting.enabled": "use_loss_weighting",
    "curriculum.loss_weighting.scheme": "loss_weighting_scheme",
    "curriculum.loss_weighting.focal_gamma": "loss_weighting_focal_gamma",
    "curriculum.loss_weighting.models": "loss_weighting_models",
    # Backward-compat aliases for older run.yaml layouts
    "pretrain.ablation": "pretrain_ablation",
    "pretrain.ablation_models": "pretrain_ablation_models",
    "pretrain.method": "pretrain_method",
    "pretrain.epochs": "pretrain_epochs",
    "pretrain.regime_aware": "pretrain_regime",
    "pretrain.max_epochs": "pretrain_max_epochs",
    "pretrain.min_epochs": "pretrain_min_epochs",
    "pretrain.handoff_patience": "pretrain_handoff_patience",
    "pretrain.handoff_min_delta": "pretrain_handoff_min_delta",
    "pretrain.handoff_loss": "pretrain_handoff_loss",
    "pretrain.lr": "pretrain_lr",
    "pretrain.batch": "pretrain_batch",
    "pretrain.projection_dim": "pretrain_projection_dim",
    "pretrain.pred_dim": "pretrain_pred_dim",
    "pretrain.ema_decay": "pretrain_ema_decay",
    "pretrain.sample_windows": "pretrain_sample_windows",
    "pretrain.blocks_per_epoch": "pretrain_blocks_per_epoch",
    "pretrain.read_windows": "pretrain_read_windows",
    "pretrain.mask_prob": "pretrain_mask_prob",
    "pretrain.recon_hidden_dim": "pretrain_recon_hidden_dim",
    "pretrain.latent_dim": "pretrain_latent_dim",
    "pretrain.vae_beta": "pretrain_vae_beta",
    "pretrain.n_clusters": "pretrain_n_clusters",
    "pretrain.forecast_horizon": "pretrain_forecast_horizon",
    "pretrain.drift_margin": "pretrain_drift_margin",
    "ensemble.enabled": "train_ensemble",
    "ensemble.epochs": "ensemble_epochs",
    "ensemble.div_weight": "ensemble_div_weight",
    "ensemble.deploy": "deploy_ensemble",
    "ensemble.explicit_diversity": "ensemble_explicit_diversity",
    "ensemble.member_seed_offset": "ensemble_member_seed_offset",
    "ensemble.member_lr_jitter": "ensemble_member_lr_jitter",
    "ensemble.member_dropout_jitter": "ensemble_member_dropout_jitter",
    "rl.enabled": "rl_train",
    "rl.algo": "rl_algo",
    "rl.episodes": "rl_episodes",
    "rl.episode_len": "rl_episode_len",
    "rl.encoder_obs": "rl_encoder_obs",
    "rl.val_frac": "rl_val_frac",
    "rl.min_val_sharpe": "rl_min_val_sharpe",
    "rl.all_models": "rl_all_models",
    "rl.deploy": "deploy_rl",
    "pretrain.temperature": "pretrain_temperature",
    "calibration.overconf_penalty": "overconf_penalty",
    "calibration.overconf_weight": "overconf_weight",
    "calibration.overconf_threshold": "overconf_threshold",
    "calibration.calibrate": "calibrate",
    "hardware.profile": "hardware_profile",
    "hardware.num_workers": "num_workers",
    "hardware.prefetch_factor": "prefetch_factor",
    "hardware.pin_memory": "pin_memory",
    "hardware.persistent_workers": "persistent_workers",
    "hardware.val_num_workers": "val_num_workers",
    "hardware.val_prefetch_factor": "val_prefetch_factor",
    "hardware.thread_prefetch_batches": "thread_prefetch_batches",
    "hardware.force_thread_prefetch": "force_thread_prefetch",
    "hardware.torch_compile": "torch_compile",
    "hardware.torch_compile_mode": "torch_compile_mode",
    "data.zarr_cname": "zarr_cname",
    "data.zarr_clevel": "zarr_clevel",
    "tracking.wandb_project": "wandb_project",
    "tracking.run_name": "run_name",
    "tracking.no_wandb": "no_wandb",
    "tracking.auto_tune": "auto_tune",
    "tracking.dry_tune": "dry_tune",
    "tracking.ollama_auto_tune": "ollama_auto_tune",
    "distillation.teacher_model": "teacher_model",
    "distillation.teacher_ckpt": "teacher_ckpt",
    "distillation.student_model": "model",  # student == --model when KD enabled
    "distillation.alpha": "distill_weight",
    "distillation.temperature": "distill_temperature",
    "quick.enabled": "quick_mode",
    "monitoring.drift_gate": "drift_gate",
    "monitoring.drift_fail_open": "drift_fail_open",
    "monitoring.drift_baseline_samples": "drift_baseline_samples",
    "monitoring.drift_live_samples": "drift_live_samples",
    "monitoring.drift_psi_threshold": "drift_psi_threshold",
    "monitoring.drift_ks_pvalue_threshold": "drift_ks_pvalue_threshold",
    "monitoring.drift_ks_statistic_threshold": "drift_ks_statistic_threshold",
    "diversity_loss.weight": "div_weight",
    "diversity_loss.same_role_mult": "same_role_mult",
    "data.integrity_gate": "integrity_gate",
    "data.auto_rebuild_on_mismatch": "auto_rebuild_on_mismatch",
    "paths.checkpoint_dir": "checkpoint_dir",
    "paths.data_cache": "data_cache",
    # XGBoost baseline
    "xgboost.enabled": "xgb_enabled",
    "xgboost.task": "xgb_task",
    "xgboost.sequence_mode": "xgb_sequence_mode",
    "xgboost.n_estimators": "xgb_n_estimators",
    "xgboost.max_depth": "xgb_max_depth",
    "xgboost.learning_rate": "xgb_learning_rate",
    "xgboost.subsample": "xgb_subsample",
    "xgboost.colsample_bytree": "xgb_colsample_bytree",
    "xgboost.min_child_weight": "xgb_min_child_weight",
    "xgboost.gamma": "xgb_gamma",
    "xgboost.reg_alpha": "xgb_reg_alpha",
    "xgboost.reg_lambda": "xgb_reg_lambda",
    "xgboost.objective": "xgb_objective",
    "xgboost.eval_metric": "xgb_eval_metric",
    "xgboost.early_stopping_rounds": "xgb_early_stopping_rounds",
    "xgboost.folds": "xgb_folds",
    "xgboost.tune": "xgb_tune",
    "xgboost.tune_trials": "xgb_tune_trials",
    "xgboost.max_samples": "xgb_max_samples",
    "xgboost.feature_importance": "xgb_feature_importance",
    "xgboost.feature_importance_top_n": "xgb_feature_importance_top_n",
    # CatBoost baseline (native keys + XGB-style aliases)
    "catboost.enabled": "cb_enabled",
    "catboost.task": "cb_task",
    "catboost.sequence_mode": "cb_sequence_mode",
    "catboost.iterations": "cb_n_estimators",
    "catboost.n_estimators": "cb_n_estimators",
    "catboost.depth": "cb_max_depth",
    "catboost.max_depth": "cb_max_depth",
    "catboost.learning_rate": "cb_learning_rate",
    "catboost.subsample": "cb_subsample",
    "catboost.colsample_bylevel": "cb_colsample_bylevel",
    "catboost.colsample_bytree": "cb_colsample_bylevel",
    "catboost.l2_leaf_reg": "cb_l2_leaf_reg",
    "catboost.reg_lambda": "cb_l2_leaf_reg",
    "catboost.early_stopping_rounds": "cb_early_stopping_rounds",
    "catboost.folds": "cb_folds",
    "catboost.tune": "cb_tune",
    "catboost.tune_trials": "cb_tune_trials",
    "catboost.max_samples": "cb_max_samples",
    "catboost.feature_importance": "cb_feature_importance",
    "catboost.feature_importance_top_n": "cb_feature_importance_top_n",
    # Validation / purged CV
    "validation.method": "validation_method",
    "validation.n_splits": "validation_n_splits",
    "validation.purge_bars": "validation_purge_bars",
    "validation.embargo_bars": "validation_embargo_bars",
    "validation.min_train_size": "validation_min_train_size",
}


def _apply_yaml_config(parser: argparse.ArgumentParser, config_path: str) -> None:
    """Load config/run.yaml and set argparse defaults from it.

    YAML parse failures are fatal: swallowing them silently drops the entire
    config (strategy ATR targets, bar_freq, curriculum, …) onto hardcoded
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


# Alias map for legacy YAML regime_scale keys → LIVE_RISK / RegimeScale names.
_REGIME_SCALE_ALIASES = {
    "volatile": "crisis",
    "ranging": "mean_rev",
    "unknown": "normal",
}


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

    # Strategy ATR → LABELING (CLI may override YAML after parse)
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

    # hardware.torch_compile → settings.GPU (and bound _GPU_CFG if same dict)
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


def parse_args():
    p = argparse.ArgumentParser(description="Forex Model ΓÇö 20M Tick GPU Trainer")
    p.set_defaults(curriculum=SETTINGS_CURRICULUM)
    p.set_defaults(execution=SETTINGS_EXECUTION)
    p.set_defaults(risk=None)
    p.set_defaults(sidecar=None)
    p.set_defaults(use_mixup=False)
    p.set_defaults(use_volatility_sampler=False)
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML run config (e.g. config/run.yaml). "
        "Values are used as defaults; explicit CLI flags override them.",
    )

    # Strategy profile
    p.add_argument(
        "--strategy-mode",
        type=str,
        default="scalping",
        choices=sorted(STRATEGY_PROFILES.keys()),
        help="Trading horizon profile. scalping=1min fast trades; normal=1h slower trades.",
    )
    p.add_argument(
        "--bar-freq", type=str, default=None, help="Bar frequency for feature/label construction, e.g. 1min, 15min, 1h."
    )
    p.add_argument(
        "--lookahead-bars",
        type=int,
        default=None,
        help="Label forward horizon in bars. Defaults to the selected strategy profile.",
    )
    p.add_argument(
        "--profit-target-atr", type=float, default=None, help="ATR profit barrier for triple-barrier/normal labels."
    )
    p.add_argument(
        "--stop-loss-atr", type=float, default=None, help="ATR stop barrier for triple-barrier/normal labels."
    )

    # Scale
    p.add_argument("--n-ticks", type=int, default=20_000_000, help="Total tick count to train on (default: 20M)")
    p.add_argument("--chunk-size", type=int, default=500_000, help="Ticks per processing chunk (RAM safety valve)")
    p.add_argument(
        "--real-data-window-days",
        type=int,
        default=0,
        help="Days per real-data ingestion window. 0 = auto from --chunk-size.",
    )
    p.add_argument(
        "--window-batch-days",
        type=int,
        default=1,
        help="Group N consecutive date windows into one batch. "
        "Effective window = real_data_window_days * window_batch_days. "
        "Larger batches give features more lookback context (default: 1).",
    )

    # Data source
    p.add_argument(
        "--data-source",
        type=str,
        default="dukascopy",
        choices=["synthetic", "dukascopy", "tds", "lmax_historical", "auto", "databento"],
        help="Which data source to use",
    )
    p.add_argument("--data-start", type=str, default="2008-01-01")
    p.add_argument("--data-end", type=str, default=(datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"))
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Cap training to the first N samples of the processed cache "
        "(cache is time-ordered; e.g. half of 94,423 = 47,211 uses the "
        "earliest ~9 years). 0 = use all samples. Reuses the cache, no rebuild.",
    )
    p.add_argument("--pair", type=str, default="EURUSD")
    p.add_argument(
        "--pairs",
        type=str,
        default=None,
        help="Comma-separated pairs for joint multi-pair training, e.g. EURUSD,GBPUSD,USDJPY. "
        "Overrides --pair when set. Can also be a list in config/run.yaml under data.pairs.",
    )
    p.add_argument(
        "--pair-embed-dim",
        type=int,
        default=0,
        help="Learnable pair embedding size (int). Appended to each pair's features before "
        "the backbone. 0 = disabled (pairs are simply concatenated on the feature axis).",
    )
    p.add_argument(
        "--corr-window",
        type=int,
        default=20,
        help="Short rolling correlation window in bars for MultiPairWrapper cross-pair features. Default: 20.",
    )
    p.add_argument(
        "--corr-window-long",
        type=int,
        default=60,
        help="Long rolling correlation window in bars for MultiPairWrapper. Default: 60.",
    )
    p.add_argument(
        "--momentum-window",
        type=int,
        default=20,
        help="Windowed relative momentum lookback in bars for MultiPairWrapper. Default: 20.",
    )
    p.add_argument(
        "--pair-align",
        type=str,
        default="inner",
        choices=["inner", "outer"],
        help="Timestamp alignment across pairs: inner=common bars only (default), outer=fill missing bars with NaN.",
    )
    p.add_argument(
        "--full-day-data",
        action="store_true",
        help="Dukascopy: load all 24h (00ΓÇô23 UTC). Default is session-only (07ΓÇô17 UTC).",
    )

    # Model
    p.add_argument(
        "--model", type=str, default="haelt", choices=["tft", "transformer", "haelt", "mamba", "gnn", "expert", "glm"]
    )
    p.add_argument("--all-models", dest="all_models", action="store_true")

    p.add_argument(
        "--no-all-models",
        dest="all_models",
        action="store_false",
        help="Force a single-model run even if config model.all_models=true.",
    )
    # store_false's default would otherwise leave all_models=True with no flags.
    p.set_defaults(all_models=False)

    p.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model list for --all-models, e.g. transformer,expert. "
        "Empty means every registered supervised architecture.",
    )

    p.add_argument(
        "--div-weight",
        type=float,
        default=0.10,
        help="C: DiversityLoss weight during post-training diversity fine-tuning",
    )
    p.add_argument(
        "--same-role-mult",
        type=float,
        default=2.0,
        help="C: Extra diversity penalty multiplier for same-role model pairs",
    )

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument(
        "--batch-size", type=int, default=2048, help="Batch size ΓÇö 2048 optimal for 20M samples on RTX 4090"
    )
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument(
        "--lr-schedule",
        type=str,
        default="warmup_cosine",
        choices=["onecycle", "warmup_cosine"],
        help="Learning-rate schedule: onecycle (legacy default) or warmup_cosine.",
    )
    p.add_argument("--lr-warmup-epochs", type=int, default=3, help="Warmup epochs used by warmup_cosine schedule.")
    p.add_argument(
        "--lr-warmup-pct",
        type=float,
        default=0.1,
        help="Warmup fraction fallback for warmup_cosine when warmup_epochs <= 0.",
    )
    p.add_argument(
        "--lr-min-ratio",
        type=float,
        default=0.05,
        help="Final LR ratio for warmup_cosine (final_lr = lr * lr_min_ratio).",
    )
    p.add_argument("--onecycle-pct-start", type=float, default=0.1, help="OneCycleLR warmup fraction (legacy path).")
    p.add_argument(
        "--onecycle-max-lr-mult",
        type=float,
        default=10.0,
        help="OneCycleLR peak multiplier over base lr (legacy path).",
    )
    p.add_argument("--seq-len", type=str, default="60")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Global random seed (A-M3: seeded by default for reproducibility). "
        "Pass a different int to vary runs; threads into numpy/torch/augmenter RNGs.",
    )
    p.add_argument(
        "--deterministic",
        dest="deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="A-M3: force fully deterministic kernels (cudnn.deterministic=True, "
        "benchmark=False, torch.use_deterministic_algorithms). Slower but reproducible.",
    )
    p.add_argument("--val-split", type=float, default=0.1, help="Validation fraction (default 0.1)")
    p.add_argument(
        "--tune-split",
        type=float,
        default=0.05,
        help="Fraction reserved for auto-tune evaluation, separate from val (default 0.05). "
        "Set to 0 to disable three-way split (reverts to val reuse).",
    )
    p.add_argument(
        "--curriculum-gate-metric",
        type=str,
        default="train_loss",
        choices=["train_loss", "val_sharpe"],
        help="Metric used for curriculum progression gating. 'train_loss' (default) "
        "prevents val set leakage into curriculum decisions (SYS-005). "
        "'val_sharpe' restores legacy behavior.",
    )
    p.add_argument(
        "--curriculum-manager",
        action="store_true",
        default=False,
        help="Enable the unified CurriculumManager (difficulty + self-paced + "
        "loss-weighted + adaptive) as an extra per-epoch sample filter "
        "(Improvement #4). Default: off (existing curriculum unchanged).",
    )
    p.add_argument(
        "--curriculum-manager-mode",
        type=str,
        default="combined",
        choices=["difficulty", "self_paced", "loss_weighting", "adaptive", "combined"],
        help="CurriculumManager combination mode used with --curriculum-manager.",
    )
    p.add_argument(
        "--curriculum-callback",
        action="store_true",
        default=False,
        help="Build the per-epoch curriculum controller through the "
        "create_curriculum_callback() factory (CustomCurriculumAdapter) "
        "instead of create_curriculum_manager(). Requires --curriculum-manager.",
    )
    p.add_argument(
        "--curriculum-miner-feedback",
        action="store_true",
        help="Enable Online Miner -> Curriculum feedback (forgetting/easy ratios inform curriculum pace).",
    )
    p.add_argument(
        "--curriculum-miner-models",
        type=str,
        default="",
        help="Comma-separated list of models to enable miner feedback for (default: tft,transformer,haelt).",
    )
    p.add_argument(
        "--curriculum-forgetting-threshold",
        type=float,
        default=0.15,
        help="Forgetting rate threshold to freeze curriculum advancement (default: 0.15).",
    )
    p.add_argument(
        "--curriculum-easy-threshold",
        type=float,
        default=0.60,
        help="Easy sample ratio threshold to accelerate curriculum (default: 0.60).",
    )
    p.add_argument(
        "--curriculum-freeze-patience",
        type=int,
        default=1,
        help="Epochs to hold curriculum after freeze trigger (default: 1).",
    )
    p.add_argument(
        "--use-self-paced",
        action="store_true",
        help="Enable SelfPacedLearning curriculum (jointly optimizes model params and sample inclusion).",
    )
    p.add_argument(
        "--use-loss-weighting",
        action="store_true",
        help="Enable LossBasedWeighting curriculum (inverse/focal/threshold/softmax weighting).",
    )
    p.add_argument(
        "--self-paced-pace",
        type=str,
        default="linear",
        choices=["linear", "exponential", "cosine"],
        help="SelfPacedLearning pace schedule (default: linear).",
    )
    p.add_argument(
        "--self-paced-lambda", type=float, default=1.0, help="SelfPacedLearning lambda parameter (default: 1.0)."
    )
    p.add_argument(
        "--loss-weighting-scheme",
        type=str,
        default="focal",
        choices=["inverse", "focal", "threshold", "softmax"],
        help="LossBasedWeighting scheme (default: focal).",
    )
    p.add_argument(
        "--loss-weighting-focal-gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma for LossBasedWeighting (default: 2.0).",
    )
    p.add_argument(
        "--self-paced-models",
        type=str,
        default="",
        help="Comma-separated list of models to enable self-paced for (default: tft,transformer,haelt).",
    )
    p.add_argument(
        "--loss-weighting-models",
        type=str,
        default="",
        help="Comma-separated list of models to enable loss weighting for (default: tft,transformer,haelt).",
    )

    p.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="Enable AMP (automatic mixed precision) for faster training. Disabled by default to avoid NaNs.",
    )
    p.add_argument(
        "--no-amp",
        action="store_true",
        default=False,
        dest="no_amp",
        help="Disable AMP ΓÇö forces FP32 training. Eliminates NaN-grad skips on 2240-feature inputs at the cost of ~30% slower throughput.",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
        help=(
            "AMP precision dtype. auto=force BF16 on all Ampere+ (CC>=8.0; "
            "full FP32 range, no GradScaler), FP16 on older GPUs. "
            "bf16: same as forced BF16 (falls back to FP16 if unsupported). "
            "fp16: needs GradScaler. fp32: no AMP (debug)."
        ),
    )
    p.add_argument(
        "--cross-asset-mode",
        type=str,
        default="auto",
        choices=["auto", "real", "synthetic", "off"],
        help="Cross-asset features source: auto=real for real FX data, synthetic for synthetic FX; "
        "real=attempt external commodities/yields download; synthetic/off disables external fetch",
    )
    p.add_argument(
        "--cross-asset-provider",
        type=str,
        default="auto",
        choices=["auto", "stooq", "yahoo", "fred", "eodhd"],
        help="Cross-asset data provider. Env CROSS_ASSET_SOURCE overrides this when set.",
    )
    p.add_argument(
        "--sentiment-mode",
        type=str,
        default="finbert",
        choices=["off", "finbert", "auto"],
        help="Sentiment feature mode: finbert=force FinBERT, auto=best available, off=disable sentiment feature columns",
    )
    p.add_argument(
        "--historical-news-mode",
        type=str,
        default="calendar",
        choices=["off", "calendar", "full"],
        help="Offline historical news mode: off=neutral, calendar=economic no-trade events, full=calendar + headline sentiment/counts.",
    )
    p.add_argument(
        "--historical-news-file",
        type=str,
        default=None,
        help="Optional CSV/JSON/JSONL historical headlines file. Defaults to data/raw/news/historical_news_combined.parquet or HISTORICAL_NEWS_FILE.",
    )
    p.add_argument(
        "--economic-calendar-file",
        type=str,
        default=None,
        help="Optional CSV/JSON/JSONL economic calendar file. Defaults to data/raw/eco_calendar/events.csv or ECONOMIC_CALENDAR_FILE.",
    )
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument(
        "--grad-accum-steps",
        type=int,
        default=2,
        help="Gradient accumulation steps; effective batch = batch_size * grad_accum_steps",
    )
    p.add_argument(
        "--swa-enabled",
        dest="swa_enabled",
        action=argparse.BooleanOptionalAction,
        default=bool(TRAINING.get("swa_enabled", False)),
        help="Enable Stochastic Weight Averaging over the final training phase.",
    )
    p.add_argument(
        "--swa-start-frac",
        type=float,
        default=float(TRAINING.get("swa_start_frac", 0.75)),
        help="Fraction of total epochs before SWA starts, e.g. 0.75.",
    )
    p.add_argument(
        "--swa-lr",
        type=float,
        default=float(TRAINING.get("swa_lr", 1e-5)),
        help="Constant learning rate used by the SWA scheduler.",
    )
    p.add_argument(
        "--training-framework",
        choices=["custom", "lightning", "composer"],
        default="custom",
        help="Training framework: custom (built-in loop), lightning (PyTorch Lightning), composer (Mosaic Composer).",
    )
    p.add_argument(
        "--rl-framework",
        choices=["custom", "cleanrl", "sb3"],
        default="custom",
        help="RL framework: custom (built-in), cleanrl, sb3 (Stable-Baselines3).",
    )
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument(
        "--label-method",
        type=str,
        default="rl_reward",
        choices=["rl_reward", "triple_barrier"],
        help="Supervised targets: RL forward P&L labels vs triple-barrier (ATR barriers + vertical)",
    )
    p.add_argument(
        "--loss",
        type=str,
        default=None,
        choices=["huber", "asymmetric", "cross_entropy", "directional_huber", "sharpe_huber"],
        help="huber/asymmetric/directional_huber/sharpe_huber on scalar targets; "
        "cross_entropy=3-class {-1,0,1} with balanced weights",
    )
    p.add_argument(
        "--direction-weight",
        type=float,
        default=0.5,
        help="Extra wrong-direction penalty multiplier for directional_huber loss",
    )
    p.add_argument("--sharpe-weight", type=float, default=0.2, help="Sharpe proxy weight for sharpe_huber loss")
    p.add_argument(
        "--sharpe-annualization-factor",
        type=float,
        default=None,
        help="Override the auto-detected Sharpe annualization factor. "
        "Default: auto from bar_freq x lookahead_bars x TRADING_DAYS. "
        "Use only when you know the exact right value.",
    )
    p.add_argument(
        "--fx-full-day",
        action="store_true",
        default=False,
        help="Treat the data as full-day FX (24h) when computing the "
        "Sharpe annualization factor. Without this flag the "
        "factor assumes a 6.5h session profile.",
    )
    p.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation improvement required to reset patience",
    )
    p.add_argument(
        "--guard-min-confidence",
        type=float,
        default=0.85,
        help="Minimum confidence required to execute a trade during validation (Disagreement Gating).",
    )
    p.add_argument("--num-workers", type=int, default=8, help="DataLoader workers ΓÇö 8 is sweet spot for H100/A100")
    p.add_argument(
        "--prefetch-factor", type=int, default=4, help="DataLoader prefetch (per worker); lower on 16GB RAM PCs"
    )
    p.add_argument(
        "--val-num-workers",
        type=int,
        default=None,
        help="Validation DataLoader workers. Default: auto from train workers.",
    )
    p.add_argument(
        "--val-prefetch-factor",
        type=int,
        default=None,
        help="Validation prefetch factor. Default: auto (lower than train).",
    )
    p.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action="store_true",
        default=None,
        help="Force DataLoader pin_memory=True for train/val.",
    )
    p.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        help="Force DataLoader pin_memory=False for train/val.",
    )
    p.add_argument(
        "--persistent-workers",
        dest="persistent_workers",
        action="store_true",
        default=None,
        help="Force DataLoader persistent_workers=True when workers > 0.",
    )
    p.add_argument(
        "--no-persistent-workers",
        dest="persistent_workers",
        action="store_false",
        help="Force DataLoader persistent_workers=False.",
    )
    p.add_argument(
        "--thread-prefetch-batches",
        type=int,
        default=8,
        help="Background-thread prefetch queue depth for train/val loaders "
        "(overlaps Zarr decompress + H2D with GPU compute; only applied "
        "when num_workers==0 or --force-thread-prefetch is set).",
    )
    p.add_argument(
        "--force-thread-prefetch",
        dest="force_thread_prefetch",
        action="store_true",
        default=False,
        help="Layer the daemon-thread prefetch queue on top of DataLoader "
        "worker-process buffering even when num_workers>0. Useful for "
        "hiding GPU-step jitter on slow disks; trades ~2x pinned memory.",
    )
    p.add_argument(
        "--zarr-cname",
        type=str,
        default="auto",
        help="Blosc codec for training-cache Zarr writes "
        "(auto|lz4|zstd|zlib|...). auto = lz4@1 on Linux, zstd@3 elsewhere.",
    )
    p.add_argument(
        "--zarr-clevel",
        type=int,
        default=None,
        help="Blosc compression level (1-9). Default: platform auto (1 on Linux with lz4, 3 with zstd fallback).",
    )
    p.add_argument(
        "--dataset-build-workers",
        type=int,
        default=1,
        help="Parallel threads for loading date windows during dataset build. "
        "1 = sequential (safe default). 2-4 overlaps tick I/O across windows.",
    )
    p.add_argument(
        "--parallel-window-workers",
        type=int,
        default=1,
        help="Parallel processes for date-window feature engineering + labeling. "
        "1 = sequential (default). 2-4 parallelises CPU-heavy chunk builds "
        "across windows using ProcessPoolExecutor.",
    )
    p.add_argument(
        "--hardware-profile",
        type=str,
        default=None,
        choices=list(HARDWARE_PROFILES.keys()) if HARDWARE_PROFILES else None,
        help="Apply tuned defaults (batch/workers/chunk/prefetch/paths). "
        "rtx_4060_16gb_ram: RTX 4060 8GB VRAM + 16GB system RAM",
    )

    # Architecture
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument(
        "--fair-sweep",
        action="store_true",
        help="Architecture bake-off: identical hyperparams from run.yaml for every model "
        "(alias for --no-model-profile).",
    )
    p.add_argument(
        "--model-profile",
        dest="model_profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply per-architecture tuned defaults from config/models.py (default: on). "
        "Use --no-model-profile or --fair-sweep for a fair architecture comparison.",
    )
    p.add_argument(
        "--feature-ablation-name",
        type=str,
        default="",
        help="Name recorded in feature_ablation_report.json for this feature ablation run.",
    )

    p.add_argument(
        "--feature-ablation-drop-groups",
        type=str,
        default="",
        help="Comma-separated curriculum feature groups to zero for this run, e.g. news,cross_asset.",
    )

    p.add_argument(
        "--feature-ablation-keep-groups",
        type=str,
        default="",
        help="Comma-separated curriculum feature groups to keep; all other grouped features are zeroed.",
    )

    p.add_argument(
        "--feature-ablation-drop-features",
        type=str,
        default="",
        help="Comma-separated exact feature names to zero for this run.",
    )

    # Pre-training & Ablation
    p.add_argument("--pretrain", action="store_true", help="Enable contrastive pre-training")
    p.add_argument("--ablate-pretrain", action="store_true", help="Run ablation test on pretraining vs no-pretraining")

    # Pre-training
    p.add_argument(
        "--pretrain-method",
        choices=["byol", "tscl", "masked", "vae", "autoencoder", "cluster", "forecast", "drift"],
        default=str(PRETRAIN.get("method", "byol")).lower(),
        help="Self-supervised pretrain: byol (default), tscl, masked, vae, cluster, forecast, drift",
    )
    p.add_argument(
        "--pretrain-framework",
        choices=["custom", "lightly", "solo"],
        default="custom",
        help="Pretraining framework: custom (built-in), lightly (lightly-ssl), solo (solo-learn).",
    )
    p.add_argument("--pretrain-epochs", type=int, default=30)
    p.add_argument(
        "--pretrain-max-epochs",
        type=int,
        default=0,
        help="Hard cap for pretraining epochs. 0 keeps pretrain_epochs unchanged.",
    )
    p.add_argument(
        "--pretrain-min-epochs",
        type=int,
        default=0,
        help="Minimum pretrain epochs before handoff checks can stop early.",
    )
    p.add_argument(
        "--pretrain-handoff-patience",
        type=int,
        default=0,
        help="Stop pretraining early after this many plateau epochs (0 disables).",
    )
    p.add_argument(
        "--pretrain-handoff-min-delta",
        type=float,
        default=0.0,
        help="Minimum pretrain loss improvement to reset handoff patience.",
    )
    p.add_argument(
        "--pretrain-handoff-loss",
        type=float,
        default=float("-inf"),
        help="Stop pretraining once loss <= threshold after min epochs. Disabled by default.",
    )
    p.add_argument(
        "--pretrain-regime",
        action="store_true",
        help="Use regime-aware TSCL: same-regime positives + cross-regime hard negatives",
    )
    p.add_argument(
        "--use-multi-task-pretrainer",
        action="store_true",
        default=False,
        help="Pretrain with the multi-task pretrainer (contrastive + masked recon + "
        "forecast + domain adaptation) from pretrain/multi_task.py instead of the "
        "built-in single-objective trainer (Improvement #3).",
    )
    p.add_argument(
        "--pretrain-lr",
        type=float,
        default=float(PRETRAIN.get("pretrain_lr", 1e-4)),
        help="Pretrain optimizer learning rate",
    )
    p.add_argument(
        "--pretrain-batch",
        type=int,
        default=int(PRETRAIN.get("pretrain_batch", 256)),
        help="Preferred pretrain batch size before VRAM safety cap",
    )
    p.add_argument(
        "--pretrain-projection-dim",
        type=int,
        default=int(PRETRAIN.get("projection_dim", 256)),
        help="Projection dimension for BYOL/TSCL heads",
    )
    p.add_argument(
        "--pretrain-pred-dim",
        type=int,
        default=int(PRETRAIN.get("pred_dim", 128)),
        help="BYOL predictor hidden dimension",
    )
    p.add_argument(
        "--pretrain-ema-decay",
        type=float,
        default=float(PRETRAIN.get("ema_decay", 0.996)),
        help="BYOL target-network EMA decay",
    )
    p.add_argument(
        "--pretrain-sample-windows",
        default="auto",
        help="Windows loaded per pretrain block, or 'auto' for RAM-based sizing",
    )
    p.add_argument(
        "--pretrain-blocks-per-epoch",
        default="auto",
        help="Fresh pretrain blocks per outer epoch, or 'auto' for effective sample volume",
    )
    p.add_argument(
        "--pretrain-mask-prob",
        type=float,
        default=float(PRETRAIN.get("mask_prob", 0.20)),
        help="Masked reconstruction probability when --pretrain-method masked",
    )
    p.add_argument(
        "--pretrain-recon-hidden-dim",
        type=int,
        default=int(PRETRAIN.get("recon_hidden_dim", 512)),
        help="Masked reconstruction decoder hidden size",
    )
    p.add_argument(
        "--pretrain-latent-dim",
        type=int,
        default=int(PRETRAIN.get("latent_dim", 64)),
        help="VAE latent dimension when --pretrain-method vae",
    )
    p.add_argument(
        "--pretrain-vae-beta",
        type=float,
        default=float(PRETRAIN.get("vae_beta", 0.001)),
        help="KL weight for VAE pretrain",
    )
    p.add_argument(
        "--pretrain-n-clusters",
        type=int,
        default=int(PRETRAIN.get("n_clusters", 3)),
        help="k-means clusters for cluster contrastive pretrain",
    )
    p.add_argument(
        "--pretrain-forecast-horizon",
        type=int,
        default=int(PRETRAIN.get("forecast_horizon", 5)),
        help="Future bars to predict in forecast pretext task",
    )
    p.add_argument(
        "--pretrain-drift-margin",
        type=float,
        default=float(PRETRAIN.get("drift_margin", 1.0)),
        help="Target L2 distance between clean and drift-augmented embeddings",
    )
    p.add_argument(
        "--force-pretrain",
        action="store_true",
        help="Delete existing contrastive encoder checkpoint and pretrain from scratch",
    )
    p.add_argument(
        "--multitask",
        action="store_true",
        help="Replace single prediction head with MultiTaskHead (direction CE + magnitude Huber + confidence BCE)",
    )
    p.add_argument(
        "--mt-w-ret", type=float, default=0.5, help="Multi-task loss weight for return_hat Huber term (default 0.5)"
    )
    p.add_argument(
        "--mt-w-conf", type=float, default=0.3, help="Multi-task loss weight for confidence BCE term (default 0.3)"
    )
    p.add_argument(
        "--direction-probe",
        dest="direction_probe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a short balanced direction probe before full supervised training.",
    )
    p.add_argument("--direction-probe-epochs", type=int, default=2, help="Epochs for the pre-training direction probe.")
    p.add_argument(
        "--direction-probe-samples", type=int, default=4096, help="Total samples used by the balanced direction probe."
    )
    p.add_argument(
        "--direction-warmup-epochs",
        type=int,
        default=2,
        help="Initial epochs trained with balanced direction-only batches.",
    )
    p.add_argument(
        "--direction-min-true-class-share",
        type=float,
        default=0.15,
        help="Minimum train/val true class share required before training.",
    )
    p.add_argument(
        "--direction-min-pred-class-share",
        type=float,
        default=0.05,
        help="Minimum validation predicted share for each direction class.",
    )
    p.add_argument(
        "--direction-max-pred-class-share",
        type=float,
        default=0.80,
        help="Maximum validation predicted share for any one direction class.",
    )
    p.add_argument(
        "--direction-min-recall",
        type=float,
        default=0.001,
        help="Minimum per-class validation recall for direction readiness gates.",
    )
    p.add_argument(
        "--overconf-penalty",
        dest="overconf_penalty",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply training-time overconfidence penalty for regression losses.",
    )
    p.add_argument("--overconf-weight", type=float, default=0.3, help="Weight for the overconfidence penalty.")
    p.add_argument(
        "--overconf-threshold",
        type=float,
        default=0.6,
        help="Absolute prediction threshold that triggers overconfidence checks.",
    )
    p.add_argument(
        "--calibrate",
        dest="calibrate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fit post-training temperature calibration on the validation set.",
    )
    p.add_argument(
        "--train-ensemble",
        action="store_true",
        default=bool(SETTINGS_ENSEMBLE.get("enabled", False)),
        help="After supervised training, train the EnsembleMetaLearner "
        "with diversity penalty across all trained base models",
    )
    p.add_argument(
        "--ensemble-epochs",
        type=int,
        default=int(SETTINGS_ENSEMBLE.get("epochs", 10)),
        help="Epochs to train the meta-learner (default 10)",
    )
    p.add_argument(
        "--ensemble-div-weight",
        type=float,
        default=float(SETTINGS_ENSEMBLE.get("div_weight", 0.1)),
        help="Diversity penalty weight for meta-learner training (default 0.1)",
    )
    p.add_argument(
        "--deploy-ensemble",
        action="store_true",
        default=bool(SETTINGS_ENSEMBLE.get("deploy", False)),
        help="After ensemble ONNX export, atomically promote it to production_best.onnx for the C++ server.",
    )
    p.add_argument(
        "--ensemble-explicit-diversity",
        action="store_true",
        default=bool(SETTINGS_ENSEMBLE.get("explicit_diversity", False)),
        help="Apply explicit per-member diversity controls during all-model training.",
    )
    p.add_argument(
        "--ensemble-member-seed-offset",
        type=int,
        default=int(SETTINGS_ENSEMBLE.get("member_seed_offset", 997)),
        help="Seed offset between ensemble members when explicit diversity is enabled.",
    )
    p.add_argument(
        "--ensemble-member-lr-jitter",
        type=float,
        default=float(SETTINGS_ENSEMBLE.get("member_lr_jitter", 0.0)),
        help="Relative LR jitter spread across members (e.g. 0.2 => +/-10%%).",
    )
    p.add_argument(
        "--ensemble-member-dropout-jitter",
        type=float,
        default=float(SETTINGS_ENSEMBLE.get("member_dropout_jitter", 0.0)),
        help="Absolute dropout jitter spread across members (clamped to [0,0.8]).",
    )

    # RL
    p.add_argument("--rl-train", action="store_true")
    p.add_argument("--rl-algo", type=str, default="dqn", choices=["dqn", "ppo"])
    p.add_argument("--rl-episodes", type=int, default=500)
    p.add_argument(
        "--rl-episode-len",
        type=int,
        default=2048,
        help="A-H1: bars per RL episode (sub-window sampled at a random offset "
        "each reset). 0 = full series each episode.",
    )
    p.add_argument(
        "--off-policy-rewards",
        action="store_true",
        default=False,
        help="Log IPS/DR OPE estimates during RL (diagnostic only - does not train).",
    )
    p.add_argument(
        "--rl-encoder-obs",
        dest="rl_encoder_obs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="A-C3: use the frozen supervised encoder embedding as the RL "
        "observation (connects supervisedΓåÆRL). --no-rl-encoder-obs falls "
        "back to raw last-timestep features.",
    )
    p.add_argument(
        "--rl-val-frac", type=float, default=0.15, help="Fraction of the RL window held out for validation rollouts."
    )
    p.add_argument(
        "--rl-min-val-sharpe",
        type=float,
        default=-999.0,
        help="Minimum validation Sharpe required to save rl_*_best.pt.",
    )
    p.add_argument(
        "--rl-use-sharpe-reward",
        dest="rl_use_sharpe_reward",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Replace env P&L reward with rolling SharpeRewardWrapper during RL training.",
    )
    p.add_argument(
        "--rl-use-her",
        dest="rl_use_her",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Hindsight Experience Replay (HERBuffer) for DQN training.",
    )
    p.add_argument(
        "--rl-all-models",
        action="store_true",
        default=False,
        help="With --all-models, run RL once per trained architecture subfolder.",
    )
    p.add_argument(
        "--deploy-rl",
        action="store_true",
        default=bool(RL.get("deploy", False)),
        help="After RL ONNX export, atomically promote it to production_best.onnx for the C++ server.",
    )
    p.add_argument(
        "--pretrain-temperature",
        type=float,
        default=float(PRETRAIN.get("temperature", 0.5)),
        help="Initial TSCL temperature (learnable during pretrain).",
    )
    p.add_argument(
        "--pretrain-read-windows",
        type=int,
        default=int(PRETRAIN.get("read_windows", 64)),
        help="Zarr span read chunk size during pretrain loading.",
    )
    p.add_argument(
        "--maturity-stage",
        type=str,
        default="paper",
        choices=["dev", "paper", "production"],
        help="Model maturity stage (gates live promotion expectations).",
    )
    p.add_argument(
        "--max-bad-frac",
        type=float,
        default=0.05,
        help="Max fraction of bad rows allowed inside a training sequence window.",
    )
    p.add_argument(
        "--max-zero-frac",
        type=float,
        default=0.80,
        help="Max fraction of all-zero feature rows allowed inside a sequence window.",
    )

    # Fine-tune / warm-start (B-C2)
    p.add_argument(
        "--finetune-warm-start",
        dest="finetune_warm_start",
        action="store_true",
        default=False,
        help="B-C2: load prior production/best weights then CONTINUE supervised "
        "training on the new window (distinct from --resume, which skips "
        "training when a best checkpoint already exists).",
    )
    p.add_argument(
        "--warm-start-from",
        type=str,
        default=None,
        help="Explicit checkpoint to warm-start from. Default: production_best.pt then the model's own _best.pt.",
    )

    # Promotion gate (B-C1)
    p.add_argument(
        "--promotion-gate",
        dest="promotion_gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="B-C1: after training, backtest the challenger on a held-out forward "
        "window and run PromotionGate to decide deployment (writes "
        "<model>_promotion.json). --no-promotion-gate disables.",
    )
    p.add_argument(
        "--force-promotion",
        action="store_true",
        help="Bypass the challenger vs production gate and force the promotion.",
    )
    p.add_argument(
        "--promote-forward-frac",
        type=float,
        default=0.1,
        help="Fraction of most-recent samples used as the held-out forward "
        "window for the promotion-gate backtest (B-C1).",
    )

    # HPO
    p.add_argument("--hparam-search", action="store_true")
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument(
        "--auto-optuna",
        dest="auto_optuna",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Overlay config/optuna/run_optuna_best_<model>_<metric>.yaml when present "
        "(default: optuna.auto_load in run.yaml, else true).",
    )

    # Tracking
    p.add_argument("--wandb-project", type=str, default="forex-scaling-model")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument(
        "--auto-run-dir",
        action="store_true",
        default=False,
        help="Generate a descriptive checkpoint folder under checkpoints/runs "
        "from model, strategy, pairs, seq_len, folds, and RL/ensemble mode.",
    )
    p.add_argument(
        "--run-dir-root",
        type=str,
        default=None,
        help="Base directory for --auto-run-dir. Defaults to <checkpoint-dir>/runs.",
    )
    p.add_argument(
        "--auto-tune",
        dest="auto_tune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write auto-tune proposal artifacts after each completed training phase. "
        "--no-auto-tune disables proposal generation and config nudges.",
    )

    p.add_argument(
        "--dry-tune",
        action="store_true",
        default=False,
        help="Write auto-tune proposals without mutating config/run.yaml.",
    )
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument(
        "--ollama-auto-tune",
        action="store_true",
        default=False,
        help="Allow Ollama to edit config and restart training after a run.",
    )
    p.add_argument("--save-every", type=int, default=5)

    # Paths
    p.add_argument("--checkpoint-dir", type=str, default=PATHS["checkpoints"])
    p.add_argument("--data-cache", type=str, default=PATHS["data_processed"])
    p.add_argument(
        "--resume",
        dest="resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resume model/optimizer state from existing checkpoints. "
        "Use --no-resume for a clean supervised run even when config/run.yaml enables resume.",
    )

    p.add_argument(
        "--training-memory",
        dest="training_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply conservative hyperparameter nudges from logs/training_memory.json. "
        "Use --no-training-memory for a clean baseline/fresh run.",
    )

    p.add_argument(
        "--retrain-completed-models",
        action="store_true",
        default=False,
        help="With --all-models --resume, retrain models that already have "
        "completed artifacts instead of skipping to unfinished models.",
    )
    p.add_argument(
        "--force-rebuild",
        "--rebuild-cache",
        dest="force_rebuild",
        action="store_true",
        help="Ignore cached Zarr/NPY store and rebuild from scratch",
    )
    p.add_argument("--build-only", action="store_true", help="Only build the dataset pipeline and exit")
    p.add_argument("--quick-mode", action="store_true", help="Fast sanity run: fewer folds/epochs, no ensemble or RL.")
    p.add_argument(
        "--drift-gate",
        dest="drift_gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run a pre-training input-distribution drift gate on cached features. "
        "Keep disabled for historical training runs.",
    )
    p.add_argument(
        "--drift-fail-open",
        dest="drift_fail_open",
        action="store_true",
        default=False,
        help="If drift gate check errors, continue training with a warning.",
    )
    p.add_argument(
        "--drift-baseline-samples",
        type=int,
        default=20_000,
        help="Baseline sample rows from cache start for drift gate.",
    )
    p.add_argument(
        "--drift-live-samples", type=int, default=5_000, help="Recent sample rows from cache end for drift gate."
    )
    p.add_argument(
        "--drift-psi-threshold",
        type=float,
        default=float(MONITORING.get("psi_threshold", 0.2)),
        help="PSI threshold for drift gate fail condition.",
    )
    p.add_argument(
        "--drift-ks-pvalue-threshold",
        type=float,
        default=float(MONITORING.get("ks_pvalue_threshold", 0.05)),
        help="KS p-value threshold for drift gate fail condition.",
    )
    p.add_argument(
        "--drift-ks-statistic-threshold",
        type=float,
        default=0.05,
        help="KS D-statistic (effect-size) floor for drift gate. "
        "KS only fails when BOTH p-value < threshold AND D-stat >= this value. "
        "Prevents false alarms on large datasets (default 0.05).",
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Run torch.profiler for 3 warm-up + 5 active batches then exit. "
            "Outputs a Chrome trace to logs/profile_<model>_<run>.json. "
            "Open in chrome://tracing or https://ui.perfetto.dev. "
            "Reveals whether you are compute-bound, memory-bound, or input-bound. "
            "Recommended before tuning batch size or enabling torch.compile."
        ),
    )
    p.add_argument(
        "--integrity-gate",
        dest="integrity_gate",
        action="store_true",
        default=True,
        help="Fail fast when cached X/y lengths are inconsistent.",
    )
    p.add_argument(
        "--no-integrity-gate",
        dest="integrity_gate",
        action="store_false",
        help="Disable strict cache integrity gate (not recommended).",
    )
    p.add_argument(
        "--feature-schema-gate",
        dest="feature_schema_gate",
        action="store_true",
        default=None,
        help=(
            "Fail dataset build when curriculum features or required market columns "
            "are missing from the built schema. Default: follows --integrity-gate."
        ),
    )
    p.add_argument(
        "--no-feature-schema-gate",
        dest="feature_schema_gate",
        action="store_false",
        help="Disable the dataset feature-schema gate (not recommended).",
    )
    p.add_argument(
        "--min-pair-years",
        type=int,
        default=2,
        help="Minimum years of data required per pair for multi-pair training (default: 2).",
    )
    p.add_argument(
        "--expected-pair-years",
        type=int,
        default=18,
        help="Expected years of data per pair - warns if less (default: 18).",
    )
    p.add_argument(
        "--coverage-report", action="store_true", help="Generate data coverage report (data_coverage_report.json)."
    )
    p.add_argument(
        "--auto-rebuild-on-mismatch",
        action="store_true",
        help="If cache integrity fails, delete cache/sidecars and rebuild automatically.",
    )
    p.add_argument(
        "--pretrain-ablation",
        type=str,
        nargs="?",
        const="true",
        choices=["true", "false", "auto"],
        default="auto",
        help="If true or auto (for transformer/haelt), runs a full training baseline with NO PRETRAIN first.",
    )
    p.add_argument(
        "--pretrain-ablation-models",
        type=str,
        default="",
        help="Comma-separated model list used when --pretrain-ablation auto. "
        "Default/config recommendation: tft,transformer,haelt.",
    )

    p.add_argument(
        "--ignore-manifest",
        action="store_true",
        help="Bypass dataset_manifest.json checks and force load existing cache.",
    )
    p.add_argument(
        "--walk-forward-cv",
        action="store_true",
        help="Purged walk-forward CV (train past / val future, embargo=seq_len+lookahead+delay) instead of one split",
    )
    p.add_argument(
        "--walk-forward-folds",
        type=int,
        default=None,
        help="Number of walk-forward folds (default: TRAINING['walk_forward_folds'])",
    )
    p.add_argument(
        "--cv-strategy",
        type=str,
        default="legacy",
        choices=["legacy", "walk_forward", "comb", "online"],
        help="CV split strategy when --walk-forward-cv is on (Improvement #11): "
        "legacy = original walk_forward_splits (default); walk_forward = "
        "WalkForwardCV; comb = combinatorial purged CV; online = rolling window.",
    )
    p.add_argument(
        "--early-stop-metric",
        type=str,
        default=None,
        choices=["loss", "sharpe"],
        help="Checkpoint early stopping on val loss or validation Sharpe proxy (default: TRAINING)",
    )
    p.add_argument(
        "--execution-delay-bars",
        type=int,
        default=1,
        help="Bars between model signal and executable entry; used for training labels and backtests.",
    )
    p.add_argument(
        "--data-quality-check",
        action="store_true",
        help="Run the data quality check script on the Zarr cache before training.",
    )
    p.add_argument(
        "--skip-training",
        action="store_true",
        help="Exit after data quality check (or dataset build) without training.",
    )
    p.add_argument(
        "--validate-config",
        action="store_true",
        help="Audit run.yaml/CLI for contradictions, estimate runtime, and exit without training.",
    )

    p.add_argument(
        "--teacher-model", type=str, default=None, help="Name of teacher model to distill from (e.g., haelt, ensemble)"
    )
    p.add_argument("--teacher-ckpt", type=str, default=None, help="Explicit teacher checkpoint path for distillation")
    p.add_argument(
        "--distill-weight", type=float, default=0.5, help="Weight of distillation loss relative to supervised loss"
    )
    p.add_argument(
        "--distill-temperature",
        type=float,
        default=float(DISTILLATION.get("temperature", 2.0)),
        help="Temperature for distillation soft targets (KL).",
    )

    # Advanced Training Mechanics (Phase 2)
    p.add_argument(
        "--enable-ewc",
        action="store_true",
        help="Enable Elastic Weight Consolidation to prevent catastrophic forgetting.",
    )
    p.add_argument("--ewc-lambda", type=float, default=1000.0, help="EWC penalty weight (default: 1000.0).")

    p.add_argument(
        "--enable-si", action="store_true", help="Enable Synaptic Intelligence (SI) to prevent catastrophic forgetting."
    )
    p.add_argument(
        "--si-lambda",
        type=float,
        default=1.0,
        help="SI penalty weight (default: 1.0). "
        "When the FeatureStabilityMonitor is active, this base lambda is "
        "scaled per epoch by 1/(1 + max_shift^2) as the SI dynamic lambda.",
    )

    p.add_argument("--enable-per", action="store_true", help="Enable Prioritized Experience Replay (PER).")

    p.add_argument(
        "--enable-adversarial",
        action="store_true",
        help="Enable adversarial training (PGD/FGSM/FreeLB) or legacy market shocks.",
    )
    p.add_argument(
        "--adversarial-method",
        type=str,
        default="pgd",
        choices=["pgd", "fgsm", "freelb", "market_shock", "graph_pgd"],
        help="Adversarial method: pgd (Projected Gradient Descent), fgsm (Fast Gradient Sign), "
        "freelb (Free Large-Batch), market_shock (legacy random shocks), "
        "graph_pgd (graph-aware PGD for GNNs; auto-selected for model_name 'gnn').",
    )
    p.add_argument(
        "--adversarial-prob",
        type=float,
        default=0.01,
        help="Probability of applying adversarial attack per batch (default: 0.01).",
    )
    p.add_argument(
        "--adversarial-eps", type=float, default=0.3, help="L-infinity perturbation budget epsilon (default: 0.3)."
    )
    p.add_argument("--adversarial-alpha", type=float, default=0.01, help="Step size for PGD/FreeLB (default: 0.01).")
    p.add_argument(
        "--adversarial-steps", type=int, default=7, help="Number of attack steps for PGD/FreeLB (default: 7)."
    )
    p.add_argument(
        "--adversarial-normalize-grad",
        action="store_true",
        help="L2 normalize gradients in PGD/Graph PGD (Madry best practice).",
    )
    p.add_argument(
        "--adversarial-warmup-steps",
        type=int,
        default=0,
        help="Gradually increase attack steps over this many training steps (0=disabled).",
    )
    p.add_argument(
        "--adversarial-eps-curriculum-scale",
        action="store_true",
        help="Scale adversarial epsilon with curriculum difficulty level (eps *= level/n_levels).",
    )
    p.add_argument(
        "--adversarial-models",
        type=str,
        default="",
        help="Comma-separated list of models to enable adversarial for (default: all except expert).",
    )

    # -- Risk engine (Improvement #1) - optional live/dry-run enforcement config --
    p.add_argument(
        "--risk-config",
        type=str,
        default=None,
        metavar="PATH",
        help="JSON/YAML file with a RiskEngine config (keys mirror "
        "config/settings.RISK) for live/dry-run enforcement during training.",
    )

    # -- Pre-parse to find --config, then apply YAML defaults before full parse --
    pre, _ = p.parse_known_args()
    if pre.config:
        _apply_yaml_config(p, pre.config)
        from training.optuna_config import apply_optuna_overlay_if_needed

        apply_optuna_overlay_if_needed(p, pre.config, getattr(pre, "auto_optuna", None), _apply_yaml_config)

    args = p.parse_args()
    # --no-amp: force FP32 regardless of --dtype or hardware profile
    if getattr(args, "no_amp", False):
        args.dtype = "fp32"
        args.amp = False
    if args.val_split is None:
        args.val_split = float(TRAINING["val_split"])
    if args.loss is None:
        args.loss = str(TRAINING.get("loss", "huber"))
    if args.walk_forward_folds is None:
        args.walk_forward_folds = int(TRAINING.get("walk_forward_folds", 6))
    if args.early_stop_metric is None:
        args.early_stop_metric = str(TRAINING.get("early_stop_metric", "sharpe"))
    if args.grad_accum_steps is None:
        args.grad_accum_steps = int(TRAINING.get("grad_accum_steps", 1))
    prof = strategy_profile(args.strategy_mode)
    scalp = strategy_profile("scalping")
    if args.bar_freq is None:
        args.bar_freq = str(prof["bar_freq"])
    if args.strategy_mode != "scalping" and int(args.seq_len) == int(scalp["seq_len"]):
        args.seq_len = int(prof["seq_len"])
    if args.lookahead_bars is None:
        args.lookahead_bars = int(prof["lookahead_bars"])
    if args.profit_target_atr is None:
        args.profit_target_atr = float(prof["profit_target_atr"])
    if args.stop_loss_atr is None:
        args.stop_loss_atr = float(prof["stop_loss_atr"])
    if args.strategy_mode != "scalping" and int(args.execution_delay_bars) == int(scalp["execution_delay_bars"]):
        args.execution_delay_bars = int(prof["execution_delay_bars"])
    print(
        f"[Strategy] {args.strategy_mode} | bars={args.bar_freq} | seq_len={args.seq_len} | "
        f"lookahead={args.lookahead_bars} | TP/SL={args.profit_target_atr}/{args.stop_loss_atr} ATR"
    )
    if args.quick_mode:
        # Synthetic/quick smokes are too small for purged walk-forward
        # (embargo ≈ seq_len+lookahead often exceeds the sample count).
        if str(getattr(args, "data_source", "")).lower() == "synthetic":
            args.walk_forward_cv = False
            args.walk_forward_folds = 1
        else:
            args.walk_forward_cv = True
            args.walk_forward_folds = min(max(int(args.walk_forward_folds), 1), 2)
        # Short windows (synthetic or brief real ranges) cannot satisfy
        # direction class-prior / probe floors - keep smoke training going.
        args.direction_probe = False
        args.ignore_preflight = True
        args.epochs = min(int(args.epochs), 8)
        args.pretrain_epochs = min(int(args.pretrain_epochs), 5)
        args.patience = min(int(args.patience), 4)
        # Quick mode always disables ensemble/RL - make the override loud when YAML
        # had them enabled so operators do not think a "full" pipeline ran.
        _ens_was = bool(getattr(args, "train_ensemble", False))
        _rl_was = bool(getattr(args, "rl_train", False))
        args.train_ensemble = False
        args.rl_train = False
        if _ens_was or _rl_was:
            print(
                "[Quick] WARN: quick.enabled forced ensemble="
                f"{'off (was on)' if _ens_was else 'off'} | "
                f"rl={'off (was on)' if _rl_was else 'off'}. "
                "Set quick.enabled: false in run.yaml for a full post-train path."
            )
        # reduce-overhead CUDAGraphs + tiny synthetic batches can RecursionError
        # on teardown; keep quick smokes in eager mode.
        args.torch_compile = False
        # Rich live display teardown has been observed to RecursionError after
        # TRAINING COMPLETE on short smoke runs - use tqdm-only progress.
        args.no_rich = True
        try:
            from config import settings as _settings

            _settings.GPU["torch_compile"] = False
        except Exception:
            pass
        try:
            if isinstance(_GPU_CFG, dict):
                _GPU_CFG["torch_compile"] = False
        except Exception:
            pass
        # Time-anchored seq_len resolution (resolve string like "6h40m" -> integer bars before quick-mode caps)
        _bf = getattr(args, "bar_freq", "5m")
        args.seq_len = _resolve_seq_len(args.seq_len, _bf)

        # Synthetic short windows (~80–200 bars from 10k–50k ticks) cannot  # noqa: RUF003
        # form sequences when seq_len + lookahead_bars exceeds bar count
        # (e.g. seq=64 + LH=30 on ~84 bars → zero samples).
        if str(getattr(args, "data_source", "")).lower() == "synthetic":
            _lh = int(getattr(args, "lookahead_bars", 30) or 30)
            _delay = int(getattr(args, "execution_delay_bars", 1) or 1)
            _cap = max(16, 48 - _lh // 2)  # keep headroom for LH=30 → seq≈32
            if int(args.seq_len) > _cap:
                print(
                    f"[Quick] synthetic seq_len {args.seq_len} → {_cap} (headroom for lookahead={_lh}+delay={_delay})"
                )
                args.seq_len = _cap
        cur = getattr(args, "curriculum", None)
        if isinstance(cur, dict):
            capped = []
            for entry in cur.get("seq_schedule") or []:
                if not isinstance(entry, dict):
                    continue
                e = dict(entry)
                if e.get("seq_len") is not None:
                    e["seq_len"] = min(int(e["seq_len"]), int(args.seq_len))
                capped.append(e)
            args.curriculum = {**cur, "seq_schedule": capped or [{"epoch_start": 0, "seq_len": int(args.seq_len)}]}
        print(
            f"[Quick] ON | folds={args.walk_forward_folds} | epochs={args.epochs} | "
            f"pretrain_epochs={args.pretrain_epochs} | ensemble=off | rl=off"
            f" | wf={'on' if args.walk_forward_cv else 'off'}"
        )
    # B-M1: a warm-start fine-tune on a short rolling window must NOT run k-fold
    # walk-forward CV (a 1-epoch fine-tune would attempt 5 folds then hit the
    # small-data fallback). Force the single-split + embargo path explicitly.
    if getattr(args, "finetune_warm_start", False) and args.walk_forward_cv:
        args.walk_forward_cv = False
        print("[FineTune] Warm-start mode: walk-forward CV disabled (single embargoed split).")
    if getattr(args, "fair_sweep", False):
        args.model_profile = False
    args._cli_profile_overrides = _collect_cli_profile_overrides()
    _sync_runtime_config(args)

    # -- Risk engine (Improvement #1): optional live/dry-run enforcement config. --
    if getattr(args, "risk_config", None):
        try:
            import json as _json
            from pathlib import Path as _Path

            rc_path = _Path(args.risk_config)
            if rc_path.is_file():
                text = rc_path.read_text()
                if rc_path.suffix.lower() in (".yaml", ".yml"):
                    import yaml as _yaml

                    rc_data = _yaml.safe_load(text) or {}
                else:
                    rc_data = _json.loads(text)
            elif args.risk_config.strip().startswith("{"):
                rc_data = _json.loads(args.risk_config)
            else:
                rc_data = {}
            from risk.risk_engine import RiskConfig, RiskEngine

            cfg = RiskConfig.from_dict(rc_data or {})
            args.risk_engine = RiskEngine(equity=float(getattr(args, "risk_equity", 10_000.0)), cfg=cfg)
            print(
                f"[Risk] Loaded risk config from {args.risk_config} | max_notional=${cfg.max_notional_usd:,.0f} "
                f"pos_cap={cfg.max_position_pct:.2%} dd_halt={cfg.max_drawdown_halt:.0%} "
                f"var={cfg.var_confidence:.0%}"
            )
        except Exception as e:
            print(f"[Risk] Failed to load --risk-config {args.risk_config}: {e}")
            args.risk_engine = None
    else:
        args.risk_engine = None

    # Time-anchored seq_len resolution
    _bf = getattr(args, "bar_freq", "5m")
    args.seq_len = _resolve_seq_len(args.seq_len, _bf)
    if hasattr(args, "curriculum") and isinstance(args.curriculum, dict):
        sched = args.curriculum.get("seq_schedule")
        if isinstance(sched, list):
            for entry in sched:
                if isinstance(entry, dict) and "seq_len" in entry:
                    entry["seq_len"] = _resolve_seq_len(entry["seq_len"], _bf)

    return args


def apply_hardware_profile(args):
    """Override training paths and loader settings for a known local GPU/RAM combo."""
    name = getattr(args, "hardware_profile", None)
    if not name:
        return
    prof = HARDWARE_PROFILES.get(name)
    if not prof:
        return
    for k, v in prof.items():
        if k == "local_project_paths":
            continue
        setattr(args, k, v)
    if prof.get("local_project_paths"):
        args.checkpoint_dir = PATHS["checkpoints"]
        args.data_cache = PATHS["data_processed"]
    print(
        f"[Hardware] profile={name} | batch={args.batch_size} | workers={args.num_workers} | "
        f"chunk={args.chunk_size} | prefetch={args.prefetch_factor}"
    )
    print(f"             checkpoint_dir={args.checkpoint_dir} | data_cache={args.data_cache}")


def _set_global_seed(seed: int | None) -> None:
    """Set all relevant RNG seeds when a seed is provided."""
    if seed is None:
        return
    s = int(seed)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _slug_part(value: object, max_len: int = 80) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    text = re.sub(r"-{2,}", "-", text)
    return text[:max_len].strip("-") or "run"


def _build_auto_run_name(args) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    pairs = _get_pairs(args)
    pair_label = f"{len(pairs)}pairs" if len(pairs) > 3 else "-".join(pairs)
    if getattr(args, "all_models", False):
        model_label = "all-models"
    elif getattr(args, "train_ensemble", False):
        model_label = f"ensemble-{getattr(args, 'model', 'base')}"
    elif getattr(args, "rl_train", False):
        model_label = f"rl-{getattr(args, 'rl_algo', 'dqn')}-{getattr(args, 'model', 'model')}"
    else:
        model_label = getattr(args, "model", "model")
    modes = []
    if getattr(args, "quick_mode", False):
        modes.append("quick")
    if getattr(args, "train_ensemble", False) and "ensemble" not in str(model_label):
        modes.append("ensemble")
    if getattr(args, "rl_train", False) and "rl" not in str(model_label):
        modes.append(f"rl-{getattr(args, 'rl_algo', 'dqn')}")
    if getattr(args, "walk_forward_cv", False):
        modes.append(f"wf{int(getattr(args, 'walk_forward_folds', 0) or 0)}")
    if getattr(args, "pretrain_ablation", None) not in (None, "false", False):
        modes.append(f"ablate-{args.pretrain_ablation}")
    if getattr(args, "deploy_ensemble", False):
        modes.append("deploy-ensemble")
    if getattr(args, "deploy_rl", False):
        modes.append("deploy-rl")
    parts = [
        ts,
        model_label,
        getattr(args, "strategy_mode", "strategy"),
        pair_label,
        f"seq{int(getattr(args, 'seq_len', 0) or 0)}",
        *modes,
    ]
    return _slug_part("_".join(str(p) for p in parts), max_len=140)


def _apply_auto_run_dir(args) -> str:
    env_dir = os.getenv("CHECKPOINT_RUN_DIR", "").strip()
    if env_dir and not getattr(args, "auto_run_dir", False):
        args.checkpoint_dir = env_dir
        run_name = args.run_name or Path(env_dir).name
        args.run_name = run_name
        args.run_name_slug = _slug_part(run_name, max_len=140)

        return run_name

    run_name = args.run_name or _build_auto_run_name(args)
    args.run_name = run_name
    args.run_name_slug = _slug_part(run_name, max_len=140)

    if getattr(args, "auto_run_dir", False):
        root = (
            Path(args.run_dir_root).expanduser()
            if getattr(args, "run_dir_root", None)
            else Path(args.checkpoint_dir).expanduser() / "runs"
        )
        run_dir = root / args.run_name_slug

        args.checkpoint_dir = str(run_dir)
        os.environ["CHECKPOINT_RUN_DIR"] = str(run_dir)
        run_doc = {
            "run_name": run_name,
            "checkpoint_dir": str(run_dir),
            "run_dir_root": str(root),
            "generated_at": datetime.now(UTC).isoformat(),
            "model": getattr(args, "model", None),
            "all_models": bool(getattr(args, "all_models", False)),
            "train_ensemble": bool(getattr(args, "train_ensemble", False)),
            "rl_train": bool(getattr(args, "rl_train", False)),
            "rl_algo": getattr(args, "rl_algo", None),
            "strategy_mode": getattr(args, "strategy_mode", None),
            "pairs": _get_pairs(args),
            "seq_len": int(getattr(args, "seq_len", 0) or 0),
            "walk_forward_folds": int(getattr(args, "walk_forward_folds", 0) or 0),
        }
        try:
            _safe_save_json(run_doc, root / "latest_run.json")
            _safe_save_json(run_doc, run_dir / "run_info.json")
        except Exception as exc:
            print(f"[RunDir] could not write run metadata: {exc}")
        print(f"[RunDir] auto-run-dir enabled -> {run_dir}")
    return run_name


_PROFILE_CLI_FLAGS = {
    "--lr": "lr",
    "--dropout": "dropout",
    "--num-layers": "num_layers",
    "--hidden-size": "hidden_size",
    "--d-model": "d_model",
    "--nhead": "nhead",
    "--seq-len": "seq_len",
    "--weight-decay": "weight_decay",
    "--batch-size": "batch_size",
    "--loss": "loss",
    "--early-stop-metric": "early_stop_metric",
    "--pretrain-method": "pretrain_method",
    "--pretrain-epochs": "pretrain_epochs",
    "--pretrain-lr": "pretrain_lr",
    "--pretrain-ablation": "pretrain_ablation",
}


def _collect_cli_profile_overrides() -> frozenset:
    """Dest names explicitly set on the CLI for profile-managed hyperparameters."""
    overrides: set[str] = set()
    argv = sys.argv[1:]
    idx = 0
    while idx < len(argv):
        tok = argv[idx]
        if tok in _PROFILE_CLI_FLAGS:
            overrides.add(_PROFILE_CLI_FLAGS[tok])
        elif tok.startswith("--") and "=" in tok:
            flag = tok.split("=", 1)[0]
            if flag in _PROFILE_CLI_FLAGS:
                overrides.add(_PROFILE_CLI_FLAGS[flag])
        idx += 1
    return frozenset(overrides)


def _normalize_architecture_profile(profile: dict, model_name: str) -> dict:
    """Map config/models.py keys to train_gpu argparse dest names."""
    key = model_name.lower().strip()
    out: dict = {}
    if "learning_rate" in profile:
        out["lr"] = float(profile["learning_rate"])
    if "dropout" in profile:
        out["dropout"] = float(profile["dropout"])
    if "seq_len" in profile:
        out["seq_len"] = int(profile["seq_len"])
    if "weight_decay" in profile:
        out["weight_decay"] = float(profile["weight_decay"])
    if "batch_size" in profile:
        out["batch_size"] = int(profile["batch_size"])
    if "loss" in profile:
        out["loss"] = str(profile["loss"]).lower()

    if "early_stop_metric" in profile:
        out["early_stop_metric"] = str(profile["early_stop_metric"]).lower()

    if "pretrain_epochs" in profile:
        out["pretrain_epochs"] = int(profile["pretrain_epochs"])
    if "pretrain_lr" in profile:
        out["pretrain_lr"] = float(profile["pretrain_lr"])
    if "pretrain_method" in profile:
        out["pretrain_method"] = str(profile["pretrain_method"]).lower()
    if "pretrain_ablation" in profile:
        out["pretrain_ablation"] = str(profile["pretrain_ablation"]).lower()

    # WIRE-002: Map dim_feedforward to dim_ff for all architectures
    if "dim_feedforward" in profile:
        out["dim_ff"] = int(profile["dim_feedforward"])

    if key == "haelt":
        if "lstm_hidden" in profile:
            out["hidden_size"] = int(profile["lstm_hidden"]) * 2
        if "d_model" in profile:
            out["d_model"] = int(profile["d_model"])
        if "nhead" in profile:
            out["nhead"] = int(profile["nhead"])
        if "num_layers" in profile:
            out["num_layers"] = int(profile["num_layers"])
        elif "n_transformer_layers" in profile:
            out["num_layers"] = int(profile["n_transformer_layers"])
    elif key == "tft":
        if "hidden_size" in profile:
            out["hidden_size"] = int(profile["hidden_size"])
        if "nhead" in profile:
            out["nhead"] = int(profile["nhead"])
        elif "attention_head_size" in profile:
            out["nhead"] = int(profile["attention_head_size"])
        if "lstm_layers" in profile:
            out["num_layers"] = int(profile["lstm_layers"])
    elif key == "gnn":
        if "hidden_channels" in profile:
            out["hidden_size"] = int(profile["hidden_channels"])
        if "num_layers" in profile:
            out["num_layers"] = int(profile["num_layers"])
        if "heads" in profile:
            out["nhead"] = int(profile["heads"])
        if "node_features" in profile:
            out["node_features"] = int(profile["node_features"])
    else:
        for field in ("d_model", "nhead", "num_layers", "hidden_size"):
            if field in profile:
                out[field] = int(profile[field])
    return out


def _apply_model_profile(args, model_name: str, *, enabled: bool = True):
    """Merge architecture_config(name) onto args; explicit CLI overrides win."""
    if not enabled:
        return args
    try:
        from config.models import architecture_config

        profile = architecture_config(model_name)
    except Exception as exc:
        print(f"[Profile] Skipped for {model_name}: {exc}")
        return args

    normalized = _normalize_architecture_profile(profile, model_name)
    if not normalized:
        return args

    cli_overrides = getattr(args, "_cli_profile_overrides", None) or _collect_cli_profile_overrides()
    log_parts: list[str] = []
    recipe_name = str(profile.get("recipe_name") or profile.get("decision_role") or "").strip()

    if recipe_name:
        args.recipe_name = recipe_name

        log_parts.append(f"recipe={recipe_name}")

    for dest, value in normalized.items():
        if dest in cli_overrides or not hasattr(args, dest):
            continue
        setattr(args, dest, value)
        if dest == "lr":
            log_parts.append(f"lr={float(value):.3e}")
        elif dest == "dropout":
            log_parts.append(f"dropout={float(value):.3f}")
        elif dest == "weight_decay":
            log_parts.append(f"weight_decay={float(value):.3e}")
        else:
            log_parts.append(f"{dest}={value}")

    # Apply training profile (adversarial, curriculum, miner, pretrain, SWA, etc.)
    _apply_training_profile(args, model_name, cli_overrides, log_parts)

    args.model = model_name
    args._profile_applied = True
    if log_parts:
        print(f"[Profile] {model_name}: " + " ".join(log_parts))
    return args


def _apply_training_profile(args, model_name: str, cli_overrides: frozenset, log_parts: list):
    """Apply per-model training dimensions from ModelTrainingProfile."""
    try:
        tprofile: ModelTrainingProfile = get_training_profile(model_name)
    except Exception as exc:
        print(f"[TrainingProfile] Skipped for {model_name}: {exc}")
        return

    # Map training profile fields to args (only if not CLI-overridden)
    training_fields = {
        # Adversarial
        "enable_adversarial": tprofile.adversarial_enabled,
        "adversarial_method": tprofile.adversarial_method,
        "adversarial_eps": tprofile.adversarial_eps,
        "adversarial_alpha": tprofile.adversarial_alpha,
        "adversarial_steps": tprofile.adversarial_steps,
        "adversarial_prob": tprofile.adversarial_prob,
        # Curriculum
        "curriculum_manager": getattr(args, "curriculum_manager", True),
        "curriculum_manager_mode": tprofile.curriculum_mode,
        # Self-paced / loss weighting flags
        "use_self_paced": tprofile.use_self_paced,
        "use_loss_weighting": tprofile.use_loss_weighting,
        # Online Miner feedback
        "curriculum_miner_feedback": tprofile.miner_feedback,
        "curriculum_forgetting_threshold": tprofile.forgetting_threshold,
        "curriculum_easy_threshold": tprofile.easy_threshold,
        "curriculum_freeze_patience": tprofile.freeze_patience,
        # Continuous Learning
        "enable_ewc": tprofile.enable_ewc,
        "ewc_lambda": tprofile.ewc_lambda,
        "enable_si": tprofile.enable_si,
        "si_lambda": tprofile.si_lambda,
        # Pretraining
        "pretrain_method": tprofile.pretrain_method,
        "pretrain_framework": tprofile.pretrain_framework,
        # SWA
        "swa_enabled": tprofile.swa_enabled,
        "swa_start_frac": tprofile.swa_start_frac,
        "swa_lr": tprofile.swa_lr,
        # EMA
        "pretrain_ema_decay": tprofile.ema_decay,
        # Framework
        "training_framework": tprofile.training_framework,
        # RL
        "rl_framework": tprofile.rl_framework,
        "rl_use_lstm": tprofile.rl_use_lstm,
    }

    features_report = {}
    report_keys = {
        "curriculum_manager": "curriculum_manager_mode",
        "self_paced": "use_self_paced",
        "loss_weighting": "use_loss_weighting",
        "miner_feedback": "curriculum_miner_feedback",
    }

    for label, dest in report_keys.items():
        yaml_val = getattr(args, dest, None)
        prof_val = training_fields.get(dest)

        if dest in cli_overrides:
            source = "CLI"
            val = getattr(args, dest, None)
        else:
            if yaml_val != prof_val:
                source = "profile"
            else:
                source = "yaml"
            val = prof_val
        features_report[label] = {"mode": val, "source": source}

    for dest, value in training_fields.items():
        if dest in cli_overrides or not hasattr(args, dest):
            continue
        setattr(args, dest, value)
        log_parts.append(f"{dest}={value}")

    args._training_features_report = features_report


def _member_training_args(base_args, model_name: str, member_idx: int, total_members: int):
    """
    Clone args, apply per-architecture profile, and optional ensemble diversity controls.
    """
    out = argparse.Namespace(**vars(base_args))
    out.model = model_name

    # Place each model's checkpoints in its own subfolder: checkpoints/<model_name>/
    base_ckpt = Path(base_args.checkpoint_dir)
    out.checkpoint_dir = str(base_ckpt / model_name)
    Path(out.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    if getattr(base_args, "model_profile", True):
        out = _apply_model_profile(out, model_name, enabled=True)

    explicit = bool(getattr(base_args, "ensemble_explicit_diversity", False))
    if not (explicit and total_members > 1):
        return out

    # Spread members across a deterministic [-0.5, +0.5] range.
    center = (total_members - 1) / 2.0
    rel = (member_idx - center) / max(1.0, float(total_members - 1))

    base_seed = getattr(base_args, "seed", None)
    if base_seed is not None:
        out.seed = int(base_seed) + int(getattr(base_args, "ensemble_member_seed_offset", 997)) * member_idx

    lr_jitter = float(getattr(base_args, "ensemble_member_lr_jitter", 0.0))
    if lr_jitter > 0:
        out.lr = float(out.lr) * max(0.25, 1.0 + rel * lr_jitter)

    drop_jitter = float(getattr(base_args, "ensemble_member_dropout_jitter", 0.0))
    if drop_jitter > 0:
        out.dropout = float(np.clip(float(out.dropout) + rel * drop_jitter, 0.0, 0.8))

    print(
        f"[EnsembleDiversity] {model_name}: seed={getattr(out, 'seed', None)} "
        f"lr={out.lr:.3e} dropout={out.dropout:.3f} (member {member_idx + 1}/{total_members})"
    )
    if getattr(out, "pretrain", False) or getattr(out, "ablate_pretrain", False):
        from config.model_training_profile import pretrain_method_for

        out.pretrain_method = pretrain_method_for(model_name)

    return out


def _model_build_args(base_args, model_name: str) -> argparse.Namespace:
    """Per-architecture args for build_model / checkpoint load (no ensemble jitter)."""
    out = argparse.Namespace(**vars(base_args))
    out.model = str(model_name).lower().strip()
    if getattr(base_args, "model_profile", True):
        out = _apply_model_profile(out, out.model, enabled=True)
    return out


def _model_completion_status(model_name: str, checkpoint_dir: str | Path) -> tuple[bool, str]:
    """Return whether an all-model member appears fully trained.

    Crash checkpoints are deliberately ignored: they prove the model started,
    not that it produced a clean resume/best artifact.
    """
    model = str(model_name).lower().strip()
    ckpt_dir = Path(checkpoint_dir)
    best_paths = [
        ckpt_dir / f"{model}_best.pt",
        ckpt_dir / model / f"{model}_best.pt",
    ]
    has_best = any(p.exists() for p in best_paths)
    has_manifest = (ckpt_dir / "manifest.json").exists()
    has_train_summary = (ckpt_dir / "train_summary.json").exists()
    has_fold_selection = (ckpt_dir / "fold_selection.json").exists()
    has_deployment = (ckpt_dir / "deployment.json").exists()
    crash_files = sorted(ckpt_dir.glob(f"*{model}*_crash.pt"))

    if has_deployment and (has_best or has_manifest):
        return True, "deployment.json + completed checkpoint metadata"
    if has_fold_selection and has_best:
        return True, "fold_selection.json + best checkpoint"
    if has_manifest and has_best:
        return True, "manifest.json + best checkpoint"
    if has_train_summary and has_best and not has_fold_selection:
        return True, "train_summary.json + best checkpoint"
    if crash_files and not has_best:
        return False, f"crash checkpoint only ({crash_files[-1].name})"
    if has_best:
        return False, "best checkpoint exists but completion metadata is missing"
    return False, "no completed artifacts"


def _baseline_ablation_completion_status(model_name: str, checkpoint_dir: str | Path, args) -> tuple[bool, str]:
    """Return whether the no-pretrain baseline proof for a model is already complete.

    Baseline ablation artifacts live under <checkpoint_dir>/baseline and do not go
    through the full model promotion/deployment path, so the generic completion
    helper is too weak here. For walk-forward runs we require every expected fold
    best checkpoint before skipping baseline on resume.
    """
    model = str(model_name).lower().strip()
    baseline_dir = Path(checkpoint_dir) / "baseline"
    if not baseline_dir.exists():
        return False, "baseline artifact directory missing"

    walk_forward = bool(getattr(args, "walk_forward_cv", False))
    if walk_forward:
        n_folds = max(1, int(getattr(args, "walk_forward_folds", 1)))
        missing = []
        for fi in range(n_folds):
            fold_best = baseline_dir / f"baseline_{model}_fold{fi}_best.pt"
            if not fold_best.exists():
                missing.append(fold_best.name)
        if not missing:
            return True, f"all {n_folds} baseline fold checkpoints present"
        return False, f"missing baseline fold checkpoints: {', '.join(missing[:3])}" + (
            " ..." if len(missing) > 3 else ""
        )

    single_best = baseline_dir / f"baseline_{model}_best.pt"
    if single_best.exists():
        return True, "single-split baseline checkpoint present"
    return False, f"missing {single_best.name}"


def _supervised_resume_status(model_name: str, checkpoint_dir: str | Path, args) -> tuple[bool, str]:
    """Return whether supervised training already started for this model."""
    model = str(model_name).lower().strip()
    ckpt_dir = Path(checkpoint_dir)
    walk_forward = bool(getattr(args, "walk_forward_cv", False))

    if walk_forward:
        n_folds = max(1, int(getattr(args, "walk_forward_folds", 1)))
        last_paths = [ckpt_dir / f"{model}_fold{fi}_last.pt" for fi in range(n_folds)]
        best_paths = [ckpt_dir / f"{model}_fold{fi}_best.pt" for fi in range(n_folds)]
        existing_last = [p for p in last_paths if p.exists()]
        if existing_last:
            latest = max(existing_last, key=lambda p: p.stat().st_mtime if p.exists() else 0.0)
            return True, f"supervised resume checkpoint present ({latest.name})"
        existing_best = [p for p in best_paths if p.exists()]
        if existing_best:
            latest = max(existing_best, key=lambda p: p.stat().st_mtime if p.exists() else 0.0)
            return True, f"supervised fold checkpoint present ({latest.name})"
        return False, "no supervised fold checkpoints found"

    last_path = ckpt_dir / f"{model}_last.pt"
    best_path = ckpt_dir / f"{model}_best.pt"
    if last_path.exists():
        return True, f"supervised resume checkpoint present ({last_path.name})"
    if best_path.exists():
        return True, f"supervised best checkpoint present ({best_path.name})"
    return False, "no supervised checkpoints found"


def _latest_resumable_fold(model_name: str, checkpoint_dir: str | Path, n_folds: int) -> int | None:
    """Return the latest fold index with a resumable supervised checkpoint."""
    model = str(model_name).lower().strip()
    ckpt_dir = Path(checkpoint_dir)
    candidates: list[tuple[float, int]] = []
    for fi in range(max(1, int(n_folds))):
        for path in (
            ckpt_dir / f"{model}_fold{fi}_last.pt",
            ckpt_dir / f"{model}_fold{fi}_best.pt",
        ):
            if path.exists():
                try:
                    candidates.append((path.stat().st_mtime, fi))
                except Exception:
                    candidates.append((0.0, fi))
                break
    if not candidates:
        return None
    candidates.sort()
    return int(candidates[-1][1])


def _effective_max_seq_len(args) -> int:
    """Max sequence length required by training config + curriculum schedule."""
    seqs = [int(getattr(args, "seq_len", 60) or 60)]
    cur = getattr(args, "curriculum", None)
    if cur is None or cur is False or cur == "none" or cur == "":
        return max(seqs)
    if not isinstance(cur, dict):
        cur = SETTINGS_CURRICULUM
    for entry in cur.get("seq_schedule") or []:
        if isinstance(entry, dict) and entry.get("seq_len") is not None:
            seqs.append(int(entry["seq_len"]))
    return max(seqs)


def _load_cv_fold_entry(
    model_name: str,
    checkpoint_dir: str | Path,
    fold_idx: int,
    early_stop_metric: str = "sharpe",
) -> dict | None:
    """Rebuild one walk-forward fold summary from its checkpoint artifacts."""
    model = str(model_name).lower().strip()
    ckpt_dir = Path(checkpoint_dir)
    fold_suffix = f"_fold{int(fold_idx)}"
    last_path = ckpt_dir / f"{model}{fold_suffix}_last.pt"
    best_path = ckpt_dir / f"{model}{fold_suffix}_best.pt"
    ckpt_path = last_path if last_path.exists() else (best_path if best_path.exists() else None)
    if ckpt_path is None:
        return None
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        return None
    history = ck.get("history")
    if not isinstance(history, dict) or not history:
        return None
    best_metric = None
    if early_stop_metric == "sharpe" and history.get("val_sharpe"):
        best_metric = float(max(history["val_sharpe"]))
    elif history.get("val_loss"):
        best_metric = float(min(history["val_loss"]))
    elif ck.get("best_sharpe") is not None:
        best_metric = float(ck["best_sharpe"])
    elif ck.get("best_val_loss") is not None:
        best_metric = float(ck["best_val_loss"])
    return {"fold": int(fold_idx), "best_metric": best_metric, "history": history}


def _load_walk_forward_resume_history(
    model_name: str,
    checkpoint_dir: str | Path,
    log_dir: Path,
    run_name_slug: str,
    model_slug: str,
    start_fold: int,
    early_stop_metric: str = "sharpe",
) -> list[dict]:
    """Load completed fold metrics for folds [0, start_fold) when resuming walk-forward CV."""
    if start_fold <= 0:
        return []
    entries: list[dict] = []
    cv_path = Path(log_dir) / f"{run_name_slug}_{model_slug}_cv.json"
    if cv_path.exists():
        try:
            data = json.loads(cv_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                entries = [e for e in data if isinstance(e, dict) and int(e.get("fold", -1)) < int(start_fold)]
        except Exception as exc:
            _log_warn(f"[WalkForward] Could not read prior cv.json ({exc}); rebuilding from checkpoints.")
    loaded_folds = {int(e.get("fold", -1)) for e in entries}
    for fi in range(int(start_fold)):
        if fi in loaded_folds:
            continue
        entry = _load_cv_fold_entry(model_name, checkpoint_dir, fi, early_stop_metric)
        if entry is not None:
            entries.append(entry)
    entries.sort(key=lambda e: int(e.get("fold", 0)))
    return entries
