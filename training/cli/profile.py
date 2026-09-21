"""Per-architecture model profile application and ensemble member arg cloning."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from config.model_training_profile import (
    ModelTrainingProfile,
    get_training_profile,
)

from .helpers import _collect_cli_profile_overrides


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
        "rl_finetune": tprofile.rl_finetune,
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
        if dest in cli_overrides:
            continue
        setattr(args, dest, value)
        log_parts.append(f"{dest}={value}")

    args._training_features_report = features_report


def _member_training_args(base_args, model_name: str, member_idx: int, total_members: int):
    """Clone args, apply per-architecture profile, and optional ensemble diversity controls."""
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
