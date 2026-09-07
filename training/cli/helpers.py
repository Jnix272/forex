"""Seed, slug, run-dir, and CLI-override-collection helpers."""

from __future__ import annotations

import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from training.cache_integrity import _get_pairs
from training.dataset_builder import _safe_save_json


def _set_global_seed(seed: int | None) -> None:
    """Set all relevant RNG seeds when a seed is provided."""
    if seed is None:
        return
    s = int(seed)
    import random
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


# Hyperparameter CLI flags that model profiles can also set (explicit CLI wins)
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
