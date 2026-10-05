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


def _warn_unsupported_training_options(args) -> list[str]:
    """One-line warnings for options that are set but have no effect in this configuration."""
    explicit = frozenset(getattr(args, "_yaml_explicit_keys", None) or ()) | frozenset(
        getattr(args, "_cli_profile_overrides", None) or ()
    )
    multitask = bool(getattr(args, "multitask", False))
    loss = str(getattr(args, "loss", "") or "").lower()
    msgs: list[str] = []
    if getattr(args, "use_volatility_sampler", False):
        msgs.append(
            "use_volatility_sampler=true is not supported by the streaming Zarr loader "
            "(a full-epoch stratified draw is just a permutation); ignored."
        )
    if multitask and "direction_weight" in explicit:
        msgs.append(
            f"direction_weight={getattr(args, 'direction_weight', None)} has no effect with multitask "
            "(only the non-multitask directional_huber loss uses it; multitask w_dir is 1.0)."
        )
    if "sharpe_weight" in explicit and loss != "sharpe_huber":
        msgs.append(
            f"sharpe_weight={getattr(args, 'sharpe_weight', None)} has no effect with loss={loss} "
            "(only loss=sharpe_huber adds the Sharpe term)."
        )
    floor = getattr(args, "mt_direction_weight_floor", None)
    if multitask and floor is not None:
        msgs.append(
            f"multitask.direction_weight_floor={floor} is inert: the multitask direction weight "
            "is fixed at 1.0 (never below the floor)."
        )
    for msg in msgs:
        print(f"[Config] WARN: {msg}")
    return msgs


def _parser_option_map(parser) -> dict[str, str]:
    """option string (incl. ``--no-x`` forms) -> dest for every parser action."""
    out: dict[str, str] = {}
    for action in getattr(parser, "_actions", ()):
        dest = getattr(action, "dest", None)
        if not dest or dest == "help":
            continue
        for opt in getattr(action, "option_strings", ()) or ():
            out[opt] = dest
    return out


def _collect_cli_profile_overrides(parser=None, argv: list[str] | None = None) -> frozenset:
    """Dest names explicitly passed on the CLI.

    With ``parser`` every registered option is detected generically (including
    ``--flag=value``, ``--no-flag`` and unambiguous argparse prefixes); without
    it only the legacy profile-managed flags are recognised.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    option_map = _parser_option_map(parser) if parser is not None else dict(_PROFILE_CLI_FLAGS)
    allow_abbrev = bool(getattr(parser, "allow_abbrev", False)) if parser is not None else False
    long_opts = [o for o in option_map if o.startswith("--")]
    overrides: set[str] = set()
    for tok in argv:
        if tok == "--":
            break
        if not tok.startswith("-") or tok == "-":
            continue
        flag = tok.split("=", 1)[0]
        dest = option_map.get(flag)
        if dest is None and allow_abbrev and flag.startswith("--"):
            matches = {option_map[o] for o in long_opts if o.startswith(flag)}
            if len(matches) == 1:
                dest = next(iter(matches))
        if dest is not None:
            overrides.add(dest)
    return frozenset(overrides)
