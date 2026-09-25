import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import optuna
import torch
import yaml

ARTIFACT_DIR = Path("logs/optuna")
OPTUNA_CONFIG_DIR = Path("config/optuna")  # isolated folder for all trial + best configs
BEST_CONFIG_DIR = OPTUNA_CONFIG_DIR  # kept for compat with helpers that reference it
ACTIVE_RUN_CONFIG = Path("config/run.yaml")  # base config; never overwritten by the tuner
DEFAULT_METRIC = "cost_sharpe"


def _direction_warmup_epochs() -> int:
    """direction_training.warmup_epochs from config/run.yaml (trainer default 2)."""
    try:
        cfg = yaml.safe_load(Path("config/run.yaml").read_text(encoding="utf-8")) or {}
        return max(0, int((cfg.get("direction_training") or {}).get("warmup_epochs", 2)))
    except Exception:
        return 2


def _fail_stale_running_trials(study: optuna.Study) -> None:
    """Mark RUNNING trials left by an interrupted process as FAIL before resuming."""
    for t in study.trials:
        if t.state == optuna.trial.TrialState.RUNNING:
            try:
                study._storage.set_trial_state_values(t._trial_id, optuna.trial.TrialState.FAIL)
                print(f"[Optuna] Marked stale RUNNING trial {t.number} as FAIL")
            except Exception as exc:
                print(f"[Optuna] Could not close stale trial {t.number}: {exc}")


def _metric_direction(metric: str) -> str:
    """Map metric to Optuna direction (val_loss -> minimize, val_sharpe -> maximize)."""
    return "minimize" if str(metric).lower() == "val_loss" else "maximize"


def _safe_slug(text: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", str(text).lower()).strip("-") or "run"


def _mode_defaults(mode: str) -> dict[str, Any]:
    # Proxies must outlast the 2 direction-warmup epochs (which train a different,
    # direction-only objective); 2-epoch proxies measured nothing else.
    if mode == "deep":
        return {
            "epochs": 12,
            "folds": 2,
            "confirm_top_k": 3,
            "full_confirm_folds": 7,
            "full_confirm_epochs": 18,
        }
    return {
        "epochs": 6,
        "folds": 2,
        "confirm_top_k": 0,
        "full_confirm_folds": 7,
        "full_confirm_epochs": 18,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna tuner for forex deep models")
    parser.add_argument("--model", required=True, choices=["tft", "haelt", "transformer"], help="Architecture to tune.")
    parser.add_argument(
        "--mode",
        default="cheap",
        choices=["cheap", "deep"],
        help="Cheap = fast proxy search. Deep = stronger proxy + confirmation.",
    )
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials.")
    parser.add_argument("--epochs", type=int, default=0, help="Override trial epochs. 0 = mode default.")
    parser.add_argument("--folds", type=int, default=0, help="Override proxy walk-forward folds. 0 = mode default.")
    parser.add_argument(
        "--metric", default=DEFAULT_METRIC, choices=["val_loss", "val_sharpe", "cost_sharpe"], help="Trial objective metric."
    )
    parser.add_argument(
        "--confirm-top-k",
        type=int,
        default=-1,
        help="After the proxy search, rerun the top K trials with fuller CV. -1 = mode default.",
    )
    parser.add_argument(
        "--full-confirm-folds", type=int, default=0, help="Full CV folds for top-K confirmation. 0 = mode default."
    )
    parser.add_argument(
        "--full-confirm-epochs", type=int, default=0, help="Epochs for top-K confirmation. 0 = mode default."
    )
    parser.add_argument("--study-name", type=str, default="", help="Optional explicit Optuna study name.")
    parser.add_argument("--seed", type=int, default=1337, help="Sampler/training seed.")
    parser.add_argument(
        "--hpo-scheduler",
        type=str,
        default="tpe",
        choices=["tpe", "pbt", "bohb", "asha"],
        help="HPO strategy (Improvement #12): tpe=default; "
        "asha=successive halving; bohb=BO+hyperband; "
        "pbt=population-based evolutionary.",
    )
    parser.add_argument(
        "--curriculum-only",
        action="store_true",
        help="Search curriculum shape + adaptation thresholds only; "
        "keep model/training hyperparams from config/run.yaml.",
    )
    parser.add_argument(
        "--launch-training",
        action="store_true",
        help="After the study finishes and best config is applied, "
        "start a full training.train_gpu run using config/run.yaml.",
    )
    parser.add_argument(
        "--auto", action="store_true", help="Shorthand for --launch-training (search -> apply best -> full train)."
    )
    return parser.parse_args()


def _trial_checkpoint_path(checkpoint_dir: Path, model_name: str, folds: int) -> Path:
    model = str(model_name).lower().strip()
    candidates: list[Path] = []
    for fi in range(max(1, int(folds))):
        candidates.extend(
            [
                checkpoint_dir / f"{model}_fold{fi}_last.pt",
                checkpoint_dir / f"{model}_fold{fi}_best.pt",
                checkpoint_dir / model / f"{model}_fold{fi}_last.pt",
                checkpoint_dir / model / f"{model}_fold{fi}_best.pt",
            ]
        )
    candidates.extend(
        [
            checkpoint_dir / f"{model}_last.pt",
            checkpoint_dir / f"{model}_best.pt",
            checkpoint_dir / model / f"{model}_last.pt",
            checkpoint_dir / model / f"{model}_best.pt",
        ]
    )
    existing = [p for p in candidates if p.exists()]
    if not existing:
        raise RuntimeError(f"No trial checkpoint found under {checkpoint_dir}")
    return max(existing, key=lambda p: p.stat().st_mtime)


def _trial_summary_path(checkpoint_dir: Path, model_name: str = "") -> Path:
    candidates = [checkpoint_dir / "train_summary.json"]
    if model_name:
        candidates.insert(0, checkpoint_dir / model_name / "train_summary.json")
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def _trial_control_report_path(checkpoint_dir: Path, model_name: str) -> Path:
    """Locate the training-control report written by train_gpu after the epoch loop."""
    candidates = [
        checkpoint_dir / model_name / f"{model_name}_training_control_report.json",
        checkpoint_dir / model_name / f"{model_name}_fold0_training_control_report.json",
        checkpoint_dir / f"{model_name}_training_control_report.json",
        checkpoint_dir / f"{model_name}_fold0_training_control_report.json",
        checkpoint_dir / f"{model_name}_control_report.json",
        checkpoint_dir / model_name / f"{model_name}_control_report.json",
        checkpoint_dir / "train_control_report.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _curriculum_diagnostics(history: dict[str, Any]) -> dict[str, Any]:
    """Extract curriculum health signals from checkpoint history (P2)."""
    stalls_curve = history.get("curriculum_stalls", [])
    total_stalls = int(stalls_curve[-1]) if stalls_curve else 0

    events = history.get("curriculum_events", [])
    advance_types = {"seq_len_increase", "difficulty_increase", "seq_len_schedule_floor", "difficulty_schedule_floor"}
    recovery_count = sum(1 for e in events if e.get("type") == "recovery")
    advance_count = sum(1 for e in events if e.get("type") in advance_types)

    # Did the schedule ever advance beyond its first stage?
    seq_hist = history.get("seq_len", [])
    diff_hist = history.get("difficulty_stage", [])
    seq_advanced = len(seq_hist) > 0 and max(seq_hist) > seq_hist[0] if seq_hist else False
    diff_advanced = len(diff_hist) > 0 and max(diff_hist) > diff_hist[0] if diff_hist else False

    return {
        "total_stalls": total_stalls,
        "advance_count": advance_count,
        "recovery_count": recovery_count,
        "seq_advanced": seq_advanced,
        "diff_advanced": diff_advanced,
    }


def _metric_score(metric: str, summary: dict[str, Any], history: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    val_sharpe = summary.get("best_val_sharpe")
    final_sharpe = summary.get("final_val_sharpe")
    val_loss = summary.get("best_val_loss")
    cost_sharpe = summary.get("best_cost_sharpe")
    gen_gap = summary.get("gen_gap_final")
    epochs_completed = int(summary.get("epochs_completed") or len(history.get("train_loss", [])) or 0)
    train_mode = summary.get("train_mode", "unknown")
    n_folds = int(summary.get("n_folds") or 1)

    if val_sharpe is None:
        sharpe_hist = history.get("val_sharpe", [])
        val_sharpe = max(sharpe_hist) if sharpe_hist else None
    if final_sharpe is None:
        sharpe_hist = history.get("val_sharpe", [])
        final_sharpe = sharpe_hist[-1] if sharpe_hist else None
    if val_loss is None:
        loss_hist = history.get("val_loss", [])
        val_loss = min(loss_hist) if loss_hist else None
    if cost_sharpe is None:
        cost_hist = history.get("cost_aware_sharpe", history.get("cost_sharpe", history.get("val_cost_sharpe", [])))
        cost_sharpe = max(cost_hist) if cost_hist else None
    if gen_gap is None:
        loss_hist = history.get("val_loss", [])
        train_hist = history.get("train_loss", [])
        if loss_hist and train_hist:
            gen_gap = float(loss_hist[-1] - train_hist[-1])

    curr_diag = _curriculum_diagnostics(history)

    diagnostics = {
        "best_val_sharpe": val_sharpe,
        "final_val_sharpe": final_sharpe,
        "best_val_loss": val_loss,
        "best_cost_sharpe": cost_sharpe,
        "gen_gap_final": gen_gap,
        "epochs_completed": epochs_completed,
        "train_mode": train_mode,
        "n_folds": n_folds,
        **curr_diag,
    }

    if metric == "val_loss":
        if val_loss is None:
            # Fragility fix: return worst score instead of raising -> trial pruned with diagnostics
            diagnostics["missing_metric"] = "val_loss"
            return float("inf"), diagnostics
        return float(val_loss), diagnostics

    if metric == "cost_sharpe":
        if cost_sharpe is None:
            diagnostics["missing_metric"] = "cost_sharpe"
            return float("-inf"), diagnostics
        return float(cost_sharpe), diagnostics

    if val_sharpe is None:
        diagnostics["missing_metric"] = "val_sharpe"
        return float("-inf"), diagnostics

    score = float(val_sharpe)
    if final_sharpe is not None:
        score -= max(0.0, float(val_sharpe) - float(final_sharpe)) * 0.25
    if gen_gap is not None and float(gen_gap) > 0:
        score -= float(gen_gap) * 0.10
    if n_folds > 1:
        score += min(0.05, 0.01 * n_folds)

    # P2: Curriculum health penalties/bonuses. Skipped for short proxies: the
    # curriculum is pinned there, so "never advanced" was a constant -0.10.
    if epochs_completed < _MIN_EPOCHS_FOR_CURRICULUM:
        return score, diagnostics
    total_stalls = curr_diag["total_stalls"]
    advance_count = curr_diag["advance_count"]
    seq_advanced = curr_diag["seq_advanced"]
    diff_advanced = curr_diag["diff_advanced"]

    # Penalty: excessive stalls -> unstable training (gradient noise / schedule too aggressive)
    if total_stalls > 2:
        score -= 0.05 * (total_stalls - 2)  # -0.05 per stall above the 2-stall tolerance

    # Penalty: schedule never advanced -> curriculum too conservative or model stuck at easy
    if not seq_advanced and not diff_advanced:
        score -= 0.10

    # Bonus: at least one advance of each kind -> healthy progression
    if seq_advanced:
        score += 0.03
    if diff_advanced:
        score += 0.03
    # Bonus: multiple advances -> curriculum is actually progressing
    if advance_count > 2:
        score += min(0.05, 0.01 * (advance_count - 2))

    return score, diagnostics


def _trial_report_path(study_name: str, trial_number: int) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACT_DIR / f"{_safe_slug(study_name)}_trial_{trial_number}.json"


def _ranked_report_path(study_name: str) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACT_DIR / f"{_safe_slug(study_name)}_ranked_report.json"


def _best_config_path(model_name: str, metric: str) -> Path:
    return BEST_CONFIG_DIR / f"run_optuna_best_{_safe_slug(model_name)}_{_safe_slug(metric)}.yaml"


def _best_summary_path(model_name: str, metric: str) -> Path:
    return ARTIFACT_DIR / f"optuna_best_{_safe_slug(model_name)}_{_safe_slug(metric)}.json"


def _hardware_safe_batch_choices(model_name: str, d_model: int, seq_len: int) -> list[int]:
    base = [128, 256, 512]
    complexity = int(d_model) * int(seq_len)
    if model_name == "transformer":
        complexity = int(complexity * 1.25)
    elif model_name == "haelt":
        complexity = int(complexity * 1.10)

    if complexity >= 90_000:
        return [64, 128]
    if complexity >= 45_000:
        return [64, 128, 256]
    return base


# ---------------------------------------------------------------------------
# Curriculum shape helpers
# ---------------------------------------------------------------------------

_CURRICULUM_PARAMS = (
    "cur_seq_start",
    "cur_seq_ramp_epoch",
    "cur_seq_target",
    "cur_collapse_drop",
    "cur_collapse_min_peak",
    "cur_advance_lr_mult",
    "cur_collapse_lr_mult",
    "cur_stable_window",
    "cur_reversal_threshold",
    "cur_recovery_window",
    "cur_min_epochs_per_stage",
    # difficulty schedule (P2)
    "cur_diff_ramp_epoch",
    "cur_diff_final_stage",
)


def _read_arch_params_from_config(base_cfg_path: Path) -> dict[str, Any]:
    """Read fixed architecture / training hyperparams from the active run config."""
    with base_cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    model = cfg.get("model", {}) or {}
    training = cfg.get("training", {}) or {}
    multitask = cfg.get("multitask", {}) or {}
    d_model = int(model.get("d_model") or model.get("hidden_size") or 256)
    return {
        "lr": float(training.get("lr", 5e-5)),
        "d_model": d_model,
        "hidden_size": int(model.get("hidden_size") or d_model),
        "nhead": int(model.get("nhead", 8)),
        "num_layers": int(model.get("num_layers", 3)),
        "dropout": float(model.get("dropout", 0.25)),
        "batch_size": int(training.get("batch_size", 256)),
        "mt_direction_weight_floor": float(multitask.get("direction_weight_floor", 0.30)),
        "mt_focal_gamma": float(multitask.get("focal_gamma", 1.5)),
        "mt_class_balance_weight": float(multitask.get("class_balance_weight", 0.15)),
    }


def _assemble_trial_params(
    raw: dict[str, Any],
    *,
    curriculum_only: bool,
    base_cfg_path: Path,
    model_name: str = "",
) -> dict[str, Any]:
    """Merge Optuna trial dict with architecture params when searching curriculum only."""
    raw = dict(raw)
    if any(k not in raw for k in ("cur_seq_start", "cur_seq_ramp_epoch", "cur_seq_target")):
        # Curriculum was pinned (short proxy) and not stored on the trial: restore the pinned values.
        for k, v in _sample_curriculum(_FixedMidTrial(), model_name).items():
            raw.setdefault(k, v)
    if not curriculum_only:
        return raw
    arch = _read_arch_params_from_config(base_cfg_path)
    curriculum = {k: raw[k] for k in _CURRICULUM_PARAMS if k in raw}
    return {**arch, **curriculum}


_SEQ_TARGET_CHOICES: dict[str, list[int]] = {
    "tft": [45, 60, 90],
    "haelt": [60, 90, 120],
    "transformer": [60, 90, 120],
}


def _seq_len_ceiling(model_name: str) -> int:
    """Fixed dataset-cache seq_len for the whole study (max of curriculum targets).

    The data cache filename is keyed by ``seq_len`` (see ``_get_cache_path`` in
    ``train_gpu.py``). If each trial built the cache at its own sampled
    ``cur_seq_target``, every distinct value would trigger a full rebuild from
    raw ticks (hours, for an 18-year multi-pair history). Pinning the cache
    size to a constant ceiling means the cache is built once (by the first
    trial) and reused by every subsequent trial; the curriculum still ramps
    up to each trial's own ``cur_seq_target`` *within* that fixed window via
    runtime batch slicing (see ``_effective_max_seq_len`` in ``train_gpu.py``).
    """
    choices = _SEQ_TARGET_CHOICES.get(str(model_name).lower().strip(), [60, 90, 120])
    return max(choices)


def _sample_curriculum(trial, model_name: str) -> dict[str, Any]:
    """Sample curriculum shape parameters that control how training complexity grows.

    These replace the static ``seq_len`` parameter so that Optuna searches
    over the *dynamics* of training, not just the final window length.
    """
    model_key = str(model_name).lower().strip()
    if model_key == "tft":
        seq_start_choices = [20, 30, 45]
    else:
        seq_start_choices = [30, 45, 60]
    seq_target_choices = _SEQ_TARGET_CHOICES.get(model_key, [60, 90, 120])

    cur_seq_start = trial.suggest_categorical("cur_seq_start", seq_start_choices)
    cur_seq_target = trial.suggest_categorical("cur_seq_target", seq_target_choices)
    # Ensure target >= start (Optuna may pick any combination)
    cur_seq_target = max(cur_seq_target, cur_seq_start)

    return {
        # Sequence length ramp --------------------------------------------------
        "cur_seq_start": cur_seq_start,
        # Epoch at which the ramp begins (midpoint of the search range)
        "cur_seq_ramp_epoch": trial.suggest_categorical("cur_seq_ramp_epoch", [6, 10, 14, 20]),
        "cur_seq_target": cur_seq_target,
        # Auto-Tuner sensitivity ------------------------------------------------
        # How aggressively the stall fires (smaller = more sensitive)
        "cur_collapse_drop": trial.suggest_categorical("cur_collapse_drop", [0.10, 0.15, 0.20, 0.25]),
        # Minimum peak the model must hit before a stall is allowed
        "cur_collapse_min_peak": trial.suggest_categorical("cur_collapse_min_peak", [0.15, 0.25, 0.35]),
        # LR multiplier when the curriculum advances to a harder market stage
        "cur_advance_lr_mult": trial.suggest_categorical("cur_advance_lr_mult", [0.75, 0.85, 0.95]),
        # LR multiplier when a Sharpe collapse is detected
        "cur_collapse_lr_mult": trial.suggest_categorical("cur_collapse_lr_mult", [0.70, 0.80, 0.90]),
        # Consecutive stable epochs needed before advancing market difficulty
        "cur_stable_window": trial.suggest_categorical("cur_stable_window", [2, 3, 4]),
        # Single-epoch Sharpe drop below this also triggers a stall (P1 unified guard)
        "cur_reversal_threshold": trial.suggest_categorical("cur_reversal_threshold", [-0.15, -0.10, -0.05]),
        # Epochs stable post-stall required before unfreezing (P1 recovery)
        "cur_recovery_window": trial.suggest_categorical("cur_recovery_window", [3, 4, 5]),
        # Minimum cooldown epochs between any two advances (P1 cooldown)
        "cur_min_epochs_per_stage": trial.suggest_categorical("cur_min_epochs_per_stage", [2, 3, 4]),
        # Difficulty schedule (P2) - when to introduce medium/hard market bars
        # epoch at which medium-difficulty bars first appear (hard always follows 4 epochs later)
        "cur_diff_ramp_epoch": trial.suggest_categorical("cur_diff_ramp_epoch", [4, 6, 8, 12]),
        # highest difficulty stage reached by end of training (0=easy only, 1=medium, 2=all bars)
        "cur_diff_final_stage": trial.suggest_categorical("cur_diff_final_stage", [1, 2]),
    }


def _build_difficulty_schedule(
    cur_diff_ramp_epoch: int,
    cur_diff_final_stage: int,
) -> list[dict]:
    """Convert difficulty params into a concrete difficulty_schedule list.

    Structure:
      epoch 0                   -> stage 0 (easy bars only)
      cur_diff_ramp_epoch       -> stage 1 (medium bars introduced)
      cur_diff_ramp_epoch + 4   -> stage cur_diff_final_stage (hard bars if final_stage == 2)
    """
    schedule = [{"epoch_start": 0, "max_difficulty": 0}]
    schedule.append({"epoch_start": int(cur_diff_ramp_epoch), "max_difficulty": 1})
    if cur_diff_final_stage >= 2:
        hard_epoch = int(cur_diff_ramp_epoch) + 4
        schedule.append({"epoch_start": hard_epoch, "max_difficulty": 2})
    # Deduplicate by epoch_start
    seen: set[int] = set()
    deduped = []
    for entry in schedule:
        if entry["epoch_start"] not in seen:
            seen.add(entry["epoch_start"])
            deduped.append(entry)
    return deduped


def _build_seq_schedule(
    cur_seq_start: int, cur_seq_ramp_epoch: int, cur_seq_target: int, total_epochs: int
) -> list[dict]:
    """Convert curriculum params into a concrete seq_schedule list.

    Structure:
      - epoch 0              -> cur_seq_start
      - cur_seq_ramp_epoch   -> midpoint between start and target
      - cur_ramp_epoch + gap -> cur_seq_target
    """
    if cur_seq_target <= cur_seq_start:
        return [{"epoch_start": 0, "seq_len": int(cur_seq_start)}]

    mid = round((cur_seq_start + cur_seq_target) / 2)
    second_ramp = min(cur_seq_ramp_epoch + max(4, (total_epochs - cur_seq_ramp_epoch) // 2), total_epochs - 1)
    schedule = [
        {"epoch_start": 0, "seq_len": int(cur_seq_start)},
        {"epoch_start": int(cur_seq_ramp_epoch), "seq_len": int(mid)},
        {"epoch_start": int(second_ramp), "seq_len": int(cur_seq_target)},
    ]
    # Keep stages inside the run and in epoch order. With few epochs the clamp to
    # total_epochs - 1 used to put the second ramp *before* the first
    # (e.g. {10: 60}, {1: 90}).
    last = max(0, int(total_epochs) - 1)
    seen: set[int] = set()
    deduped = []
    for entry in sorted(schedule, key=lambda e: e["epoch_start"]):
        if entry["epoch_start"] > last or entry["epoch_start"] in seen:
            continue
        if deduped and entry["seq_len"] <= deduped[-1]["seq_len"]:
            continue
        seen.add(entry["epoch_start"])
        deduped.append(entry)
    return deduped


# ---------------------------------------------------------------------------
# Model + curriculum param sampling
# ---------------------------------------------------------------------------


_MIN_EPOCHS_FOR_CURRICULUM = 12


class _FixedMidTrial:
    """Stand-in trial that returns the middle choice without registering a param."""

    @staticmethod
    def suggest_categorical(name, choices):
        return list(choices)[len(choices) // 2]


def _sample_params(
    trial,
    model_name: str,
    *,
    curriculum_only: bool = False,
    base_cfg_path: Path | None = None,
    proxy_epochs: int | None = None,
) -> dict[str, Any]:
    model_name = str(model_name).lower().strip()
    if proxy_epochs is not None and int(proxy_epochs) < _MIN_EPOCHS_FOR_CURRICULUM:
        # Curriculum ramps start at epoch 4-20; a short proxy never reaches them, so
        # sampling ~14 inert params only adds noise dimensions to TPE. Pin to mid-choices.
        curriculum = _sample_curriculum(_FixedMidTrial(), model_name)
    else:
        curriculum = _sample_curriculum(trial, model_name)
    if curriculum_only:
        path = base_cfg_path or Path("config/run.yaml")
        if not path.exists():
            raise FileNotFoundError(f"Base config {path} not found (required for --curriculum-only).")
        return {**_read_arch_params_from_config(path), **curriculum}

    # Use ceiling (max) seq_len for batch-size safety, not cur_seq_target (60 vs 120 -> 5% underestimate -> 512 OOM after 2D reshape)
    representative_seq = _seq_len_ceiling(model_name)

    if model_name == "tft":
        d_model = trial.suggest_categorical("d_model", [64, 128, 256])
        batch_choices = _hardware_safe_batch_choices(model_name, d_model, representative_seq)
        arch = {
            "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            "d_model": d_model,
            "hidden_size": d_model,
            "nhead": trial.suggest_categorical("nhead", [4, 8]),
            "num_layers": trial.suggest_categorical("num_layers", [2, 3, 4]),
            "dropout": trial.suggest_float("dropout", 0.05, 0.35),
            "batch_size": trial.suggest_categorical("batch_size", batch_choices),
            "mt_direction_weight_floor": trial.suggest_float("mt_direction_weight_floor", 0.20, 0.50),
            "mt_focal_gamma": trial.suggest_float("mt_focal_gamma", 1.0, 2.5),
            "mt_class_balance_weight": trial.suggest_float("mt_class_balance_weight", 0.05, 0.30),
        }
    elif model_name == "haelt":
        d_model = trial.suggest_categorical("d_model", [128, 256, 512])
        batch_choices = _hardware_safe_batch_choices(model_name, d_model, representative_seq)
        arch = {
            "lr": trial.suggest_float("lr", 5e-5, 2e-3, log=True),
            "d_model": d_model,
            "hidden_size": d_model,
            "nhead": trial.suggest_categorical("nhead", [4, 8]),
            "num_layers": trial.suggest_int("num_layers", 2, 6),
            "dropout": trial.suggest_float("dropout", 0.1, 0.45),
            "batch_size": trial.suggest_categorical("batch_size", batch_choices),
            "mt_direction_weight_floor": trial.suggest_float("mt_direction_weight_floor", 0.20, 0.50),
            "mt_focal_gamma": trial.suggest_float("mt_focal_gamma", 1.0, 2.5),
            "mt_class_balance_weight": trial.suggest_float("mt_class_balance_weight", 0.05, 0.30),
        }
    elif model_name == "transformer":
        d_model = trial.suggest_categorical("d_model", [128, 256, 512])
        batch_choices = _hardware_safe_batch_choices(model_name, d_model, representative_seq)
        arch = {
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "d_model": d_model,
            "hidden_size": d_model,
            "nhead": trial.suggest_categorical("nhead", [4, 8, 16]),
            "num_layers": trial.suggest_int("num_layers", 2, 8),
            "dropout": trial.suggest_float("dropout", 0.1, 0.4),
            "batch_size": trial.suggest_categorical("batch_size", batch_choices),
            "mt_direction_weight_floor": trial.suggest_float("mt_direction_weight_floor", 0.20, 0.50),
            "mt_focal_gamma": trial.suggest_float("mt_focal_gamma", 1.0, 2.5),
            "mt_class_balance_weight": trial.suggest_float("mt_class_balance_weight", 0.05, 0.30),
        }
    else:
        raise ValueError(f"Unsupported model for Optuna tuning: {model_name}")

    return {**arch, **curriculum}


def _build_trial_config(
    base_cfg_path: Path,
    args,
    params: dict[str, Any],
    *,
    epochs: int,
    curriculum_only: bool = False,
    disable_pretrain: bool = False,
) -> tuple[dict[str, Any], Path]:
    with base_cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg.setdefault("model", {})
    cfg.setdefault("training", {})
    cfg.setdefault("tracking", {})
    cfg.setdefault("multitask", {})
    cfg.setdefault("ensemble", {})
    cfg.setdefault("rl", {})
    cfg.setdefault("curriculum", {})
    cfg.setdefault("pretrain", {})

    cfg["model"]["name"] = args.model
    cfg["model"]["all_models"] = False
    if not curriculum_only:
        cfg["model"]["hidden_size"] = int(params["hidden_size"])
        cfg["model"]["d_model"] = int(params["d_model"])
        cfg["model"]["nhead"] = int(params["nhead"])
        cfg["model"]["num_layers"] = int(params["num_layers"])
        cfg["model"]["dropout"] = float(f"{params['dropout']:.3f}")

        cfg["training"]["lr"] = float(f"{params['lr']:.2e}")
        cfg["training"]["batch_size"] = int(params["batch_size"])

        cfg["multitask"]["direction_weight_floor"] = float(f"{params['mt_direction_weight_floor']:.3f}")
        cfg["multitask"]["focal_gamma"] = float(f"{params['mt_focal_gamma']:.3f}")
        cfg["multitask"]["class_balance_weight"] = float(f"{params['mt_class_balance_weight']:.3f}")

    cfg["training"]["epochs"] = int(epochs)
    cfg["training"]["resume"] = False
    # Pin the dataset-cache seq_len to a constant ceiling (not the per-trial
    # cur_seq_target) so every trial in the study reuses the same cache
    # instead of triggering a full rebuild from raw ticks. See
    # _seq_len_ceiling() docstring for why this must stay fixed.
    cfg["training"]["seq_len"] = _seq_len_ceiling(args.model)

    cfg["tracking"]["no_wandb"] = True
    cfg["ensemble"]["enabled"] = False
    cfg["rl"]["enabled"] = False

    if disable_pretrain:
        # Proxy/confirmation trials score curriculum + architecture quickly;
        # running a full unsupervised pretrain (and the no-pretrain baseline
        # ablation proof) on every trial multiplies cost ~3x for no search
        # benefit. Pretrain still runs normally in the final launched/exported
        cfg["pretrain"]["enabled"] = False
        cfg["pretrain"]["ablation"] = "false"

    # P3: stamp so the post-run auto-tuner knows not to override Optuna's curriculum choices
    cfg.setdefault("optuna", {})
    cfg["optuna"]["applied"] = True
    cfg["optuna"]["model"] = str(args.model)
    cfg["optuna"]["metric"] = str(getattr(args, "metric", DEFAULT_METRIC))
    cfg["optuna"]["curriculum_only"] = bool(getattr(args, "curriculum_only", False))
    cfg["optuna"]["hpo_scheduler"] = str(getattr(args, "hpo_scheduler", "tpe") or "tpe")
    cfg["optuna"]["auto_load"] = True
    cfg["optuna"]["auto_launch"] = bool(getattr(args, "auto", False) or getattr(args, "launch_training", False))

    # -----------------------------------------------------------------------
    # Curriculum: Optuna designs the schedule; Auto-Tuner guards it at runtime
    # -----------------------------------------------------------------------
    # Build a concrete seq_schedule from Optuna's curriculum shape params
    seq_schedule = _build_seq_schedule(
        cur_seq_start=int(params["cur_seq_start"]),
        cur_seq_ramp_epoch=int(params["cur_seq_ramp_epoch"]),
        cur_seq_target=int(params["cur_seq_target"]),
        total_epochs=int(epochs),
    )
    cfg["curriculum"]["seq_schedule"] = seq_schedule

    # Override adaptation sensitivity with Optuna's chosen thresholds
    cfg["curriculum"].setdefault("adaptation", {})
    cfg["curriculum"]["adaptation"].update(
        {
            "collapse_drop": float(params["cur_collapse_drop"]),
            "collapse_min_peak": float(params["cur_collapse_min_peak"]),
            "advance_lr_mult": float(params["cur_advance_lr_mult"]),
            "collapse_lr_mult": float(params["cur_collapse_lr_mult"]),
            "stable_window": int(params["cur_stable_window"]),
            "collapse_reversal_threshold": float(params.get("cur_reversal_threshold", -0.10)),
            "recovery_window": int(params.get("cur_recovery_window", 4)),
            "min_epochs_per_stage": int(params.get("cur_min_epochs_per_stage", 3)),
        }
    )

    # P2: build difficulty schedule from Optuna's chosen params
    if "cur_diff_ramp_epoch" in params and "cur_diff_final_stage" in params:
        difficulty_schedule = _build_difficulty_schedule(
            cur_diff_ramp_epoch=int(params["cur_diff_ramp_epoch"]),
            cur_diff_final_stage=int(params["cur_diff_final_stage"]),
        )
        cfg["curriculum"]["difficulty_schedule"] = difficulty_schedule
    # -----------------------------------------------------------------------

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    OPTUNA_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    # Trial configs are ephemeral -> ARTIFACT_DIR, not OPTUNA_CONFIG_DIR (which is for best only)
    trial_cfg_path = (
        ARTIFACT_DIR / f"run_optuna_{_safe_slug(args.model)}_{getattr(args, '_trial_suffix', 'trial')}.yaml"
    )
    return cfg, trial_cfg_path


def _run_trial_process(
    args, params: dict[str, Any], *, trial_number: int, epochs: int, folds: int, phase: str = "proxy"
) -> tuple[Path, dict[str, Any]]:
    base_cfg_path = Path("config/run.yaml")
    if not base_cfg_path.exists():
        raise FileNotFoundError(f"Base config {base_cfg_path} not found.")

    args._trial_suffix = f"{phase}_{trial_number}"
    cfg, trial_cfg_path = _build_trial_config(
        base_cfg_path,
        args,
        params,
        epochs=epochs,
        curriculum_only=bool(getattr(args, "curriculum_only", False)),
        disable_pretrain=True,
    )
    with trial_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)

    checkpoint_dir = Path(f"checkpoints/optuna_{_safe_slug(args.model)}_{phase}_{trial_number}")
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    run_name = f"optuna_{args.model}_{phase}_{trial_number}"
    cmd = [
        sys.executable,
        "-m",
        "training.train_gpu",
        "--config",
        str(trial_cfg_path),
        "--model",
        args.model,
        "--run-name",
        run_name,
        "--no-all-models",
        "--walk-forward-folds",
        str(int(folds)),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--no-wandb",
        "--no-resume",
        "--no-training-memory",
        "--no-auto-tune",
    ]

    print(f"[Optuna] Executing: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    live_epoch = 0
    latest_loss = None
    latest_sharpe = None
    latest_cost = None
    honest_seen = False
    live_lines: list[str] = []
    try:
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip("\n")
            live_lines.append(line)
            print(line)

            m_epoch = re.search(r"\[Epoch\s+(\d+)/(\d+)\]", line)
            if m_epoch:
                live_epoch = max(live_epoch, int(m_epoch.group(1)))

            m_loss = re.search(r"val(?:_loss| loss)[=:]\s*([-+]?\d+(?:\.\d+)?)", line, re.IGNORECASE)
            if m_loss:
                latest_loss = float(m_loss.group(1))

            m_sh = re.search(r"(?:val[_ ]sharpe|sharpe_proxy)[=:]\s*([-+]?\d+(?:\.\d+)?)", line, re.IGNORECASE)
            if m_sh:
                latest_sharpe = float(m_sh.group(1))
            # cost_sharpe objective must prune on the same metric it is scored on:
            # the honest net-PnL Sharpe (falls back to the label-based cost_sharpe line).
            if "[Val][honest] enabled" in line:
                honest_seen = True  # printed before epoch 1: never report label-based values
            m_hon = re.search(r"\[Val\]\[honest\] net_sharpe=([-+]?\d+(?:\.\d+)?)", line)
            if m_hon:
                honest_seen = True
                latest_cost = float(m_hon.group(1))
            elif not honest_seen:
                m_cs = re.search(r"\bcost_sharpe=([-+]?\d+(?:\.\d+)?)", line)
                if m_cs:
                    latest_cost = float(m_cs.group(1))

            report_value = None
            if args.metric == "val_loss" and latest_loss is not None:
                report_value = latest_loss
            elif args.metric == "cost_sharpe" and latest_cost is not None:
                report_value = latest_cost
            elif args.metric == "val_sharpe" and latest_sharpe is not None:
                report_value = latest_sharpe
            if report_value is not None and live_epoch > 0:
                yield_payload = {
                    "epoch": live_epoch,
                    "latest_val_loss": latest_loss,
                    "latest_val_sharpe": latest_sharpe,
                }
                yield_payload["process"] = process
                yield_payload["checkpoint_dir"] = checkpoint_dir
                yield_payload["trial_cfg_path"] = trial_cfg_path
                yield_payload["live_lines"] = live_lines
                yield_payload["report_value"] = report_value
                yield yield_payload

        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)
    finally:
        if process.poll() is None:
            process.kill()


def _evaluate_trial_artifacts(checkpoint_dir: Path, model_name: str, folds: int, metric: str) -> dict[str, Any]:
    ckpt_path = _trial_checkpoint_path(checkpoint_dir, model_name, folds)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    history = ckpt.get("history", {})
    summary = _read_json(_trial_summary_path(checkpoint_dir, model_name))

    # Merge curriculum_events from control report or resume checkpoint state
    ctrl = _read_json(_trial_control_report_path(checkpoint_dir, model_name))
    if ctrl.get("curriculum_events"):
        history["curriculum_events"] = ctrl["curriculum_events"]
    if ctrl.get("curriculum_stalls") is not None and not history.get("curriculum_stalls"):
        history["curriculum_stalls"] = [int(ctrl["curriculum_stalls"])]
    curr_state = ckpt.get("curriculum_state", {})
    if curr_state.get("curriculum_events") and not history.get("curriculum_events"):
        history["curriculum_events"] = curr_state["curriculum_events"]
    if curr_state.get("curriculum_stalls") is not None and not history.get("curriculum_stalls"):
        history["curriculum_stalls"] = [int(curr_state["curriculum_stalls"])]

    score, diagnostics = _metric_score(metric, summary, history)
    return {
        "checkpoint_path": str(ckpt_path),
        "objective_score": float(score),
        "summary": summary,
        "history": history,
        "diagnostics": diagnostics,
    }


def _write_trial_report(study_name: str, trial_number: int, payload: dict[str, Any]) -> None:
    _trial_report_path(study_name, trial_number).write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )


def _sorted_trial_rows(study: optuna.Study) -> list[dict[str, Any]]:
    rows = []
    for t in study.trials:
        rows.append(
            {
                "trial": int(t.number),
                "state": str(t.state),
                "value": t.value,
                "params": dict(t.params),
                "user_attrs": dict(t.user_attrs),
            }
        )

    # Sort according to study direction (val_loss -> asc, val_sharpe -> desc)
    is_maximize = str(getattr(study.direction, "name", "")).upper() == "MAXIMIZE"

    def _sort_key(row):
        value = row.get("value")
        if value is None:
            return float("inf") if not is_maximize else float("-inf")
        v = float(value)
        return -v if is_maximize else v

    return sorted(rows, key=_sort_key)


def _write_ranked_report(study: optuna.Study, study_name: str, args) -> None:
    rows = _sorted_trial_rows(study)
    report = {
        "study_name": study_name,
        "model": args.model,
        "mode": args.mode,
        "metric": args.metric,
        "metric_direction": _metric_direction(args.metric),
        "trials_requested": args.trials,
        "rows": rows,
        "best_trial": study.best_trial.number if study.best_trial else None,
    }
    _ranked_report_path(study_name).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def _production_epochs_from_config(base_cfg_path: Path) -> int:
    """Return training.epochs from the active run config (full production budget)."""
    with base_cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return int((cfg.get("training") or {}).get("epochs", 40))


def _launch_training_run(args) -> int:
    """Run full production training with the exported best config.

    It used to launch from config/run.yaml, which the tuner never changes, so the
    study's result was silently ignored.
    """
    cfg_path = Path(getattr(args, "_exported_config", "") or "")
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"Cannot launch training: exported config {cfg_path} not found. Did _export_best_config run?"
        )

    run_name = f"optuna_launch_{_safe_slug(args.model)}_{_safe_slug(args.study_name)}"
    cmd = [
        sys.executable,
        "-m",
        "training.train_gpu",
        "--config",
        str(cfg_path),
        "--model",
        args.model,
        "--run-name",
        run_name,
        "--no-all-models",
        "--no-auto-tune",
    ]

    print("\n[Optuna] Launching full training run...")
    print(f"[Optuna] Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    print("[Optuna] Full training finished successfully (exit 0).")
    return int(result.returncode)


def _export_best_config(args, study: optuna.Study, confirmed_trial: int | None = None) -> Path:
    # Prefer the full-budget confirmation winner over the proxy winner.
    chosen = study.trials[confirmed_trial] if confirmed_trial is not None else study.best_trial
    best_params = chosen.params
    base_cfg_path = Path("config/run.yaml")
    curriculum_only = bool(getattr(args, "curriculum_only", False))
    params = _assemble_trial_params(
        best_params, curriculum_only=curriculum_only, base_cfg_path=base_cfg_path, model_name=args.model
    )
    args._trial_suffix = f"best_{_safe_slug(args.metric)}"
    # Keep production epoch budget from run.yaml - not the short confirm-trial count.
    production_epochs = _production_epochs_from_config(base_cfg_path)
    cfg, export_path = _build_trial_config(
        base_cfg_path,
        args,
        params,
        epochs=production_epochs,
        curriculum_only=curriculum_only,
    )
    with export_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)

    # Not applied to config/run.yaml automatically; --launch-training trains
    # from export_path directly.
    scope = "curriculum" if curriculum_only else "full"
    print(f"[Optuna] [OK] Best {scope} config archived -> {export_path}")
    print(f"[Optuna]   To apply: copy {export_path} {ACTIVE_RUN_CONFIG}")
    # ------------------------------------------------------------------

    best_summary = {
        "model": args.model,
        "metric": args.metric,
        "mode": args.mode,
        "curriculum_only": curriculum_only,
        "study_name": args.study_name,
        "best_trial_number": int(chosen.number),
        "best_trial_value": chosen.value,
        "selected_by": "confirmation" if confirmed_trial is not None else "proxy",
        "best_params": best_params,
        "exported_config": str(export_path),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    _best_summary_path(args.model, args.metric).write_text(
        json.dumps(best_summary, indent=2, default=str),
        encoding="utf-8",
    )
    args._exported_config = export_path
    return export_path


def _confirm_top_trials(args, study: optuna.Study) -> int | None:
    """Retrain the top-K proxy trials at full budget; return the best confirmed trial number."""
    if args.confirm_top_k <= 0:
        return None
    # study.best_trials is only the Pareto front (1 trial for single-objective);
    # rank all completed trials to confirm the real top-K.
    _done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    _done.sort(key=lambda t: t.value, reverse=_metric_direction(args.metric) == "maximize")
    top_trials = _done[: args.confirm_top_k]
    if not top_trials:
        return None

    base_cfg_path = Path("config/run.yaml")
    curriculum_only = bool(getattr(args, "curriculum_only", False))
    confirm_rows = []
    for t in top_trials:
        params = _assemble_trial_params(
            t.params,
            curriculum_only=curriculum_only,
            base_cfg_path=base_cfg_path,
            model_name=args.model,
        )
        trial_cfg_path = None
        checkpoint_dir = None
        try:
            for payload in _run_trial_process(
                args,
                params,
                trial_number=int(t.number),
                epochs=int(args.full_confirm_epochs),
                folds=int(args.full_confirm_folds),
                phase="confirm",
            ):
                trial_cfg_path = payload.get("trial_cfg_path")
                checkpoint_dir = payload.get("checkpoint_dir")
            # checkpoint_dir is a Path from the generator; fall back only if loop never ran
            if checkpoint_dir is None:
                checkpoint_dir = Path(f"checkpoints/optuna_{_safe_slug(args.model)}_confirm_{int(t.number)}")
            else:
                checkpoint_dir = Path(checkpoint_dir)
            confirm = _evaluate_trial_artifacts(checkpoint_dir, args.model, int(args.full_confirm_folds), args.metric)
            confirm_rows.append(
                {
                    "trial": int(t.number),
                    "objective_score": confirm["objective_score"],
                    "diagnostics": confirm["diagnostics"],
                    "checkpoint_path": confirm["checkpoint_path"],
                    "params": params,
                }
            )
        except (subprocess.CalledProcessError, optuna.exceptions.TrialPruned) as _ce:
            print(f"[Optuna] Confirmation trial {t.number} failed/pruned ({type(_ce).__name__}); skipping.")
            continue
        finally:
            if trial_cfg_path and Path(trial_cfg_path).exists():
                Path(trial_cfg_path).unlink()

    path = ARTIFACT_DIR / f"{_safe_slug(args.study_name)}_confirm_report.json"
    path.write_text(json.dumps({"rows": confirm_rows}, indent=2, default=str), encoding="utf-8")
    scored = [r for r in confirm_rows if math.isfinite(float(r["objective_score"]))]
    if not scored:
        return None
    pick = (max if _metric_direction(args.metric) == "maximize" else min)(scored, key=lambda r: r["objective_score"])
    print(f"[Optuna] Confirmation winner: trial {pick['trial']} (score {pick['objective_score']:.4f})")
    return int(pick["trial"])


def objective(trial, args):
    base_cfg_path = Path("config/run.yaml")
    curriculum_only = bool(getattr(args, "curriculum_only", False))
    params = _sample_params(
        trial,
        args.model,
        curriculum_only=curriculum_only,
        base_cfg_path=base_cfg_path,
        proxy_epochs=int(args.epochs),
    )
    searched = {k: v for k, v in params.items() if k in _CURRICULUM_PARAMS} if curriculum_only else params
    print(f"\n[Optuna] Starting Trial {trial.number} for {args.model} with: {searched}")

    checkpoint_dir = None
    live_lines: list[str] = []
    try:
        trial_gen = _run_trial_process(
            args,
            params,
            trial_number=int(trial.number),
            epochs=int(args.epochs),
            folds=int(args.folds),
            phase="proxy",
        )
        for payload in trial_gen:
            live_lines = payload["live_lines"]
            payload["trial_cfg_path"]
            checkpoint_dir = payload["checkpoint_dir"]
            trial.report(float(payload["report_value"]), step=int(payload["epoch"]))
            if trial.should_prune():
                proc = payload["process"]
                if proc.poll() is None:
                    proc.kill()
                raise optuna.exceptions.TrialPruned()

        # Use checkpoint_dir from the live payload if the generator ran; fall back
        # to the expected path only when the subprocess exited before reporting
        # any epoch (e.g. the run was completed without intermediate prune checks).
        if checkpoint_dir is None:
            checkpoint_dir = Path(f"checkpoints/optuna_{_safe_slug(args.model)}_proxy_{int(trial.number)}")
        else:
            checkpoint_dir = Path(checkpoint_dir)
        result = _evaluate_trial_artifacts(checkpoint_dir, args.model, int(args.folds), args.metric)
        trial.set_user_attr("checkpoint_path", result["checkpoint_path"])
        trial.set_user_attr("diagnostics", result["diagnostics"])
        trial.set_user_attr("train_summary", result["summary"])
        trial.set_user_attr("mode", args.mode)
        trial.set_user_attr("metric_direction", _metric_direction(args.metric))
        trial.set_user_attr("stdout_tail", live_lines[-40:])
        _write_trial_report(
            args.study_name,
            int(trial.number),
            {
                "trial": int(trial.number),
                "params": params,
                "objective_metric": args.metric,
                "objective_score": result["objective_score"],
                "checkpoint_path": result["checkpoint_path"],
                "diagnostics": result["diagnostics"],
                "train_summary": result["summary"],
            },
        )
        return float(result["objective_score"])
    except subprocess.CalledProcessError as exc:
        print(f"[Optuna] Trial {trial.number} failed with exit code {exc.returncode}")
        trial.set_user_attr("stdout_tail", live_lines[-40:])
        raise
    except Exception as exc:
        # Fragility fix: store tail for any failure (e.g. missing checkpoint/history) before pruning
        print(f"[Optuna] Trial {trial.number} failed ({type(exc).__name__}: {exc})")
        trial.set_user_attr("stdout_tail", live_lines[-40:])
        trial.set_user_attr("diagnostics", {"error": str(exc), "missing_metric": True})
        raise
    finally:
        # Trial configs now live in ARTIFACT_DIR (not OPTUNA_CONFIG_DIR) to avoid polluting best dir
        for base in (ARTIFACT_DIR, OPTUNA_CONFIG_DIR):
            cleanup_cfg = base / f"run_optuna_{_safe_slug(args.model)}_proxy_{int(trial.number)}.yaml"
            if cleanup_cfg.exists():
                try:
                    cleanup_cfg.unlink()
                except Exception:
                    pass


def main():
    args = parse_args()
    if args.auto:
        args.launch_training = True
    elif not args.launch_training:
        from training.optuna_config import read_run_yaml_optuna_section

        if read_run_yaml_optuna_section().get("auto_launch"):
            args.launch_training = True
            print("[Optuna] auto_launch=true in config/run.yaml - full training will start after study.")
    defaults = _mode_defaults(args.mode)
    if int(args.epochs) <= 0:
        args.epochs = int(defaults["epochs"])
    if int(args.folds) <= 0:
        args.folds = int(defaults["folds"])
    if int(args.confirm_top_k) < 0:
        args.confirm_top_k = int(defaults["confirm_top_k"])
    if int(args.full_confirm_folds) <= 0:
        args.full_confirm_folds = int(defaults["full_confirm_folds"])
    if int(args.full_confirm_epochs) <= 0:
        args.full_confirm_epochs = int(defaults["full_confirm_epochs"])
    if not args.study_name:
        suffix = "_curriculum" if args.curriculum_only else ""
        cfg_bytes = Path("config/run.yaml").read_bytes() if Path("config/run.yaml").exists() else b""
        cfg_hash = hashlib.sha256(cfg_bytes).hexdigest()[:10]
        args.study_name = f"optuna_{args.model}_{args.mode}_{args.metric}{suffix}_{cfg_hash}"

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    study_db_path = ARTIFACT_DIR / f"{_safe_slug(args.study_name)}.db"
    storage = f"sqlite:///{study_db_path.as_posix()}"

    from training.hpo import build_optuna_search

    hpo_scheduler = str(getattr(args, "hpo_scheduler", "tpe") or "tpe").lower()
    _warm = _direction_warmup_epochs()
    if int(args.epochs) < _warm + 2:
        raise SystemExit(
            f"[Optuna] --epochs {args.epochs} leaves < 2 epochs after the {_warm} direction-warmup epochs; "
            f"trials would only measure warmup. Use --epochs >= {_warm + 2}."
        )
    sampler, pruner = build_optuna_search(
        hpo_scheduler,
        seed=int(args.seed),
        min_resource=_warm + 2,
        max_resource=max(int(args.epochs), _warm + 2),
    )
    print(f"[Optuna] HPO scheduler={hpo_scheduler} sampler={type(sampler).__name__} pruner={type(pruner).__name__}")

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction=_metric_direction(args.metric),
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    _fail_stale_running_trials(study)
    if int(args.epochs) < _MIN_EPOCHS_FOR_CURRICULUM:
        print(
            f"[Optuna] Proxy epochs={args.epochs} < {_MIN_EPOCHS_FOR_CURRICULUM}: curriculum params pinned "
            "(ramps never trigger in short proxies)."
            + (" WARNING: --curriculum-only now searches nothing; raise --epochs." if args.curriculum_only else "")
        )
    search_scope = "curriculum-only" if args.curriculum_only else "architecture + curriculum"
    print(f"[Optuna] Commencing {args.trials} trials for {args.model}")
    print(
        f"[Optuna] Scope={search_scope} mode={args.mode} metric={args.metric} folds={args.folds} epochs={args.epochs}"
    )
    if args.curriculum_only:
        print("[Optuna] Architecture/training hyperparams fixed from config/run.yaml")
    print(f"[Optuna] Study stored in {study_db_path}")
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.trials,
        catch=(Exception,),  # any single-trial failure is recorded, not study-fatal
    )

    _finite = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None and math.isfinite(t.value)
    ]
    if not _finite:
        raise RuntimeError(
            "Optuna produced no completed trial with a finite score; refusing to export a configuration"
        )

    confirmed = _confirm_top_trials(args, study)
    _write_ranked_report(study, args.study_name, args)
    _export_best_config(args, study, confirmed_trial=confirmed)

    print("\n=== OPTUNA STUDY FINISHED ===")
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Objective Score: {study.best_value}")
    print("Best Params:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    print(f"Best config exported -> {_best_config_path(args.model, args.metric)}")
    print(f"Ranked report -> {_ranked_report_path(args.study_name)}")

    if args.launch_training:
        _launch_training_run(args)
    else:
        print(
            "\n[Optuna] Next step: python -m training.train_gpu --config config/run.yaml "
            f"--model {args.model} --no-all-models"
        )
        print("[Optuna] Or re-run with --auto / --launch-training to chain full training.")


if __name__ == "__main__":
    main()
