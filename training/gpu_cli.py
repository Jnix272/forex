"""CLI / argparse / YAML / run-dir helpers for GPU training.

This module is now a thin shim. All logic lives in training/cli/.
Existing callers (training/train_gpu.py etc.) continue to work unchanged.

See docs/CONTINUE.md.
"""

from __future__ import annotations

# Re-export everything from the sub-package so callers need no changes.
from training.cli import (  # noqa: F401
    _YAML_MAP,
    _apply_auto_run_dir,
    _apply_model_profile,
    _apply_training_profile,
    _apply_yaml_config,
    _apply_yaml_risk_to_live_risk,
    _baseline_ablation_completion_status,
    _build_auto_run_name,
    _collect_cli_profile_overrides,
    _latest_resumable_fold,
    _load_cv_fold_entry,
    _load_walk_forward_resume_history,
    _member_training_args,
    _model_build_args,
    _model_completion_status,
    _normalize_architecture_profile,
    _resolve_seq_len,
    _set_global_seed,
    _slug_part,
    _supervised_resume_status,
    _sync_runtime_config,
    apply_hardware_profile,
    parse_args,
)

__all__ = [
    "parse_args",
    "apply_hardware_profile",
    "_YAML_MAP",
    "_apply_yaml_config",
    "_apply_yaml_risk_to_live_risk",
    "_sync_runtime_config",
    "_resolve_seq_len",
    "_set_global_seed",
    "_slug_part",
    "_build_auto_run_name",
    "_apply_auto_run_dir",
    "_collect_cli_profile_overrides",
    "_normalize_architecture_profile",
    "_apply_model_profile",
    "_apply_training_profile",
    "_member_training_args",
    "_model_build_args",
    "_model_completion_status",
    "_baseline_ablation_completion_status",
    "_supervised_resume_status",
    "_latest_resumable_fold",
    "_load_cv_fold_entry",
    "_load_walk_forward_resume_history",
]
