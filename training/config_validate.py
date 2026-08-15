"""Preflight audit for train_gpu runs — no torch/dataset dependencies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml as _yaml
    _YAML = True
except ImportError:
    _YAML = False

from config.models import SUPPORTED_SUPERVISED
from config.settings import CURRICULUM as SETTINGS_CURRICULUM
from config.settings import LIVE_RISK, SIZING

# Keep aligned with training.train_gpu loss choices + common eval aliases.
SUPPORTED_LOSSES = frozenset({
    "cross_entropy",
    "sharpe_huber",
    "huber",
    "mse",
    "focal",
    "directional_huber",
    "asymmetric",
    "rmse",
    "multi_task",
    "asymmetric_directional",
})


def _get_pairs(args) -> list[str]:
    raw = getattr(args, "pairs", None)
    if not raw:
        return [str(getattr(args, "pair", "EURUSD")).upper()]
    if isinstance(raw, list):
        return [p.strip().upper() for p in raw if p and str(p).strip()]
    return [p.strip().upper() for p in str(raw).split(",") if p.strip()]


def _effective_max_seq_len(args) -> int:
    seqs = [int(getattr(args, "seq_len", 60) or 60)]
    cur = getattr(args, "curriculum", None)
    if cur is None or cur is False or cur == "none" or cur == "":
        return max(seqs)
    if not isinstance(cur, dict):
        cur = SETTINGS_CURRICULUM
    for entry in (cur.get("seq_schedule") or []):
        if isinstance(entry, dict) and entry.get("seq_len") is not None:
            seqs.append(int(entry["seq_len"]))
    return max(seqs)


def _parse_pretrain_ablation_models(value) -> set[str]:
    default = {"tft", "transformer", "haelt"}
    if value is None:
        return default
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return default
        if raw.lower() in {"none", "false", "off", "disabled"}:
            return set()
        return {part.strip().lower() for part in raw.split(",") if part.strip()}
    if isinstance(value, (list, tuple, set)):
        return {str(part).strip().lower() for part in value if str(part).strip()}
    return default


def _curriculum_max_epoch(curriculum: dict | None, key: str) -> int:
    if not isinstance(curriculum, dict):
        return 0
    schedule = curriculum.get(key) or []
    epochs = [
        int(e.get("epoch_start", 0))
        for e in schedule
        if isinstance(e, dict) and e.get("epoch_start") is not None
    ]
    return max(epochs) if epochs else 0


def _model_checkpoint_dir(base_ckpt: Path, model_name: str) -> Path:
    return base_ckpt / model_name


def _model_looks_complete(ckpt_dir: Path, model_name: str) -> tuple[bool, str]:
    model = model_name.lower().strip()
    best = [
        ckpt_dir / f"{model}_best.pt",
        ckpt_dir / model / f"{model}_best.pt",
    ]
    if any(p.is_file() for p in best):
        if (ckpt_dir / "train_summary.json").is_file() or (ckpt_dir / model / "train_summary.json").is_file():
            return True, "train_summary + best checkpoint"
        return True, "best checkpoint present"
    return False, "no best checkpoint"


def resolve_models_to_train(
    args,
    *,
    apply_resume_filter: bool = False,
) -> tuple[list[str], list[tuple[str, str]]]:
    if getattr(args, "all_models", False):
        requested = [
            m.strip().lower()
            for m in str(getattr(args, "models", "") or "").split(",")
            if m.strip()
        ]
        if requested:
            unknown = [m for m in requested if m not in SUPPORTED_SUPERVISED]
            if unknown:
                raise ValueError(
                    f"Unknown model(s) {unknown}; expected one of {sorted(SUPPORTED_SUPERVISED)}"
                )
            models = requested
        else:
            models = sorted(SUPPORTED_SUPERVISED)
    else:
        models = [str(getattr(args, "model", "haelt")).lower().strip()]

    skipped: list[tuple[str, str]] = []
    base_ckpt = Path(str(getattr(args, "checkpoint_dir", "checkpoints"))).expanduser()
    if apply_resume_filter and getattr(args, "all_models", False) and getattr(args, "resume", False):
        if not getattr(args, "retrain_completed_models", False):
            pending: list[str] = []
            for name in models:
                mdir = _model_checkpoint_dir(base_ckpt, name)
                done, reason = _model_looks_complete(mdir, name)
                if done:
                    skipped.append((name, reason))
                else:
                    pending.append(name)
            models = pending
    return models, skipped


def estimate_run_minutes(args, models: list[str]) -> dict[str, float]:
    """Rough wall-clock estimate (conservative, single-GPU, multi-pair WF)."""
    folds = int(getattr(args, "walk_forward_folds", 1) or 1) if getattr(args, "walk_forward_cv", False) else 1
    epochs = int(getattr(args, "epochs", 40) or 40)
    warmup = int(getattr(args, "lr_warmup_epochs", 3) or 3)
    patience = int(getattr(args, "patience", 10) or 10)
    avg_sup_epochs = float(min(epochs, max(warmup + patience + 3, int(epochs * 0.55))))

    min_pretrain_epoch = 4.5
    min_sup_epoch = 11.0

    abl_arg = str(getattr(args, "pretrain_ablation", "auto")).lower()
    abl_models = _parse_pretrain_ablation_models(getattr(args, "pretrain_ablation_models", ""))
    run_ablation = abl_arg == "true" or abl_arg == "auto"

    pretrain_min = 0.0
    supervised_min = 0.0
    ablation_min = 0.0

    for _name in models:
        pt_epochs = int(getattr(args, "pretrain_epochs", 0) or 0)
        if getattr(args, "pretrain", False) and pt_epochs > 0:
            pretrain_min += pt_epochs * min_pretrain_epoch
        supervised_min += folds * avg_sup_epochs * min_sup_epoch
        if run_ablation and (abl_arg == "true" or _name in abl_models):
            ablation_min += folds * avg_sup_epochs * min_sup_epoch

    post_min = 0.0
    if getattr(args, "all_models", False) and len(models) >= 2:
        post_min += 25.0
    if getattr(args, "train_ensemble", False):
        post_min += 50.0
    if getattr(args, "rl_train", False):
        rl_models = (
            models
            if (getattr(args, "all_models", False) and getattr(args, "rl_all_models", False))
            else [str(getattr(args, "model", models[0] if models else "haelt"))]
        )
        ep = int(getattr(args, "rl_episodes", 500) or 500)
        post_min += len(rl_models) * max(45.0, ep * 0.12)

    total = pretrain_min + supervised_min + ablation_min + post_min
    return {
        "pretrain_min": pretrain_min,
        "supervised_min": supervised_min,
        "ablation_min": ablation_min,
        "post_min": post_min,
        "total_min": total,
        "avg_sup_epochs": avg_sup_epochs,
        "folds": float(folds),
        "n_models": float(len(models)),
    }


def collect_config_issues(
    args,
    models: list[str],
    skipped: list[tuple[str, str]],
) -> tuple[list[str], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    info: list[str] = []

    pairs = _get_pairs(args)
    ckpt_root = Path(str(getattr(args, "checkpoint_dir", "checkpoints"))).expanduser()
    data_cache = Path(str(getattr(args, "data_cache", "data/processed"))).expanduser()
    curriculum = getattr(args, "curriculum", None)
    if not isinstance(curriculum, dict):
        curriculum = SETTINGS_CURRICULUM

    # Preserve explicit zeros (e.g. epochs=0) so range checks can reject them.
    def _int_field(name: str, default: int) -> int:
        raw = getattr(args, name, default)
        return default if raw is None else int(raw)

    def _float_field(name: str, default: float) -> float:
        raw = getattr(args, name, default)
        return default if raw is None else float(raw)

    epochs = _int_field("epochs", 40)
    batch_size = _int_field("batch_size", 256)
    lr = _float_field("lr", 1e-4)
    patience = _int_field("patience", 10)
    warmup = _int_field("lr_warmup_epochs", 3)
    max_seq = _effective_max_seq_len(args)
    base_seq = _int_field("seq_len", 60)
    cur_seq_epoch = _curriculum_max_epoch(curriculum, "seq_schedule")
    cur_diff_epoch = _curriculum_max_epoch(curriculum, "difficulty_schedule")

    # Range validation on critical hyperparameters
    if epochs <= 0:
        errors.append(f"training.epochs={epochs} must be > 0.")
    if batch_size <= 0:
        errors.append(f"training.batch_size={batch_size} must be > 0.")
    if lr <= 0 or lr >= 1.0:
        errors.append(f"training.lr={lr} out of sane range (0, 1.0).")
    if patience <= 0:
        errors.append(f"training.patience={patience} must be > 0.")
    if base_seq <= 0:
        errors.append(f"training.seq_len={base_seq} must be > 0.")

    # Warmup vs epochs ratio check
    if warmup >= epochs:
        errors.append(
            f"lr_warmup_epochs={warmup} >= epochs={epochs} — "
            "model never trains at target LR."
        )
    elif warmup > epochs * 0.5:
        warnings.append(
            f"lr_warmup_epochs={warmup} is >{50}% of epochs={epochs} — "
            "consider reducing warmup so the model has enough post-warmup epochs."
        )

    # Patience vs effective training epochs
    effective_train_epochs = epochs - warmup
    if patience >= effective_train_epochs and effective_train_epochs > 0:
        warnings.append(
            f"patience={patience} >= post-warmup epochs ({effective_train_epochs}) — "
            "early stopping will never trigger."
        )

    # Model name validation
    model_name = str(getattr(args, "model", "haelt")).lower().strip()
    if model_name not in SUPPORTED_SUPERVISED:
        errors.append(
            f"Unknown model '{model_name}'. Valid: {sorted(SUPPORTED_SUPERVISED)}"
        )

    # Framework Availability Check
    import importlib.util
    import os

    def _is_installed(pkg_name: str) -> bool:
        try:
            return importlib.util.find_spec(pkg_name) is not None
        except Exception:
            return False

    tf = str(getattr(args, "training_framework", "custom")).lower()
    if tf == "lightning" and not (_is_installed("pytorch_lightning") or _is_installed("lightning")):
        errors.append(f"Framework: training_framework='{tf}' requires 'pytorch-lightning', which is not installed.")
    elif tf == "composer" and not _is_installed("composer"):
        errors.append(f"Framework: training_framework='{tf}' requires 'mosaicml' (composer), which is not installed.")

    pf = str(getattr(args, "pretrain_framework", "custom")).lower()
    if pf == "lightly" and not _is_installed("lightly"):
        errors.append(f"Framework: pretrain_framework='{pf}' requires 'lightly', which is not installed.")
    elif pf == "solo" and not _is_installed("solo"):
        errors.append(f"Framework: pretrain_framework='{pf}' requires 'solo-learn', which is not installed.")

    rf = str(getattr(args, "rl_framework", "custom")).lower()
    if rf == "sb3" and not _is_installed("stable_baselines3"):
        errors.append(f"Framework: rl_framework='{rf}' requires 'stable-baselines3', which is not installed.")
    elif rf == "cleanrl":
        _local_cleanrl = os.path.exists(os.path.join(os.path.dirname(__file__), "..", "cleanrl"))
        if not (_is_installed("cleanrl") or _local_cleanrl):
            errors.append(f"Framework: rl_framework='{rf}' requires 'cleanrl' package or local cleanrl/ dir.")

    # Loss function validation
    loss_fn = str(getattr(args, "loss", "sharpe_huber")).lower().strip()
    if loss_fn not in SUPPORTED_LOSSES:
        warnings.append(
            f"training.loss='{loss_fn}' not in known set {sorted(SUPPORTED_LOSSES)} — "
            "will fall back to default if unrecognized."
        )

    # Risk parameter validation (SYS-004)
    kelly_f = float(LIVE_RISK.get("kelly_fraction", 0.25) or 0.25)
    if kelly_f <= 0 or kelly_f >= 1.0:
        errors.append(f"LIVE_RISK.kelly_fraction={kelly_f} must be in (0, 1.0).")
    elif kelly_f > 0.5:
        warnings.append(f"LIVE_RISK.kelly_fraction={kelly_f} > 0.5 is dangerously aggressive.")

    max_dd = float(LIVE_RISK.get("max_drawdown_pct", 20) or 20)
    if max_dd <= 0 or max_dd > 100:
        errors.append(f"LIVE_RISK.max_drawdown_pct={max_dd} must be in (0, 100].")

    risk_pct = float(SIZING.get("risk_pct", 1.0) or 1.0)
    if risk_pct <= 0 or risk_pct > 10:
        errors.append(f"SIZING.risk_pct={risk_pct} must be in (0, 10].")

    grad_clip = float(getattr(args, "grad_clip", 1.0) or 1.0)
    if grad_clip <= 0:
        errors.append(f"training.grad_clip={grad_clip} must be > 0.")

    info.append(
        f"Queue: {len(models)} model(s) — {', '.join(m.upper() for m in models) or '(none)'}"
    )
    if skipped:
        info.append(
            f"Resume skip: {len(skipped)} completed — "
            + ", ".join(f"{n} ({r})" for n, r in skipped[:4])
            + (" ..." if len(skipped) > 4 else "")
        )

    if not models and not skipped:
        errors.append("No models to train (empty queue).")
    elif not models and skipped:
        warnings.append(
            "All models appear complete; use --retrain-completed-models to force a full retrain."
        )

    if len(pairs) > 1 and int(getattr(args, "pair_embed_dim", 0) or 0) == 0:
        warnings.append(
            f"Multi-pair training ({len(pairs)} pairs) with pair_embed_dim=0 — "
            "consider pair_embed_dim: 16 in run.yaml."
        )

    eff_batch = int(getattr(args, "batch_size", 256)) * int(getattr(args, "grad_accum_steps", 1) or 1)
    if eff_batch > 2048 and len(pairs) >= 3:
        warnings.append(
            f"Effective batch {eff_batch} with {len(pairs)} pairs may OOM — "
            "lower batch_size or grad_accum_steps if you hit CUDA OOM."
        )

    if cur_seq_epoch >= epochs:
        warnings.append(
            f"Curriculum seq_schedule reaches epoch {cur_seq_epoch} but training.epochs={epochs} — "
            "final seq_len may never be reached before the epoch cap."
        )
    if cur_diff_epoch >= epochs:
        warnings.append(
            f"Curriculum difficulty_schedule reaches epoch {cur_diff_epoch} but epochs={epochs} — "
            "hardest bars may be excluded."
        )
    if max_seq > base_seq:
        info.append(
            f"Cache/manifest must cover max seq_len {max_seq} (training.seq_len={base_seq}, curriculum schedule)."
        )

    if getattr(args, "all_models", False):
        if not getattr(args, "dry_tune", False) and getattr(args, "auto_tune", True):
            warnings.append(
                "all_models + auto_tune with dry_tune=false can mutate config/run.yaml mid-run — "
                "set tracking.dry_tune: true (code also forces dry proposals when all_models=true)."
            )
        if getattr(args, "rl_train", False) and not getattr(args, "rl_all_models", False):
            warnings.append(
                f"all_models=true but rl.all_models=false — RL runs once for model.name="
                f"{getattr(args, 'model', '?')} only. Set rl.all_models: true to train RL per architecture."
            )

    if getattr(args, "train_ensemble", False) and len(models) < 2 and not getattr(args, "all_models", False):
        warnings.append("ensemble.enabled with a single model — meta-learner needs >=2 base checkpoints.")

    if getattr(args, "pretrain", False) and getattr(args, "model_profile", True) and len(models) > 1:
        info.append(
            f"Global pretrain.method={getattr(args, 'pretrain_method', 'byol')}; "
            "per-model methods come from config/models.py when model_profile=true."
        )

    if getattr(args, "teacher_model", None) and float(getattr(args, "distill_weight", 0) or 0) > 0:
        tname = str(args.teacher_model).lower().strip()
        tcands = [
            ckpt_root / tname / f"{tname}_best.pt",
            ckpt_root / f"{tname}_best.pt",
        ]
        if not any(p.is_file() for p in tcands):
            warnings.append(
                f"Distillation teacher '{tname}' checkpoint not found — train teacher first or disable --teacher-model."
            )

    cfg_path = Path(getattr(args, "config", "config/run.yaml") or "config/run.yaml")
    if cfg_path.is_file() and _YAML:
        try:
            with cfg_path.open("r", encoding="utf-8-sig") as fh:
                raw = _yaml.safe_load(fh) or {}
            teacher_ckpt = (raw.get("distillation") or {}).get("teacher_ckpt")
            if teacher_ckpt and not Path(str(teacher_ckpt)).expanduser().is_file():
                warnings.append(f"distillation.teacher_ckpt missing on disk: {teacher_ckpt}")
            if bool((raw.get("optuna") or {}).get("applied")):
                info.append("optuna.applied=true — auto-tuner records proposals but skips YAML mutations.")
            # Curriculum / FEATURE_MASK consistency (mismatch C/D)
            try:
                from config.curriculum_audit import (
                    audit_curriculum_feature_groups,
                    audit_required_market_columns,
                    audit_settings_yaml_curriculum_drift,
                )
                from config.feature_mask import FEATURE_MASK
                curr = raw.get("curriculum") if isinstance(raw.get("curriculum"), dict) else {}
                groups = curr.get("feature_groups") or {}
                audit = audit_curriculum_feature_groups(
                    schema=[k for k, v in FEATURE_MASK.items() if v],
                    feature_groups=groups,
                    feature_mask=FEATURE_MASK,
                )
                warnings.extend(audit.get("warnings") or [])
                drift = audit_settings_yaml_curriculum_drift(
                    SETTINGS_CURRICULUM,
                    curr,
                    yaml_path=str(cfg_path),
                )
                errors.extend(drift.get("errors") or [])
                warnings.extend(drift.get("warnings") or [])
                mkt = audit_required_market_columns(feature_mask=FEATURE_MASK)
                warnings.extend(mkt.get("warnings") or [])
                errors.extend(mkt.get("errors") or [])
                # settings.py ↔ YAML shared-key mismatches (section parts)
                try:
                    from config.config_mismatch_audit import (
                        audit_args_vs_yaml_mismatches,
                        audit_settings_yaml_section_mismatches,
                    )
                    sec = audit_settings_yaml_section_mismatches(
                        raw, yaml_path=str(cfg_path)
                    )
                    errors.extend(sec.get("errors") or [])
                    warnings.extend(sec.get("warnings") or [])
                    arg_m = audit_args_vs_yaml_mismatches(
                        args, raw, yaml_path=str(cfg_path)
                    )
                    errors.extend(arg_m.get("errors") or [])
                    warnings.extend(arg_m.get("warnings") or [])
                except Exception as mism_exc:
                    warnings.append(f"settings/YAML mismatch audit failed: {mism_exc}")
                # Strategy block must actually parse (mismatch B regression guard)
                strat = raw.get("strategy") if isinstance(raw.get("strategy"), dict) else {}
                if not strat:
                    warnings.append(
                        f"{cfg_path}: strategy: block missing or not a mapping — "
                        "profit_target_atr/stop_loss_atr/bar_freq will not load from YAML."
                    )
                else:
                    for key in ("profit_target_atr", "stop_loss_atr", "bar_freq", "mode"):
                        if key not in strat:
                            warnings.append(f"{cfg_path}: strategy.{key} missing")
            except Exception as audit_exc:
                warnings.append(f"Curriculum/FEATURE_MASK audit failed: {audit_exc}")
        except Exception as exc:
            # YAML parse failure is blocking — do not train on silent defaults.
            errors.append(f"Could not parse {cfg_path}: {exc}")

    if bool(getattr(args, "training_memory", True)) and not getattr(args, "all_models", False):
        warnings.append(
            "training_memory=true on a single-model run — global or per-model nudges may apply at startup."
        )
    elif bool(getattr(args, "training_memory", True)) and getattr(args, "all_models", False):
        info.append(
            "training_memory=true — all-models uses per-architecture memory only; "
            "epochs are not capped below curriculum milestones."
        )

    if getattr(args, "drift_gate", False):
        warnings.append(
            "drift_gate=true during training often false-alarms on long historical windows — "
            "keep drift_gate off for offline training."
        )

    try:
        ckpt_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        errors.append(f"checkpoint_dir not writable: {ckpt_root} ({exc})")

    if not data_cache.exists():
        info.append(f"data_cache not found yet ({data_cache}) — first run will build/download.")

    return errors, warnings, info


def _fmt_hours(minutes: float) -> str:
    if minutes < 90:
        return f"{minutes:.0f} min"
    return f"{minutes / 60:.1f} h ({minutes:.0f} min)"


def build_config_preflight_report(args: Any) -> dict[str, Any]:
    """Structured preflight report used by CLI and unit tests."""
    try:
        models, skipped = resolve_models_to_train(
            args, apply_resume_filter=bool(getattr(args, "resume", False))
        )
    except ValueError as exc:
        return {
            "ok": False,
            "errors": [str(exc)],
            "warnings": [],
            "info": [],
            "supported_losses": sorted(SUPPORTED_LOSSES),
            "models": [],
            "skipped": [],
        }

    errors, warnings, info = collect_config_issues(args, models, skipped)
    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "info": info,
        "supported_losses": sorted(SUPPORTED_LOSSES),
        "models": models,
        "skipped": list(skipped),
    }


def validate_run_config(args: Any, *, verbose: bool = True) -> int:
    """Print config audit + runtime estimate. Returns 0=ok, 1=blocking errors."""
    try:
        models, skipped = resolve_models_to_train(
            args, apply_resume_filter=bool(getattr(args, "resume", False))
        )
    except ValueError as exc:
        print(f"\n[ValidateConfig] ERROR: {exc}")
        return 1

    errors, warnings, info = collect_config_issues(args, models, skipped)
    est_models = models if models else sorted(SUPPORTED_SUPERVISED)
    est = estimate_run_minutes(args, est_models)

    lines: list[str] = [
        "",
        "=" * 62,
        "  Config preflight — validate-config",
        "=" * 62,
        f"  Config file : {getattr(args, 'config', 'config/run.yaml')}",
        f"  Checkpoints : {getattr(args, 'checkpoint_dir', '?')}",
        f"  Data cache  : {getattr(args, 'data_cache', '?')}",
        f"  Strategy    : {getattr(args, 'strategy_mode', '?')} | "
        f"{getattr(args, 'data_start', '?')} → {getattr(args, 'data_end', '?')}",
        f"  Pairs       : {', '.join(_get_pairs(args))}",
        f"  Walk-forward: {'ON' if getattr(args, 'walk_forward_cv', False) else 'OFF'} "
        f"({int(est['folds'])} folds)",
        f"  Epochs      : {getattr(args, 'epochs', '?')} "
        f"(est. ~{est['avg_sup_epochs']:.0f} effective w/ early-stop)",
        f"  Quick mode  : {getattr(args, 'quick_mode', False)}",
        "-" * 62,
        "  Runtime estimate (single GPU, conservative):",
        f"    Pretrain     : {_fmt_hours(est['pretrain_min'])}",
        f"    Supervised   : {_fmt_hours(est['supervised_min'])}",
        f"    Ablation     : {_fmt_hours(est['ablation_min'])}",
        f"    Post (ens/RL): {_fmt_hours(est['post_min'])}",
        f"    TOTAL        : {_fmt_hours(est['total_min'])}",
        "-" * 62,
    ]
    
    try:
        from training.gpu_cli import _apply_model_profile
        import copy
        has_printed_header = False
        for m in models:
            m_args = copy.copy(args)
            m_args = _apply_model_profile(m_args, m, enabled=True)
            if hasattr(m_args, "_training_features_report"):
                if not has_printed_header:
                    lines.append("  Training-Features Report (Precedence):")
                    has_printed_header = True
                report = m_args._training_features_report
                lines.append(f"    [{m.upper()}]")
                for k, v in report.items():
                    lines.append(f"      {k:20s}: {str(v['mode']):<6s} (source: {v['source']})")
        if has_printed_header:
            lines.append("-" * 62)
    except Exception as e:
        lines.append(f"  [Warning] Failed to generate features report: {e}")
        lines.append("-" * 62)

    if info:
        lines.append("  Info:")
        lines.extend(f"    • {msg}" for msg in info)
    if warnings:
        lines.append("  Warnings:")
        lines.extend(f"    ⚠ {msg}" for msg in warnings)
    if errors:
        lines.append("  Errors:")
        lines.extend(f"    ✗ {msg}" for msg in errors)

    # Data coverage check
    try:
        from training.data_coverage import validate_pair_coverage
        _pairs = _get_pairs(args)
        if len(_pairs) > 1:
            _src = getattr(args, "data_source", "dukascopy")
            _min_y = getattr(args, "min_pair_years", 2)
            _exp_y = getattr(args, "expected_pair_years", 18)
            _valid, _rep = validate_pair_coverage(_pairs, data_source=_src, min_years=_min_y, expected_years=_exp_y)
            _skipped = [p for p in _pairs if p not in _valid]
            if _skipped:
                lines.append("  Data Coverage:")
                lines.append(f"    ⚠ {len(_skipped)}/{len(_pairs)} pairs lack data (<{_min_y} years): {', '.join(_skipped)}")
                lines.append(f"    → Download with: python scripts/download_data.py --pairs {' '.join(_skipped)}")
                errors.append(f"Data coverage: {len(_skipped)} pair(s) lack sufficient data")
            else:
                _low = [r["pair"] for r in _rep if r["status"] == "LOW"]
                if _low:
                    lines.append(f"    ℹ Low coverage pairs ({len(_low)}): {', '.join(_low)}")
    except Exception:
        pass  # Non-blocking

    if not errors and not warnings:
        lines.append("  ✓ No blocking issues detected.")
    elif not errors:
        lines.append("  ✓ No blocking errors (review warnings before a multi-day run).")
    else:
        lines.append("  ✗ Fix errors before starting training.")

    lines.extend(["=" * 62, ""])

    if verbose:
        print("\n".join(lines))

    return 1 if errors else 0
