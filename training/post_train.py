"""Ensemble meta-training, promotion gate, auto-tune, and artifact helpers.\n\nSee docs/CONTINUE.md."""
from __future__ import annotations

import json
import os
import pickle
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.feature_ablation import _atomic_copy, _feature_ablation_config

_HOST = None
_BOUND = False
_HOST_DEPS = (
    '_log_error',
    '_log_warn',
    '_log_info',
    '_core_model',
    '_strict_load_report',
    '_slug_part',
    'build_model',
    '_model_build_args',
    '_apply_model_profile',
    '_on_disk_sequence_count',
    'ZarrStreamDataset',
    '_ThreadPrefetchLoader',
    '_x_path',
    '_y_path',
    '_zarr_open_group',
    'ZARR',
    '_get_pairs',
    '_promotion_holdout_n',
    '_embargo_bars',
    '_purge_bars',
    '_read_json_dict',
    '_fold_history_summary',
    '_deploy_onnx_to_cpp_server',
    '_feature_schema_payload',
    '_verify_onnx_schema_deployment',
    '_atomic_copy',
    'ENSEMBLE',
    'EnsembleMetaLearner',
    'train_meta_learner',
    'WANDB',
    '_safe_wandb_log',
    'PATHS',
    'LABELING',
    'FEATURES',
    '_TRAIN_LOGGER',
    'MODEL_REGISTRY',
    'MODEL_ROLES',
    'validate_epoch',
    'train_epoch',
    'build_criterion',
    'TemperatureScaler',
)


def bind_host(host_mod) -> None:
    global _HOST, _BOUND
    _HOST = host_mod
    g = globals()
    for name in _HOST_DEPS:
        if hasattr(host_mod, name):
            g[name] = getattr(host_mod, name)
    _BOUND = True


def _ensure_bound() -> None:
    import training.train_gpu as tg
    bind_host(tg)

# -----------------------------------------------------------------------------
# ENSEMBLE META-LEARNER TRAINING
# -----------------------------------------------------------------------------

def run_ensemble_meta(
    cache_path:  str,
    n_features:  int,
    args,
    device:      torch.device,
) -> None:
    """
    Load all trained base model checkpoints, build an EnsembleMetaLearner,
    then train the meta-network with a diversity penalty.

    The diversity penalty has two components:
      1. Weight entropy maximisation ΓÇö prevents the meta from collapsing to
         a single model (all weight on the best base model).
      2. Base-output correlation penalty ΓÇö rewards the meta for up-weighting
         models whose predictions disagree with each other.

    Only runs when --train-ensemble is passed and at least 2 base checkpoints
    are found in the checkpoint directory.
    """
    _ensure_bound()
    if not ENSEMBLE:
        print("[EnsembleMeta] models.ensemble not available ΓÇö skipping.")
        return

    ckpt_dir = Path(args.checkpoint_dir)
    loaded_bases: list = []
    loaded_names: list = []
    loaded_ckpts: list[Path] = []
    loaded_seq_lens: list[int] = []

    for model_name in MODEL_REGISTRY:
        # Per-model subfolder layout first, then legacy flat layout
        ckpt = ckpt_dir / model_name / f"{model_name}_best.pt"
        if not ckpt.exists():
            ckpt = ckpt_dir / f"{model_name}_best.pt"
        if not ckpt.exists():
            continue
        try:
            model_args = _model_build_args(args, model_name)
            base = build_model(model_name, n_features, model_args).to(device)

            # Load checkpoint with flexibility for wrapper nesting
            ckpt_data = torch_load_safe(ckpt, map_location=device)
            state = ckpt_data.get("model_state_dict", ckpt_data.get("state_dict", ckpt_data))

            # Attempt to load into base directly (if checkpoint was a MultiTaskWrapper)
            # or into core backbone (if checkpoint was only the base architecture).
            try:
                base.load_state_dict(state, strict=True)
            except Exception:
                # If strict loading failed, load into backbone with an asserting
                # report so a near-empty load fails loudly (A-H2).
                core = base.backbone if hasattr(base, "backbone") else base
                _strict_load_report(core, state, f"EnsembleMeta:{model_name}", min_frac_loaded=0.6)

            base.eval()

            loaded_bases.append(base)
            loaded_names.append(model_name)
            loaded_ckpts.append(ckpt)
            loaded_seq_lens.append(int(getattr(model_args, "seq_len", getattr(args, "seq_len", 0)) or 0))
            print(f"  [EnsembleMeta] Loaded {model_name} from {ckpt.name}")
        except Exception as e:
            print(f"  [EnsembleMeta] Could not load {model_name}: {e}")

    if len(loaded_bases) < 2:
        print("[EnsembleMeta] Need >= 2 trained base models ΓÇö skipping "
              f"(found {len(loaded_bases)}: {loaded_names}). "
              "Train with --all-models first.")
        return

    print(f"\n[EnsembleMeta] Training meta-learner on {len(loaded_bases)} bases: "
          f"{loaded_names}")

    meta = EnsembleMetaLearner(
        loaded_bases,
        context_dim=32,
        hidden=64,
        base_names=loaded_names,
        base_seq_lens=loaded_seq_lens,
    ).to(device)

    # Random 10% of the *trainable* prefix only — never the promotion holdout.
    _total = _on_disk_sequence_count(cache_path) or 10_000
    _holdout = int(_promotion_holdout_n(_total, args))
    _embargo = int(_embargo_bars(args))
    _trainable = max(0, int(_total) - _holdout - _embargo)
    if _trainable < 100:
        print(f"[Ensemble] Trainable prefix too small ({_trainable}); skipping meta training.")
        return
    n_meta   = min(200_000, max(1, int(0.1 * _trainable)))
    meta_idx = np.random.choice(_trainable, n_meta, replace=False)
    meta_ds = ZarrStreamDataset(cache_path, meta_idx, shuffle_chunks=True)
    meta_dl = DataLoader(
        meta_ds, batch_size=min(args.batch_size, 512),
        shuffle=False, num_workers=min(4, args.num_workers),
        pin_memory=(os.name != "nt") if getattr(args, "pin_memory", None) is None else bool(args.pin_memory),
    )

    ensemble_dir = ckpt_dir / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    out = ensemble_dir / "ensemble_meta_best.pt"
    history = train_meta_learner(
        meta,
        meta_dl,
        epochs=getattr(args, "ensemble_epochs", 10),
        lr=1e-3,
        diversity_weight=getattr(args, "ensemble_div_weight", 0.1),
        device=str(device),
        verbose=True,
        checkpoint_path=str(out),
        checkpoint_meta={
            "base_names": loaded_names,
            "base_seq_lens": loaded_seq_lens,
            "n_features": int(n_features),
            "seq_len": int(getattr(args, "seq_len", 60)),
            "cache_path": str(cache_path),
        },
    )

    final = ensemble_dir / "ensemble_meta_final.pt"
    _safe_save(meta.state_dict(), final)
    _meta_payload = {
        "epoch": len(history),
        "loss": float(history[-1]) if history else None,
        "history": history,
        "meta": {
            "base_names": loaded_names,
            "base_seq_lens": loaded_seq_lens,
            "n_features": int(n_features),
            "seq_len": int(getattr(args, "seq_len", 60)),
            "cache_path": str(cache_path),
        },
    }
    import json as _json
    final.with_suffix(final.suffix + ".json").write_text(
        _json.dumps(_meta_payload, indent=2), encoding="utf-8"
    )
    ensemble_manifest = {
        "kind": "ensemble_meta_learner",
        "created_at": datetime.now(UTC).isoformat(),
        "base_models": [
            {"name": name, "checkpoint": str(path), "seq_len": int(seq_len)}
            for name, path, seq_len in zip(loaded_names, loaded_ckpts, loaded_seq_lens)
        ],
        "artifacts": {
            "best_checkpoint": str(out),
            "final_checkpoint": str(final),
            "best_onnx": str(ensemble_dir / "ensemble_meta_best.onnx"),
            "promotion_gate": str(ensemble_dir / "promotion_gate.json"),
        },
        "schema": {
            "n_features": int(n_features),
            "seq_len": int(getattr(args, "seq_len", 60)),
            "feature_names": list(getattr(args, "_feat_names", []) or []),
        },
        "training": {
            "cache_path": str(cache_path),
            "samples": int(n_meta),
            "epochs": int(getattr(args, "ensemble_epochs", 10)),
            "diversity_weight": float(getattr(args, "ensemble_div_weight", 0.1)),
            "history": history,
            "final_loss": float(history[-1]) if history else None,
        },
    }
    _safe_save_json(ensemble_manifest, ensemble_dir / "ensemble_manifest.json")
    try:
        from inference.onnx_inference import _wrap_ensemble_logits, core_onnx_export

        ensemble_onnx = ensemble_dir / "ensemble_meta_best.onnx"
        wrapped = _wrap_ensemble_logits(meta).to(device)
        wrapped.eval()
        core_onnx_export(
            model=wrapped,
            n_features=int(n_features),
            seq_len=int(getattr(args, "seq_len", 60)),
            output_path=str(ensemble_onnx),
            output_name="logits",
        )
        print(f"[EnsembleMeta] Exported ONNX -> {ensemble_onnx}")
        if bool(getattr(args, "deploy_ensemble", False)):
            args._n_features = int(n_features)
            _deploy_onnx_to_cpp_server(
                ensemble_onnx,
                args,
                model_name="ensemble",
                artifact_dir=ensemble_dir,
                source_checkpoint=out if out.exists() else final,
            )
    except Exception as exc:
        print(f"[EnsembleMeta] ONNX export/deploy skipped: {exc}")
    best_msg = f"best={out}" if out.exists() else "best=not written"
    print(f"[EnsembleMeta] Saved -> {best_msg} | final={final} | Final loss: {history[-1]:.6f}")

def run_profiler(model, loader, device, amp_dtype, use_amp, log_dir: str, run_name: str,
                 seq_len: int | None = None) -> None:
    """
    Run torch.profiler for a short burst and write a Chrome / Perfetto trace.

    Usage:
        python training/train_gpu.py --profile --model haelt --quick-mode

    Opens trace in:
        ΓÇó Chrome: chrome://tracing -> Load -> select logs/profile_*.json
        ΓÇó Perfetto: https://ui.perfetto.dev  (richer, recommended)

    What the trace shows:
        ΓÇó CPU-GPU overlap (are kernels launching without gaps?)
        ΓÇó DataLoader stall time (input-bound vs compute-bound)
        ΓÇó torch.compile kernel fusion vs interpreted ops
        ΓÇó Memory copies and fragmentation
    """
    _ensure_bound()
    from torch.profiler import ProfilerActivity, profile, record_function, tensorboard_trace_handler

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    trace_path = str(Path(log_dir) / f"profile_{run_name}")

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    print(f"\n[Profiler] Warmup 3 batches, active 5 batches ΓÇö trace -> {trace_path}/")
    _log_info(f"[Profiler] trace path: {trace_path}")

    model.train()
    data_iter = iter(loader)

    with profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=0, warmup=3, active=5, repeat=1),
        on_trace_ready=tensorboard_trace_handler(trace_path),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,        # stack traces slow things down; enable for deep dives
    ) as prof:
        for step in range(8):
            try:
                xb, yb = next(data_iter)
            except StopIteration:
                break
            xb = xb.to(device, non_blocking=True)
            xb = _crop_to_seq_len(xb, seq_len)
            yb = yb.to(device, non_blocking=True)
            with record_function("forward"):
                if use_amp and device.type == "cuda":
                    with autocast("cuda", dtype=amp_dtype):
                        model(xb)
                else:
                    model(xb)
            prof.step()

    # Print a short table to stdout so you don't have to open the trace to see the hottest ops
    print(prof.key_averages().table(sort_by="cuda_time_total" if device.type == "cuda" else "cpu_time_total", row_limit=15))
    print(f"[Profiler] Full trace written to {trace_path}/  (open in Perfetto or chrome://tracing)")
    _log_info("[Profiler] Complete")

def _safe_save(obj, path, metadata=None) -> None:
    """Safe wrapper for torch.save that immediately verifies integrity via atomic tempfile."""
    _ensure_bound()
    import json
    import os
    import tempfile
    from pathlib import Path

    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(obj, tmp)
        if os.path.getsize(tmp) <= 0:
            raise ValueError(f"[SafeSave] Temporary checkpoint has 0 bytes: {tmp}")
        _ = torch.load(tmp, map_location="cpu", weights_only=True)
        os.replace(tmp, path)
        if metadata is not None:
            meta = dict(metadata)
            meta.update({
                "artifact_path": str(path),
                "artifact_bytes": int(path.stat().st_size),
                "verified_loadable": True,
                "verified_at": datetime.now(UTC).isoformat(),
            })
            meta_path = path.with_suffix(path.suffix + ".metadata.json")
            fd_meta, tmp_meta = tempfile.mkstemp(prefix=f".{meta_path.name}.", suffix=".tmp", dir=str(meta_path.parent))
            os.close(fd_meta)
            try:
                with open(tmp_meta, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2, default=str)
                with open(tmp_meta, encoding="utf-8") as f:
                    loaded_meta = json.load(f)
                for key, expected in metadata.items():
                    if str(loaded_meta.get(key)) != str(expected):
                        raise ValueError(f"[SafeSave] metadata mismatch for {key}: {loaded_meta.get(key)!r} != {expected!r}")
                os.replace(tmp_meta, meta_path)
            finally:
                if os.path.exists(tmp_meta):
                    try:
                        os.remove(tmp_meta)
                    except OSError:
                        pass
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass

def _safe_save_json(data, path) -> None:
    """Safely write JSON to `path` using atomic tempfile replacement."""
    _ensure_bound()
    import json
    import os
    import tempfile
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass

def torch_load_safe(path, map_location=None) -> Any:
    """Load a PyTorch checkpoint with weights_only=True, falling back to
    weights_only=False ONLY for legacy checkpoints that require arbitrary
    objects. New checkpoints are always deserialized in safe mode."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except (torch.SerializationWarning, pickle.UnpicklingError, AttributeError,
            TypeError, KeyError, ValueError, ModuleNotFoundError):
        return torch.load(path, map_location=map_location, weights_only=False)

def _generate_model_card(model_name: str, args, history_or_cv, ckpt_dir: str, n_features: int) -> None:
    """Generates a standard Model Card JSON documenting the architecture, features, and performance."""
    card = {
        "model_name": model_name,
        "timestamp": datetime.now(UTC).isoformat(),
        "architecture": getattr(args, "model", "unknown"),
        "data_window": f"{getattr(args, 'start_date', 'unknown')} to {getattr(args, 'end_date', 'unknown')}",
        "pairs": getattr(args, "pairs", []),
        "features_count": n_features,
        "label_method": getattr(args, "label_method", "unknown"),
        "promotion_status": "candidate",
        "known_weaknesses": ["Needs stress evaluation"],
    }

    # Extract best validation stats
    if isinstance(history_or_cv, list):  # CV run
        best_fold = max(history_or_cv, key=lambda x: x.get('best_metric', -999) if x.get('best_metric') is not None else -999)
        card["validation_results"] = {
            "best_val_sharpe_proxy": best_fold.get('best_metric'),
            "fold": best_fold.get('fold')
        }
        card["forward_holdout_results"] = "See fold validation metrics"
    elif isinstance(history_or_cv, dict):
        best_val = max(history_or_cv.get('val_sharpe', [0.0])) if history_or_cv.get('val_sharpe') else None
        best_loss = min(history_or_cv.get('val_loss', [999.0])) if history_or_cv.get('val_loss') else None
        card["validation_results"] = {
            "best_val_sharpe_proxy": best_val,
            "best_val_loss": best_loss
        }
        card["forward_holdout_results"] = "Pending"

    out_path = Path(ckpt_dir) / f"{model_name}_model_card.json"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(card, f, indent=2)
        print(f"\n[ModelCard] Generated -> {out_path}")
    except Exception as e:
        print(f"\n[ModelCard] Warning: Failed to generate model card: {e}")

def _stability_adjusted_score(score, sharpe_curve, gen_gap):
    """Apply stability penalty to a sharpe-based fold score.

    Penalises folds where the best-seen Sharpe is far above the final value
    (peak-drop), where the validation Sharpe was volatile across epochs, or
    where the generalization gap (val_loss - train_loss) is large.

    Returns (adjusted_score, volatility, gap_penalty).
    """
    import numpy as np
    volatility = 0.0
    gap_penalty = 0.0
    if len(sharpe_curve) >= 4 and score is not None:
        volatility = float(np.std(sharpe_curve, ddof=1))
        final_val = sharpe_curve[-1]
        gap_penalty = max(0.0, score - final_val)
        train_val_gap_penalty = (gen_gap * 0.1) if gen_gap is not None and gen_gap > 0 else 0.0
        score = score - (gap_penalty * 0.5) - (volatility * 0.5) - train_val_gap_penalty
    return score, volatility, gap_penalty

def _promote_best_fold(
    model_name: str,
    checkpoint_dir: str,
    cv_hist: list,
    early_stop_metric: str = "sharpe",
    alerter = None,
    force_promotion: bool = False
) -> None:
    """
    After walk-forward CV, scan all fold configs in checkpoint_dir, pick the
    fold with the best metric, and copy its checkpoint to
    <checkpoint_dir>/<model_name>_best.pt.
    """
    _ensure_bound()
    ckpt_dir = Path(checkpoint_dir)
    use_sharpe = early_stop_metric == "sharpe"
    best_fold = None
    best_score = None
    best_tie_breaker = None
    best_tie_breaker2 = None
    best_metrics = {}

    candidate_folds = []

    # Read metrics from the saved config JSON files
    for entry in cv_hist:
        fi = entry["fold"]
        cfg_candidates = [
            ckpt_dir / f"{model_name}_fold{fi}_config.json",
            ckpt_dir / model_name / f"{model_name}_fold{fi}_config.json"
        ]
        cfg_path = next((p for p in cfg_candidates if p.exists()), None)

        score = None
        tie_breaker = None
        tie_breaker2 = None

        history = entry.get("history", {})
        gen_gap = None
        if history and "train_loss" in history and "val_loss" in history:
            try:
                # Calculate generalization gap at the last epoch
                gen_gap = history["val_loss"][-1] - history["train_loss"][-1]
            except Exception:
                pass

        sharpe_val = None
        loss_val = None
        volatility = 0.0
        gap_penalty = 0.0

        if cfg_path:
            try:
                with open(cfg_path, encoding="utf-8") as f:
                    cfg = json.load(f)
                sharpe_val = cfg.get("best_val_sharpe_proxy")
                loss_val = cfg.get("best_val_loss")

                if use_sharpe:
                    score = sharpe_val
                    sharpe_curve = history.get("val_sharpe", [])
                    score, volatility, gap_penalty = _stability_adjusted_score(
                        score, sharpe_curve, gen_gap)

                    tie_breaker = -loss_val if loss_val is not None else None
                else:
                    score = -loss_val if loss_val is not None else None
                    tie_breaker = sharpe_val
            except Exception:
                pass

        if score is None:
            raw = entry.get("best_metric")
            if raw is not None:
                if use_sharpe:
                    score = raw
                    sharpe_curve = history.get("val_sharpe", [])
                    score, volatility, gap_penalty = _stability_adjusted_score(
                        score, sharpe_curve, gen_gap)
                else:
                    score = -raw
                tie_breaker = 0.0

        if score is None:
            continue

        tie_breaker2 = -gen_gap if gen_gap is not None else 0.0

        candidate_folds.append({
            "fold": fi,
            "score": score,
            "tie_breaker": tie_breaker,
            "tie_breaker2": tie_breaker2,
            "gen_gap": gen_gap,
            "volatility": volatility if use_sharpe else 0.0,
            "gap_penalty": gap_penalty if use_sharpe else 0.0
        })

        is_better = False
        if best_score is None or score > best_score:
            is_better = True
        elif score == best_score:
            if tie_breaker is not None and best_tie_breaker is not None:
                if tie_breaker > best_tie_breaker:
                    is_better = True
                elif tie_breaker == best_tie_breaker:
                    if tie_breaker2 is not None and best_tie_breaker2 is not None:
                        if tie_breaker2 > best_tie_breaker2:
                            is_better = True

        if is_better:
            best_score = score
            best_tie_breaker = tie_breaker
            best_tie_breaker2 = tie_breaker2
            best_fold = fi
            best_metrics = {"sharpe": sharpe_val or 0.0, "val_loss": loss_val or 0.0, "gen_gap": gen_gap}

    if best_fold is None:
        print(f"[BestFold] {model_name}: could not determine best fold ΓÇö skipping promotion.")
        return

    src_flat = ckpt_dir / f"{model_name}_fold{best_fold}_best.pt"
    src_nested = ckpt_dir / model_name / f"{model_name}_fold{best_fold}_best.pt"

    src = src_nested if src_nested.exists() else src_flat

    if not src.exists():
        print(f"[BestFold] {model_name}: fold {best_fold} checkpoint not found at {src_flat} or {src_nested}")
        return

    metric_label = "sharpe" if use_sharpe else "val_loss"
    metric_val   = best_score if use_sharpe else -best_score

    # Challenger vs Production Gate
    accepted = True
    reject_reason = ""
    deployment_json = ckpt_dir / "deployment.json"
    if deployment_json.exists() and not force_promotion:
        try:
            with open(deployment_json) as f:
                prod_data = json.load(f)
            # If there's an existing metric value
            if "metric_value" in prod_data and prod_data.get("metric", "") == metric_label:
                prod_metric = prod_data["metric_value"]
                # F1 fix: positive min_delta in BOTH directions.
                # For sharpe (higher=better): reject if challenger <= prod + min_delta
                #   (must beat prod by strictly more than min_delta)
                # For loss (lower=better): reject if challenger >= prod - min_delta
                #   (must be lower than prod by strictly more than min_delta)
                min_delta = 0.001
                if use_sharpe:
                    # higher is better — challenger wins only if strictly exceeds prod
                    if metric_val <= prod_metric + min_delta:
                        reject_reason = f"Rejected: new sharpe {metric_val:.4f} is not significantly better than deployed {prod_metric:.4f} (needs >{prod_metric + min_delta:.4f})"
                        accepted = False
                else:
                    # loss direction: lower is better — challenger wins only if
                    # strictly lower than prod by at least min_delta
                    if metric_val >= prod_metric - min_delta:
                        reject_reason = f"Rejected: new loss {metric_val:.4f} is not significantly lower than deployed {prod_metric:.4f} (needs <{prod_metric - min_delta:.4f})"
                        accepted = False
                
                if accepted:
                    print(f"[ChallengerGate] Accepted: new {metric_label} {metric_val:.4f} vs deployed {prod_metric:.4f}")
                else:
                    print(f"[ChallengerGate] {reject_reason}")
        except Exception as e:
            print(f"[ChallengerGate] Warning: failed to parse existing deployment.json: {e}")

    # M11: emit challenger-vs-prod decision JSONL telemetry
    try:
        _tl = getattr(_HOST, "_TRAIN_LOGGER", None)
        if _tl is not None and hasattr(_tl, "on_promotion_decision"):
            _prod_metric_val = None
            try:
                if deployment_json.exists() and not force_promotion:
                    with open(deployment_json) as _f:
                        _pd = json.load(_f)
                    if "metric_value" in _pd and _pd.get("metric", "") == metric_label:
                        _prod_metric_val = _pd["metric_value"]
            except Exception:
                pass
            
            _tl.on_promotion_decision(
                model_name=model_name,
                promoted=accepted,
                metric_name=metric_label,
                metric_value=float(metric_val) if metric_val is not None else None,
                gate_summary=f"challenger accepted ({metric_label}={metric_val:.4f})" if accepted else reject_reason,
                gate_reasons=[reject_reason] if not accepted else None,
                gate_details={"selected_fold": best_fold, "n_candidates": len(candidate_folds)},
                challenger_vs_prod={
                    "prod_metric": _prod_metric_val,
                    "challenger_metric": float(metric_val) if metric_val is not None else None,
                    "direction": "sharpe" if use_sharpe else "loss",
                    "min_delta": 0.001,
                    "accepted": accepted,
                },
            )
    except Exception:
        pass

    if not accepted:
        return

    dst_flat = ckpt_dir / f"{model_name}_best.pt"
    dst_nested = ckpt_dir / model_name / f"{model_name}_best.pt"

    dst_nested.parent.mkdir(parents=True, exist_ok=True)
    _atomic_copy(src, dst_flat)
    if src != dst_nested:
        _atomic_copy(src, dst_nested)
    print(f"[BestFold] {model_name}: fold {best_fold} is best ({metric_label}={metric_val:.4f}) -> promoted to {dst_flat.name} & {dst_nested.name}")

    if alerter:
        try:
            alerter.send_fold_selected(model_name, best_fold, best_metrics)
        except Exception as e:
            print(f"[Discord] Failed to send fold_selected: {e}")

    summary = {
        "model": model_name,
        "selected_fold": best_fold,
        "metric": metric_label,
        "metric_value": round(metric_val, 6),
        "secondary_metric": "val_loss" if use_sharpe else "val_sharpe",
        "secondary_value": round(best_metrics.get("val_loss", 0.0), 6),
        "gen_gap": round(best_metrics.get("gen_gap") or 0.0, 6),
        "n_candidates": len(candidate_folds),
        "source_checkpoint": str(src.name),
        "selected_at": datetime.now(UTC).isoformat(),
        "candidates": [
            {
                "fold": c["fold"],
                "score": round(c["score"], 6),
                "tie_breaker": round(c["tie_breaker"], 6) if c.get("tie_breaker") is not None else None,
                "gen_gap": round(c["gen_gap"], 6) if c.get("gen_gap") is not None else None,
            }
            for c in candidate_folds
        ],
    }
    try:
        _safe_save_json(summary, ckpt_dir / "fold_selection.json")
        print(f"[BestFold] fold_selection.json written -> {ckpt_dir / 'fold_selection.json'}")
    except Exception:
        try:
            with open(ckpt_dir / "fold_selection.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        except Exception:
            pass

def _evaluate_forward_gate(model_name, cache_path, n_samples, n_features, args, device,
                           fold_sharpes=None) -> dict:
    """B-C1: execution-aware forward holdout gate for a freshly trained challenger.

    Runs ``scripts.backtest_model.run_execution_backtest`` on the chronological
    promotion holdout window, then feeds Sharpe / PF / DD / n_trades into
    ``PromotionGate.evaluate``. Result includes ``gate_input_type:
    execution_backtest`` for promotion audit.
    """
    _ensure_bound()
    try:
        from validation.promotion_gate import GateConfig, PromotionGate
    except Exception as e:
        return {"promoted": False, "details": {}, "reasons": [f"gate import failed: {e}"],
                "summary": "REJECT (gate unavailable)"}

    fwd_n = _promotion_holdout_n(n_samples, args)
    start = max(0, n_samples - fwd_n)
    n_fwd = n_samples - start
    if n_fwd < 50:
        return {"promoted": False, "details": {"n_trades": float(n_fwd)},
                "reasons": ["forward window too small"],
                "summary": "REJECT (insufficient forward data)"}

    ckpt_dir   = Path(args.checkpoint_dir)
    getattr(args, "loss", "") in ("cross_entropy", "multi_task", "asymmetric_directional")

    if model_name == "ensemble":
        ckpt_path = ckpt_dir / "ensemble" / "ensemble_meta_best.pt"
        if not ckpt_path.exists():
            return {"promoted": False, "details": {}, "reasons": ["no ensemble checkpoint to gate"], "summary": "REJECT (no checkpoint)"}

        meta_json_path = ckpt_dir / "ensemble" / "ensemble_meta_final.json"
        base_names = []
        if meta_json_path.exists():
            import json as _json
            meta_data = _json.loads(meta_json_path.read_text())
            base_names = meta_data.get("meta", {}).get("base_names", base_names)

        from models.ensemble import EnsembleMetaLearner
        loaded_bases = []
        for b_name in base_names:
            b_ckpt = next((p for p in [ckpt_dir / b_name / f"{b_name}_best.pt", ckpt_dir / f"{b_name}_best.pt"] if p.exists()), None)
            if b_ckpt:
                b_model = build_model(b_name, n_features, _model_build_args(args, b_name)).to(device)
                _dummy = torch.zeros(2, getattr(args, "seq_len", 60), n_features, device=device)
                _ = b_model(_dummy)
                b_state = torch_load_safe(b_ckpt, map_location=device)
                if isinstance(b_state, dict) and "model_state_dict" in b_state:
                    b_state = b_state["model_state_dict"]
                elif isinstance(b_state, dict) and "state_dict" in b_state:
                    b_state = b_state["state_dict"]
                b_model.load_state_dict(b_state, strict=False)
                loaded_bases.append(b_model)

        if not loaded_bases:
            return {"promoted": False, "details": {}, "reasons": ["no ensemble base models loaded"], "summary": "REJECT (no bases)"}

        model = EnsembleMetaLearner(
            loaded_bases,
            context_dim=32,
            hidden=64,
            base_names=base_names,
        ).to(device)

        # Initialize LazyLinear modules before loading state dict
        _dummy = torch.zeros(2, getattr(args, "seq_len", 60), n_features, device=device)
        _ = model(_dummy)

        model.load_state_dict(torch_load_safe(ckpt_path, map_location=device), strict=False)
        model.eval()
        core = model
        state = {} # dummy state to pass the strict load report
    else:
        candidates = [ckpt_dir / model_name / f"{model_name}_best.pt",
                      ckpt_dir / f"{model_name}_best.pt"]
        ckpt_path  = next((p for p in candidates if p.exists()), None)
        if ckpt_path is None:
            return {"promoted": False, "details": {}, "reasons": ["no checkpoint to gate"],
                    "summary": "REJECT (no checkpoint)"}

        model = build_model(model_name, n_features, _model_build_args(args, model_name)).to(device)
        core  = _core_model(model)

        # Initialize LazyLinear modules before loading state dict
        _dummy = torch.zeros(2, getattr(args, "seq_len", 60), n_features, device=device)
        _ = model(_dummy)

        state = torch_load_safe(ckpt_path, map_location=device)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
    try:
        if model_name != "ensemble":
            _strict_load_report(core, state, f"Gate:{model_name}", min_frac_loaded=0.6)
    except Exception as e:
        return {"promoted": False, "details": {}, "reasons": [f"checkpoint load failed: {e}"],
                "summary": "REJECT (load failed)"}
    model.eval()

    if ZARR and cache_path.endswith(".zarr") and Path(cache_path).is_dir():
        _z = _zarr_open_group(cache_path, mode="r"); _Xs = _z["X"]
        def _rd(a, b): return np.asarray(_Xs[start + a:start + b], dtype=np.float32)
    else:
        _Xm = np.load(_x_path(cache_path), mmap_mode="r")
        def _rd(a, b): return np.asarray(_Xm[start + a:start + b], dtype=np.float32)

    # Calculate holdout dates from the TRAINING DATA window, not wall-clock run time.
    try:
        import pandas as pd

        _start_raw = (
            getattr(args, "start_date", None)
            or getattr(args, "data_start", None)
            or getattr(args, "start", None)
        )
        _end_raw = (
            getattr(args, "end_date", None)
            or getattr(args, "data_end", None)
            or getattr(args, "end", None)
        )
        if not _start_raw or not _end_raw:
            raise ValueError(f"missing training data window start/end ({_start_raw!r}, {_end_raw!r})")

        s_dt = pd.Timestamp(_start_raw)
        e_dt = pd.Timestamp(_end_raw)
        total_days = max(1, (e_dt - s_dt).days)
        holdout_frac = getattr(args, "promote_forward_frac", 0.1)
        holdout_days = max(1, int(total_days * holdout_frac))
        holdout_start = (e_dt - pd.Timedelta(days=holdout_days)).strftime("%Y-%m-%d")
        holdout_end = e_dt.strftime("%Y-%m-%d")

    except Exception as e:
        return {
            "promoted": False,
            "details": {"gate_input_type": "execution_backtest"},
            "gate_input_type": "execution_backtest",
            "reasons": [f"holdout dates unavailable: {e}"],
            "summary": "REJECT (missing holdout dates)",
        }



    pairs = list(_get_pairs(args))

    if not pairs:

        pairs = ["EURUSD"]



    print(f"\n[PromotionGate] Running EXECUTION-AWARE Backtest for {model_name} on {holdout_start} -> {holdout_end}")



    try:

        import sys

        _ROOT = Path(__file__).resolve().parent.parent

        if str(_ROOT) not in sys.path:

            sys.path.insert(0, str(_ROOT))

        from scripts.backtest_model import run_execution_backtest



        bt_metrics = run_execution_backtest(
            model=model,
            pair_list=pairs,
            start_date=holdout_start,
            end_date=holdout_end,
            seq_len=getattr(args, "seq_len", 60),
            n_features=n_features,
            device=device,
            bar_freq=getattr(args, "bar_freq", "1Min"),
            data_source=getattr(args, "data_source", "dukascopy"),
            stop_pips=15.0, # Will be overridden by ATR tracking ideally, using defaults for gate
            take_pips=20.0,
            inference_batch_size=getattr(args, "batch_size", 4096),
        )

    except Exception as e:

        print(f"[PromotionGate] Execution backtest failed: {e}")

        bt_metrics = {"error": str(e)}



    if bt_metrics.get("error"):

        return {"promoted": False, "details": {"n_trades": 0.0, "error": bt_metrics["error"]},

                "reasons": [f"Execution backtest failed: {bt_metrics['error']}"],

                "summary": "REJECT (backtest error)"}



    # Extract metrics for PromotionGate

    pnls = bt_metrics.pop("signals_df", pd.DataFrame())["pnl_pips"].tolist() if "signals_df" in bt_metrics and "pnl_pips" in bt_metrics["signals_df"] else []

    bt_metrics.pop("equity_curve", [10000.0])



    if bt_metrics["n_trades"] < 1:

        return {"promoted": False, "details": {"n_trades": 0.0},
                "reasons": ["challenger took no trades on forward window"],
                "summary": "REJECT (no trades)"}

    folds = list(fold_sharpes) if fold_sharpes else []
    n_trials = max(1, len(folds))
    sharpe_std = float(np.std(folds)) if len(folds) > 1 else 0.0
    gate = PromotionGate(GateConfig(strict_psr=True))   # B-M4: deflated Sharpe for retrain selection


    # Overwrite the gate evaluate call with the execution-aware metrics directly
    # P1 fix (2026-08-07): stop substituting net_pnl for gross_pnl and 0.0 for
    # transaction costs — that defeated the cost gate (cost_pct = 0.0 always
    # passed max_cost_pct=0.30). The backtester now exposes gross_pnl_usd
    # (sum of gross trade P&L = wins + losses *before* costs) and
    # total_commission_usd separately; we pass both so the cost gate fires.
    # The cost gate is `transaction_costs / max(abs(gross_pnl), 1e-9) <= 0.30`
    # so it now measures real cost drag.
    gross_pnl_value = float(bt_metrics.get("gross_pnl", 0.0) or 0.0)
    transaction_costs_value = float(bt_metrics.get("total_commission", 0.0) or 0.0)
    # If the backtester didn't populate gross_pnl (old cache/pre-2026-08-07),
    # fall back to deriving gross from net + total_commission.
    if gross_pnl_value == 0.0 and transaction_costs_value > 0.0:
        # gross_pnl_usd = net_pnl + total_commission (since commission subtracted to get net)
        gross_pnl_value = float(bt_metrics.get("net_pnl", 0.0) or 0.0) + transaction_costs_value

    # Only winners contribute to gross_profit (the gate expects gross *winnings*
    # in the denominator, not signed sums). The backtester emits `gross_pnl_usd`
    # as the sum of gross trade P&L (can be negative if losses exceed wins).
    # For the cost gate we use the correct denominator: only *winning* gross.
    # When winners-side is unavailable, use abs(gross_pnl) which is a stronger
    # (smaller) denominator so cost_pct is still meaningful.
    gross_for_cost_gate = abs(gross_pnl_value) if gross_pnl_value != 0.0 else None

    if gross_for_cost_gate is None:
        # Flag: caller should fail the cost gate explicitly when gross profit
        # information is unavailable. PromotionGate raises on gross_pnl=None,
        # which converts into a REJECT — a fail-closed signal to the operator
        # that the forward backtest didn't expose cost data. We catch here so
        # the rest of the chain can run; the gate's reject message explains.
        try:
            result = gate.evaluate(
                sharpe=bt_metrics["sharpe"],
                profit_factor=bt_metrics.get("profit_factor", 1.0),
                max_drawdown=bt_metrics["max_drawdown"],
                n_trades=bt_metrics["n_trades"],
                regime_pnl={},
                gross_pnl=None,  # triggers fail-closed ValueError in gate
                transaction_costs=0.0,
                n_backtest_trials=n_trials,
                backtest_sharpe_std=sharpe_std,
                emergency_retrain=bool(getattr(args, "finetune_warm_start", False)),
                n_obs=max(1, int(bt_metrics.get("n_trades", 0) or 0)),
            )
        except ValueError as _ve:
            print(f"[PromotionGate] forward gate rejected: gross_pnl not exposed "
                  f"by backtest — {_ve}")
            return {
                "promoted": False,
                "details": {"n_trades": bt_metrics["n_trades"],
                            "error": "gross_pnl unavailable"},
                "reasons": ["forward backtest missing gross_pnl (cannot run cost gate)"],
                "summary": "REJECT (no gross_pnl — cost gate cannot run)",
            }
    else:
        result = gate.evaluate(

            sharpe=bt_metrics["sharpe"],

            profit_factor=bt_metrics.get("profit_factor", 1.0),

            max_drawdown=bt_metrics["max_drawdown"],

            n_trades=bt_metrics["n_trades"],

            regime_pnl={}, # Not tracking regime pnl in backtest_model return yet

            gross_pnl=gross_for_cost_gate,  # P1: real gross_pnl, not net_pnl

            transaction_costs=transaction_costs_value,  # P1: real costs, not 0.0

            n_backtest_trials=n_trials,
            backtest_sharpe_std=sharpe_std,
            emergency_retrain=bool(getattr(args, "finetune_warm_start", False)),
            n_obs=max(1, int(bt_metrics.get("n_trades", 0) or 0)),
        )
    result["gate_input_type"] = "execution_backtest"
    result.setdefault("details", {}).update({

        "gate_input_type": "execution_backtest",

        "forward_window": float(n_fwd),

        "holdout_start": holdout_start,

        "holdout_end": holdout_end,

        "sharpe": float(bt_metrics.get("sharpe", 0.0)),

        "profit_factor": float(bt_metrics.get("profit_factor", 0.0)),

        "max_drawdown": float(bt_metrics.get("max_drawdown", 0.0)),

        "n_trades": int(bt_metrics.get("n_trades", 0)),

        "net_pnl": float(bt_metrics.get("net_pnl", 0.0)),

    })

    print(f"[PromotionGate] {model_name}: {result.get('summary','?')} "
          f"| trades={len(pnls)} | forward_n={n_fwd}")

    # M11: emit on_promotion_decision JSONL event for audit trail
    try:
        _tl = getattr(_HOST, "_TRAIN_LOGGER", None)
        if _tl is not None and hasattr(_tl, "on_promotion_decision"):
            _tl.on_promotion_decision(
                model_name=model_name,
                promoted=bool(result.get("promoted", False)),
                metric_name="val_sharpe",
                metric_value=float(bt_metrics.get("sharpe", 0.0)),
                gate_summary=str(result.get("summary", "")),
                gate_reasons=result.get("reasons"),
                gate_details=result.get("details"),
                challenger_vs_prod={
                    "gate_input_type": "execution_backtest",
                    "forward_window": int(n_fwd),
                    "n_trades": int(bt_metrics.get("n_trades", 0)),
                    "net_pnl": float(bt_metrics.get("net_pnl", 0.0)),
                },
            )
    except Exception as _e:
        print(f"[PromotionGate] telemetry emit failed: {_e}")


    # ╬ô├╢├ç╬ô├╢├ç Confidence threshold sweep ╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç

    # Sweep [0.35 ╬ô├ç┬¬ 0.60] and write {model_name}_threshold_tuning.json next to

    # the checkpoint. Only meaningful when confidence has real variance.

    try:
        from validation.promotion_gate import write_threshold_tuning_json

        conf_scores = bt_metrics.get("confidence_scores", []) or []

        conf_traded = np.array(conf_scores, dtype=float) if len(conf_scores) >= 30 else np.array([])

        if len(conf_traded) > 0 and conf_traded.std() > 1e-6:

            sweep = gate.sweep_confidence_threshold(

                trade_pnls=pnls,

                confidence_scores=conf_traded.tolist(),

                annualization=252.0,

                min_trades=max(30, len(pnls) // 20),

            )

            thr_path = (ckpt_dir / model_name / f"{model_name}_threshold_tuning.json"

                        if (ckpt_dir / model_name).is_dir()

                        else ckpt_dir / f"{model_name}_threshold_tuning.json")

            written = write_threshold_tuning_json(

                sweep,

                str(thr_path),

                model_name=model_name,

                extra_meta={"forward_n": int(n_fwd), "n_traded": len(pnls)},

            )

            opt_thr = sweep.get("optimal_threshold")

            opt_sr  = sweep.get("optimal_sharpe")

            result.setdefault("details", {})["optimal_confidence_threshold"] = opt_thr

            print(f"[ThresholdSweep] {model_name}: optimal={opt_thr} "

                  f"(Sharpe={opt_sr}) ╬ô├Ñ├å {written}")

        else:

            print(f"[ThresholdSweep] {model_name}: skipped ╬ô├ç├╢ regression confidence proxy "

                  f"has uniform variance; use a softmax head for meaningful tuning.")

    except Exception as _thr_exc:

        print(f"[ThresholdSweep] {model_name}: sweep failed ({_thr_exc}); continuing.")



    return result

def _auto_tune_next_run(
    config_path: str,
    history: dict,
    gate_result: dict,
    best_epoch: int,
    total_epochs: int,
    run_name: str = "unknown_run",
    dry_tune: bool = False,
) -> None:
    """Audit every hyperparameter proposal and (optionally) apply it.

    Always writes ``logs/auto_tune/<run_name>_proposal.json`` with structured
    records ΓÇö whether or not dry_tune is set and whether or not any changes
    are made.  High-risk fields (data_range, label_method, checkpoint_dir,
    production thresholds) are never mutated.
    """
    _ensure_bound()
    import json
    from datetime import datetime
    from pathlib import Path

    # ΓöÇΓöÇ constants ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    _HIGH_RISK = {
        "start_date", "end_date", "data_source", "label_method",
        "checkpoint_dir", "data_cache", "production",
    }

    def _proposal(issue, section, key, prev, new, reason, confidence):
        return {
            "issue":        issue,
            "section":      section,
            "key":          key,
            "prev_value":   prev,
            "new_value":    new,
            "reason":       reason,
            "confidence":   confidence,   # "low" | "medium" | "high"
        }

    cfg_path = Path(config_path) if config_path else Path("config/run.yaml")
    log_dir  = Path("logs/auto_tune")
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_run_name = _slug_part(run_name, max_len=180)

    # Use versioned config path instead of overwriting run.yaml
    ts_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    versioned_cfg_path = cfg_path.parent / f"run_{ts_str}.yaml"

    prop_file = log_dir / f"{safe_run_name}_proposal.json"


    proposals: list[dict] = []

    # ΓöÇΓöÇ try to load config ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    data = None
    if cfg_path.exists():
        try:
            from ruamel.yaml import YAML
            yaml_io = YAML()
            yaml_io.preserve_quotes = True
            with open(cfg_path, encoding="utf-8") as f:
                data = yaml_io.load(f)
            backup = cfg_path.with_name(cfg_path.name + ".bak")
            shutil.copy2(cfg_path, backup)
        except ImportError:
            print("[Auto-Tune] ruamel.yaml not installed ΓÇö proposal-only mode.")
        except Exception as e:
            print(f"[Auto-Tune] Could not read config: {e}")
    else:
        print(f"[Auto-Tune] Config not found at {cfg_path} ΓÇö proposal-only mode.")

    # P3: detect whether this config was written by Optuna; if so skip heuristics
    # that would clobber the curriculum schedule Optuna already optimised.
    _optuna_applied = bool((data or {}).get("optuna", {}).get("applied", False)) if data else False
    if _optuna_applied:
        print("[Auto-Tune] Optuna-applied config detected ΓÇö all config mutations skipped "
              "(proposals still recorded for audit).")

    def _get(section, key, fallback):
        if data is None:
            return fallback
        return data.get(section, {}).get(key, fallback)

    def _class_balance_diagnostics(hist: dict) -> dict:
        pred_curve = hist.get("val_pred_counts") if isinstance(hist, dict) else None
        true_curve = hist.get("val_true_counts") if isinstance(hist, dict) else None
        pred = pred_curve[-1] if pred_curve else None
        true = true_curve[-1] if true_curve else None
        out = {
            "available": bool(pred),
            "quarantined": False,
            "pred_counts": pred,
            "true_counts": true,
            "pred_shares": None,
            "true_shares": None,
            "reason": None,
        }
        if not pred:
            return out
        pred_total = max(1, sum(int(x) for x in pred))
        pred_shares = [float(x) / pred_total for x in pred]
        out["pred_shares"] = pred_shares
        if true:
            true_total = max(1, sum(int(x) for x in true))
            out["true_shares"] = [float(x) / true_total for x in true]
        if min(pred_shares) < 0.02:
            out["quarantined"] = True
            out["reason"] = "missing_or_near_missing_prediction_class"
        elif max(pred_shares) > 0.85:
            out["quarantined"] = True
            out["reason"] = "dominant_prediction_class"
        return out

    _class_diag = _class_balance_diagnostics(history)
    _auto_tune_quarantined = bool(_class_diag.get("quarantined"))
    if _auto_tune_quarantined:
        proposals.append(_proposal(
            issue="class_balance_quarantine",
            section="tracking",
            key="auto_tune",
            prev="normal",
            new="proposal_only",
            reason=(
                f"validation prediction distribution unhealthy: "
                f"pred_counts={_class_diag.get('pred_counts')} "
                f"reason={_class_diag.get('reason')}; block config mutations"
            ),
            confidence="high",
        ))

    def _set(section, key, value):
        if _optuna_applied or _auto_tune_quarantined:
            return
        if data is None or section in _HIGH_RISK or key in _HIGH_RISK:
            return
        if section not in data:
            data[section] = {}
        data[section][key] = value

    # SYS-002: when tune_eval metrics are available, prefer them over val metrics
    # to prevent validation data from leaking into hyperparameter decisions.
    _use_tune_eval = bool(history.get("_tune_eval_isolated")) if isinstance(history, dict) else False
    if _use_tune_eval:
        print("[Auto-Tune] Using isolated tune-eval metrics (SYS-002 three-way split active)")

    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    # HEURISTIC 1 — Overfitting: val_loss >> train_loss at final epoch
    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    if (
        history and isinstance(history, dict)
        and history.get("train_loss") and (history.get("tune_loss") if _use_tune_eval else history.get("val_loss"))
    ):
        t_loss = history["train_loss"][-1]
        v_loss = history["tune_loss"] if _use_tune_eval else history["val_loss"][-1]
        if t_loss and v_loss and v_loss > t_loss * 1.15:
            gen_gap = v_loss - t_loss
            old_do = float(_get("model", "dropout", 0.25))
            new_do = min(0.50, old_do + 0.05)
            proposals.append(_proposal(
                issue       = "overfitting",
                section     = "model",
                key         = "dropout",
                prev        = old_do,
                new         = round(new_do, 2),
                reason      = f"val_loss ({v_loss:.4f}) > train_loss ({t_loss:.4f}) * 1.15 "
                              f"(gen_gap={gen_gap:.4f})",
                confidence  = "high" if gen_gap > t_loss * 0.30 else "medium",
            ))
            _set("model", "dropout", float(f"{new_do:.2f}"))

    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    # HEURISTIC 2 ΓÇö Premature early stopping
    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    if total_epochs > 0 and best_epoch is not None and best_epoch < total_epochs * 0.25:
        old_lr = float(_get("training", "lr", 5e-5))
        new_lr = old_lr * 0.5
        proposals.append(_proposal(
            issue      = "premature_early_stop",
            section    = "training",
            key        = "lr",
            prev       = old_lr,
            new        = float(f"{new_lr:.2e}"),
            reason     = f"best_epoch={best_epoch} < 25% of total={total_epochs}",
            confidence = "medium",
        ))
        _set("training", "lr", float(f"{new_lr:.2e}"))

    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    # HEURISTIC 3 ΓÇö Sharpe collapse: peaked early then degraded
    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    val_sharpe_curve = history.get("val_sharpe", []) if isinstance(history, dict) else []
    if val_sharpe_curve and len(val_sharpe_curve) >= 4:
        peak_val   = max(val_sharpe_curve)
        peak_epoch = val_sharpe_curve.index(peak_val)
        final_val  = val_sharpe_curve[-1]
        collapse   = (peak_val - final_val) / max(abs(peak_val), 1e-9)
        if collapse > 0.20 and peak_epoch < len(val_sharpe_curve) * 0.60:
            # LR was likely too high ΓåÆ reduce warmup peak
            old_lr = float(_get("training", "lr", 5e-5))
            new_lr = max(1e-6, old_lr * 0.70)
            proposals.append(_proposal(
                issue      = "sharpe_collapse",
                section    = "training",
                key        = "lr",
                prev       = old_lr,
                new        = float(f"{new_lr:.2e}"),
                reason     = (f"val_sharpe peaked at epoch {peak_epoch} ({peak_val:.4f}) "
                              f"then collapsed to {final_val:.4f} "
                              f"(drop={collapse:.1%})"),
                confidence = "high" if collapse > 0.40 else "medium",
            ))
            _set("training", "lr", float(f"{new_lr:.2e}"))

            # Also nudge patience down so we stop before the collapse
            old_pat = int(_get("training", "patience", 6))
            new_pat = max(3, min(old_pat, peak_epoch + 2))
            if new_pat != old_pat:
                proposals.append(_proposal(
                    issue      = "sharpe_collapse",
                    section    = "training",
                    key        = "patience",
                    prev       = old_pat,
                    new        = new_pat,
                    reason     = f"stop before collapse; peak at epoch {peak_epoch}",
                    confidence = "medium",
                ))
                _set("training", "patience", new_pat)

    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    # HEURISTIC 4 ΓÇö Gate failure on drawdown
    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    reasons = gate_result.get("reasons", []) if gate_result else []
    if any("drawdown" in str(r).lower() for r in reasons):
        if data and "rl" in data and "reward" in data.get("rl", {}):
            old_pen = float(data["rl"]["reward"].get("drawdown", 0.5))
            new_pen = min(2.0, old_pen + 0.25)
            proposals.append(_proposal(
                issue      = "gate_fail_drawdown",
                section    = "rl.reward",
                key        = "drawdown",
                prev       = old_pen,
                new        = round(new_pen, 2),
                reason     = "promotion gate rejected on drawdown criterion",
                confidence = "medium",
            ))
            if data is not None and not _optuna_applied and not _auto_tune_quarantined:
                data["rl"]["reward"]["drawdown"] = float(f"{new_pen:.2f}")

    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    # HEURISTIC 5 ΓÇö Gate failure on profit factor
    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    if any("profit factor" in str(r).lower() for r in reasons):
        old_tx = float(_get("execution", "slippage_vol_alpha", 0.5))
        new_tx = min(1.0, old_tx + 0.10)
        proposals.append(_proposal(
            issue      = "gate_fail_profit_factor",
            section    = "execution",
            key        = "slippage_vol_alpha",
            prev       = old_tx,
            new        = round(new_tx, 2),
            reason     = "promotion gate rejected on profit factor; tighten slippage model",
            confidence = "low",
        ))
        _set("execution", "slippage_vol_alpha", float(f"{new_tx:.2f}"))

    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    # HEURISTIC 6 ΓÇö Frequent Curriculum Stalls (Noisy Gradients / Hard Data)
    # P3: skipped when Optuna already optimised the curriculum schedule.
    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    stalls_curve = history.get("curriculum_stalls", []) if isinstance(history, dict) else []
    if stalls_curve and stalls_curve[-1] >= 5 and not _optuna_applied and not _auto_tune_quarantined:
        # 1. Increase batch_size to smooth gradients
        old_bs = int(_get("training", "batch_size", 256))
        new_bs = min(512, int(old_bs * 1.25))
        if new_bs != old_bs:
            proposals.append(_proposal(
                issue      = "frequent_stalls",
                section    = "training",
                key        = "batch_size",
                prev       = old_bs,
                new        = new_bs,
                reason     = f"Total stalls reached {stalls_curve[-1]}; increasing batch size to smooth gradients",
                confidence = "medium",
            ))
            _set("training", "batch_size", new_bs)

        # 2. Decrease seq_len to simplify the learning task
        old_seq = int(_get("training", "seq_len", 60))
        new_seq = max(15, int(old_seq * 0.75))
        if new_seq != old_seq:
            proposals.append(_proposal(
                issue      = "frequent_stalls",
                section    = "training",
                key        = "seq_len",
                prev       = old_seq,
                new        = new_seq,
                reason     = f"Total stalls reached {stalls_curve[-1]}; reducing seq_len to simplify the task",
                confidence = "medium",
            ))
            _set("training", "seq_len", new_seq)

            # Disable the seq_schedule to avoid conflicts
            old_sched = _get("curriculum", "seq_schedule", None)
            if old_sched:
                proposals.append(_proposal(
                    issue      = "frequent_stalls",
                    section    = "curriculum",
                    key        = "seq_schedule",
                    prev       = "active",
                    new        = "disabled",
                    reason     = "Disabled sequence schedule because base seq_len was dynamically reduced",
                    confidence = "medium",
                ))
                _set("curriculum", "seq_schedule", [])
    elif stalls_curve and stalls_curve[-1] >= 5 and (_optuna_applied or _auto_tune_quarantined):
        pass  # all mutations blocked when optuna.applied or class balance is quarantined.

    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    # HEURISTIC 7 ΓÇö Perfect Stability (Too Easy / Underfitting)
    # P3: skipped when Optuna already optimised the curriculum schedule.
    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    if stalls_curve and stalls_curve[-1] == 0 and not _optuna_applied and not _auto_tune_quarantined:
        if isinstance(history, dict) and history.get("train_loss") and history.get("val_loss"):
            t_loss = history["train_loss"][-1]
            v_loss = history["val_loss"][-1]
            if t_loss and v_loss and v_loss < t_loss * 1.05:
                old_seq = int(_get("training", "seq_len", 60))
                new_seq = min(120, int(old_seq * 1.25))
                if new_seq != old_seq:
                    proposals.append(_proposal(
                        issue      = "perfect_stability",
                        section    = "training",
                        key        = "seq_len",
                        prev       = old_seq,
                        new        = new_seq,
                        reason     = "Zero stalls and tight gen_gap; increasing seq_len to challenge the model",
                        confidence = "low",
                    ))
                    _set("training", "seq_len", new_seq)

    # ΓöÇΓöÇ always write proposal JSON ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    proposal_doc = {
        "run_name":       run_name,
        "generated_at":   datetime.now(UTC).isoformat(),
        "dry_tune":       dry_tune,
        "optuna_applied": _optuna_applied,   # P3: flag recorded in proposal for auditability
        "class_balance":  _class_diag,
        "auto_tune_quarantined": _auto_tune_quarantined,
        "applied":        (not dry_tune and not _auto_tune_quarantined
                           and data is not None and len(proposals) > 0),
        "n_proposals":    len(proposals),
        "proposals":      proposals,
    }
    try:
        _safe_save_json(proposal_doc, prop_file)
    except Exception:
        try:
            with open(prop_file, "w", encoding="utf-8") as f:
                json.dump(proposal_doc, f, indent=2, default=str)
        except Exception:
            pass

    # ΓöÇΓöÇ print summary ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    if proposals:
        action = "(Dry-Run) Would modify" if dry_tune else "Modified"
        if _auto_tune_quarantined:
            action = "Quarantined; proposal-only"
        print(f"\n[Auto-Tune] {action} {cfg_path.name} for the next run "
              f"({len(proposals)} proposal(s)):")
        for p in proposals:
            print(f"  [{p['confidence'].upper()}] {p['issue']}: "
                  f"{p['section']}.{p['key']} "
                  f"{p['prev_value']} -> {p['new_value']}  ({p['reason']})")
        print(f"  Proposal written -> {prop_file}")
    else:
        print("\n[Auto-Tune] No hyperparameter changes recommended. "
              f"Proposal written -> {prop_file}")

    # ── write back to config (live mode only) ──────────────────────────
    if not dry_tune and not _auto_tune_quarantined and data is not None and proposals:
        try:
            yaml_io = YAML()
            yaml_io.preserve_quotes = True
            with open(versioned_cfg_path, "w", encoding="utf-8") as f:
                yaml_io.dump(data, f)
            print(f"[Auto-Tune] Wrote tuned config -> {versioned_cfg_path}")
        except Exception as e:
            print(f"[Auto-Tune] Failed to write back config: {e}")

def _best_epoch_from_history(history: dict) -> int:

    """Pick the epoch index used by auto-tune without depending on outer locals."""

    if isinstance(history, dict) and history.get("val_sharpe"):

        return int(history["val_sharpe"].index(max(history["val_sharpe"])))

    if isinstance(history, dict) and history.get("val_loss"):

        return int(history["val_loss"].index(min(history["val_loss"])))

    return 0

def _history_for_auto_tune(history_or_folds) -> dict:

    """Normalize single-split or walk-forward histories into one curve dict.

    Walk-forward: use the **last completed fold** only. Concatenating all fold
    curves creates a fake multi-hundred-epoch run and breaks auto-tune heuristics.
    """

    if isinstance(history_or_folds, dict):

        return history_or_folds

    if not isinstance(history_or_folds, list) or not history_or_folds:

        return {}

    last_entry = history_or_folds[-1]

    if isinstance(last_entry, dict) and isinstance(last_entry.get("history"), dict):

        return dict(last_entry["history"])

    return {}

def _evaluate_tune_split(model, cache_path: str, tune_idx: np.ndarray, args,
                         device, amp_dtype) -> dict:
    """SYS-002: Evaluate best model on the held-out tune split.

    Returns a dict with 'tune_loss' and 'tune_sharpe' that can be injected into
    the auto-tune history, preventing val-set reuse for hyperparameter decisions.
    """
    import torch
    from torch.utils.data import DataLoader

    model.eval()
    tune_idx_sorted = np.sort(tune_idx)
    seq_len = int(getattr(args, "seq_len", 64) or 64)
    batch_size = int(getattr(args, "batch_size", 256) or 256)

    try:
        tune_ds = ZarrStreamDataset(
            cache_path, tune_idx_sorted,
            multitask_targets=bool(getattr(args, "classification", False) or getattr(args, "multitask", False)),
        )
        tune_loader = DataLoader(tune_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=0, pin_memory=False)
    except Exception as e:
        print(f"[TuneEval] Could not create tune dataloader: {e}")
        return {}

    criterion = torch.nn.CrossEntropyLoss() if getattr(args, "classification", False) else torch.nn.MSELoss()
    total_loss = 0.0
    total_samples = 0
    all_returns = []

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
        for batch in tune_loader:
            x = batch[0].to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True)
            try:
                out = model(x)
                if isinstance(out, tuple):
                    out = out[0]
                loss = criterion(out.squeeze(-1) if out.dim() > y.dim() else out, y)
                bs = x.size(0)
                total_loss += loss.item() * bs
                total_samples += bs
                if not getattr(args, "classification", False):
                    all_returns.append(out.squeeze(-1).cpu())
            except Exception:
                continue

    if total_samples == 0:
        return {}

    tune_loss = total_loss / total_samples
    tune_sharpe = 0.0
    if all_returns:
        rets = torch.cat(all_returns)
        std = rets.std()
        if std > 1e-8:
            tune_sharpe = float((rets.mean() / std) * (252 ** 0.5))

    print(f"[TuneEval] tune_loss={tune_loss:.6f}  tune_sharpe={tune_sharpe:.4f}  "
          f"samples={total_samples:,}")
    return {"tune_loss": tune_loss, "tune_sharpe": tune_sharpe, "tune_samples": total_samples}

def _metric_from_gate(details: dict, key: str, default=None):

    for candidate in (key, f"forward_{key}", f"holdout_{key}"):

        if isinstance(details, dict) and candidate in details:

            return details.get(candidate)

    return default

def _append_model_comparison_report(args, model_name: str, train_summary: dict, gate_result: dict) -> None:

    """Write a shared same-forward-holdout comparison file across model runs."""

    root = Path(args.checkpoint_dir)

    path = root / "model_comparison.json"

    existing = _read_json_dict(path)

    details = gate_result.get("details", {}) if isinstance(gate_result, dict) else {}

    row = {

        "model_name": model_name,

        "recipe_name": getattr(args, "recipe_name", None),

        "loss": getattr(args, "loss", None),

        "seq_len": int(getattr(args, "seq_len", 0) or 0),

        "pretrain_enabled": bool(getattr(args, "pretrain", False)),

        "feature_ablation": (_feature_ablation_config(args).get("name") or "full_features"),

        "validation": {

            "best_val_sharpe": train_summary.get("best_val_sharpe"),

            "best_val_loss": train_summary.get("best_val_loss"),

            "final_val_sharpe": train_summary.get("final_val_sharpe"),

            "gen_gap_final": train_summary.get("gen_gap_final"),

        },

        "forward_holdout": {

            "promoted": bool(gate_result.get("promoted", False)) if isinstance(gate_result, dict) else False,

            "summary": gate_result.get("summary") if isinstance(gate_result, dict) else None,

            "sharpe": _metric_from_gate(details, "sharpe"),

            "profit_factor": _metric_from_gate(details, "profit_factor"),

            "max_drawdown": _metric_from_gate(details, "max_drawdown"),

            "n_trades": _metric_from_gate(details, "n_trades"),

            "forward_window": details.get("forward_window") if isinstance(details, dict) else None,

            "reasons": gate_result.get("reasons", []) if isinstance(gate_result, dict) else [],

        },

        "updated_at": datetime.now(UTC).isoformat(),

    }



    rows = [r for r in existing.get("models", []) if r.get("model_name") != model_name]

    rows.append(row)



    def _score(r):

        fwd = r.get("forward_holdout", {})

        val = r.get("validation", {})

        sharpe = fwd.get("sharpe")

        if sharpe is None:

            sharpe = val.get("best_val_sharpe")

        try:

            return float(sharpe)

        except Exception:

            return float("-inf")



    rows = sorted(rows, key=_score, reverse=True)

    report = {

        "run_name": getattr(args, "run_name", None),

        "checkpoint_dir": str(root),

        "comparison_basis": "same promotion forward holdout window from config/promote_forward_frac",

        "models": rows,

        "leader": rows[0].get("model_name") if rows else None,

        "updated_at": datetime.now(UTC).isoformat(),

    }

    _safe_save_json(report, path)

    print(f"[ModelComparison] Updated -> {path}")

def _maybe_auto_tune_next_run(

    args,

    history: dict,

    gate_result: dict | None = None,

    *,

    phase: str = "main",

    model_name: str = "model",

    force_dry: bool = False,

) -> None:

    """Centralized auto-tune entrypoint for every training phase.



    Baseline/pretrain-ablation phases write proposal artifacts too, but force

    dry-run mode so comparison runs do not mutate config before the main model

    has trained and passed through promotion.

    """

    _ensure_bound()
    if not bool(getattr(args, "auto_tune", True)):

        print(f"[Auto-Tune] Disabled for {model_name}/{phase} (--no-auto-tune or tracking.auto_tune=false)")

        return

    hist = history if isinstance(history, dict) else {}

    best_epoch = _best_epoch_from_history(hist)

    total_epochs = len(hist.get("train_loss", [])) or int(getattr(args, "epochs", 0) or 0)

    run_base = str(getattr(args, "run_name_slug", "") or getattr(args, "run_name", "unknown_run") or "unknown_run")

    phase_run_name = _slug_part(f"{run_base}_{model_name}_{phase}", max_len=180)

    _auto_tune_next_run(

        getattr(args, "config", "config/run.yaml"),

        hist,

        gate_result or {"promoted": True, "reasons": []},

        best_epoch=best_epoch,

        total_epochs=total_epochs,

        run_name=phase_run_name,

        dry_tune=bool(
            force_dry
            or getattr(args, "dry_tune", False)
            or getattr(args, "all_models", False)
        ),

    )
