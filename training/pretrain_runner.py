"""Contrastive / multi-task pretraining runner.\n\nSee docs/CONTINUE.md."""
from __future__ import annotations

import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from config.settings import PRETRAIN
from training.direction_control import _coerce_auto_int
from pretrain.contrastive import (
    BYOLTrainer,
    MaskedReconstructionTrainer,
    RegimeAwareTSCLTrainer,
    RepresentationCollapseError,
    TimeSeriesAugmenter,
    TSCLTrainer,
)
from pretrain.extended_trainers import (
    ClusterContrastiveTrainer,
    DriftContrastiveTrainer,
    ForecastPretextTrainer,
    VAESeqTrainer,
)
from pretrain.hard_example_mining import PretrainHardExampleMiner

_HOST = None
_BOUND = False
_HOST_DEPS = (
'_log_error',
    '_log_warn',
    '_log_info',
    '_log_nan',
    '_log_oom',
    '_crop_to_seq_len',
    '_core_model',
    '_strict_load_report',
    '_on_disk_sequence_count',
    '_clamp_n_samples_to_disk',
    '_x_path',
    '_y_path',
    '_zarr_open_group',
    'ZARR',
    'ZarrStreamDataset',
    '_ThreadPrefetchLoader',
    'build_model',
    '_model_build_args',
    '_apply_model_profile',
    '_embargo_bars',
    '_purge_bars',
    '_slug_part',
    '_safe_wandb_log',
    '_safe_save',
    '_safe_save_json',
    'WANDB',
    'PATHS',
    'LABELING',
    'FEATURES',
    '_TRAIN_LOGGER',
    '_sharpe_ann_factor',
    '_pbar',
    '_trainable_max_index',
    '_load_diff_array',
    '_promotion_holdout_n',
    '_multitask_head_in',
    'TimeSeriesAugmenter',
    'BYOLTrainer',
    'MaskedReconstructionTrainer',
    'RegimeAwareTSCLTrainer',
    'TSCLTrainer',
    'VAESeqTrainer',
    'ForecastPretextTrainer',
    'ClusterContrastiveTrainer',
    'DriftContrastiveTrainer',
    'RepresentationCollapseError',
    'pretrain_multi_task',
    'create_multi_task_pretrainer'
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

_PRETRAIN_SINGLE_PASS = frozenset({"byol", "masked", "vae", "forecast", "drift"})
_PRETRAIN_MULTI_BLOCK = _PRETRAIN_SINGLE_PASS
_PRETRAIN_STD_QUALITY = _PRETRAIN_SINGLE_PASS
_VALID_PRETRAIN_METHODS = _PRETRAIN_SINGLE_PASS | {"tscl", "cluster"}


def _normalize_pretrain_method(method: str) -> str:
    _ensure_bound()
    aliases = {
        "autoencoder": "vae",
        "regime_cluster": "cluster",
        "cluster_tscl": "cluster",
        "drift_pretrain": "drift",
    }
    return aliases.get(str(method or "byol").lower(), str(method or "byol").lower())

def _pretrain_channel_chunk(args, n_features: int) -> int | None:
    """Per-pair feature block size for channel-shuffle augmentation."""
    fpp = getattr(args, "_f_per_pair", None)
    if fpp is not None and int(fpp) > 0:
        embed = int(getattr(args, "pair_embed_dim", 0) or 0)
        n_pairs = int(getattr(args, "_n_pairs", 1) or 1)
        if n_pairs > 1 and embed > 0:
            return int(fpp) + embed
        return int(fpp)
    return None

def _make_pretrain_augmenter(args, n_features: int) -> TimeSeriesAugmenter:
    """Build augmenter from PRETRAIN defaults + optional YAML overrides."""
    _ensure_bound()
    aug_cfg = getattr(args, "pretrain_augmentations", None) or PRETRAIN.get("augmentations") or {}
    scale_rng = aug_cfg.get("scaling_range", (0.8, 1.2))
    if isinstance(scale_rng, list):
        scale_rng = tuple(scale_rng)
    crop_rng = aug_cfg.get("crop_ratio", (0.7, 1.0))
    if isinstance(crop_rng, list):
        crop_rng = tuple(crop_rng)
    return TimeSeriesAugmenter(
        jitter_std=float(aug_cfg.get("jitter_std", 0.02)),
        scale_range=scale_rng,
        feature_drop_p=float(aug_cfg.get("feature_drop_p", 0.3)),
        crop_ratio=crop_rng,
        seed=getattr(args, "seed", None),
        channel_chunk=_pretrain_channel_chunk(args, n_features),
    )

def _make_pretrain_span_plan(
    n_total: int,
    n_windows: int,
    *,
    diff: np.ndarray | None = None,
    max_spans: int = 8,
    rng: np.random.Generator | None = None,
) -> list[tuple[int, int]]:
    """
    Plan chunk-friendly pretrain reads as multiple contiguous spans.

    If difficulty labels are available, starts are selected across difficulty
    buckets before falling back to timeline-spread spans. The return value is a
    list of (start, length) slices sorted by start for sequential-ish reads.
    """
    n_total = max(0, int(n_total))
    n_windows = min(max(1, int(n_windows)), n_total) if n_total else 0
    if n_windows <= 0:
        return []
    rng = rng or np.random.default_rng()
    max_spans = max(1, min(int(max_spans), n_windows, n_total))
    span_len = max(1, int(math.ceil(n_windows / max_spans)))

    starts: list[int] = []
    if diff is not None:
        try:
            d = np.asarray(diff[:n_total], dtype=np.uint8)
            buckets = [np.flatnonzero(d == level) for level in (0, 1, 2)]
            buckets = [b for b in buckets if len(b)]
            if buckets:
                while len(starts) < max_spans:
                    bucket = buckets[len(starts) % len(buckets)]
                    anchor = int(bucket[int(rng.integers(0, len(bucket)))])
                    starts.append(max(0, min(anchor - span_len // 2, n_total - span_len)))
        except Exception:
            starts = []

    if not starts:
        if max_spans == 1:
            starts = [int(rng.integers(0, max(1, n_total - span_len + 1)))]
        else:
            edges = np.linspace(0, max(0, n_total - span_len), num=max_spans, dtype=int)
            jitter = max(1, span_len // 2)
            starts = [
                int(np.clip(edge + int(rng.integers(-jitter, jitter + 1)), 0, max(0, n_total - span_len)))
                for edge in edges
            ]

    spans: list[tuple[int, int]] = []
    remaining = n_windows
    for start in sorted(dict.fromkeys(starts)):
        if remaining <= 0:
            break
        length = min(span_len, remaining, n_total - start)
        if length > 0:
            spans.append((int(start), int(length)))
            remaining -= length

    while remaining > 0:
        length = min(span_len, remaining)
        start = int(rng.integers(0, max(1, n_total - length + 1)))
        spans.append((start, length))
        remaining -= length

    return sorted(spans, key=lambda x: x[0])

def _read_pretrain_spans(
    x_reader,
    y_reader,
    spans: list[tuple[int, int]],
    *,
    seq_len: int,
    n_features: int,
    progress_desc: str = "[Pretrain] Loading spans",
) -> tuple[np.ndarray, np.ndarray]:
    total = int(sum(length for _, length in spans))
    w_out = np.zeros((total, int(seq_len), int(n_features)), dtype=np.float32)
    y_out = np.zeros(total, dtype=np.float32)
    pos = 0
    read_step = max(1, int(PRETRAIN.get("read_windows", 64)))
    with _pbar(total=total, desc=progress_desc, unit="win") as pb:
        for start, length in spans:
            span_end = start + length
            cursor = start
            while cursor < span_end:
                end = min(span_end, cursor + read_step)
                chunk_len = end - cursor
                w_chunk = _crop_to_seq_len(np.asarray(x_reader[cursor:end]), seq_len).copy()
                np.nan_to_num(w_chunk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                w_out[pos:pos + chunk_len] = w_chunk
                y_chunk = np.asarray(y_reader[cursor:end], dtype=np.float32).copy()
                np.nan_to_num(y_chunk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                y_out[pos:pos + chunk_len] = y_chunk
                pos += chunk_len
                cursor = end
                pb.update(chunk_len)
                if _TRAIN_LOGGER:
                    _TRAIN_LOGGER.heartbeat()
    return w_out, y_out

def _select_pretrain_trainer_class(method: str, regime_aware: bool):
    method = _normalize_pretrain_method(method)
    if method == "masked":
        return MaskedReconstructionTrainer
    if method == "tscl":
        return RegimeAwareTSCLTrainer if regime_aware else TSCLTrainer
    if method == "vae":
        return VAESeqTrainer
    if method == "cluster":
        return ClusterContrastiveTrainer
    if method == "forecast":
        return ForecastPretextTrainer
    if method == "drift":
        return DriftContrastiveTrainer
    return BYOLTrainer

def _parse_pretrain_ablation_models(value) -> set[str]:

    """Return model names that should run no-pretrain proof when ablation=auto."""

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

# -----------------------------------------------------------------------------
# CONTRASTIVE PRE-TRAINING
# -----------------------------------------------------------------------------

def _pretrain_report_path(args) -> Path:
    return Path(getattr(args, "checkpoint_dir", ".")) / "pretrain_report.json"


def _read_json_dict(path: Path) -> dict:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"[Pretrain] WARN: could not read {path}: {exc}")
    return {}


def _update_pretrain_report(args, updates: dict) -> None:
    _ensure_bound()
    path = _pretrain_report_path(args)
    report = _read_json_dict(path)
    report.update(updates or {})
    report["updated_at"] = datetime.now(UTC).isoformat()
    try:
        _safe_save_json(report, path)
    except NameError:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def _recommended_pretrain_method(model_name: str) -> str:
    name = str(model_name or "").lower()
    if "haelt" in name:
        return "masked_or_byol"
    if "mamba" in name:
        return "forecast_or_drift"
    if "tft" in name:
        return "masked"
    if "gnn" in name:
        return "cluster"
    if "expert" in name:
        return "tscl"
    if "transformer" in name:
        return "byol_or_tscl"
    return "masked"


def _fold_history_summary(folds: list | None, metric_name: str = "sharpe") -> dict:
    folds = folds or []
    best_metrics = []
    best_sharpes = []
    best_losses = []
    final_sharpes = []
    final_losses = []
    final_dir_acc = []
    final_gaps = []

    for entry in folds:
        if not isinstance(entry, dict):
            continue
        metric = entry.get("best_metric")
        if metric is not None:
            best_metrics.append(float(metric))
        hist = entry.get("history", {})
        if not isinstance(hist, dict):
            continue
        sharpe = hist.get("val_sharpe") or []
        loss = hist.get("val_loss") or []
        train_loss = hist.get("train_loss") or []
        dir_acc = hist.get("dir_acc") or []
        if sharpe:
            best_sharpes.append(float(max(sharpe)))
            final_sharpes.append(float(sharpe[-1]))
        if loss:
            best_losses.append(float(min(loss)))
            final_losses.append(float(loss[-1]))
        if dir_acc:
            final_dir_acc.append(float(dir_acc[-1]))
        if loss and train_loss:
            final_gaps.append(float(loss[-1]) - float(train_loss[-1]))

    maximize_metric = str(metric_name).lower() == "sharpe"
    best_metric = None
    if best_metrics:
        best_metric = max(best_metrics) if maximize_metric else min(best_metrics)
    return {
        "fold_count": len(folds),
        "best_metric": best_metric,
        "mean_best_metric": float(np.mean(best_metrics)) if best_metrics else None,
        "best_val_sharpe": float(max(best_sharpes)) if best_sharpes else None,
        "mean_best_val_sharpe": float(np.mean(best_sharpes)) if best_sharpes else None,
        "best_val_loss": float(min(best_losses)) if best_losses else None,
        "mean_best_val_loss": float(np.mean(best_losses)) if best_losses else None,
        "final_val_sharpe": float(np.mean(final_sharpes)) if final_sharpes else None,
        "final_val_loss": float(np.mean(final_losses)) if final_losses else None,
        "final_dir_acc": float(np.mean(final_dir_acc)) if final_dir_acc else None,
        "final_train_val_gap": float(np.mean(final_gaps)) if final_gaps else None,
    }


def _pretrain_ablation_verdict(baseline: dict, pretrained: dict) -> tuple[str, dict]:
    def _delta(key):
        a = pretrained.get(key)
        b = baseline.get(key)
        return (float(a) - float(b)) if a is not None and b is not None else None

    deltas = {
        "best_metric": _delta("best_metric"),
        "best_val_sharpe": _delta("best_val_sharpe"),
        "mean_best_val_sharpe": _delta("mean_best_val_sharpe"),
        "best_val_loss": _delta("best_val_loss"),
        "mean_best_val_loss": _delta("mean_best_val_loss"),
        "final_dir_acc": _delta("final_dir_acc"),
        "final_train_val_gap": _delta("final_train_val_gap"),
    }
    sharpe_delta = deltas.get("best_val_sharpe")
    loss_delta = deltas.get("best_val_loss")
    gap_delta = deltas.get("final_train_val_gap")
    if sharpe_delta is None and loss_delta is None:
        verdict = "unknown"
    elif (sharpe_delta is not None and sharpe_delta > 0.0) and (
        loss_delta is None or loss_delta <= 0.0
    ) and (gap_delta is None or gap_delta <= 0.02):
        verdict = "pretrain_helped"
    elif (sharpe_delta is not None and sharpe_delta < 0.0) or (
        loss_delta is not None and loss_delta > 0.0 and (sharpe_delta is None or sharpe_delta <= 0.0)
    ):
        verdict = "pretrain_hurt"
    else:
        verdict = "mixed"
    return verdict, deltas


def _run_multi_task_pretrain(model, windows, ckpt, n_features, args, device):
    """
    Pretrain the model backbone with the multi-task pretrainer (Improvement #3).

    Samples one window block (already loaded by the caller), runs contrastive +
    masked-recon + forecast + optional domain-adaptation pretraining via
    ``pretrain_multi_task``, copies compatible encoder weights into the model
    backbone, and saves a ``model_state`` checkpoint at ``ckpt`` so the standard
    supervised-transfer path can load it.
    """
    _ensure_bound()
    from pretrain.multi_task import pretrain_multi_task

    _epochs = max(1, int(getattr(args, "pretrain_epochs", 30) or 30))
    _bs = max(4, min(int(getattr(args, "pretrain_batch", 256) or 256), int(n_features) * args.seq_len, 2048))
    _bs = max(4, _bs // 8 * 8) if _bs > 8 else _bs
    print(f"[Pretrain] Multi-task pretrainer (Improvement #3) | epochs={_epochs} "
          f"batch={_bs} windows={len(windows)}")

    try:
        trainer, history = pretrain_multi_task(
            windows,
            seq_len=args.seq_len,
            n_features=n_features,
            epochs=_epochs,
            batch_size=_bs,
            device=device,
            silent=False,
        )
    except Exception as exc:
        print(f"[Pretrain] Multi-task pretrainer failed ({exc}); falling back to "
              f"built-in pretrain.")
        return None

    target = model.backbone if hasattr(model, "backbone") else model
    if hasattr(target, "module"):
        target = target.module
    _enc_state = trainer.encoder.state_dict()
    try:
        _missing, _unexpected = target.load_state_dict(_enc_state, strict=False)
        _frac = 1.0 - (len(_missing) / max(1, len(_enc_state)))
        print(f"[Pretrain] Multi-task encoder → backbone | loaded={_frac:.0%} "
              f"missing={len(_missing)} unexpected={len(_unexpected)}")
    except Exception as exc:
        print(f"[Pretrain] Encoder copy skipped ({exc}).")

    Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": _enc_state, "method": "multi_task"}, ckpt)
    print(f"[Pretrain] Saved multi-task encoder checkpoint → {ckpt}")

    try:
        _update_pretrain_report(args, {
            "status": "completed",
            "method": "multi_task",
            "epochs": int(_epochs),
            "checkpoint_path": str(ckpt),
            "final_loss": float(history.get("loss", 0.0)) if isinstance(history, dict) else 0.0,
        })
    except Exception as exc:
        print(f"[Pretrain] WARN: failed to update multi_task pretrain report: {exc}")
    return model


def _run_pretrain_via_adapter(model, cache_path, n_features, args, device,
                              framework="lightly", run=None):
    """Opt-in pretraining fast path: route through ``create_pretrain_adapter``.

    Used only when ``--pretrain-framework`` is explicitly non-"custom". The
    adapter trains its own encoder (e.g. TS2Vec/Lightly) over the trainable
    window and returns a metrics dict; the supervised ``model`` is returned
    unchanged (external frameworks do not consume the in-house encoder).
    """
    from training.pretrain_adapter import PretrainConfig, create_pretrain_adapter, run_pretrain_with_adapter
    from training.cache_integrity import _zarr_open_group

    _method = str(getattr(args, "pretrain_method", PRETRAIN.get("method", "byol"))).lower()
    print(f"\n[Pretrain] framework={framework} | method={_method} | model={getattr(args, 'model', None)}")

    if not (ZARR and cache_path.endswith(".zarr") and Path(cache_path).is_dir()):
        raise RuntimeError(
            "[Pretrain] adapter path requires a .zarr cache; either rebuild the cache "
            "as zarr or run with --pretrain-framework custom."
        )

    _z = _zarr_open_group(cache_path, mode="r")
    n_total = min(int(_z["X"].shape[0]), int(_z["y"].shape[0]))
    _source_n_total = int(n_total)
    _cap = int(_trainable_max_index(n_total, args))
    if 0 < _cap < n_total:
        n_total = _cap
        print(f"[Pretrain] Holdout-safe index cap: {n_total:,} trainable windows")

    _holdout_n = _promotion_holdout_n(_source_n_total, args)
    if not bool(n_total <= max(0, _source_n_total - _holdout_n)):
        raise RuntimeError(
            f"Pretrain window overlaps promotion holdout "
            f"(pretrain_end={n_total}, holdout_start={max(0, _source_n_total - _holdout_n)})"
        )

    train_indices = np.arange(n_total, dtype=np.int64)
    seq_len = int(args.seq_len)
    _ckpt_dir = Path(args.checkpoint_dir)
    _ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg = PretrainConfig(
        input_dims=int(n_features),
        output_dims=int(getattr(args, "pretrain_embed_dim", 320) or 320),
        hidden_dims=int(getattr(args, "pretrain_hidden_dim", 64) or 64),
        batch_size=int(getattr(args, "pretrain_batch", PRETRAIN.get("pretrain_batch", 256))),
        max_epochs=max(1, int(getattr(args, "pretrain_epochs", 30) or 30)),
        max_train_length=int(seq_len) if seq_len > 0 else None,
        device=str(device),
        save_path=str(_ckpt_dir / f"pretrain_{framework}_encoder.pt"),
        verbose=True,
    )
    adapter = create_pretrain_adapter(framework, cfg)

    try:
        results = run_pretrain_with_adapter(
            adapter, cache_path, train_indices,
            method=_method,
            seq_len=seq_len,
            n_features=int(n_features),
        )
    except Exception as _ad_exc:
        print(f"[Pretrain] Adapter run failed: {_ad_exc}")
        raise

    if getattr(adapter, "model", None) is not None and hasattr(adapter, "save"):
        try:
            adapter.save(cfg.save_path)
            print(f"[Pretrain] Saved adapter encoder -> {cfg.save_path}")
        except Exception as _save_exc:
            print(f"[Pretrain] Adapter encoder save skipped: {_save_exc}")

    try:
        _update_pretrain_report(args, {
            "model_name": getattr(args, "model", None),
            "pretrain_enabled": True,
            "status": "completed",
            "method": _method,
            "framework": framework,
            "cache_path": str(cache_path),
            "source_windows": int(_source_n_total),
            "trainable_windows_used_by_pretrain": int(n_total),
            "adapter_metrics": results,
            "checkpoint_path": str(cfg.save_path),
            "loads_into_supervised_model": False,
        })
    except Exception as _rp_err:
        print(f"[Pretrain] WARN: failed to update pretrain report: {_rp_err}")
    print(f"[Pretrain] Adapter run complete | framework={framework} method={_method} | windows={n_total:,}")
    return model


def run_pretrain(model, cache_path, n_features, args, device, run=None):
    _ensure_bound()
    _pf = str(getattr(args, "pretrain_framework", "custom") or "custom").lower()
    if _pf != "custom":
        try:
            return _run_pretrain_via_adapter(
                model, cache_path, n_features, args, device,
                framework=_pf, run=run,
            )
        except Exception as _pf_exc:
            print(f"[Pretrain] Adapter path unavailable ({_pf_exc}); falling back to built-in trainer.")
    _ensure_bound()
    _method = _normalize_pretrain_method(
        str(getattr(args, "pretrain_method", PRETRAIN.get("method", "byol"))).lower()
    )
    if _method not in _VALID_PRETRAIN_METHODS:
        print(f"[Pretrain] WARN: unknown method={_method!r}; falling back to byol")
        _method = "byol"
    use_regime = getattr(args, "pretrain_regime", False) and _method == "tscl"
    _mode_labels = {
        "byol": "BYOL",
        "masked": "MaskedRecon",
        "vae": "VAE",
        "cluster": "ClusterTSCL",
        "forecast": "ForecastPretext",
        "drift": "DriftContrastive",
        "tscl": "RegimeAware-TSCL" if use_regime else "TSCL",
    }
    mode_str = _mode_labels.get(_method, "BYOL")
    target_epochs = max(1, int(getattr(args, "pretrain_epochs", 1)))
    max_epochs = int(getattr(args, "pretrain_max_epochs", 0) or 0)
    if max_epochs > 0:
        target_epochs = min(target_epochs, max_epochs)
    min_epochs = max(0, int(getattr(args, "pretrain_min_epochs", 0)))
    handoff_patience = max(0, int(getattr(args, "pretrain_handoff_patience", 0)))
    handoff_min_delta = float(getattr(args, "pretrain_handoff_min_delta", 0.0))
    handoff_loss = float(getattr(args, "pretrain_handoff_loss", float("-inf")))
    handoff_enabled = handoff_patience > 0 or handoff_loss > float("-inf")
    print(f"\n[Pretrain] {mode_str} | target_epochs={target_epochs}"
          f"{' (handoff enabled)' if handoff_enabled else ''}")

    # Reduce VRAM fragmentation (recommended by PyTorch OOM diagnostics)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    encoder = model.backbone if hasattr(model, "backbone") else model

    # --force-pretrain: wipe previous encoder checkpoint so we start fresh
    if getattr(args, "force_pretrain", False):
        for suffix in ("", "_regime"):
            old = Path(args.checkpoint_dir) / f"contrastive_encoder{suffix}.pt"
            if old.exists():
                old.unlink()
                print(f"[Pretrain] Deleted old checkpoint: {old}")

    # Cap windows-per-block to available RAM unless explicitly configured.
    # Reads are planned as several contiguous spans so zarr/memmap I/O remains
    # chunk-friendly while each epoch sees more of the timeline than one slice.
    _bytes_per_window = int(n_features) * args.seq_len * 4
    try:
        import psutil as _psutil
        _free = _psutil.virtual_memory().available
    except Exception:
        _free = 4 * 1024 ** 3                  # conservative fallback: 4 GiB
    _ram_budget = max(1 * 1024 ** 3, int(_free * 0.75))  # use 75% of free RAM
    auto_windows = min(100_000, max(512, _ram_budget // _bytes_per_window))
    n_windows = _coerce_auto_int(
        getattr(args, "pretrain_sample_windows", "auto"),
        auto_windows,
        minimum=128,
    )
    _seed = getattr(args, "seed", None)
    _rng = np.random.default_rng(_seed)

    if ZARR and cache_path.endswith(".zarr") and Path(cache_path).is_dir():
        _z      = _zarr_open_group(cache_path, mode="r")
        n_total = min(int(_z["X"].shape[0]), int(_z["y"].shape[0]))
        X_reader, y_reader = _z["X"], _z["y"]
    else:
        X_mmap   = np.load(_x_path(cache_path), mmap_mode="r")
        y_mmap   = np.load(_y_path(cache_path), mmap_mode="r")
        n_total = min(len(X_mmap), len(y_mmap))
        X_reader, y_reader = X_mmap, y_mmap

    _source_n_total = int(n_total)
    _pretrain_cap = _trainable_max_index(n_total, args)
    if 0 < _pretrain_cap < n_total:
        n_total = _pretrain_cap
        print(f"[Pretrain] Holdout-safe index cap: {n_total:,} trainable windows")

    n_windows = min(int(n_windows), int(n_total))
    diff_for_sampling = _load_diff_array(str(cache_path), n_total)
    _hard_examples_injected = 0

    def _sample_pretrain_block(desc: str = "[Pretrain] Loading spans"):
        nonlocal _hard_examples_injected
        spans = _make_pretrain_span_plan(
            n_total,
            n_windows,
            diff=diff_for_sampling,
            max_spans=8 if n_windows >= 1024 else 4,
            rng=_rng,
        )

        # Hard-example mining integration
        try:
            import json
            _he_path = Path("logs/hard_examples.json")
            if _he_path.exists():
                _he_data = json.loads(_he_path.read_text(encoding="utf-8"))
                _he_indices = _he_data.get("indices", [])
                if _he_indices:
                    # FIX: Validate hard example indices belong to trainable window
                    # (not holdout or embargo) to prevent data leakage
                    _trainable_end = int(n_total)  # n_total already capped by _trainable_max_index
                    _valid_he = [i for i in _he_indices if 0 <= i < _trainable_end]
                    if len(_valid_he) < len(_he_indices):
                        print(f"[Pretrain] Discarded {len(_he_indices) - len(_valid_he)} hard examples outside trainable window (leakage prevention)")
                    _he_spans = [(i, i+1) for i in _valid_he[:int(n_windows * 0.2)]]  # max 20% hard examples
                    if _he_spans:
                        print(f"[Pretrain] Injected {len(_he_spans):,} hard examples from {len(_he_indices):,} total ({len(_valid_he)} valid).")
                        _hard_examples_injected += len(_he_spans)
                        spans.extend(_he_spans)
        except Exception as _he_e:
            print(f"[Pretrain] Hard example reuse failed: {_he_e}")

        return _read_pretrain_spans(
            X_reader, y_reader, spans,
            seq_len=args.seq_len,
            n_features=n_features,
            progress_desc=desc,
        )

    windows, y_sample = _sample_pretrain_block()

    print(f"[Pretrain] Sampled {len(windows):,} windows | shape {windows.shape[1:]}")

    if getattr(args, "use_multi_task_pretrainer", False):
        _ckpt_path = str(Path(args.checkpoint_dir) / "contrastive_encoder.pt")
        if getattr(args, "resume", False) and os.path.exists(_ckpt_path):
            print(f"[Pretrain] Resume: loading existing multi-task encoder {Path(_ckpt_path).name}")
            return model
        _mt_model = _run_multi_task_pretrain(
            model, windows, _ckpt_path, n_features, args, device,
        )
        if _mt_model is not None:
            return _mt_model
        print("[Pretrain] Multi-task pretrainer unavailable; continuing with built-in trainer.")

    ckpt   = str(Path(args.checkpoint_dir) / f"contrastive_encoder{'_regime' if use_regime else ''}.pt")
    _holdout_n = _promotion_holdout_n(_source_n_total, args)
    _embargo_n = _embargo_bars(args)
    _update_pretrain_report(args, {
        "model_name": getattr(args, "model", None),
        "pretrain_enabled": True,
        "status": "started",
        "method": _method,
        "recommended_method_for_model": _recommended_pretrain_method(getattr(args, "model", "")),
        "regime_aware": bool(use_regime),
        "cache_path": str(cache_path),
        "source_windows": int(_source_n_total),
        "trainable_windows_used_by_pretrain": int(n_total),
        "pretrain_window": {"start_index": 0, "end_index_exclusive": int(n_total)},
        "supervised_window": {
            "start_index": 0,
            "end_index_exclusive": int(max(0, _source_n_total - _holdout_n)),
            "embargo_bars": int(_embargo_n),
        },
        "promotion_holdout_window": {
            "start_index": int(max(0, _source_n_total - _holdout_n)),
            "end_index_exclusive": int(_source_n_total),
            "reserved_windows": int(_holdout_n),
        },
        "holdout_safe": bool(n_total <= max(0, _source_n_total - _holdout_n)),
        "sample_windows_per_block": int(n_windows),
        "loaded_into_supervised_training": False,
    })
    try:
        from pretrain.guardrails import PretrainGuardrails
        PretrainGuardrails().enforce_no_holdout_leakage(
            (0, int(n_total)),
            (int(max(0, _source_n_total - _holdout_n)), int(_source_n_total)),
        )
    except RuntimeError:
        raise
    except Exception as _gr_err:
        print(f"[Pretrain] Guardrail check skipped ({_gr_err})")
    if not bool(n_total <= max(0, _source_n_total - _holdout_n)):
        raise RuntimeError(
            f"Pretrain window overlaps promotion holdout "
            f"(pretrain_end={n_total}, holdout_start={max(0, _source_n_total - _holdout_n)})"
        )
    if getattr(args, "resume", False) and os.path.exists(ckpt):
        print(f"[Pretrain] Resume: skipping, loading existing checkpoint {Path(ckpt).name}")
        # A-H2: load with a report + assertion instead of silently swallowing
        # the exception (a failed load here would leave a random encoder).
        _enc = encoder.module if hasattr(encoder, "module") else encoder
        try:
            _state = torch.load(ckpt, map_location=device, weights_only=True)
        except Exception:
            _state = torch.load(ckpt, map_location=device, weights_only=True)
        _strict_load_report(_enc, _state, "PretrainResume", min_frac_loaded=0.6)
        _update_pretrain_report(args, {
            "status": "resume_loaded_existing",
            "checkpoint_path": str(ckpt),
            "quality_gate_result": "loaded_existing_checkpoint",
        })
        return model
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    # Compute a VRAM-safe batch size for the contrastive trainer.
    # With no_grad on pos/neg views, only 1 encoder pass retains activation graphs.
    # But the LSTM still allocates large hidden-state buffers proportional to
    # input_size ├ù hidden_size ├ù seq_len, so we budget conservatively.
    try:
        import torch as _torch
        _torch.cuda.synchronize(device)
        torch.cuda.empty_cache()   # reclaim any fragmented VRAM before pretraining
        _vram_total = _torch.cuda.get_device_properties(device).total_memory
        _vram_free = _vram_total - _torch.cuda.memory_allocated(device)
    except Exception:
        _vram_total = 8 * 1024 ** 3
        _vram_free = 4 * 1024 ** 3
    _bytes_per_sample = args.seq_len * n_features * 4     # FP32 input tensors
    # Single-pass methods: 1 grad pass. Cluster: 2. TSCL: 2ΓÇô3.
    if _method in _PRETRAIN_SINGLE_PASS:
        _n_passes = 1
        _safety_factor = 15
    elif _method == "cluster":
        _n_passes = 2
        _safety_factor = 20
    else:
        _n_passes = 3 if use_regime else 2
        _safety_factor = 30 if use_regime else 20
    _pt_bs_vram = max(16, int(_vram_free * 0.25) // (_bytes_per_sample * _n_passes * _safety_factor))
    _pt_bs_vram = (_pt_bs_vram // 8) * 8
    _vram_gb = _vram_total / (1024 ** 3)
    if _vram_gb <= 10:
        _max_bs = 256 if _method in _PRETRAIN_SINGLE_PASS else (128 if not use_regime else 64)
    elif _vram_gb <= 16:
        _max_bs = 512 if _method in _PRETRAIN_SINGLE_PASS else (256 if not use_regime else 128)
    else:
        _max_bs = 1024 if _method in _PRETRAIN_SINGLE_PASS else (512 if not use_regime else 256)
    _cfg_bs = int(getattr(args, "pretrain_batch", PRETRAIN.get("pretrain_batch", 256)))
    pt_bs = max(4, min(_cfg_bs, _pt_bs_vram, _max_bs))
    if _method == "tscl" and pt_bs < 32:
        print(
            f"[Pretrain] WARN: TSCL estimated batch_size={pt_bs} is too small for useful "
            "contrastive negatives; falling back to BYOL for this run."
        )
        _method = "byol"
        use_regime = False
        mode_str = "BYOL"
        _n_passes = 1
        _safety_factor = 15
        _pt_bs_vram = max(16, int(_vram_free * 0.25) // (_bytes_per_sample * _n_passes * _safety_factor))
        _pt_bs_vram = (_pt_bs_vram // 8) * 8
        _max_bs = 256 if _vram_gb <= 10 else (512 if _vram_gb <= 16 else 1024)
        pt_bs = max(4, min(_cfg_bs, _pt_bs_vram, _max_bs))
        ckpt = str(Path(args.checkpoint_dir) / "contrastive_encoder.pt")
    print(
        f"[Pretrain] method={_method} | VRAM {_vram_gb:.1f} GB | batch_size={pt_bs} "
        f"(configured={_cfg_bs}, budget={_pt_bs_vram}, cap={_max_bs})"
    )

    # Infer encoder output dim via dummy forward. Strip the prediction head first ΓÇö
    # same as BYOLTrainer / TSCLTrainer ΓÇö or backbones like GNN return (B,) logits
    # and we would mis-read encoder_dim as batch size (e.g. 1) instead of hidden dim.
    _saved_head = None
    if hasattr(encoder, "head"):
        _saved_head = encoder.head
        encoder.head = nn.Identity()
    try:
        with torch.no_grad():
            _dummy = torch.zeros(2, args.seq_len, n_features, device=device)
            _out   = encoder(_dummy)
            if _out.ndim == 2:
                encoder_dim = int(_out.shape[-1])
            elif _out.ndim >= 3:
                encoder_dim = int(_out[:, -1, :].shape[-1])
            elif _out.ndim == 1:
                # Multitask + GNN used to return (B,) when inner head was Identity but forward
                # still squeezed; fixed in GNNCrossAsset. Fallback: derive dim from multitask head_in.
                if getattr(args, "multitask", False) and getattr(args, "model", None):
                    n_pairs = getattr(args, "_n_pairs", 1)
                    f_per_pair = getattr(args, "_f_per_pair", n_features)
                    embed_dim = getattr(args, "pair_embed_dim", 0)
                    use_pair_emb = n_pairs > 1 and embed_dim > 0
                    if use_pair_emb:
                        _n_cross = n_pairs * (n_pairs - 1) // 2
                        _n_interaction = 3 * _n_cross + n_pairs + 2
                        backbone_input = n_pairs * (f_per_pair + embed_dim) + _n_interaction
                    else:
                        backbone_input = n_features
                    encoder_dim = _multitask_head_in(
                        str(args.model).lower(), args, backbone_input
                    )
                else:
                    raise RuntimeError(
                        f"[Pretrain] Encoder output is 1D {_out.shape} with head=Identity; "
                        "cannot infer representation width."
                    )
            else:
                raise RuntimeError(
                    f"[Pretrain] Unexpected encoder output ndim={_out.ndim} "
                    f"shape={tuple(_out.shape)}"
                )
    finally:
        if _saved_head is not None:
            encoder.head = _saved_head

    _pt_aug = _make_pretrain_augmenter(args, n_features)
    common = dict(
        d_model     = encoder_dim,
        proj_dim    = int(getattr(args, "pretrain_projection_dim", PRETRAIN.get("projection_dim", 256))),
        temperature = float(getattr(args, "pretrain_temperature", PRETRAIN.get("temperature", 0.5))),
        lr          = float(getattr(args, "pretrain_lr", PRETRAIN.get("pretrain_lr", 1e-4))),
        device      = str(device),
        seed        = _seed,
        aug         = _pt_aug,
    )

    def _fresh_windows():
        """Sample a fresh multi-span block for pretraining."""
        return _sample_pretrain_block()

    def _regime_labels(y, w=None):
        _fn = getattr(args, "_feat_names", None)
        if _fn is not None and w is not None:
            try:
                idx = _fn.index("regime_label")
                scores = w[:, -1, idx]
                return np.where(scores > 0.5, 1, np.where(scores < -0.5, -1, 0)).astype(np.int8)
            except ValueError:
                pass
        return np.where(y > 0.1, 1, np.where(y < -0.1, -1, 0)).astype(np.int8)

    # Build trainer
    trainer_cls = _select_pretrain_trainer_class(_method, use_regime)
    if trainer_cls is BYOLTrainer:
        trainer = BYOLTrainer(
            encoder    = encoder,
            d_model    = encoder_dim,
            proj_dim   = int(getattr(args, "pretrain_projection_dim", PRETRAIN.get("projection_dim", 256))),
            pred_dim   = int(getattr(args, "pretrain_pred_dim", 128)),
            ema_decay  = float(getattr(args, "pretrain_ema_decay", PRETRAIN.get("ema_decay", 0.996))),
            lr         = float(getattr(args, "pretrain_lr", PRETRAIN.get("pretrain_lr", 1e-4))),
            device     = str(device),
            seed       = _seed,
            aug        = _pt_aug,
        )
    elif trainer_cls is MaskedReconstructionTrainer:
        trainer = MaskedReconstructionTrainer(
            encoder=encoder,
            d_model=encoder_dim,
            seq_len=args.seq_len,
            n_features=n_features,
            hidden_dim=int(getattr(args, "pretrain_recon_hidden_dim", PRETRAIN.get("recon_hidden_dim", 512))),
            mask_prob=float(getattr(args, "pretrain_mask_prob", PRETRAIN.get("mask_prob", 0.20))),
            lr=float(getattr(args, "pretrain_lr", PRETRAIN.get("pretrain_lr", 1e-4))),
            device=str(device),
            seed=_seed,
        )
    elif trainer_cls is RegimeAwareTSCLTrainer:
        trainer = RegimeAwareTSCLTrainer(
            encoder=encoder,
            regime_labels=_regime_labels(y_sample, windows),
            hard_negative_weight=1.0,
            **common,
        )
    elif trainer_cls is VAESeqTrainer:
        trainer = VAESeqTrainer(
            encoder=encoder,
            d_model=encoder_dim,
            seq_len=args.seq_len,
            n_features=n_features,
            latent_dim=int(getattr(args, "pretrain_latent_dim", PRETRAIN.get("latent_dim", 64))),
            hidden_dim=int(getattr(args, "pretrain_recon_hidden_dim", PRETRAIN.get("recon_hidden_dim", 512))),
            beta=float(getattr(args, "pretrain_vae_beta", PRETRAIN.get("vae_beta", 0.001))),
            lr=float(getattr(args, "pretrain_lr", PRETRAIN.get("pretrain_lr", 1e-4))),
            device=str(device),
            seed=_seed,
        )
    elif trainer_cls is ClusterContrastiveTrainer:
        trainer = ClusterContrastiveTrainer(
            encoder=encoder,
            d_model=encoder_dim,
            proj_dim=int(getattr(args, "pretrain_projection_dim", PRETRAIN.get("projection_dim", 256))),
            n_clusters=int(getattr(args, "pretrain_n_clusters", PRETRAIN.get("n_clusters", 3))),
            temperature=float(getattr(args, "pretrain_temperature", PRETRAIN.get("temperature", 0.5))),
            lr=float(getattr(args, "pretrain_lr", PRETRAIN.get("pretrain_lr", 1e-4))),
            device=str(device),
            seed=_seed,
            aug=_pt_aug,
        )
    elif trainer_cls is ForecastPretextTrainer:
        trainer = ForecastPretextTrainer(
            encoder=encoder,
            d_model=encoder_dim,
            seq_len=args.seq_len,
            n_features=n_features,
            horizon=int(getattr(args, "pretrain_forecast_horizon", PRETRAIN.get("forecast_horizon", 5))),
            hidden_dim=int(getattr(args, "pretrain_recon_hidden_dim", PRETRAIN.get("recon_hidden_dim", 512))),
            lr=float(getattr(args, "pretrain_lr", PRETRAIN.get("pretrain_lr", 1e-4))),
            device=str(device),
            seed=_seed,
        )
    elif trainer_cls is DriftContrastiveTrainer:
        trainer = DriftContrastiveTrainer(
            encoder=encoder,
            d_model=encoder_dim,
            margin=float(getattr(args, "pretrain_drift_margin", PRETRAIN.get("drift_margin", 1.0))),
            lr=float(getattr(args, "pretrain_lr", PRETRAIN.get("pretrain_lr", 1e-4))),
            device=str(device),
            seed=_seed,
        )
    else:
        trainer = TSCLTrainer(encoder=encoder, **common)

    # Train epoch-by-epoch with fresh windows each time so the encoder sees
    # n_windows ├ù n_epochs unique samples instead of repeating the same n_windows.
    #
    # BYOL multi-block: each outer epoch runs _n_blocks independent blocks so the
    # encoder sees _n_blocks ├ù n_windows windows per epoch instead of just n_windows.
    # With n_windowsΓëê2270 (RAM limit on 16 GB), 3 blocks gives ~6,800 windows and
    # ~26 batches per epoch ΓÇö enough gradient signal to actually move the loss.
    auto_blocks = max(1, 6_000 // max(1, n_windows)) if _method in _PRETRAIN_MULTI_BLOCK else 1
    _n_blocks = _coerce_auto_int(
        getattr(args, "pretrain_blocks_per_epoch", "auto"),
        auto_blocks,
        minimum=1,
    ) if _method in _PRETRAIN_MULTI_BLOCK else 1
    effective_windows = int(_n_blocks * n_windows)
    print(
        f"[Pretrain] windows_per_block={n_windows:,} | blocks_per_epoch={_n_blocks} "
        f"| effective_windows_per_epoch={effective_windows:,}"
    )

    all_losses = []
    final_align = None
    final_unif = None
    final_embed_std = None
    _latest_diag = {}

    _best_loss = float("inf")
    _stale_epochs = 0
    _stopped_early = False
    try:
        _last_w = windows
        for _ep in range(target_epochs):
            if _method in _PRETRAIN_MULTI_BLOCK:
                # Multi-block: run _n_blocks fresh slices per epoch, silent except last
                _block_losses = []
                for _blk in range(_n_blocks):
                    _w, _y = _fresh_windows()
                    _last_w = _w
                    _is_last_block = (_blk == _n_blocks - 1)
                    _h = trainer.pretrain(
                        _w, epochs=1, batch_size=pt_bs,
                        checkpoint_path=ckpt,
                        silent=not _is_last_block,
                    )
                    _block_losses.append(_h["loss"][0])
                if hasattr(trainer, "save_encoder"):
                    trainer.save_encoder(ckpt)
                ls = sum(_block_losses) / len(_block_losses)
            else:
                _w, _y = _fresh_windows()
                _last_w = _w
                if use_regime:
                    rl = _regime_labels(_y, _w)
                    extreme_mask = rl != 0
                    if extreme_mask.any():
                        _ext_idx = np.flatnonzero(extreme_mask)
                        _max_extra = min(len(_ext_idx), max(256, min(len(_w) // 10, 4096)))
                        if _max_extra > 0:
                            _ext_idx = _ext_idx[:_max_extra]
                            _w = np.concatenate([_w, _w[_ext_idx]], axis=0)
                            _y = np.concatenate([_y, _y[_ext_idx]], axis=0)
                            rl = np.concatenate([rl, rl[_ext_idx]], axis=0)
                            print(f"[Pretrain] Regime oversample capped at {_max_extra:,} windows "
                                  f"(base={len(extreme_mask):,}, extreme={int(extreme_mask.sum()):,})")
                    trainer.regime_labels = rl
                _h = trainer.pretrain(_w, epochs=1, batch_size=pt_bs, checkpoint_path=ckpt)
                ls = _h["loss"][0]

            all_losses.append(ls)

            if _method in _PRETRAIN_MULTI_BLOCK:
                _diag = {}
                if hasattr(trainer, "diagnostics"):
                    _diag = trainer.diagnostics(_last_w)
                _latest_diag = dict(_diag or {})
                if "align" in _latest_diag:
                    final_align = float(_latest_diag.get("align", 0.0))
                if "unif" in _latest_diag:
                    final_unif = float(_latest_diag.get("unif", 0.0))
                if "embed_std" in _latest_diag:
                    final_embed_std = float(_latest_diag.get("embed_std", 0.0))
                _extra = ""
                if _diag:
                    if "align" in _diag:
                        _extra = (
                            f" | align={_diag.get('align', 0):.3f} "
                            f"unif={_diag.get('unif', 0):.3f}"
                        )
                    elif "masked_mse" in _diag:
                        _extra = f" | masked_mse={_diag.get('masked_mse', 0):.4f}"
                    elif "recon_loss" in _diag:
                        _extra = (
                            f" | recon={_diag.get('recon_loss', 0):.4f} "
                            f"kl={_diag.get('kl', 0):.4f}"
                        )
                    elif "forecast_mse" in _diag:
                        _extra = f" | forecast_mse={_diag.get('forecast_mse', 0):.4f}"
                    elif "drift_margin" in _diag:
                        _extra = f" | drift_dist={_diag.get('drift_margin', 0):.4f}"
                print(f"[Pretrain] Ep {_ep+1:2d}/{target_epochs} | loss={ls:.4f}"
                      f"  ({_n_blocks} blocks ├ù {n_windows:,} windows = "
                      f"{_n_blocks * n_windows:,} total){_extra}")
                if run and hasattr(run, "log"):
                    _log = {
                        "pt_loss": ls,
                        "pt_effective_windows": effective_windows,
                        "pt_windows_per_block": n_windows,
                        "pt_blocks": _n_blocks,
                        "pt_lr": float(getattr(args, "pretrain_lr", PRETRAIN.get("pretrain_lr", 1e-4))),
                    }
                    if _diag:
                        _log.update({f"pt_{k}": v for k, v in _diag.items()
                                     if k in (
                                         "align", "unif", "embed_std", "masked_mse",
                                         "recon_loss", "kl", "forecast_mse", "drift_margin",
                                     )})
                    _safe_wandb_log(run, _log)
            else:
                al = _h.get("align", [0.0])[-1]
                un = _h.get("unif",  [0.0])[-1]
                std = _h.get("embed_std", [0.0])[-1] if "embed_std" in _h else 0.0
                final_align = float(al)
                final_unif = float(un)
                final_embed_std = float(std)
                _latest_diag = {"align": final_align, "unif": final_unif, "embed_std": final_embed_std}
                print(f"[Pretrain] Ep {_ep+1:2d}/{target_epochs} | loss={ls:.3f} | align={al:.3f} | unif={un:.3f}")
                _safe_wandb_log(run, {
                    "pt_loss": ls, "pt_align": al, "pt_unif": un,
                    "pt_temp": trainer.temp.item(),
                })

            _improved = ls < (_best_loss - handoff_min_delta)
            if _improved:
                _best_loss = ls
                _stale_epochs = 0
                if hasattr(trainer, "save_encoder"):
                    chk_path = Path(ckpt)
                    ep_ckpt = chk_path.with_name(f"{chk_path.stem}_ep{_ep+1}{chk_path.suffix}")
                    trainer.save_encoder(str(ep_ckpt))
            else:
                _stale_epochs += 1

            _epochs_done = _ep + 1
            if _epochs_done >= max(1, min_epochs):
                if handoff_loss > float("-inf") and ls <= handoff_loss:
                    print(f"[Pretrain] Handoff: reached loss threshold {handoff_loss:.4f} at epoch {_epochs_done}.")
                    _stopped_early = True
                    break

                if handoff_patience > 0 and _stale_epochs >= handoff_patience:
                    # Resolve metrics for handoff logic
                    cur_std = _diag.get("embed_std", 0.0) if _method in _PRETRAIN_MULTI_BLOCK and _diag else (std if _method not in _PRETRAIN_MULTI_BLOCK else 0.0)
                    cur_unif = _diag.get("unif", 0.0) if _method in _PRETRAIN_MULTI_BLOCK and _diag else (un if _method not in _PRETRAIN_MULTI_BLOCK else 0.0)

                    from pretrain.handoff_logic import PretrainHandoffGate
                    # Match runner discard cutoffs (stricter than class defaults).
                    _handoff_gate = PretrainHandoffGate(std_threshold=0.015, max_uniformity=-0.1)
                    # uniformity here is negative for good TSCL; gate treats high as bad.
                    # For plateau handoff we only require std diversity when available.
                    quality_ok = (cur_std > 0.015 or cur_std == 0.0) and (
                        cur_unif < -0.5 or cur_unif == 0.0
                    )
                    if quality_ok and (
                        cur_std == 0.0 or _handoff_gate.evaluate_representation_quality(cur_std, None)
                    ):
                        print(f"[Pretrain] Handoff: plateau ({_stale_epochs} stale epochs). Quality met (std={cur_std:.4f}, unif={cur_unif:.2f}).")
                        _stopped_early = True
                        break
                    else:
                        print(f"[Pretrain] Handoff skipped: plateau reached but quality not met (std={cur_std:.4f}, unif={cur_unif:.2f}).")

    except RepresentationCollapseError as e:
        print(f"\n[Pretrain] ABORTED: {e}")
        print("[Pretrain] Falling back to random initialization.")
        _update_pretrain_report(args, {
            "status": "aborted",
            "quality_gate_result": "representation_collapse",
            "error": str(e),
            "epochs_completed": len(all_losses),
            "loss_history": [float(x) for x in all_losses],
            "checkpoint_path": str(ckpt),
        })
        model = build_model(args.model, n_features, args).to(device)
        return model

    # Quality gate — embedding spread for reconstruction-style methods; uniformity for contrastive
    _quality_gate = "passed"
    _handoff_gate = None
    try:
        from pretrain.handoff_logic import PretrainHandoffGate
        _handoff_gate = PretrainHandoffGate(std_threshold=0.015, max_uniformity=-0.1)
    except Exception:
        _handoff_gate = None
    if _method in _PRETRAIN_STD_QUALITY:
        _final_diag = trainer.diagnostics(_last_w) if hasattr(trainer, "diagnostics") else {}
        _std = float(_final_diag.get("embed_std", 0.0))
        _latest_diag.update(_final_diag or {})
        final_embed_std = _std
        _collapsed = bool(_final_diag.get("collapsed", False)) or not np.isfinite(_std)
        if _collapsed:
            print(
                f"\n[Pretrain] Quality Gate Failed: {_method.upper()} embeddings collapsed "
                f"(std={_std:.6f}). Discarding pretrain weights."
            )
            _update_pretrain_report(args, {
                "status": "discarded",
                "quality_gate_result": "failed_embedding_collapse",
                "epochs_completed": len(all_losses),
                "average_pretrain_loss": float(sum(all_losses) / max(len(all_losses), 1)),
                "final_embedding_std": final_embed_std,
                "diagnostics": _latest_diag,
                "checkpoint_path": str(ckpt),
            })
            model = build_model(args.model, n_features, args).to(device)
            return model
        if _handoff_gate is not None:
            # Logs via handoff module; soft threshold matches runner warning band.
            _handoff_gate.evaluate_representation_quality(_std, None)
        if _std < 0.015:
            _quality_gate = "warning_low_embedding_std"
            print(
                f"\n[Pretrain] WARNING: Low {_method.upper()} embedding spread (std={_std:.6f}); "
                "continuing because it is above the collapse threshold."
            )
    elif _method in {"tscl", "cluster"}:
        _unif = float(final_unif or 0.0)
        if _unif > -0.5:
            _quality_gate = "warning_low_uniformity"
            print(f"\n[Pretrain] WARNING: Low uniformity ({_unif:.2f}). "
                  "Embeddings are clustered.")
            if _unif > -0.1:
                print("[Pretrain] Quality Gate Failed. Discarding pretrain weights.")
                _update_pretrain_report(args, {
                    "status": "discarded",
                    "quality_gate_result": "failed_low_uniformity",
                    "epochs_completed": len(all_losses),
                    "average_pretrain_loss": float(sum(all_losses) / max(len(all_losses), 1)),
                    "alignment": final_align,
                    "uniformity": final_unif,
                    "final_embedding_std": final_embed_std,
                    "diagnostics": _latest_diag,
                    "checkpoint_path": str(ckpt),
                })
                model = build_model(args.model, n_features, args).to(device)
                return model

    avg_loss = sum(all_losses) / max(len(all_losses), 1)
    if _stopped_early:
        print(f"[Pretrain] Done (early handoff). avg_loss={avg_loss:.4f}")
    else:
        print(f"[Pretrain] Done. avg_loss={avg_loss:.4f}")

    # Compute feature vulnerability from hard examples for adversarial training (Task 2)
    try:
        _he_path = Path("logs/hard_examples.json")
        if _he_path.exists():
            import json
            _he_data = json.loads(_he_path.read_text(encoding="utf-8"))
            _he_indices = _he_data.get("indices", [])
            if _he_indices:
                # Load full training data to compute vulnerability
                # Use the same data reader as pretraining
                _X_full = np.asarray(x_reader[:n_total])  # (n_total, seq_len, n_features)
                _miner = PretrainHardExampleMiner()
                _vuln = _miner.compute_feature_vulnerability(_X_full, _he_indices, method="gradient_norm")
                _miner.save_vulnerability_scores(_vuln)
                print(f"[Pretrain] Computed feature vulnerability for {len(_vuln)} features -> logs/hard_feature_dims.json")
    except Exception as _vuln_e:
        print(f"[Pretrain] Feature vulnerability computation failed: {_vuln_e}")

    _update_pretrain_report(args, {
        "status": "completed",
        "method": _method,
        "regime_aware": bool(use_regime),
        "epochs_requested": int(target_epochs),
        "epochs_completed": len(all_losses),
        "stopped_early_for_handoff": bool(_stopped_early),
        "handoff": {
            "enabled": bool(handoff_enabled),
            "patience": int(handoff_patience),
            "min_delta": float(handoff_min_delta),
            "loss_threshold": None if handoff_loss == float("-inf") else float(handoff_loss),
        },
        "average_pretrain_loss": float(avg_loss),
        "final_pretrain_loss": float(all_losses[-1]) if all_losses else None,
        "loss_history": [float(x) for x in all_losses],
        "alignment": final_align,
        "uniformity": final_unif,
        "final_embedding_std": final_embed_std,
        "diagnostics": _latest_diag,
        "quality_gate_result": _quality_gate,
        "checkpoint_path": str(ckpt),
        "batch_size": int(pt_bs),
        "blocks_per_epoch": int(_n_blocks),
        "effective_windows_per_epoch": int(effective_windows),
        "hard_examples_injected": int(_hard_examples_injected),
    })
    return model
