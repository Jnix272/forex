"""Direction-class helpers and balance gates for supervised multi-task training.

See docs/CONTINUE.md.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from infrastructure.numerics import sanitize_array
from training.gpu_cache_io import (
    ZARR,
    _diff_path,
    _pq_path,
    _y_cls_path,
    _y_path,
    _zarr_open_group,
)

_HOST = None
_BOUND = False
_HOST_DEPS = (
    "_log_error",
    "_log_warn",
    "_log_info",
    "_slug_part",
    "_safe_save_json",
    "_cache_length_snapshot",
    "_on_disk_sequence_count",
    "ZarrStreamDataset",
    # train_epoch / validate_epoch / build_criterion come from supervised_loop
    # (see _ensure_bound) — not via train_gpu, to avoid a circular import.
    "build_model",
    "_match_target_shape",
    "_crop_to_seq_len",
    "_is_uninitialized_parameter",
    "PATHS",
    "ZARR",
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
    """Bind host deps without forcing a circular import of train_gpu.

    Prefer symbols already present on a partially-initialized ``train_gpu``
    module (``sys.modules``) and pull ``train_epoch`` / ``validate_epoch`` from
    ``supervised_loop`` directly so probe code does not re-enter train_gpu.
    """
    import sys

    global _BOUND
    g = globals()

    sl = sys.modules.get("training.supervised_loop")
    if sl is None:
        from training import supervised_loop as sl
    for name in ("train_epoch", "validate_epoch", "build_criterion"):
        if hasattr(sl, name):
            g[name] = getattr(sl, name)

    tg = sys.modules.get("training.train_gpu")
    if tg is not None:
        bind_host(tg)
        return

    # train_gpu not loaded yet — mark bound with what we have; later bind_host
    # from train_gpu.main / import side will overlay remaining helpers.
    _BOUND = True


def direction_recall_from_confusion(confusion: list[list[int]]) -> list[float]:
    recalls = []
    for cls_idx in range(3):
        denom = max(1, int(sum(confusion[cls_idx])))
        recalls.append(float(confusion[cls_idx][cls_idx]) / denom)
    return recalls


def direction_gate_failed(diag: dict[str, Any], args) -> tuple[bool, str]:
    pred = [int(x) for x in diag.get("pred", [0, 0, 0])]
    total = max(1, sum(pred))
    shares = [x / total for x in pred]
    recalls = [float(x) for x in diag.get("recall", [0.0, 0.0, 0.0])]
    min_pred = float(getattr(args, "direction_min_pred_class_share", 0.05))
    max_pred = float(getattr(args, "direction_max_pred_class_share", 0.80))
    min_recall = float(getattr(args, "direction_min_recall", 0.001))
    if min(shares) < min_pred:
        return True, f"min_pred_share {min(shares):.4f} < {min_pred:.4f}"
    if max(shares) > max_pred:
        return True, f"max_pred_share {max(shares):.4f} > {max_pred:.4f}"
    if min(recalls) < min_recall:
        return True, f"min_recall {min(recalls):.4f} < {min_recall:.4f}"
    return False, "ok"


_direction_recall_from_confusion = direction_recall_from_confusion
_direction_gate_failed = direction_gate_failed

def _load_diff_array(cache_path: str, n_samples: int) -> np.ndarray | None:
    """
    B: Load the full per-sample difficulty array (uint8) from cache.
    Returns None if the sidecar does not exist (graceful fallback -> train on all).
    """
    if ZARR and str(cache_path).endswith(".zarr") and Path(cache_path).is_dir():
        try:
            z = _zarr_open_group(cache_path, mode="r")
            if "diff" in z:
                return np.asarray(z["diff"][:n_samples], dtype=np.uint8)
        except Exception:
            pass
    np_path = _diff_path(cache_path)
    if Path(np_path).exists():
        try:
            return np.load(np_path, mmap_mode="r").astype(np.uint8)[:n_samples]
        except Exception:
            pass
    return None

def _load_feature_schema(cache_path: str, n_features: int) -> list[str] | None:
    """Load the ordered feature schema saved next to a processed cache."""
    schema_path = Path(str(cache_path) + "_feature_schema.json")
    if not schema_path.exists():
        return None
    try:
        import json
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        if isinstance(schema, list) and len(schema) == int(n_features):
            return [str(c) for c in schema]
    except Exception:
        pass
    return None

def _coerce_auto_int(value, auto_value: int, *, minimum: int = 1) -> int:
    """Return an integer config value, accepting 'auto' as a computed default."""
    if value is None:
        return max(minimum, int(auto_value))
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"", "auto", "none"}:
            return max(minimum, int(auto_value))
        value = raw
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError):
        return max(minimum, int(auto_value))

def _read_y_cls_indices(cache_path: str, indices: np.ndarray, chunk: int = 500_000) -> np.ndarray | None:
    """Read direction-class sidecar; None if missing (legacy caches)."""
    if ZARR and cache_path.endswith(".zarr") and Path(cache_path).is_dir():
        z = _zarr_open_group(cache_path, mode="r")
        if "y_cls" not in z:
            return None
        y = z["y_cls"]
        parts: list[np.ndarray] = []
        for s in range(0, len(indices), chunk):
            sl = indices[s : s + chunk]
            parts.append(np.asarray(y.oindex[sl]))
        return np.concatenate(parts) if parts else np.array([])
    ym = Path(_y_cls_path(cache_path))
    if not ym.exists():
        return None
    arr = np.load(str(ym), mmap_mode="r")
    parts = [np.asarray(arr[indices[s : s + chunk]]) for s in range(0, len(indices), chunk)]
    return np.concatenate(parts) if parts else np.array([])

def _read_pq_indices(cache_path: str, indices: np.ndarray, chunk: int = 500_000) -> np.ndarray | None:
    if ZARR and cache_path.endswith(".zarr") and Path(cache_path).is_dir():
        z = _zarr_open_group(cache_path, mode="r")
        if "pq" not in z:
            return None
        pq = z["pq"]
        parts: list[np.ndarray] = []
        for s in range(0, len(indices), chunk):
            sl = indices[s : s + chunk]
            parts.append(np.asarray(pq.oindex[sl]))
        return np.concatenate(parts) if parts else np.array([])
    pm = Path(_pq_path(cache_path))
    if not pm.exists():
        return None
    arr = np.load(str(pm), mmap_mode="r")
    parts = [np.asarray(arr[indices[s : s + chunk]]) for s in range(0, len(indices), chunk)]
    return np.concatenate(parts) if parts else np.array([])

def _read_y_indices(cache_path: str, indices: np.ndarray, chunk: int = 500_000) -> np.ndarray:
    parts: list[np.ndarray] = []
    if ZARR and cache_path.endswith(".zarr") and Path(cache_path).is_dir():
        z = _zarr_open_group(cache_path, mode="r")
        y = z["y"]
        for s in range(0, len(indices), chunk):
            sl = indices[s : s + chunk]
            parts.append(np.asarray(y.oindex[sl]))   # oindex = fancy/out-of-order indexing
    else:
        ym = np.load(_y_path(cache_path), mmap_mode="r")
        for s in range(0, len(indices), chunk):
            sl = indices[s : s + chunk]
            parts.append(np.asarray(ym[sl]))
    y = np.concatenate(parts) if parts else np.array([])
    return sanitize_array(y, context="cached labels")

def _class_weights_tensor(
    cache_path: str, train_idx: np.ndarray, device: torch.device, max_samples: int = 2_000_000,
    use_direction_sidecar: bool = False,
) -> torch.Tensor:
    prior = _class_prior_array(
        cache_path,
        train_idx,
        max_samples=max_samples,
        use_direction_sidecar=use_direction_sidecar,
    )
    inv = 1.0 / np.clip(prior, 1e-6, None)
    weights = inv / max(float(inv.mean()), 1e-6)
    weights = np.clip(weights, 0.85, 1.15).astype(np.float32)
    return torch.tensor(weights, dtype=torch.float32, device=device)

def _class_prior_array(
    cache_path: str, train_idx: np.ndarray, max_samples: int = 2_000_000,
    use_direction_sidecar: bool = False,
) -> np.ndarray:
    if len(train_idx) > max_samples:
        sub = np.random.choice(train_idx, max_samples, replace=False)
    else:
        sub = train_idx
    sub = np.sort(sub)
    if use_direction_sidecar:
        y_raw = _read_y_cls_indices(cache_path, sub)
        if y_raw is None:
            y_raw = _read_y_indices(cache_path, sub)
    else:
        y_raw = _read_y_indices(cache_path, sub)

    # Round to the nearest integer label and cast to int8 so sklearn
    # never hits float32 vs float64 comparison issues.
    y = np.round(y_raw.astype(np.float64)).astype(np.int8)
    # Keep only the three valid direction labels; drop any noise values.
    y = y[np.isin(y, [-1, 0, 1])]

    counts = np.ones(3, dtype=np.float64)  # Laplace smoothing keeps absent classes finite
    for label, count in zip(*np.unique(y, return_counts=True)):
        idx = int(label) + 1
        if 0 <= idx < 3:
            counts[idx] += float(count)
    prior = counts / max(float(counts.sum()), 1.0)
    return prior.astype(np.float32)

def _class_prior_tensor(
    cache_path: str, train_idx: np.ndarray, device: torch.device,
    use_direction_sidecar: bool = False,
) -> torch.Tensor:
    prior = _class_prior_array(
        cache_path, train_idx, use_direction_sidecar=use_direction_sidecar,
    )
    return torch.tensor(prior, dtype=torch.float32, device=device)

def _class_counts_from_y_cls(cache_path: str, indices: np.ndarray) -> dict:
    """Return S/H/B counts and shares from authoritative y_cls sidecar."""
    idx = np.asarray(indices, dtype=np.int64)
    y_raw = _read_y_cls_indices(cache_path, np.sort(idx))
    if y_raw is None:
        raise RuntimeError("[DirectionPreflight] Cache is missing y_cls direction sidecar.")
    y = np.round(np.asarray(y_raw, dtype=np.float64)).astype(np.int8)
    invalid = y[~np.isin(y, [-1, 0, 1])]
    counts = np.zeros(3, dtype=np.int64)
    for label, count in zip(*np.unique(y[np.isin(y, [-1, 0, 1])], return_counts=True)):
        counts[int(label) + 1] = int(count)
    total = max(1, int(counts.sum()))
    return {
        "counts": [int(x) for x in counts.tolist()],
        "shares": [float(x) / total for x in counts.tolist()],
        "invalid_count": len(invalid),
        "total": int(total),
    }

def _balanced_direction_indices(
    cache_path: str,
    indices: np.ndarray,
    *,
    total_samples: int | None = None,
    seed: int = 1337,
) -> np.ndarray:
    """Build an approximately class-balanced index list using y_cls labels."""
    _ensure_bound()
    idx = np.asarray(indices, dtype=np.int64)
    if len(idx) == 0:
        return idx
    y_raw = _read_y_cls_indices(cache_path, np.sort(idx))
    if y_raw is None:
        raise RuntimeError("[DirectionBalance] Cannot balance batches without y_cls sidecar.")
    sorted_idx = np.sort(idx)
    y = np.round(np.asarray(y_raw, dtype=np.float64)).astype(np.int8)
    rng = np.random.default_rng(int(seed))
    buckets = []
    for label in (-1, 0, 1):
        bucket = sorted_idx[y == label]
        if len(bucket) == 0:
            raise RuntimeError(f"[DirectionBalance] Missing class {label} in training fold.")
        buckets.append(bucket)
    if total_samples is None:
        per_class = min(len(b) for b in buckets)
    else:
        per_class = max(1, int(total_samples) // 3)
    parts = [
        rng.choice(bucket, size=per_class, replace=(len(bucket) < per_class))
        for bucket in buckets
    ]
    out = np.concatenate(parts).astype(np.int64)
    rng.shuffle(out)
    return out

def _direction_preflight(cache_path: str, train_idx: np.ndarray, val_idx: np.ndarray, args) -> dict:
    """Hard gate before supervised direction training starts."""
    _ensure_bound()
    snap = _cache_length_snapshot(cache_path)
    required = ("zarr_X", "zarr_y", "zarr_y_cls", "zarr_pq", "zarr_diff", "zarr_close", "zarr_atr", "zarr_spread")
    if snap and any(k.startswith("zarr_") for k in snap):
        missing = [k for k in required if k not in snap]
        if missing:
            raise RuntimeError(f"[DirectionPreflight] Missing cache arrays: {missing}")
        x_len = int(snap["zarr_X"])
        bad = {k: int(v) for k, v in snap.items() if k.startswith("zarr_") and int(v) != x_len}
        if bad:
            raise RuntimeError(f"[DirectionPreflight] Cache array length mismatch: X={x_len}, bad={bad}")

    train_stats = _class_counts_from_y_cls(cache_path, train_idx)
    val_stats = _class_counts_from_y_cls(cache_path, val_idx)
    min_share = float(getattr(args, "direction_min_true_class_share", 0.15))
    for split, stats in (("train", train_stats), ("val", val_stats)):
        if stats["invalid_count"]:
            raise RuntimeError(f"[DirectionPreflight] {split} y_cls has {stats['invalid_count']} invalid labels.")
        if min(stats["shares"]) < min_share:
            raise RuntimeError(
                f"[DirectionPreflight] {split} class prior too thin: "
                f"S/H/B={stats['shares']} min_required={min_share}"
            )

    forced = torch.tensor([[9.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 9.0]])
    if forced.argmax(-1).tolist() != [0, 1, 2]:
        raise RuntimeError("[DirectionPreflight] Forced logit class order is broken.")
    report = {"train": train_stats, "val": val_stats, "class_order": ["Sell", "Hold", "Buy"]}
    print(
        "[DirectionPreflight] PASS | "
        f"train S/H/B={train_stats['shares'][0]:.3f}/{train_stats['shares'][1]:.3f}/{train_stats['shares'][2]:.3f} | "
        f"val S/H/B={val_stats['shares'][0]:.3f}/{val_stats['shares'][1]:.3f}/{val_stats['shares'][2]:.3f}"
    )
    return report

def _write_class_balance_failure(run_name: str, model_name: str, epoch: int, diag: dict, reason: str) -> Path:
    _ensure_bound()
    out_dir = Path("logs/tests")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"class_balance_failure_{_slug_part(run_name, 120)}_{_slug_part(model_name, 40)}_ep{epoch+1}.json"
    payload = {
        "run_name": run_name,
        "model": model_name,
        "epoch": int(epoch + 1),
        "reason": reason,
        "diagnostics": diag,
        "class_order": ["Sell", "Hold", "Buy"],
        "generated_at": datetime.now(UTC).isoformat(),
    }
    _safe_save_json(payload, out)
    return out

def _direction_probe(
    model,
    cache_path: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    args,
    device,
    *,
    model_name: str,
    n_features: int,
    amp_dtype: torch.dtype,
) -> dict:
    """Short balanced probe that must pass before full supervised training."""
    _ensure_bound()
    if not bool(getattr(args, "direction_probe", True)):
        return {"enabled": False, "passed": True}
    epochs = max(1, int(getattr(args, "direction_probe_epochs", 2)))
    samples = max(96, int(getattr(args, "direction_probe_samples", 4096)))
    seed = int(getattr(args, "seed", 1337))
    probe_train_idx = _balanced_direction_indices(cache_path, train_idx, total_samples=samples, seed=seed)
    probe_val_idx = _balanced_direction_indices(cache_path, val_idx, total_samples=max(96, samples // 2), seed=seed + 17)
    train_ds = ZarrStreamDataset(cache_path, probe_train_idx, shuffle_chunks=True, multitask_targets=True)
    val_ds = ZarrStreamDataset(cache_path, probe_val_idx, shuffle_chunks=False, multitask_targets=True)
    bs = min(max(32, int(getattr(args, "batch_size", 128))), 256)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=False)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=False)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=max(float(getattr(args, "lr", 1e-4)), 1e-4), weight_decay=0.0)
    print(f"[DirectionProbe] START | samples={len(probe_train_idx):,} val={len(probe_val_idx):,} epochs={epochs}")
    last = {}
    classification = True
    for ep in range(epochs):
        tl = train_epoch(
            model, train_dl, opt, crit, GradScaler(enabled=False), device,
            use_amp=False, classification=classification,
            grad_clip=float(getattr(args, "grad_clip", 1.0)),
            amp_dtype=amp_dtype, seq_len=int(getattr(args, "seq_len", 80)),
            multitask=True, epoch=ep, direction_only=True,
        )
        vl, da, _ = validate_epoch(
            model, val_dl, crit, device, classification,
            amp=False, amp_dtype=amp_dtype,
            seq_len=int(getattr(args, "seq_len", 80)), multitask=True,
            direction_only=True,
        )
        diag = getattr(validate_epoch, "last_class_diag", {})
        failed, reason = _direction_gate_failed(diag, args)
        last = {"train_loss": float(tl), "val_loss": float(vl), "dir_acc": float(da), "diag": diag, "failed": failed, "reason": reason}
        print(
            f"[DirectionProbe] Epoch {ep+1}/{epochs} train={tl:.4f} val={vl:.4f} acc={da:.4f} "
            f"pred={diag.get('pred')} recall={[round(x, 4) for x in diag.get('recall', [])]} reason={reason}"
        )
    if last.get("failed", True):
        out = _write_class_balance_failure(getattr(args, "run_name", "direction_probe"), model_name, 0, last.get("diag", {}), last.get("reason", "probe_failed"))
        raise RuntimeError(f"[DirectionProbe] FAILED: {last.get('reason')} | diagnostics -> {out}")
    print("[DirectionProbe] PASS")
    return {"enabled": True, "passed": True, **last}

def _init_multitask_direction_bias(model: nn.Module, class_prior: torch.Tensor) -> None:
    """Start the direction head from the fold label prior instead of a random class bias."""
    target = class_prior.detach().float().cpu().clamp_min(1e-6)
    target = target / target.sum().clamp_min(1e-6)
    bias = target.log()
    modules = [model]
    if isinstance(model, nn.DataParallel):
        modules.append(model.module)
    for root in modules:
        mt_head = getattr(root, "mt_head", None)
        direction = getattr(mt_head, "direction", None)
        if direction is None:
            continue
        for layer in reversed(list(direction.modules())):
            if isinstance(layer, nn.Linear) and layer.out_features == 3 and layer.bias is not None:
                layer.bias.copy_(bias.to(layer.bias.device, dtype=layer.bias.dtype))
                return

def labels_to_class_index(yb: torch.Tensor) -> torch.Tensor:
    """Map {-1,0,+1} direction labels to CE indices {0,1,2}.

    Non-finite labels are **not** silently mapped to hold (0.0). Callers should
    sanitize/drop first; remaining non-finites warn and stay invalid until
    ``round().long().clamp`` (prefer drop via ``_sanitize_batch_tensors``).
    """
    _ensure_bound()
    yb = yb.float()
    if not torch.isfinite(yb).all():
        n_bad = int((~torch.isfinite(yb)).sum().item())
        print(
            f"[labels_to_class_index] WARN: {n_bad} non-finite label(s) "
            "(not zeroed to hold — drop via sanitize)"
        )
    return (yb + 1.0).round().long().clamp(0, 2)

def _direction_class_index(

    yb: torch.Tensor,

    y_cls: torch.Tensor | None = None,

    *,

    classification: bool = True,

) -> torch.Tensor:

    """Return class indices, preferring the explicit direction sidecar.



    For rl_reward labels, ``yb`` is a continuous reward. Rounding it as if it

    were {-1, 0, +1} can collapse classification training and validation. The

    y_cls sidecar is the authoritative direction target when present.

    """

    if y_cls is not None:

        return labels_to_class_index(y_cls)

    if classification:

        return labels_to_class_index(yb)

    return _reward_to_class_index(yb)

def _reward_to_class_index(y_reward: torch.Tensor, hold_eps: float = 0.5) -> torch.Tensor:
    """Fallback when y_cls sidecar is absent: threshold continuous rewards to classes."""
    r = torch.nan_to_num(y_reward.float(), nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)
    cls = torch.ones(r.shape[0], dtype=torch.long, device=r.device)
    cls[r > hold_eps] = 2
    cls[r < -hold_eps] = 0
    return cls

def _gradients_are_finite(model: nn.Module) -> bool:
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return False
    return True

def _recover_nonfinite_training_state(model: nn.Module, opt: torch.optim.Optimizer) -> None:
    for p in model.parameters():
        if _is_uninitialized_parameter(p):
            continue
        if not torch.isfinite(p).all():
            p.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    for state in opt.state.values():
        for value in state.values():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                value.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    opt.zero_grad(set_to_none=True)

