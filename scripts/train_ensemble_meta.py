"""
Train only the EnsembleMeta learner from existing base checkpoints.

This is the recovery/iteration path after HAELT, Mamba, and GNN have already
been trained. It streams the cached training windows from Zarr instead of
loading the full dataset into RAM.

Example:
    .venv-gpu\\Scripts\\python.exe scripts/train_ensemble_meta.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import PATHS
from models.ensemble import EnsembleMetaLearner, train_meta_learner
from training.train_gpu import ZarrStreamDataset, _on_disk_sequence_count, build_model

DEFAULT_CACHE = (
    ROOT
    / "data"
    / "processed"
    / "dataset_AUDUSD-EURGBP-EURJPY-EURUSD-GBPJPY-GBPUSD-NZDUSD-USDCAD-USDCHF-USDJPY_20000000_dukascopy_60_rl_reward.zarr"
)


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EnsembleMeta from existing base checkpoints.")
    p.add_argument("--cache", type=str, default=str(DEFAULT_CACHE), help="Training Zarr cache")
    p.add_argument("--checkpoint-dir", type=str, default=PATHS["checkpoints"])
    p.add_argument("--models", nargs="+", default=["haelt", "mamba", "gnn"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--samples", type=int, default=200_000, help="Random cached samples to use")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--div-weight", type=float, default=0.1)
    p.add_argument("--output", type=str, default=None, help="Best checkpoint path")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state", "model_state_dict", "state_dict"):
            state = checkpoint.get(key)
            if isinstance(state, dict):
                return state
    return checkpoint


def resolve_checkpoint(model_name: str, checkpoint_dir: Path) -> Path | None:
    candidates = [
        checkpoint_dir / model_name / f"{model_name}_best.pt",
        checkpoint_dir / f"{model_name}_best.pt",
        checkpoint_dir / model_name / f"{model_name}_fold0_best.pt",
        checkpoint_dir / f"{model_name}_fold0_best.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    model_dir = checkpoint_dir / model_name
    folds = sorted(model_dir.glob(f"{model_name}_fold*_best.pt"))
    return folds[-1] if folds else None


def load_training_config(model_name: str, ckpt_path: Path) -> dict:
    candidates = [
        ckpt_path.with_name(ckpt_path.name.replace("_best.pt", "_config.json")),
        ckpt_path.parent / f"{model_name}_config.json",
    ]
    summary = ckpt_path.parent / f"{model_name}_best_fold_summary.json"
    if summary.exists():
        try:
            best_fold = int(json.loads(summary.read_text(encoding="utf-8")).get("best_fold", 0))
            candidates.insert(0, ckpt_path.parent / f"{model_name}_fold{best_fold}_config.json")
        except Exception:
            pass
    candidates.extend(sorted(ckpt_path.parent.glob(f"{model_name}_fold*_config.json")))
    for path in candidates:
        if path.exists():
            cfg = json.loads(path.read_text(encoding="utf-8"))
            cfg["_config_path"] = str(path)
            return cfg
    return {}


def make_model_args(model_name: str, cfg: dict, state: dict, n_features: int, seq_len: int) -> SimpleNamespace:
    keys = tuple(state.keys()) if isinstance(state, dict) else ()
    has_pair_embed = any("pair_embed" in k for k in keys)
    has_multitask = any("mt_head" in k or ".mt_head." in k for k in keys)
    pair_embed_shape = None
    pair_norm_shape = None
    if isinstance(state, dict):
        for key, value in state.items():
            shape = getattr(value, "shape", None)
            if shape is None:
                continue
            if key.endswith("pair_embeds.weight") and len(shape) == 2:
                pair_embed_shape = tuple(int(v) for v in shape)
            elif key.endswith("pair_norms.0.weight") and len(shape) == 1:
                pair_norm_shape = tuple(int(v) for v in shape)

    n_pairs = int(cfg.get("_n_pairs") or cfg.get("n_pairs") or 0)
    if n_pairs <= 0 and pair_embed_shape is not None:
        n_pairs = int(pair_embed_shape[0])
    if n_pairs <= 0:
        n_pairs = 1

    f_per_pair = int(cfg.get("_f_per_pair") or cfg.get("f_per_pair") or 0)
    if f_per_pair <= 0 and pair_norm_shape is not None:
        f_per_pair = int(pair_norm_shape[0])
    if f_per_pair <= 0:
        f_per_pair = max(1, n_features // n_pairs)

    pair_embed_dim = int(cfg.get("pair_embed_dim", 0))
    if pair_embed_shape is not None:
        pair_embed_dim = int(pair_embed_shape[1])
    elif has_pair_embed and pair_embed_dim <= 0:
        pair_embed_dim = 16

    if has_pair_embed and n_pairs * f_per_pair != n_features:
        if n_features % n_pairs == 0 and pair_norm_shape is None:
            f_per_pair = n_features // n_pairs
        else:
            raise ValueError(
                f"{model_name} checkpoint expects {n_pairs} pairs x {f_per_pair} features "
                f"but cache has {n_features} features"
            )

    return SimpleNamespace(
        model=model_name,
        hidden_size=int(cfg.get("hidden_size", 256)),
        d_model=int(cfg.get("d_model", 256)),
        nhead=int(cfg.get("nhead", 8)),
        num_layers=int(cfg.get("num_layers", 3)),
        dropout=float(cfg.get("dropout", 0.1)),
        seq_len=int(cfg.get("seq_len", seq_len)),
        multitask=bool(cfg.get("multitask", has_multitask)),
        pair_embed_dim=pair_embed_dim,
        loss=str(cfg.get("loss", "cross_entropy")),
        corr_window=int(cfg.get("corr_window", 20)),
        corr_window_long=int(cfg.get("corr_window_long", 60)),
        momentum_window=int(cfg.get("momentum_window", 20)),
        _n_pairs=n_pairs,
        _f_per_pair=f_per_pair,
    )


def load_base_model(model_name: str, ckpt_path: Path, n_features: int, seq_len: int, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = checkpoint_state_dict(ckpt)
    cfg = load_training_config(model_name, ckpt_path)
    model_args = make_model_args(model_name, cfg, state, n_features, seq_len)
    model = build_model(model_name, n_features, model_args).to(device)
    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        core = model.backbone if hasattr(model, "backbone") else model
        has_prefix = any(k.startswith("backbone.") for k in state.keys())
        core_has_prefix = any(p.startswith("backbone.") for p, _ in core.named_parameters())
        if has_prefix and not core_has_prefix:
            state = {k.replace("backbone.", "", 1): v for k, v in state.items()}
        core.load_state_dict(state, strict=False)
    model.eval()
    return model, cfg


def infer_cache_shape(cache_path: Path) -> tuple[int, int, int]:
    try:
        import zarr

        z = zarr.open(str(cache_path), mode="r")
        n_samples, seq_len, n_features = z["X"].shape
        return int(n_samples), int(seq_len), int(n_features)
    except Exception as exc:
        raise RuntimeError(f"Could not inspect cache {cache_path}: {exc}") from exc


def main() -> int:
    args = parse_args()
    cache_path = Path(args.cache)
    ckpt_dir = Path(args.checkpoint_dir)
    if args.output:
        out = Path(args.output)
    else:
        out = ckpt_dir / "ensemble" / "ensemble_meta_best.pt"

    if not cache_path.exists():
        log(f"[EnsembleMeta] Cache not found: {cache_path}")
        return 1

    total, seq_len, n_features = infer_cache_shape(cache_path)
    total = _on_disk_sequence_count(str(cache_path)) or total
    n_meta = min(int(args.samples), int(total))
    rng = np.random.default_rng(int(args.seed))
    meta_idx = np.sort(rng.choice(total, n_meta, replace=False))

    device = torch.device(args.device)
    log(f"[EnsembleMeta] cache={cache_path}")
    log(f"[EnsembleMeta] samples={n_meta:,}/{total:,} | seq_len={seq_len} | n_features={n_features}")
    log(f"[EnsembleMeta] device={device} | output={out}")

    bases = []
    names = []
    configs = {}
    checkpoints = {}
    for name in args.models:
        ckpt = resolve_checkpoint(name, ckpt_dir)
        if ckpt is None:
            log(f"  [EnsembleMeta] Missing {name}: no checkpoint found")
            continue
        log(f"  [EnsembleMeta] Loading {name} from {ckpt}")
        try:
            model, cfg = load_base_model(name, ckpt, n_features, seq_len, device)
        except Exception as exc:
            log(f"  [EnsembleMeta] Could not load {name} from {ckpt}: {exc}")
            continue
        bases.append(model)
        names.append(name)
        configs[name] = cfg.get("_config_path")
        checkpoints[name] = str(ckpt)
        log(f"  [EnsembleMeta] Loaded {name} from {ckpt}")

    if len(bases) < 2:
        log(f"[EnsembleMeta] Need at least 2 base models, found {len(bases)}: {names}")
        return 1

    dataset = ZarrStreamDataset(str(cache_path), meta_idx, shuffle_chunks=True)
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=False,
    )

    meta = EnsembleMetaLearner(bases, context_dim=32, hidden=64, base_names=names).to(device)
    log(f"[EnsembleMeta] Training on bases: {names}")
    history = train_meta_learner(
        meta=meta,
        loader=loader,
        epochs=int(args.epochs),
        lr=float(args.lr),
        diversity_weight=float(args.div_weight),
        device=str(device),
        verbose=True,
        checkpoint_path=str(out),
        checkpoint_meta={
            "base_names": names,
            "base_configs": configs,
            "cache_path": str(cache_path),
            "n_features": n_features,
            "seq_len": seq_len,
            "samples": n_meta,
        },
    )

    final = out.parent / "ensemble_meta_final.pt"
    torch.save(meta.state_dict(), final)
    final.with_suffix(final.suffix + ".json").write_text(
        json.dumps(
            {
                "epoch": len(history),
                "loss": float(history[-1]) if history else None,
                "history": history,
                "meta": {
                    "base_names": names,
                    "base_configs": configs,
                    "cache_path": str(cache_path),
                    "n_features": n_features,
                    "seq_len": seq_len,
                    "samples": n_meta,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = {
        "kind": "ensemble_meta_learner",
        "created_at": datetime.now(UTC).isoformat(),
        "base_models": [
            {
                "name": name,
                "checkpoint": checkpoints.get(name),
                "config": configs.get(name),
            }
            for name in names
        ],
        "artifacts": {
            "best_checkpoint": str(out),
            "latest_checkpoint": str(out.with_name(f"{out.stem}_latest{out.suffix}")),
            "final_checkpoint": str(final),
            "best_onnx": str(out.with_suffix(".onnx")),
        },
        "schema": {
            "n_features": int(n_features),
            "seq_len": int(seq_len),
        },
        "training": {
            "cache_path": str(cache_path),
            "samples": int(n_meta),
            "epochs": int(args.epochs),
            "diversity_weight": float(args.div_weight),
            "history": history,
            "final_loss": float(history[-1]) if history else None,
        },
    }
    (out.parent / "ensemble_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    log(f"[EnsembleMeta] Saved best/latest via trainer: {out}")
    log(f"[EnsembleMeta] Saved final: {final}")
    log(f"[EnsembleMeta] Saved manifest: {out.parent / 'ensemble_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
