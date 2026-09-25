"""Train EnsembleMetaLearner with 120-bar Temporal Attention Pooling.

Loads trained base models (HAELT, Mamba, GNN, TFT), reads cached sequences,
and optimizes TemporalAttentionPooling context encoder and meta weights.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import zarr

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.ensemble import EnsembleMetaLearner, train_meta_learner
from trading.live_engine import build_inference_agents


def main():
    print("=" * 70, flush=True)
    print("   TRAINING ENSEMBLE META-LEARNER (TEMPORAL ATTENTION POOLING)        ", flush=True)
    print("=" * 70, flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}", flush=True)

    # 1. Resolve Zarr Dataset Cache
    proc_dir = ROOT / "data" / "processed"
    zpath = None
    for entry in os.scandir(proc_dir):
        if entry.name.startswith("dataset_scalping_") and entry.name.endswith(".zarr") and entry.is_dir():
            zpath = Path(entry.path)
            break
    if zpath is None:
        raise FileNotFoundError("Could not find processed Zarr dataset")
    print(f"[Data] Found dataset: {zpath.name}", flush=True)

    z = zarr.open(str(zpath), mode="r")
    total_samples, seq_len, n_features = z["X"].shape
    print(f"[Data] Total samples: {total_samples:,} | seq_len: {seq_len} | features: {n_features}", flush=True)

    # 2. Build inference ensemble to load trained base models
    print("[Models] Loading base architectures into ensemble...", flush=True)
    _, slow, meta_info = build_inference_agents("ensemble", runtime="pytorch")
    ensemble = slow.model
    while hasattr(ensemble, "model") or hasattr(ensemble, "inner"):
        if hasattr(ensemble, "inner"):
            ensemble = ensemble.inner
        elif hasattr(ensemble, "model"):
            ensemble = ensemble.model

    print(f"[Models] Base architectures loaded: {len(ensemble.bases)} bases", flush=True)
    for i, b in enumerate(ensemble.bases):
        print(f"  Base {i}: {b.__class__.__name__}", flush=True)

    # Ensure EnsembleMetaLearner uses TemporalAttentionPooling
    assert hasattr(ensemble, "context_enc"), "Ensemble missing context_enc"
    print(f"[Meta] Context encoder: {ensemble.context_enc.__class__.__name__}", flush=True)

    # 3. Load contiguous slice for meta-training (e.g. 20,000 samples before out-of-sample test)
    n_train = 15_000
    n_val = 2_500
    start_train = max(0, total_samples - n_train - n_val - 500)
    end_train = start_train + n_train

    print(f"[Data] Slicing {n_train:,} training samples [{start_train}:{end_train}]...", flush=True)
    t0 = time.time()
    X_train = np.asarray(z["X"][start_train:end_train], dtype=np.float32)
    y_train = np.asarray(z["y"][start_train:end_train], dtype=np.float32)
    print(f"[Data] Loaded slice in {time.time() - t0:.2f}s | Target std: {y_train.std():.4f}", flush=True)

    # 4. Prepare DataLoader
    ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = DataLoader(ds, batch_size=256, shuffle=True)

    # 5. Train meta learner
    ckpt_dir = ROOT / "checkpoints" / "ensemble"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "ensemble_meta_best.pt"

    print(f"[Train] Training TemporalAttentionPooling + MetaLearner with Diversity Regularization (unfrozen base heads)...", flush=True)
    history = train_meta_learner(
        meta=ensemble,
        loader=loader,
        epochs=10,
        lr=1e-3,
        diversity_weight=0.15,
        unfreeze_base_heads=True,
        device=str(device),
        verbose=True,
        checkpoint_path=str(ckpt_path),
        checkpoint_meta={
            "n_features": n_features,
            "seq_len": seq_len,
            "context_dim": 32,
            "hidden": 64,
            "trained_samples": n_train,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    print(f"[Train] Loss history: {[round(h, 4) for h in history]}", flush=True)
    print(f"[Save] Checkpoint successfully saved to {ckpt_path}", flush=True)
    print("=" * 70, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
