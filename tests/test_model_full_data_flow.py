"""End-to-end model smoke tests using real cached data when available, else synthetic."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from data.data_ingestion import generate_synthetic_tick_data
from data.sources import _enforce_schema
from features.feature_engineering import FeatureEngineer
from models.architectures import MODEL_REGISTRY
from training.train_gpu import _build_chunk, build_model

SEQ_LEN = 60
FULL_TEST_SAMPLES = 160
TRAIN_SPLIT = 0.8
EPOCHS = 3
BATCH_SIZE = 16
_LOG_DIR = Path(__file__).resolve().parent.parent / "logs" / "tests"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_PATH = _LOG_DIR / "model_full_data_flow.log"


def _get_logger() -> logging.Logger:
    logger = logging.getLogger("tests.model_full_data_flow")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(_LOG_PATH, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


LOGGER = _get_logger()


def _load_real_cached_ticks(max_hour_files: int = 8) -> tuple[pd.DataFrame, str] | None:
    raw_root = Path(__file__).resolve().parent.parent / "data" / "raw" / "dukascopy"
    if not raw_root.exists():
        return None

    for pair_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        files = sorted(pair_dir.rglob("*.parquet"))
        if not files:
            continue

        frames = []
        for fp in files[:max_hour_files]:
            try:
                frames.append(pd.read_parquet(fp))
            except Exception:
                continue
        if not frames:
            continue

        pair = pair_dir.name.upper()
        df = _enforce_schema(pd.concat(frames, copy=False), pair, "dukascopy")
        if len(df) > 1_000:
            return df, pair
    return None


def _load_ticks_real_or_synthetic() -> tuple[pd.DataFrame, str, str]:
    real = _load_real_cached_ticks()
    if real is not None:
        ticks, pair = real
        LOGGER.info("Using real cached Dukascopy data for full model test: pair=%s rows=%s", pair, len(ticks))
        return ticks, pair, "real"

    ticks = generate_synthetic_tick_data(
        n_rows=120_000,
        pair="EURUSD",
        seed=7,
    )
    LOGGER.info("Using synthetic data for full model test: pair=EURUSD rows=%s", len(ticks))
    return ticks, "EURUSD", "synthetic"


def _args(seq_len: int, n_features: int) -> argparse.Namespace:
    return argparse.Namespace(
        loss="cross_entropy",
        multitask=False,
        seq_len=seq_len,
        hidden_size=32,
        d_model=32,
        nhead=2,
        num_layers=2,
        dropout=0.0,
        pair_embed_dim=0,
        corr_window=20,
        corr_window_long=60,
        momentum_window=20,
        _n_pairs=1,
        _f_per_pair=n_features,
    )


@pytest.fixture(scope="module")
def prepared_sequences():
    import training.train_gpu as _tg

    _tg._FIRST_CHUNK_COLS = None  # isolate from other test modules

    ticks, pair, source = _load_ticks_real_or_synthetic()
    fe = FeatureEngineer()
    scaler = StandardScaler()

    def build_chunk_for(test_ticks):
        _tg._FIRST_CHUNK_COLS = None
        return _build_chunk(
            ticks_chunk=test_ticks,
            fe=fe,
            scaler=scaler,
            seq_len=SEQ_LEN,
            chunk_idx=0,
            label_method="rl_reward",
            cross_asset=None,
        )

    chunk = build_chunk_for(ticks)
    X_seq, y_seq, diff_seq, pq_seq = chunk[:4]
    n_features = chunk.n_features

    if len(X_seq) < FULL_TEST_SAMPLES and source == "real":
        ticks = generate_synthetic_tick_data(
            n_rows=120_000,
            pair="EURUSD",
            seed=7,
        )
        pair = "EURUSD"
        source = "synthetic"
        scaler = StandardScaler()
        chunk = build_chunk_for(ticks)
        X_seq, y_seq, diff_seq, pq_seq = chunk[:4]
        n_features = chunk.n_features

    assert len(X_seq) >= FULL_TEST_SAMPLES, (
        f"Need at least {FULL_TEST_SAMPLES} sequences for the full model test, got {len(X_seq)} from {source} data"
    )
    LOGGER.info(
        "Prepared full-test batch: source=%s pair=%s sequences=%s features=%s",
        source,
        pair,
        len(X_seq),
        n_features,
    )
    x_arr = X_seq[:FULL_TEST_SAMPLES]
    y_cls = np.clip(y_seq[:FULL_TEST_SAMPLES] + 1, 0, 2).astype(np.int64)
    batch_x = torch.tensor(x_arr, dtype=torch.float32)
    batch_y = torch.tensor(y_cls, dtype=torch.long)
    return {
        "x": batch_x,
        "y": batch_y,
        "pair": pair,
        "source": source,
        "n_features": n_features,
        "diff": diff_seq[:8],
        "path_quality": pq_seq[:8],
    }


def test_prepared_sequences_use_real_or_synthetic_data(prepared_sequences):
    assert prepared_sequences["source"] in {"real", "synthetic"}
    assert prepared_sequences["x"].ndim == 3
    assert prepared_sequences["x"].shape[1] == SEQ_LEN
    assert prepared_sequences["x"].shape[2] == prepared_sequences["n_features"]
    assert prepared_sequences["x"].shape[0] == FULL_TEST_SAMPLES
    assert torch.isfinite(prepared_sequences["x"]).all()


@pytest.mark.slow
@pytest.mark.parametrize("model_name", sorted(MODEL_REGISTRY.keys()))
def test_models_run_full_training_cycle_on_real_or_fake_data(model_name, prepared_sequences):
    x = prepared_sequences["x"]
    y = prepared_sequences["y"]
    args = _args(SEQ_LEN, prepared_sequences["n_features"])
    n_train = int(len(x) * TRAIN_SPLIT)
    x_train, x_val = x[:n_train], x[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    LOGGER.info(
        "Running full model test: model=%s source=%s samples=%s train=%s val=%s seq_len=%s features=%s",
        model_name,
        prepared_sequences["source"],
        x.shape[0],
        len(x_train),
        len(x_val),
        x.shape[1],
        x.shape[2],
    )
    model = build_model(model_name, prepared_sequences["n_features"], args)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_train_losses: list[float] = []
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            assert out.shape == (xb.shape[0], 3), f"{model_name} returned {out.shape}, expected ({xb.shape[0]}, 3)"
            assert torch.isfinite(out).all(), (
                f"{model_name} produced non-finite logits on {prepared_sequences['source']} data"
            )
            loss = F.cross_entropy(out, yb)
            assert torch.isfinite(loss), (
                f"{model_name} produced non-finite train loss on {prepared_sequences['source']} data"
            )
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(float(loss.detach().cpu()))

        model.eval()
        epoch_val_losses: list[float] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb)
                assert out.shape == (xb.shape[0], 3)
                assert torch.isfinite(out).all(), (
                    f"{model_name} produced non-finite validation logits on {prepared_sequences['source']} data"
                )
                loss = F.cross_entropy(out, yb)
                assert torch.isfinite(loss), (
                    f"{model_name} produced non-finite validation loss on {prepared_sequences['source']} data"
                )
                epoch_val_losses.append(float(loss.detach().cpu()))

        mean_train = float(np.mean(epoch_train_losses))
        mean_val = float(np.mean(epoch_val_losses))
        train_losses.append(mean_train)
        val_losses.append(mean_val)
        LOGGER.info(
            "Model epoch complete: model=%s epoch=%s/%s train_loss=%.6f val_loss=%.6f",
            model_name,
            epoch + 1,
            EPOCHS,
            mean_train,
            mean_val,
        )

    grad_found = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    assert grad_found, f"{model_name} did not produce usable gradients"
    assert len(train_losses) == EPOCHS
    assert len(val_losses) == EPOCHS
    assert np.isfinite(train_losses).all()
    assert np.isfinite(val_losses).all()
    assert max(train_losses) < 10.0, f"{model_name} train loss exploded: {train_losses}"
    assert max(val_losses) < 10.0, f"{model_name} val loss exploded: {val_losses}"
    LOGGER.info(
        "Model full test passed: model=%s final_train_loss=%.6f final_val_loss=%.6f source=%s",
        model_name,
        train_losses[-1],
        val_losses[-1],
        prepared_sequences["source"],
    )
