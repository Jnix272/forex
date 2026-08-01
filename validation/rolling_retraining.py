"""Production-like expanding-window splits with purge, embargo and lockbox."""
from __future__ import annotations
import numpy as np


def rolling_retraining_splits(n_samples: int, *, retrain_every: int, validation_size: int,
                              min_train_size: int, purge_bars: int, lockbox_size: int = 0):
    usable_end = max(0, n_samples - lockbox_size)
    train_end = min_train_size
    while train_end + purge_bars + validation_size <= usable_end:
        val_start = train_end + purge_bars
        yield np.arange(0, train_end), np.arange(val_start, val_start + validation_size)
        train_end += retrain_every


def untouched_lockbox_indices(n_samples: int, lockbox_size: int):
    start = max(0, n_samples - max(0, lockbox_size))
    return np.arange(start, n_samples)
