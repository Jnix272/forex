"""
training/hard_example_miner.py
================================
Identifies and persists "hard examples" from validation — samples where the
model was confident but wrong, or missed large reward opportunities.

On the next training run these indices are lightly oversampled so the model
gets more exposure to its blind spots.

Enhancements
------------
- Loss-weighted mining: ranks failures by per-sample loss magnitude
- Boundary/uncertainty mining: captures samples near decision boundary
- Regime-aware tracking: stores distribution of hard examples per market regime
- Online in-batch mining: tracks per-sample loss rolling history during training
- Forgetting/decay tracking: detects samples that were learned then forgotten

Usage
-----
miner = OnlineHardExampleMiner(n_samples=n_train_samples)

# During training, feed each batch's per-sample losses:
miner.update_batch(sample_indices=batch_idx, per_sample_losses=per_sample_loss)

# Next epoch, build the dataloader from the oversampled index set:
base_idx = np.arange(n_train_samples)
aug_idx  = miner.get_oversampled_indices(base_idx)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

_DEFAULT_LOG_DIR = Path("logs/hard_examples")
_MAX_OVERSAMPLE_RATIO = 2.0   # hard examples can at most double their share
_CONFIDENCE_THRESHOLD = 0.65  # min predicted probability to count as "confident"
_WRONG_FRAC_CAP = 0.15        # cap hard examples at 15% of val set
_BOUNDARY_THRESHOLD = 0.60    # max confidence to be considered "uncertain"


# ═══════════════════════════════════════════════════════════════════════════════
# FORGETTING TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class ForgettingTracker:
    """Tracks per-sample loss trajectory to detect forgetting and decay.

    Forgetting: a sample whose loss decreased (learned) then increased (forgotten).
    Easy: a sample whose loss has been consistently low.

    Parameters
    ----------
    n_samples   : total number of samples in the dataset
    max_history : maximum number of past epochs to retain
    """

    def __init__(self, n_samples: int, max_history: int = 10):
        self.n_samples   = n_samples
        self.max_history = max_history
        self._history: list[np.ndarray] = []  # each entry: array of shape (n_samples,)

    def update(self, per_sample_losses: np.ndarray) -> None:
        """Append a new epoch's per-sample losses.

        Parameters
        ----------
        per_sample_losses : array of shape (n_samples,) or subset thereof.
                            If shorter, only that many positions are updated
                            (remaining positions get NaN).
        """
        losses = np.asarray(per_sample_losses, dtype=np.float32).ravel()
        if len(losses) < self.n_samples:
            full = np.full(self.n_samples, np.nan, dtype=np.float32)
            full[:len(losses)] = losses
            losses = full
        self._history.append(losses)
        if len(self._history) > self.max_history:
            self._history.pop(0)

    def get_forgotten_mask(self, recent_window: int = 3) -> np.ndarray:
        """Return boolean mask: True for samples whose loss increased recently
        after an earlier decrease.

        A sample is "forgotten" if:
        - Its loss was low `recent_window` epochs ago (or at its minimum)
        - Its loss has been increasing over the last `recent_window` epochs
        """
        if len(self._history) < recent_window * 2:
            return np.zeros(self.n_samples, dtype=bool)

        old = np.nanmean(
            [self._history[-(recent_window + i + 1)] for i in range(recent_window)],
            axis=0,
        )
        recent = np.nanmean(
            [self._history[-(i + 1)] for i in range(recent_window)],
            axis=0,
        )

        forgotten = (recent > old + 0.01) & ~np.isnan(old) & ~np.isnan(recent)
        return forgotten

    def get_easy_mask(self, recent_window: int = 3, quantile: float = 0.3) -> np.ndarray:
        """Return boolean mask: True for consistently low-loss samples.

        These can be deprioritised during oversampling.
        """
        if len(self._history) < recent_window:
            return np.zeros(self.n_samples, dtype=bool)

        recent = np.nanmean(
            [self._history[-(i + 1)] for i in range(recent_window)],
            axis=0,
        )
        threshold = np.nanquantile(recent, quantile)
        return (recent <= threshold) & ~np.isnan(recent)

    @property
    def current_epoch(self) -> int:
        return len(self._history)

    def state_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "max_history": self.max_history,
            "history": [h.tolist() for h in self._history],
        }

    def load_state_dict(self, state: dict) -> None:
        self.n_samples = state["n_samples"]
        self.max_history = state.get("max_history", 10)
        self._history = [np.array(h, dtype=np.float32) for h in state.get("history", [])]


# ═══════════════════════════════════════════════════════════════════════════════
# ONLINE HARD-EXAMPLE MINER
# ═══════════════════════════════════════════════════════════════════════════════

class OnlineHardExampleMiner:
    """Tracks per-sample loss over epochs and identifies hard/forgotten samples
    during training, enabling online data augmentation.

    Unlike an offline pass that mines a single validation run, this miner
    maintains rolling per-sample statistics across *all* training
    epochs so consistently difficult or forgotten samples can be oversampled
    before they cause overfitting.

    Parameters
    ----------
    n_samples       : total number of training samples
    window_size     : number of recent epochs used for decay tracking
    hard_quantile   : loss quantile above which a sample is considered "hard"
    forget_window   : number of recent epochs to inspect for forgetting
    easy_quantile   : loss quantile below which a sample is "easy" (deprioritised)
    boost_factor    : multiplier for forgotten/hard sample replication
    decay_factor    : EMA decay applied to per-sample scores each epoch
    """

    def __init__(
        self,
        n_samples: int,
        window_size: int = 5,
        hard_quantile: float = 0.85,
        forget_window: int = 3,
        easy_quantile: float = 0.30,
        boost_factor: float = 2.0,
        decay_factor: float = 0.90,
    ):
        self.n_samples     = n_samples
        self.window_size   = max(2, window_size)
        self.hard_quantile = min(max(float(hard_quantile), 0.5), 1.0)
        self.forget_window = max(2, forget_window)
        self.easy_quantile = min(max(float(easy_quantile), 0.0), 0.5)
        self.boost_factor  = max(1.0, float(boost_factor))
        self.decay_factor  = min(max(float(decay_factor), 0.0), 1.0)

        # Rolling per-sample loss buffer: shape (window_size, n_samples)
        # NaN = no data for that epoch/sample
        self._loss_buffer = np.full((window_size, n_samples), np.nan, dtype=np.float32)
        # EMA score: high score = consistently hard
        self._ema_score = np.zeros(n_samples, dtype=np.float32)
        self._epoch = 0
        self._forgetting_tracker = ForgettingTracker(n_samples, max_history=window_size + 2)

    def begin_epoch(self) -> None:
        """Prepare a new epoch: roll the buffer, clear per-epoch accumulators."""
        if self._epoch > 0:
            self._loss_buffer = np.roll(self._loss_buffer, -1, axis=0)
            self._loss_buffer[-1, :] = np.nan
        self._epoch += 1

    def update_batch(self, sample_indices: np.ndarray, per_sample_losses: np.ndarray) -> None:
        """Accumulate per-sample losses from one training batch."""
        indices = np.asarray(sample_indices, dtype=np.int64).ravel()
        losses  = np.asarray(per_sample_losses, dtype=np.float32).ravel()
        if len(indices) == 0 or len(losses) == 0:
            return
        valid = (indices >= 0) & (indices < self.n_samples)
        if not np.any(valid):
            return
        self._loss_buffer[-1, indices[valid]] = losses[valid]

    def end_epoch(self) -> None:
        """Finalise the current epoch: update forgetting tracker with this epoch's losses."""
        # Use only the current epoch's loss row, not a smoothed window average,
        # so ForgettingTracker sees the correct per-epoch loss trajectory.
        epoch_losses = self._loss_buffer[-1].copy()
        self._forgetting_tracker.update(epoch_losses)

    def get_hard_mask(self) -> np.ndarray:
        """Return boolean mask of consistently hard samples."""
        if self._epoch < 2:
            return np.zeros(self.n_samples, dtype=bool)

        recent = self._loss_buffer[-min(self._epoch, self.window_size):, :]
        mean_recent = np.nanmean(recent, axis=0)
        threshold = np.nanquantile(mean_recent, self.hard_quantile)
        return (mean_recent >= threshold) & ~np.isnan(mean_recent)

    def get_forgotten_mask(self) -> np.ndarray:
        """Return boolean mask of samples that were learned then forgotten."""
        return self._forgetting_tracker.get_forgotten_mask(recent_window=self.forget_window)

    def get_easy_mask(self) -> np.ndarray:
        """Return boolean mask of consistently easy samples (can deprioritise)."""
        return self._forgetting_tracker.get_easy_mask(
            recent_window=self.forget_window,
            quantile=self.easy_quantile,
        )

    def get_oversampled_indices(
        self,
        base_indices: np.ndarray,
        hard_factor: float = 1.5,
        forgotten_factor: float = 2.0,
        easy_downsample: bool = True,
    ) -> np.ndarray:
        """Return base_indices augmented with hard and forgotten samples."""
        base = np.asarray(base_indices, dtype=np.int64)
        if len(base) == 0:
            return base

        hard_mask = self.get_hard_mask()
        forgotten_mask = self.get_forgotten_mask()
        easy_mask = self.get_easy_mask()
        rng = np.random.default_rng()

        augmented = list(base)

        # Oversample hard
        hard_idx = np.where(hard_mask)[0]
        if len(hard_idx) > 0:
            n_extra_hard = max(0, int(len(hard_idx) * (hard_factor - 1.0)))
            if n_extra_hard > 0:
                extra = rng.choice(hard_idx, size=n_extra_hard, replace=True)
                augmented.extend(extra.tolist())

        # Oversample forgotten (higher priority)
        forget_idx = np.where(forgotten_mask)[0]
        if len(forget_idx) > 0:
            n_extra_forget = max(0, int(len(forget_idx) * (forgotten_factor - 1.0)))
            if n_extra_forget > 0:
                extra = rng.choice(forget_idx, size=n_extra_forget, replace=True)
                augmented.extend(extra.tolist())

        # Optionally downsample easy
        if easy_downsample and np.any(easy_mask):
            easy_idx_set = set(np.where(easy_mask)[0].tolist())
            keep_frac = min(1.0, 0.5 * len(base) / max(int(np.sum(easy_mask)), 1))
            n_keep = max(1, int(int(np.sum(easy_mask)) * keep_frac))
            keep_set = set(rng.choice(np.where(easy_mask)[0], size=n_keep, replace=False).tolist())
            augmented = [i for i in augmented if i not in easy_idx_set or i in keep_set]

        result = np.array(augmented, dtype=np.int64)
        rng.shuffle(result)
        print(
            f"[OnlineMiner] Epoch {self._epoch}: "
            f"hard={int(hard_mask.sum())} forgotten={int(forgotten_mask.sum())} "
            f"easy={int(easy_mask.sum())} "
            f"-> {len(result)} indices "
            f"(base={len(base)})"
        )
        return result

    def get_hard_indices(self) -> np.ndarray:
        """Return indices of consistently hard samples (convenience)."""
        return np.where(self.get_hard_mask())[0]

    @property
    def current_epoch(self) -> int:
        return self._epoch

    def state_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "window_size": self.window_size,
            "hard_quantile": self.hard_quantile,
            "forget_window": self.forget_window,
            "easy_quantile": self.easy_quantile,
            "boost_factor": self.boost_factor,
            "decay_factor": self.decay_factor,
            "loss_buffer": self._loss_buffer.tolist(),
            "ema_score": self._ema_score.tolist(),
            "epoch": self._epoch,
            "forgetting_tracker": self._forgetting_tracker.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.n_samples = state["n_samples"]
        self.window_size = state.get("window_size", 5)
        self.hard_quantile = state.get("hard_quantile", 0.85)
        self.forget_window = state.get("forget_window", 3)
        self.easy_quantile = state.get("easy_quantile", 0.30)
        self.boost_factor = state.get("boost_factor", 2.0)
        self.decay_factor = state.get("decay_factor", 0.90)
        self._loss_buffer = np.array(state.get("loss_buffer", [[], []]), dtype=np.float32)
        self._ema_score = np.array(state.get("ema_score", []), dtype=np.float32)
        self._epoch = state.get("epoch", 0)
        ft_state = state.get("forgetting_tracker", {})
        if ft_state:
            self._forgetting_tracker.load_state_dict(ft_state)
