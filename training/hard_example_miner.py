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
miner = HardExampleMiner(run_name="haelt_0611", model_name="haelt")
miner.collect(
    val_indices   = np.arange(1000),
    predictions   = np.array([...]),   # shape (N,) or (N, C), float logits/probs
    labels        = np.array([...]),   # shape (N,) int or float
    rewards       = np.array([...]),   # shape (N,) float reward signal (optional)
    losses        = np.array([...]),   # shape (N,) float per-sample loss (optional)
    regime_labels = np.array([...]),   # shape (N,) int regime id (optional)
)
miner.save()

# Next run:
base_idx = np.arange(n_train_samples)
aug_idx  = miner.get_oversampled_indices(base_idx)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np


_DEFAULT_LOG_DIR = Path("logs/hard_examples")
_MAX_OVERSAMPLE_RATIO = 2.0   # hard examples can at most double their share
_CONFIDENCE_THRESHOLD = 0.65  # min predicted probability to count as "confident"
_WRONG_FRAC_CAP = 0.15        # cap hard examples at 15% of val set
_BOUNDARY_THRESHOLD = 0.60    # max confidence to be considered "uncertain"


class HardExampleMiner:
    """Collects hard examples from validation and persists them for next run.

    Parameters
    ----------
    run_name    : unique run identifier (used in filename)
    model_name  : model being evaluated
    log_dir     : directory for .npz output files
    max_ratio   : maximum oversampling multiplier for hard examples
    """

    def __init__(
        self,
        run_name:   str,
        model_name: str,
        log_dir:    str | Path = _DEFAULT_LOG_DIR,
        max_ratio:  float = _MAX_OVERSAMPLE_RATIO,
    ):
        self.run_name   = run_name
        self.model_name = model_name
        self.log_dir    = Path(log_dir)
        self.max_ratio  = max_ratio

        # populated by collect()
        self.hard_indices:    Optional[np.ndarray] = None
        self.hard_reasons:    list[str]            = []
        self.n_val:           int                  = 0
        self.n_hard:          int                  = 0
        self.metadata:        dict                 = {}
        # regime-aware tracking
        self.regime_distribution: Optional[dict] = None

    # ── collection ───────────────────────────────────────────────────────────

    def collect(
        self,
        val_indices: np.ndarray,
        predictions: np.ndarray,
        labels:      np.ndarray,
        rewards:     Optional[np.ndarray] = None,
        losses:      Optional[np.ndarray] = None,
        regime_labels: Optional[np.ndarray] = None,
        confidence_threshold: float = _CONFIDENCE_THRESHOLD,
        boundary_threshold: Optional[float] = None,
        loss_weight: float = 0.3,
    ) -> "HardExampleMiner":
        """Identify hard examples from a validation pass.

        Parameters
        ----------
        val_indices  : original dataset indices for the validation samples
        predictions  : model output — shape (N,) raw logits/scores OR
                       (N, C) class probabilities
        labels       : ground-truth — shape (N,) int or float
        rewards      : optional per-sample reward signal (N,); positive = profit
        losses       : optional per-sample loss values (N,) for loss-weighted mining
        regime_labels: optional per-sample regime label (N,) int
        confidence_threshold : min confidence to call a sample "confident"
        boundary_threshold   : max confidence to call a sample "uncertain"
                               (defaults to _BOUNDARY_THRESHOLD when None)
        loss_weight  : fraction of selection driven by loss vs. confidence
                       (0.0 = pure confidence error, 1.0 = pure loss)
        """
        val_indices = np.asarray(val_indices)
        predictions = np.asarray(predictions, dtype=np.float32)
        labels      = np.asarray(labels)
        N           = len(val_indices)
        self.n_val  = N

        if boundary_threshold is None:
            boundary_threshold = _BOUNDARY_THRESHOLD

        hard_mask = np.zeros(N, dtype=bool)
        reasons   = []

        # ── 1a. high-confidence wrong predictions ──────────────────────────
        try:
            if predictions.ndim == 2:
                pred_class = np.argmax(predictions, axis=1)
                pred_conf  = predictions[np.arange(N), pred_class]
            else:
                pred_class = (predictions >= 0.5).astype(int)
                pred_conf  = np.abs(predictions - 0.5) * 2.0

            int_labels = labels.astype(int) if labels.dtype.kind in "iuf" else labels

            wrong        = (pred_class != int_labels)
            confident    = (pred_conf  >= confidence_threshold)
            conf_wrong   = wrong & confident

            n_conf_wrong = int(conf_wrong.sum())
            if n_conf_wrong > 0:
                hard_mask |= conf_wrong
                reasons.append(
                    f"confident_wrong: {n_conf_wrong} "
                    f"(conf>={confidence_threshold:.2f})"
                )
        except Exception as e:
            print(f"[HardMiner] confident-wrong pass failed: {e}")

        # ── 1b. boundary/uncertainty mining ────────────────────────────────
        try:
            if predictions.ndim == 2:
                max_conf = np.max(predictions, axis=1)
            else:
                max_conf = pred_conf  # already computed above

            uncertain = (max_conf < boundary_threshold)
            # only count uncertain samples that aren't already flagged
            uncertain_new = uncertain & ~hard_mask

            n_uncertain = int(uncertain_new.sum())
            if n_uncertain > 0:
                hard_mask |= uncertain_new
                reasons.append(
                    f"boundary_uncertain: {n_uncertain} "
                    f"(max_conf<{boundary_threshold:.2f})"
                )
        except Exception as e:
            print(f"[HardMiner] boundary-mining pass failed: {e}")

        # ── 2. large missed reward opportunities ──────────────────────────
        if rewards is not None:
            try:
                rewards_arr = np.asarray(rewards, dtype=np.float32)
                if predictions.ndim == 2:
                    pred_flat = np.argmax(predictions, axis=1).astype(float)
                else:
                    pred_flat = predictions.astype(np.float32)

                reward_threshold = np.percentile(np.abs(rewards_arr), 85)
                large_reward     = np.abs(rewards_arr) >= reward_threshold
                pred_sign        = (pred_flat >= 0.5).astype(float) * 2 - 1
                reward_sign      = np.sign(rewards_arr)
                missed_dir       = (pred_sign * reward_sign < 0) & large_reward

                n_missed = int(missed_dir.sum())
                if n_missed > 0:
                    hard_mask |= missed_dir
                    reasons.append(
                        f"missed_large_reward: {n_missed} "
                        f"(|reward|>={reward_threshold:.4f})"
                    )
            except Exception as e:
                print(f"[HardMiner] missed-reward pass failed: {e}")

        # ── 3. loss-weighted ranking (re-ranks within existing mask) ──────
        if losses is not None and loss_weight > 0.0:
            try:
                losses_arr = np.asarray(losses, dtype=np.float32).ravel()
                if len(losses_arr) == N:
                    # Normalise losses to [0, 1] across the full set
                    _min_l, _max_l = float(losses_arr.min()), float(losses_arr.max())
                    if _max_l > _min_l:
                        loss_score = (losses_arr - _min_l) / (_max_l - _min_l)
                    else:
                        loss_score = np.zeros(N, dtype=np.float32)

                    # Blend confidence error with loss score
                    if predictions.ndim == 2:
                        conf_score = pred_conf
                    else:
                        conf_score = np.abs(predictions - 0.5) * 2.0
                    conf_score = np.clip(conf_score, 0.0, 1.0)

                    # Blended score: high confidence wrong OR high loss
                    wrong_float = wrong.astype(float) if 'wrong' in dir() else (pred_class != int_labels).astype(float)
                    if 'wrong' not in dir():
                        wrong_float = (pred_class != (labels.astype(int) if labels.dtype.kind in 'iuf' else labels)).astype(float)

                    blend = (
                        (1.0 - loss_weight) * conf_score * wrong_float
                        + loss_weight * loss_score
                    )

                    # Re-rank hard candidates by blended score
                    cap = max(1, int(N * _WRONG_FRAC_CAP))
                    hard_positions = np.where(hard_mask)[0]
                    if len(hard_positions) > cap:
                        top_k = np.argsort(blend[hard_positions])[-cap:]
                        hard_positions = hard_positions[top_k]
                        # Rebuild mask from re-ranked subset
                        new_mask = np.zeros(N, dtype=bool)
                        new_mask[hard_positions] = True
                        hard_mask = new_mask
                        reasons.append(
                            f"loss_ranked: capped at {cap} "
                            f"(loss_weight={loss_weight:.2f})"
                        )
            except Exception as e:
                print(f"[HardMiner] loss-weighted pass failed: {e}")

        # ── cap at _WRONG_FRAC_CAP of val set ────────────────────────────
        cap = max(1, int(N * _WRONG_FRAC_CAP))
        hard_positions = np.where(hard_mask)[0]
        if len(hard_positions) > cap:
            try:
                if predictions.ndim == 2:
                    conf_at_hard = pred_conf[hard_positions]
                else:
                    conf_at_hard = np.abs(predictions[hard_positions] - 0.5) * 2.0
                top_k = np.argsort(conf_at_hard)[-cap:]
                hard_positions = hard_positions[top_k]
            except Exception as e:
                print(f"[HardMiner] WARNING: Fallback triggered on hard example sorting due to error: {e}")
                hard_positions = hard_positions[:cap]

        self.hard_indices = val_indices[hard_positions] if len(hard_positions) else np.array([], dtype=np.int64)
        self.hard_reasons = reasons
        self.n_hard       = len(self.hard_indices)

        # ── regime-aware tracking ─────────────────────────────────────────
        self.regime_distribution = None
        if regime_labels is not None and len(hard_positions) > 0:
            try:
                regime_arr = np.asarray(regime_labels, dtype=np.int32).ravel()
                if len(regime_arr) == N:
                    hard_regimes = regime_arr[hard_positions]
                    unique, counts = np.unique(hard_regimes, return_counts=True)
                    self.regime_distribution = {
                        int(k): int(v) for k, v in zip(unique, counts)
                    }
            except Exception as e:
                print(f"[HardMiner] regime tracking failed: {e}")

        self.metadata = {
            "run_name":   self.run_name,
            "model_name": self.model_name,
            "n_val":      N,
            "n_hard":     self.n_hard,
            "frac_hard":  round(self.n_hard / max(N, 1), 4),
            "reasons":    reasons,
        }
        if self.regime_distribution is not None:
            self.metadata["regime_distribution"] = self.regime_distribution

        print(
            f"[HardMiner] {self.model_name}: "
            f"{self.n_hard}/{N} hard examples "
            f"({self.metadata['frac_hard']:.1%}) — "
            + (", ".join(reasons) if reasons else "none found")
        )
        return self

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self) -> Optional[Path]:
        """Atomically write hard examples to .npz."""
        if self.hard_indices is None or self.n_hard == 0:
            print(f"[HardMiner] {self.model_name}: nothing to save.")
            return None

        self.log_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.log_dir / f"{self.run_name}_{self.model_name}_hard_examples.npz"

        fd, tmp = tempfile.mkstemp(
            prefix=f".hard_{self.model_name}.", suffix=".tmp.npz",
            dir=str(self.log_dir),
        )
        os.close(fd)
        try:
            np.savez_compressed(
                tmp,
                hard_indices = self.hard_indices,
                n_val        = np.array([self.n_val]),
                n_hard       = np.array([self.n_hard]),
            )
            os.replace(tmp, out_path)
            print(f"[HardMiner] Saved {self.n_hard} hard examples -> {out_path}")
            return out_path
        except Exception as e:
            print(f"[HardMiner] Save failed: {e}")
            return None
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    @classmethod
    def load(cls, path: str | Path) -> Optional[np.ndarray]:
        """Load hard example indices from a saved .npz file."""
        p = Path(path)
        if not p.exists():
            return None
        try:
            data = np.load(p)
            return data["hard_indices"]
        except Exception as e:
            print(f"[HardMiner] Could not load {p}: {e}")
            return None

    # ── oversampling ─────────────────────────────────────────────────────────

    def get_oversampled_indices(
        self,
        base_indices: np.ndarray,
        oversample_factor: float = 1.5,
    ) -> np.ndarray:
        """Return base_indices with hard examples lightly repeated.

        Parameters
        ----------
        base_indices      : normal training index array
        oversample_factor : how many extra copies of each hard example (e.g.
                            1.5 → each hard example appears 1.5× more often
                            than average)

        Returns
        -------
        Shuffled index array of the same dtype as base_indices.
        """
        if self.hard_indices is None or self.n_hard == 0:
            return base_indices

        factor = min(float(oversample_factor), self.max_ratio)
        extra_count = max(0, int(self.n_hard * (factor - 1.0)))
        if extra_count == 0:
            return base_indices

        rng        = np.random.default_rng()
        extra_pool = np.tile(self.hard_indices, (extra_count // self.n_hard) + 2)
        extra      = rng.choice(extra_pool, size=extra_count, replace=False)

        augmented = np.concatenate([base_indices, extra])
        rng.shuffle(augmented)
        print(
            f"[HardMiner] Oversampled {self.n_hard} hard examples "
            f"(+{extra_count} copies, factor={factor:.2f}) "
            f"-> total {len(augmented)} indices"
        )
        return augmented

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def find_latest(
        log_dir:    str | Path,
        model_name: str,
    ) -> Optional[Path]:
        """Return the most-recently-modified hard-example file for a model."""
        log_dir = Path(log_dir)
        if not log_dir.exists():
            return None
        candidates = sorted(
            log_dir.glob(f"*_{model_name}_hard_examples.npz"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None


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

    Unlike :class:`HardExampleMiner` which operates on a single validation pass,
    this miner maintains rolling per-sample statistics across *all* training
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
        """Finalise the current epoch: update forgetting tracker with epoch mean."""
        epoch_losses = np.nanmean(self._loss_buffer, axis=0)
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
