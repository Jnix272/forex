"""
training/train_utils.py
========================
Shared training utilities:
  - MixupBatch      : time-series MixUp augmentation
  - VolatilityStratifiedSampler : equal-tier batch sampler across vol regimes
  - RegimeTierTracker : per-difficulty-tier Sharpe tracking for conditional early stopping

Usage
-----
from training.train_utils import MixupBatch, VolatilityStratifiedSampler, RegimeTierTracker
"""

from __future__ import annotations

import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
# MixupBatch - time-series MixUp augmentation
# ─────────────────────────────────────────────────────────────────────────────


class MixupBatch:
    """
    Applies MixUp augmentation to a batch of time-series sequences.

    Randomly blends two training samples and their labels with coefficient
    lambda ~ Beta(alpha, alpha). Typical alpha=0.2 for moderate mixing.

    Parameters
    ----------
    alpha : float
        Beta distribution shape parameter. 0.0 = disabled, 0.2 = recommended.
    p : float
        Probability of applying MixUp to any given batch. 0.5 = apply half the time.

    Example
    -------
    mixup = MixupBatch(alpha=0.2, p=0.5)
    xb, yb, y_cls_b = mixup(xb, yb, y_cls_b)
    """

    def __init__(self, alpha: float = 0.2, p: float = 0.5):
        self.alpha = alpha
        self.p = p

    def __call__(
        self,
        xb: torch.Tensor,
        yb: torch.Tensor,
        y_cls_b: torch.Tensor | None = None,
        y_conf_b: torch.Tensor | None = None,
    ) -> tuple:
        if self.alpha <= 0.0 or np.random.random() > self.p:
            return xb, yb, y_cls_b, y_conf_b

        bsz = xb.size(0)
        lam = float(np.random.beta(self.alpha, self.alpha))

        # Random permutation of the batch for pairing
        idx = torch.randperm(bsz, device=xb.device)

        xb_mixed = lam * xb + (1 - lam) * xb[idx]
        yb_mixed = lam * yb + (1 - lam) * yb[idx]

        y_cls_mixed = None
        y_conf_mixed = None

        if y_cls_b is not None:
            # For classification targets: soft blend (returns soft labels)
            # If integer labels, convert to one-hot then blend
            if y_cls_b.dtype in (torch.long, torch.int32, torch.int64):
                n_cls = 3  # Sell / Hold / Buy
                # Clamp OOR labels so scatter never writes past one-hot width;
                # invalid rows stay as all-zeros (uniform / ignored by CE soft).
                y_a = y_cls_b.long()
                y_b = y_cls_b[idx].long()
                valid_a = (y_a >= 0) & (y_a < n_cls)
                valid_b = (y_b >= 0) & (y_b < n_cls)
                one_hot_a = torch.zeros(bsz, n_cls, device=xb.device, dtype=torch.float32)
                one_hot_b = torch.zeros(bsz, n_cls, device=xb.device, dtype=torch.float32)
                if bool(valid_a.any()):
                    one_hot_a.scatter_(1, y_a.clamp(0, n_cls - 1).unsqueeze(1), 1.0)
                    one_hot_a = one_hot_a * valid_a.unsqueeze(1).float()
                if bool(valid_b.any()):
                    one_hot_b.scatter_(1, y_b.clamp(0, n_cls - 1).unsqueeze(1), 1.0)
                    one_hot_b = one_hot_b * valid_b.unsqueeze(1).float()
                y_cls_mixed = lam * one_hot_a + (1 - lam) * one_hot_b
            else:
                # Continuous / soft labels: clamp into a sane range before blend
                y_cls_mixed = lam * y_cls_b.clamp(-1.0, 1.0) + (1 - lam) * y_cls_b[idx].clamp(-1.0, 1.0)

        if y_conf_b is not None:
            y_conf_mixed = lam * y_conf_b + (1 - lam) * y_conf_b[idx]

        return xb_mixed, yb_mixed, y_cls_mixed, y_conf_mixed


# ─────────────────────────────────────────────────────────────────────────────
# VolatilityStratifiedSampler - equal-tier batch sampling across vol regimes
# ─────────────────────────────────────────────────────────────────────────────


class VolatilityStratifiedSampler:
    """
    Samples training indices so that each batch contains an equal fraction of
    low-, medium-, and high-volatility bars.

    This prevents low-vol periods (e.g. 2012–2017 quiet markets) from dominating
    training and causing the model to under-learn volatile/trending regime patterns.

    Parameters
    ----------
    train_idx : np.ndarray
        Base training indices.
    diff_array : np.ndarray or None
        Difficulty sidecar (_diff.npy) values per sample. 0=easy, 1=medium, 2=hard.
        If None, falls back to uniform sampling.
    n_samples : int
        Number of samples to draw. Defaults to len(train_idx).
    seed : int

    Example
    -------
    sampler = VolatilityStratifiedSampler(train_idx, diff_array=diff_npy)
    stratified_idx = sampler.sample()
    """  # noqa: RUF002

    def __init__(
        self,
        train_idx: np.ndarray,
        diff_array: np.ndarray | None = None,
        n_samples: int | None = None,
        seed: int = 42,
    ):
        self.train_idx = train_idx
        self.diff_array = diff_array
        self.n_samples = n_samples or len(train_idx)
        self.rng = np.random.default_rng(seed)

        if diff_array is not None and len(diff_array) > 0:
            self._tier_indices = self._build_tier_indices()
        else:
            self._tier_indices = None

    def _build_tier_indices(self) -> dict[int, np.ndarray]:
        """Partition train_idx by difficulty tier."""
        tiers: dict[int, np.ndarray] = {}
        for tier in (0, 1, 2):
            mask = self.diff_array[self.train_idx] == tier
            tier_idx = self.train_idx[mask]
            if len(tier_idx) > 0:
                tiers[tier] = tier_idx
        return tiers

    def sample(self) -> np.ndarray:
        """Return a stratified index array of length n_samples."""
        if self._tier_indices is None or len(self._tier_indices) < 2:
            # Fallback: uniform shuffle
            shuffled = self.train_idx.copy()
            self.rng.shuffle(shuffled)
            return shuffled[: self.n_samples]

        # Equal split across available tiers
        n_tiers = len(self._tier_indices)
        per_tier = self.n_samples // n_tiers
        remainder = self.n_samples % n_tiers

        parts = []
        for i, (_tier, idx) in enumerate(sorted(self._tier_indices.items())):
            k = per_tier + (1 if i < remainder else 0)
            if k <= 0:
                continue
            if k <= len(idx):
                chosen = self.rng.choice(idx, size=k, replace=False)
            else:
                # Never duplicate within an epoch: take all of this tier, then
                # top up from the remaining train pool without replacement.
                chosen = idx.copy()
                need = k - len(chosen)
                pool = self.train_idx[~np.isin(self.train_idx, chosen)]
                if len(pool) >= need:
                    chosen = np.concatenate(
                        [
                            chosen,
                            self.rng.choice(pool, size=need, replace=False),
                        ]
                    )
                elif len(pool) > 0:
                    chosen = np.concatenate([chosen, pool])
                # If still short, leave under-filled rather than replace=True
            parts.append(chosen)

        if not parts:
            shuffled = self.train_idx.copy()
            self.rng.shuffle(shuffled)
            return shuffled[: self.n_samples]

        combined = np.concatenate(parts)
        # Deduplicate if top-up overlapped another tier's draw
        _, uniq_idx = np.unique(combined, return_index=True)
        combined = combined[np.sort(uniq_idx)]
        if len(combined) < self.n_samples:
            pool = self.train_idx[~np.isin(self.train_idx, combined)]
            need = self.n_samples - len(combined)
            if len(pool) > 0:
                take = self.rng.choice(pool, size=min(need, len(pool)), replace=False)
                combined = np.concatenate([combined, take])
        self.rng.shuffle(combined)
        return combined[: self.n_samples]


# ─────────────────────────────────────────────────────────────────────────────
# RegimeTierTracker - per-difficulty-tier Sharpe for conditional early stopping
# ─────────────────────────────────────────────────────────────────────────────


class RegimeTierTracker:
    """
    Tracks validation Sharpe separately for easy / medium / hard bars.

    A checkpoint is only considered "improved" if Sharpe is positive across
    ALL difficulty tiers, not just on the global average.

    Parameters
    ----------
    min_sharpe_per_tier : float
        Minimum acceptable Sharpe on each individual difficulty tier.
        Default 0.0 (just requires non-negative).

    Example
    -------
    tracker = RegimeTierTracker(min_sharpe_per_tier=0.0)
    tracker.update(epoch=5, tier_sharpes={0: 0.45, 1: 0.12, 2: -0.05})
    passes = tracker.all_tiers_pass()
    """

    def __init__(self, min_sharpe_per_tier: float = 0.0):
        self.min_sharpe = min_sharpe_per_tier
        self._history: list[dict] = []
        self._current: dict[int, float] = {}

    def update(self, epoch: int, tier_sharpes: dict[int, float]) -> None:
        """Record per-tier Sharpe for the current epoch."""
        self._current = dict(tier_sharpes)
        self._history.append({"epoch": epoch, **{f"tier_{k}": v for k, v in tier_sharpes.items()}})

    def all_tiers_pass(self) -> bool:
        """Return True if every tracked tier exceeds min_sharpe_per_tier."""
        if not self._current:
            return True  # no data yet → don't block
        return all(v >= self.min_sharpe for v in self._current.values())

    def summary(self) -> str:
        parts = [f"tier{k}={v:.3f}" for k, v in sorted(self._current.items())]
        passing = self.all_tiers_pass()
        return f"[RegimeTiers] {' | '.join(parts)} | {'PASS' if passing else 'FAIL'}"

    def best_epoch(self) -> int | None:
        """Return the epoch where the sum of tier Sharpes was highest."""
        if not self._history:
            return None
        best = max(self._history, key=lambda r: sum(v for k, v in r.items() if k != "epoch"))
        return best["epoch"]
