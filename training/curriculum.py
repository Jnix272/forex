"""
Curriculum Learning Module (Improvement #9)
===========================================
Difficulty curriculum, self-paced learning, and loss-based sample weighting
for supervised and RL training.

Extends the existing CurriculumController with:
  1. Difficulty curriculum - progressive data difficulty for supervised learning
  2. Self-paced learning - adjust sample inclusion based on model competence
  3. Loss-based sample weighting - reweight samples by difficulty/loss
  4. Integration with existing CurriculumController for unified control
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset

from training.curriculum_controller import CurriculumController

# ════════════════════════════════════════════════════════════════════════════
# 1. Difficulty Curriculum for Supervised Learning
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class DifficultyCurriculumConfig:
    """Configuration for difficulty-based curriculum."""
    # Difficulty levels (0=easiest, 1=hardest)
    n_levels: int = 10
    # Start from easiest level
    start_level: float = 0.0
    # How fast to advance (per epoch)
    advance_rate: float = 0.1
    # Minimum competence before advancing
    min_competence: float = 0.7
    # Competence measured on validation
    competence_metric: str = "accuracy"  # "accuracy", "loss", "f1"
    # Curriculum pacing function: "linear", "exp", "sqrt", "step"
    pace_function: str = "linear"
    # Hardest samples included at this pace
    max_level: float = 1.0


class DifficultyCurriculum:
    """
    Difficulty-based curriculum for supervised learning.
    
    Orders training samples by difficulty and progressively includes
    harder samples as model competence improves.
    
    Usage:
        curriculum = DifficultyCurriculum(config, difficulty_scores)
        for epoch in range(epochs):
            mask = curriculum.get_inclusion_mask(epoch, val_metric)
            train_loader = DataLoader(dataset[mask], ...)
    """

    def __init__(
        self,
        config: DifficultyCurriculumConfig,
        difficulty_scores: np.ndarray,  # per-sample difficulty in [0, 1]
        sample_indices: np.ndarray | None = None,
    ):
        self.config = config
        self.difficulty = np.asarray(difficulty_scores, dtype=float)
        self.indices = sample_indices if sample_indices is not None else np.arange(len(difficulty_scores))
        self.current_level = config.start_level
        self.current_epoch = 0

        # Sort indices by difficulty
        self.sorted_idx = np.argsort(self.difficulty)
        self.sorted_difficulty = self.difficulty[self.sorted_idx]

    def _pace(self, epoch: int) -> float:
        """Compute current difficulty level based on pace function."""
        # Guard against advance_rate=0 which would cause ZeroDivisionError
        advance_rate = max(1e-6, self.config.advance_rate)
        total_epochs = int(1 / advance_rate)
        e = min(epoch / max(1, total_epochs), 1.0)
        if self.config.pace_function == "linear":
            return self.config.start_level + e * (self.config.max_level - self.config.start_level)
        elif self.config.pace_function == "exp":
            return self.config.start_level + (self.config.max_level - self.config.start_level) * (1 - np.exp(-5 * e))
        elif self.config.pace_function == "sqrt":
            return self.config.start_level + (self.config.max_level - self.config.start_level) * np.sqrt(e)
        elif self.config.pace_function == "step":
            n_steps = int(1 / advance_rate)
            step = min(epoch // max(1, total_epochs // n_steps), n_steps - 1)
            return self.config.start_level + (self.config.max_level - self.config.start_level) * step / (n_steps - 1)
        else:
            return self.config.start_level + e * (self.config.max_level - self.config.start_level)

    def update(self, epoch: int, val_metric: float = 0.0) -> float:
        """Update curriculum level based on epoch and validation metric."""
        self.current_epoch = epoch
        self.current_level = self._pace(epoch)

        # Optionally adjust based on validation performance
        if val_metric > 0 and self.config.min_competence > 0:
            if val_metric < self.config.min_competence:
                # Slow down if competence is low
                self.current_level *= 0.9

        self.current_level = min(self.current_level, self.config.max_level)
        return self.current_level

    def get_inclusion_mask(self, epoch: int = None) -> np.ndarray:
        """Boolean mask of samples to include at current level."""
        if epoch is not None:
            self.update(epoch)
        n = len(self.difficulty)
        n_include = int(self.current_level * n)
        n_include = max(1, min(n_include, n))
        return np.isin(self.indices, self.sorted_idx[:n_include])

    def get_difficulty_weights(self, epoch: int = None) -> np.ndarray:
        """Sample weights inversely proportional to difficulty (easier samples weighted more)."""
        if epoch is not None:
            self.update(epoch)
        # Weight = 1 - difficulty * current_level
        weights = 1.0 - self.difficulty * self.current_level
        weights = np.clip(weights, 0.1, 1.0)  # minimum weight 0.1
        return weights

    def get_sorted_indices(self, epoch: int = None) -> np.ndarray:
        """Return indices sorted by difficulty up to current level."""
        if epoch is not None:
            self.update(epoch)
        n = len(self.difficulty)
        n_include = int(self.current_level * n)
        n_include = max(1, min(n_include, n))
        return self.sorted_idx[:n_include]


# ════════════════════════════════════════════════════════════════════════════
# 2. Self-Paced Learning (Kumar et al., 2010)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class SelfPacedConfig:
    """Configuration for self-paced learning."""
    # Pacing function: "linear", "log", "root"
    pace: str = "linear"
    # Regularization parameter (controls pace)
    lambda_pace: float = 1.0
    # Number of epochs to reach full dataset
    total_epochs: int = 100
    # Minimum fraction of data to use
    min_fraction: float = 0.1
    # Whether to use loss-based weighting within included samples
    use_loss_weighting: bool = True
    # Temperature for loss-based soft weighting
    loss_temp: float = 1.0


class SelfPacedLearning:
    """
    Self-paced learning (Kumar et al., 2010).
    
    Jointly optimizes model parameters and sample inclusion:
    min_{w,v} sum_i v_i * L(x_i, y_i; w) + f(v; lambda)
    
    where v_i in [0,1] are sample inclusion weights.
    
    This implementation uses an alternating optimization:
    1. Fix v, update w (model training)
    2. Fix w, update v (easy samples first)
    
    Usage:
        spl = SelfPacedLearning(config, losses_per_sample)
        for epoch in range(epochs):
            weights = spl.get_weights(epoch)
            # Train with sample weights
    """

    def __init__(
        self,
        config: SelfPacedConfig,
        n_samples: int,
        initial_losses: np.ndarray | None = None,
    ):
        self.config = config
        self.n_samples = n_samples
        self.current_epoch = 0

        # Sample inclusion weights v_i in [0, 1]
        self.v = np.full(n_samples, config.min_fraction, dtype=float)

        if initial_losses is not None:
            self.update_weights(initial_losses)

    def update_weights(self, losses: np.ndarray) -> np.ndarray:
        """
        Update sample inclusion weights based on current losses.
        
        SPL objective: v_i = 1 if L_i <= lambda, else 0 (hard)
        Soft version: v_i = exp(-L_i / tau)
        """
        losses = np.asarray(losses, dtype=float)
        losses = np.clip(losses, 0, 100)  # clip extreme losses

        if self.config.use_loss_weighting:
            # Soft weighting based on loss
            # v_i proportional to exp(-L_i / (lambda * tau))
            tau = self.config.loss_temp
            lam = self.config.lambda_pace * (1 + self.current_epoch / max(1, self.config.total_epochs))
            self.v = np.exp(-losses / (lam * tau + 1e-8))
        else:
            # Hard thresholding
            lam = self.config.lambda_pace * (1 + self.current_epoch / max(1, self.config.total_epochs))
            self.v = (losses <= lam).astype(float)

        # Ensure minimum fraction
        self.v = np.clip(self.v, self.config.min_fraction, 1.0)
        return self.v

    def get_weights(self, epoch: int, losses: np.ndarray | None = None) -> np.ndarray:
        """Get sample weights for current epoch."""
        self.current_epoch = epoch
        # If losses provided, update weights; otherwise use pace-based uniform weights
        if losses is not None:
            self.update_weights(losses)
        else:
            # Default: uniform weights based on pace
            pace = self.get_pace(epoch)
            self.v = np.full(self.n_samples, pace, dtype=float)
        return self.v

    def get_pace(self, epoch: int) -> float:
        """Current pace (fraction of data included)."""
        e = min(epoch / max(1, self.config.total_epochs), 1.0)
        if self.config.pace == "linear":
            return self.config.min_fraction + (1 - self.config.min_fraction) * e
        elif self.config.pace == "log":
            return self.config.min_fraction + (1 - self.config.min_fraction) * np.log(1 + 9 * e) / np.log(10)
        elif self.config.pace == "root":
            return self.config.min_fraction + (1 - self.config.min_fraction) * np.sqrt(e)
        return self.config.min_fraction + (1 - self.config.min_fraction) * e


# ════════════════════════════════════════════════════════════════════════════
# 3. Loss-Based Sample Weighting
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class LossWeightingConfig:
    """Configuration for loss-based sample weighting."""
    # Weighting scheme: "inverse", "focal", "threshold", "softmax"
    scheme: str = "focal"
    # Focal loss gamma
    focal_gamma: float = 2.0
    # Focal loss alpha
    focal_alpha: float = 0.25
    # Threshold for hard examples
    hard_threshold: float = 0.7
    # Curriculum temperature
    temperature: float = 1.0
    # Minimum weight
    min_weight: float = 0.01
    # Maximum weight
    max_weight: float = 10.0
    # EMA decay for online weight updates
    ema_decay: float = 0.99


class LossBasedWeighting:
    """
    Loss-based sample weighting for curriculum learning.
    
    Supports multiple weighting schemes:
    - "inverse": weight = 1 / (loss + eps)
    - "focal": weight = (1 - exp(-loss))^gamma
    - "threshold": weight = 1 if loss > threshold else 0.1
    - "softmax": weight = exp(loss / T) / sum(exp(loss / T))
    
    Usage:
        weighting = LossBasedWeighting(config)
        for epoch in range(epochs):
            weights = weighting.compute_weights(losses, epoch)
            # Train with sample weights
    """

    def __init__(self, config: LossWeightingConfig):
        self.config = config
        self.ema_losses = None

    def compute_weights(
        self,
        losses: np.ndarray,
        epoch: int = 0,
        total_epochs: int = 100,
    ) -> np.ndarray:
        """Compute sample weights from per-sample losses."""
        losses = np.asarray(losses, dtype=float)
        losses = np.clip(losses, 0, 100)

        # EMA of losses for smoothing
        if self.ema_losses is None:
            self.ema_losses = losses.copy()
        else:
            self.ema_losses = self.config.ema_decay * self.ema_losses + (1 - self.config.ema_decay) * losses

        # Use EMA losses for weight computation
        L = self.ema_losses

        if self.config.scheme == "inverse":
            w = 1.0 / (L + 1e-6)
        elif self.config.scheme == "focal":
            p = np.exp(-L)
            w = (1 - p) ** self.config.focal_gamma
        elif self.config.scheme == "threshold":
            w = np.where(self.config.hard_threshold < L, 1.0, 0.1)
        elif self.config.scheme == "softmax":
            T = self.config.temperature
            w = np.exp(L / T)
            w = w / (w.sum() + 1e-8) * len(L)
        else:
            w = np.ones_like(L)

        # Normalize and clip
        w = np.clip(w, self.config.min_weight, self.config.max_weight)
        # Normalize to mean 1.0
        w = w / (w.mean() + 1e-8)

        return w


# ════════════════════════════════════════════════════════════════════════════
# 4. Integrated Curriculum Manager
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class CurriculumManagerConfig:
    """Unified configuration for all curriculum components."""
    # Difficulty curriculum
    difficulty: DifficultyCurriculumConfig | None = None
    # Self-paced learning
    self_paced: SelfPacedConfig | None = None
    # Loss-based weighting
    loss_weighting: LossWeightingConfig | None = None
    # Performance-gated curriculum (existing)
    adaptive: Any | None = None  # AdaptiveCurriculumConfig
    # Combination mode
    mode: str = "adaptive"  # "difficulty", "self_paced", "loss_weighting", "adaptive", "combined"
    # Combination weights
    difficulty_weight: float = 0.4
    self_paced_weight: float = 0.3
    loss_weight: float = 0.3


class CurriculumManager:
    """
    Unified curriculum manager combining all curriculum strategies.
    
    Supports multiple modes:
    - "difficulty": Progressive difficulty inclusion
    - "self_paced": Self-paced learning with loss-based pacing
    - "loss_weighting": Dynamic loss-based sample weighting
    - "adaptive": Performance-gated curriculum (existing CurriculumController)
    - "combined": Weighted combination of all strategies
    """

    def __init__(
        self,
        config: CurriculumManagerConfig,
        n_samples: int,
        difficulty_scores: np.ndarray | None = None,
    ):
        self.config = config
        self.n_samples = n_samples
        self.current_epoch = 0

        # Initialize components
        self.difficulty_curriculum = None
        self.self_paced = None
        self.loss_weighting = None
        self.adaptive_controller = None

        if config.mode in ("difficulty", "combined") and config.difficulty:
            if difficulty_scores is not None:
                self.difficulty_curriculum = DifficultyCurriculum(config.difficulty, difficulty_scores)

        if config.mode in ("self_paced", "combined") and config.self_paced:
            self.self_paced = SelfPacedLearning(config.self_paced, n_samples)

        if config.mode in ("loss_weighting", "combined") and config.loss_weighting:
            self.loss_weighting = LossBasedWeighting(config.loss_weighting)

        if config.mode in ("adaptive", "combined") and config.adaptive:
            self.adaptive_controller = CurriculumController(config=config.adaptive)

    def update(self, epoch: int, val_metrics: dict[str, float] = None, losses: np.ndarray = None) -> dict[str, Any]:
        """Update all curriculum components and return combined sample weights."""
        self.current_epoch = epoch
        weights = np.ones(self.n_samples, dtype=float)
        info = {"epoch": epoch}

        # Update difficulty curriculum
        if self.difficulty_curriculum:
            level = self.difficulty_curriculum.update(epoch, val_metrics.get("val_accuracy", 0.0) if val_metrics else 0.0)
            mask = self.difficulty_curriculum.get_inclusion_mask()
            diff_weights = self.difficulty_curriculum.get_difficulty_weights()
            info["difficulty_level"] = self.difficulty_curriculum.current_level
            info["inclusion_rate"] = mask.mean()

        # Update self-paced learning
        if self.self_paced and losses is not None:
            self.self_paced.update_weights(losses)
            sp_weights = self.self_paced.get_weights(epoch)
            info["self_paced_pace"] = self.self_paced.get_pace(epoch)

        # Update loss-based weighting
        if self.loss_weighting and losses is not None:
            lw_weights = self.loss_weighting.compute_weights(losses, epoch)
            info["loss_weighting_stats"] = {
                "mean": float(self.loss_weighting.ema_losses.mean()) if self.loss_weighting.ema_losses is not None else 0,
                "weight_mean": float(lw_weights.mean()),
                "weight_std": float(lw_weights.std()),
            }

        # Update adaptive controller
        if self.adaptive_controller and val_metrics:
            val_sharpe = val_metrics.get("val_sharpe", 0.0)
            val_loss = val_metrics.get("val_loss", 0.0)
            actions = self.adaptive_controller.evaluate_epoch(epoch, val_sharpe, val_loss)
            info["adaptive_actions"] = actions

        # Combine weights
        if self.config.mode == "difficulty" and self.difficulty_curriculum:
            weights = self.difficulty_curriculum.get_difficulty_weights()
        elif self.config.mode == "self_paced" and self.self_paced:
            weights = self.self_paced.get_weights(epoch)
        elif self.config.mode == "loss_weighting" and self.loss_weighting and losses is not None:
            weights = self.loss_weighting.compute_weights(losses, epoch)
        elif self.config.mode == "adaptive" and self.adaptive_controller:
            # Adaptive doesn't directly provide sample weights
            weights = np.ones(self.n_samples)
        elif self.config.mode == "combined":
            # Weighted combination
            w = np.ones(self.n_samples)
            if self.difficulty_curriculum:
                w *= self.difficulty_curriculum.get_difficulty_weights() ** self.config.difficulty_weight
            if self.self_paced:
                w *= self.self_paced.get_weights(epoch) ** self.config.self_paced_weight
            if self.loss_weighting and losses is not None:
                w *= self.loss_weighting.compute_weights(losses, epoch) ** self.config.loss_weight
            # Normalize
            w = w / (w.mean() + 1e-8)
            weights = w

        info["weights"] = weights
        info["epoch"] = epoch
        return info

    def get_sample_weights(self) -> np.ndarray:
        """Get current combined sample weights."""
        return self.update(self.current_epoch).get("weights", np.ones(self.n_samples))

    def get_inclusion_mask(self) -> np.ndarray:
        """Get current sample inclusion mask (for data filtering)."""
        if self.difficulty_curriculum:
            return self.difficulty_curriculum.get_inclusion_mask()
        return np.ones(self.n_samples, dtype=bool)

    def state_dict(self) -> dict[str, Any]:
        """Get full state for checkpointing."""
        return {
            "epoch": self.current_epoch,
            "config_mode": self.config.mode,
            "difficulty": self.difficulty_curriculum.current_level if self.difficulty_curriculum else None,
            "self_paced_pace": self.self_paced.get_pace(self.current_epoch) if self.self_paced else None,
            "adaptive_state": self.adaptive_controller.state_dict() if self.adaptive_controller else None,
        }


# ════════════════════════════════════════════════════════════════════════════
# 5. Integration with DataLoader
# ════════════════════════════════════════════════════════════════════════════

class CurriculumDataLoader:
    """
    DataLoader wrapper that applies curriculum sample weights and inclusion masks.
    
    Usage:
        curriculum = CurriculumManager(config, n_samples, difficulty_scores)
        loader = CurriculumDataLoader(dataset, curriculum, batch_size=32)
        for epoch in range(epochs):
            curriculum.update(epoch, val_metrics, losses)
            for batch in loader:
                # batch includes sample weights
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        curriculum_manager: CurriculumManager,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        self.dataset = dataset
        self.curriculum = curriculum_manager
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        """Set current epoch (call at start of each epoch)."""
        self.current_epoch = epoch

    def __iter__(self):
        # Get current sample weights and inclusion mask
        weights = self.curriculum.get_sample_weights()
        mask = self.curriculum.get_inclusion_mask()

        # Create weighted sampler
        included_indices = np.where(mask)[0]
        if len(included_indices) == 0:
            included_indices = np.arange(len(self.dataset))

        sample_weights = weights[mask]
        sample_weights = sample_weights / (sample_weights.sum() + 1e-8)

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(included_indices),
            replacement=True,
        )

        # Create subset dataset
        subset = Subset(self.dataset, included_indices)

        loader = torch.utils.data.DataLoader(
            subset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return iter(loader)

    def __len__(self):
        mask = self.curriculum.get_inclusion_mask()
        n_included = max(1, mask.sum())
        return (n_included + self.batch_size - 1) // self.batch_size




# ════════════════════════════════════════════════════════════════════════════
# 6. Difficulty Scoring Utilities
# ════════════════════════════════════════════════════════════════════════════

def compute_difficulty_scores(
    features: np.ndarray,
    labels: np.ndarray,
    method: str = "margin",
    model: nn.Module | None = None,
) -> np.ndarray:
    """
    Compute per-sample difficulty scores in [0, 1].
    
    Methods:
    - "margin": 1 - |model_output_margin| (for classifier)
    - "loss": per-sample loss (requires model)
    - "entropy": label entropy (for multi-class)
    - "noise": label noise estimate
    - "distance": distance to decision boundary
    - "heuristic": feature-based (e.g., volatility, spread)
    
    Returns:
        difficulty scores in [0, 1] (0=easiest, 1=hardest)
    """
    n = len(labels)
    if method == "heuristic":
        # Feature-based difficulty for financial data:
        # Higher volatility (std across timesteps) = harder bar.
        # Works on both 2D (samples, features) and 3D (samples, time, features) inputs.
        if features.ndim == 3:
            vol = features.std(axis=1).mean(axis=1)          # (N,)
        elif features.ndim == 2:
            vol = features.std(axis=1)                         # (N,)
        else:
            vol = np.abs(features).ravel()[:len(labels)]
        vol = vol.astype(np.float64)
        rng = vol.max() - vol.min()
        if rng < 1e-9:
            return np.full(len(labels), 0.5)
        difficulty = (vol - vol.min()) / rng
        return np.clip(difficulty, 0.0, 1.0)

    if method == "margin" and model is not None:
        with torch.no_grad():
            model.eval()
            logits = model(torch.tensor(features, dtype=torch.float32))
            probs = F.softmax(logits, dim=-1)
            # Margin = max_prob - second_max_prob
            sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
            margin = sorted_probs[:, 0] - sorted_probs[:, 1]
            difficulty = 1.0 - margin.numpy()
            return np.clip(difficulty, 0, 1)

    if method == "loss" and model is not None:
        with torch.no_grad():
            model.eval()
            criterion = nn.CrossEntropyLoss(reduction="none")
            losses = criterion(
                model(torch.tensor(features, dtype=torch.float32)),
                torch.tensor(labels, dtype=torch.long)
            )
            difficulty = losses.numpy()
            # Normalize
            difficulty = (difficulty - difficulty.min()) / (difficulty.max() - difficulty.min() + 1e-8)
            return np.clip(difficulty, 0, 1)

    if method == "entropy":
        # Label entropy – only meaningful for soft/probabilistic labels.
        if labels.ndim > 1:
            # Soft labels: proper entropy
            entropy = -np.sum(labels * np.log(labels + 1e-8), axis=1)
            n_classes = labels.shape[1]
        else:
            # Hard integer labels: all samples have zero label entropy by definition.
            # Fall back to uniform difficulty (0.5 for all).
            return np.full(len(labels), 0.5)
        denom = np.log(max(n_classes, 2))   # guard log(1) == 0
        difficulty = entropy / denom
        return np.clip(difficulty, 0, 1)

    if method == "distance" and model is not None:
        with torch.no_grad():
            model.eval()
            # Distance to decision boundary in feature space
            embeddings = model.encoder(torch.tensor(features, dtype=torch.float32))
            # Simple: distance to centroid of own class
            centroids = {}
            for c in np.unique(labels):
                centroids[c] = embeddings[labels == c].mean(0)
            distances = np.array([torch.norm(emb - centroids[l]).item() for emb, l in zip(embeddings, labels)])
            difficulty = distances / (distances.max() + 1e-8)
            return np.clip(difficulty, 0, 1)

    # Default: uniform difficulty
    return np.full(len(labels), 0.5)


# ════════════════════════════════════════════════════════════════════════════
# 6. Training Integration Helpers
# ════════════════════════════════════════════════════════════════════════════

def create_curriculum_manager(
    mode: str = "combined",
    n_samples: int = 10000,
    difficulty_scores: np.ndarray | None = None,
    **kwargs,
) -> CurriculumManager:
    """
    Factory function to create a CurriculumManager with sensible defaults.
    
    Args:
        mode: "difficulty", "self_paced", "loss_weighting", "adaptive", "combined"
        n_samples: Number of training samples
        difficulty_scores: Per-sample difficulty scores in [0, 1]
        **kwargs: Additional config parameters
        
    Returns:
        Configured CurriculumManager
    """
    difficulty_cfg = None
    self_paced_cfg = None
    loss_weighting_cfg = None
    adaptive_cfg = None

    if mode in ("difficulty", "combined"):
        difficulty_cfg = DifficultyCurriculumConfig(
            n_levels=kwargs.get("n_levels", 10),
            advance_rate=kwargs.get("advance_rate", 0.1),
            min_competence=kwargs.get("min_competence", 0.7),
            pace_function=kwargs.get("pace_function", "linear"),
        )

    if mode in ("self_paced", "combined"):
        self_paced_cfg = SelfPacedConfig(
            pace=kwargs.get("sp_pace", "linear"),
            lambda_pace=kwargs.get("sp_lambda", 1.0),
            total_epochs=kwargs.get("total_epochs", 100),
        )

    if mode in ("loss_weighting", "combined"):
        loss_weighting_cfg = LossWeightingConfig(
            scheme=kwargs.get("lw_scheme", "focal"),
            focal_gamma=kwargs.get("focal_gamma", 2.0),
        )

    if mode in ("adaptive", "combined"):
        from training.curriculum_controller import AdaptiveCurriculumConfig
        adaptive_cfg = AdaptiveCurriculumConfig(
            stages=("easy", "medium", "hard"),
            seq_lens=(30, 60, 90, 120),
            stable_epochs_required=3,
        )

    config = CurriculumManagerConfig(
        difficulty=difficulty_cfg,
        self_paced=self_paced_cfg,
        loss_weighting=loss_weighting_cfg,
        adaptive=adaptive_cfg,
        mode=mode,
        difficulty_weight=kwargs.get("difficulty_weight", 0.4),
        self_paced_weight=kwargs.get("self_paced_weight", 0.3),
        loss_weight=kwargs.get("loss_weight", 0.3),
    )

    return CurriculumManager(config, n_samples, difficulty_scores)


# ════════════════════════════════════════════════════════════════════════════
# 7. Export
# ════════════════════════════════════════════════════════════════════════════

__all__ = [
    "CurriculumDataLoader",
    "CurriculumManager",
    "CurriculumManagerConfig",
    "DifficultyCurriculum",
    "DifficultyCurriculumConfig",
    "LossBasedWeighting",
    "LossWeightingConfig",
    "SelfPacedConfig",
    "SelfPacedLearning",
    "compute_difficulty_scores",
    "create_curriculum_manager",
]
