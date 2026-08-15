"""
Curriculum Learning Callbacks for Composer and PyTorch Lightning.

This module provides standardized curriculum learning callbacks that can be used
with both MosaicML Composer and PyTorch Lightning trainers, replacing the
custom curriculum implementations with proven library patterns.

Key concepts:
- Difficulty-based curriculum: progressively include harder samples
- Self-paced learning: weight samples by their loss (easy samples first)
- Pace functions: linear, exponential, sqrt, step, log, root
- Integration with existing custom curriculum via adapter pattern

Usage with Composer:
    from training.curriculum_callbacks import DifficultyCurriculumCallback
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        callbacks=[DifficultyCurriculumCallback(difficulty_scores, pace_fn="linear")],
    )

Usage with PyTorch Lightning:
    from training.curriculum_callbacks import PLCurriculumCallback
    trainer = Trainer(
        callbacks=[PLCurriculumCallback(difficulty_scores, pace_fn="linear")],
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union, Callable
import warnings

import numpy as np

try:
    import torch
    import torch.utils.data as torch_data
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    torch_data = None

try:
    import pytorch_lightning as pl
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    pl = None


# ════════════════════════════════════════════════════════════════════════════
# Core Curriculum Logic (framework-agnostic)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    # Pace function: "linear", "exp", "sqrt", "step", "log", "root"
    pace_function: str = "linear"
    # Total epochs to reach full dataset
    total_epochs: int = 100
    # Minimum fraction of data to start with
    min_fraction: float = 0.1
    # Maximum fraction of data (for early stopping before full dataset)
    max_fraction: float = 1.0
    # For step function: number of discrete steps
    n_steps: int = 10
    # For exp function: exponential rate
    exp_rate: float = 5.0
    # Whether to use loss-based weighting within included samples
    use_loss_weighting: bool = False
    # Temperature for loss-based soft weighting
    loss_temperature: float = 1.0


class BaseCurriculum:
    """Framework-agnostic curriculum logic."""
    
    def __init__(
        self,
        config: CurriculumConfig,
        difficulty_scores: np.ndarray,
        sample_indices: Optional[np.ndarray] = None,
    ):
        self.config = config
        self.difficulty = np.asarray(difficulty_scores, dtype=float)
        self.n_samples = len(self.difficulty)
        self.indices = sample_indices if sample_indices is not None else np.arange(self.n_samples)
        self.current_epoch = 0
        
        # Sort indices by difficulty (easiest first)
        self.sorted_idx = np.argsort(self.difficulty)
        self.sorted_difficulty = self.difficulty[self.sorted_idx]
        
        # Current inclusion mask and weights
        self._inclusion_mask = np.zeros(self.n_samples, dtype=bool)
        self._sample_weights = np.ones(self.n_samples, dtype=float)
        
    def _pace(self, epoch: int) -> float:
        """Compute current pace (fraction of data included) based on pace function."""
        e = min(epoch / max(1, self.config.total_epochs), 1.0)
        min_frac = self.config.min_fraction
        max_frac = self.config.max_fraction
        span = max_frac - min_frac
        
        if self.config.pace_function == "linear":
            return min_frac + span * e
        elif self.config.pace_function == "exp":
            return min_frac + span * (1 - np.exp(-self.config.exp_rate * e))
        elif self.config.pace_function == "sqrt":
            return min_frac + span * np.sqrt(e)
        elif self.config.pace_function == "step":
            n_steps = self.config.n_steps
            step = min(int(e * n_steps), n_steps - 1)
            return min_frac + span * step / max(1, n_steps - 1)
        elif self.config.pace_function == "log":
            return min_frac + span * np.log(1 + 9 * e) / np.log(10)
        elif self.config.pace_function == "root":
            return min_frac + span * np.sqrt(e)
        else:
            warnings.warn(f"Unknown pace function: {self.config.pace_function}, using linear")
            return min_frac + span * e
    
    def update(self, epoch: int, losses: Optional[np.ndarray] = None) -> dict[str, Any]:
        """Update curriculum state for given epoch."""
        self.current_epoch = epoch
        pace = self._pace(epoch)
        n_include = int(pace * self.n_samples)
        n_include = max(1, min(n_include, self.n_samples))
        
        # Update inclusion mask
        self._inclusion_mask[:] = False
        self._inclusion_mask[self.sorted_idx[:n_include]] = True
        
        # Update sample weights
        if self.config.use_loss_weighting and losses is not None:
            # Soft weighting: weight = exp(-loss / temperature)
            # Only for included samples
            included_losses = losses[self._inclusion_mask]
            if len(included_losses) > 0:
                included_losses = np.clip(included_losses, 0, 100)
                weights = np.exp(-included_losses / (self.config.loss_temperature + 1e-8))
                self._sample_weights[:] = 0.0
                self._sample_weights[self._inclusion_mask] = weights
            else:
                self._sample_weights[:] = 1.0
        else:
            # Uniform weighting for included samples
            self._sample_weights[:] = 0.0
            self._sample_weights[self._inclusion_mask] = 1.0
        
        return {
            "pace": pace,
            "n_included": n_include,
            "inclusion_rate": n_include / self.n_samples,
            "weights_mean": float(self._sample_weights.mean()),
            "current_epoch": epoch,
        }
    
    def get_inclusion_mask(self) -> np.ndarray:
        """Boolean mask of samples to include."""
        return self._inclusion_mask.copy()
    
    def get_sample_weights(self) -> np.ndarray:
        """Sample weights for weighted sampling."""
        return self._sample_weights.copy()
    
    def get_state(self) -> dict[str, Any]:
        """Get curriculum state for checkpointing."""
        return {
            "current_epoch": self.current_epoch,
            "config": {
                "pace_function": self.config.pace_function,
                "total_epochs": self.config.total_epochs,
                "min_fraction": self.config.min_fraction,
                "max_fraction": self.config.max_fraction,
                "n_steps": self.config.n_steps,
                "exp_rate": self.config.exp_rate,
                "use_loss_weighting": self.config.use_loss_weighting,
                "loss_temperature": self.config.loss_temperature,
            },
        }
    
    def load_state(self, state: dict[str, Any]) -> None:
        """Load curriculum state from checkpoint."""
        self.current_epoch = state.get("current_epoch", 0)


# ════════════════════════════════════════════════════════════════════════════
# PyTorch Lightning Callback
# ════════════════════════════════════════════════════════════════════════════

class PLCurriculumCallback(pl.Callback):
    """
    PyTorch Lightning callback for curriculum learning.
    
    This callback works by updating the DataModule's sampler/weights at epoch start.
    Requires the DataModule to implement `set_curriculum_weights(weights)` or
    have a `train_sampler` attribute that accepts weights.
    
    Usage:
        # In your DataModule:
        class MyDataModule(LightningDataModule):
            def __init__(self, ...):
                self.curriculum_weights = None
            
            def set_curriculum_weights(self, weights):
                self.curriculum_weights = weights
            
            def train_dataloader(self):
                if self.curriculum_weights is not None:
                    sampler = WeightedRandomSampler(
                        weights=self.curriculum_weights,
                        num_samples=len(self.curriculum_weights),
                        replacement=True
                    )
                    return DataLoader(dataset, sampler=sampler, ...)
                return DataLoader(dataset, shuffle=True, ...)
        
        # In your trainer:
        trainer = Trainer(
            callbacks=[PLCurriculumCallback(difficulty_scores, pace_fn="linear")]
        )
    """
    
    def __init__(
        self,
        difficulty_scores: np.ndarray,
        pace_function: str = "linear",
        total_epochs: int = 100,
        min_fraction: float = 0.1,
        max_fraction: float = 1.0,
        use_loss_weighting: bool = False,
        loss_temperature: float = 1.0,
        n_steps: int = 10,
        exp_rate: float = 5.0,
        verbose: bool = True,
    ):
        super().__init__()
        config = CurriculumConfig(
            pace_function=pace_function,
            total_epochs=total_epochs,
            min_fraction=min_fraction,
            max_fraction=max_fraction,
            use_loss_weighting=use_loss_weighting,
            loss_temperature=loss_temperature,
            n_steps=n_steps,
            exp_rate=exp_rate,
        )
        self.curriculum = BaseCurriculum(config, difficulty_scores)
        self.verbose = verbose
        self._last_weights = None
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Called at the start of each training epoch."""
        epoch = trainer.current_epoch
        
        # Try to get losses from previous epoch for loss-based weighting
        losses = None
        if self.curriculum.config.use_loss_weighting:
            # Look for per-sample losses in callback_metrics or pl_module
            if hasattr(pl_module, "last_train_losses"):
                losses = pl_module.last_train_losses
            elif "train_loss_per_sample" in trainer.callback_metrics:
                losses = trainer.callback_metrics["train_loss_per_sample"]
        
        info = self.curriculum.update(epoch, losses)
        weights = self.curriculum.get_sample_weights()
        mask = self.curriculum.get_inclusion_mask()
        
        # Try to update datamodule
        datamodule = trainer.datamodule
        if datamodule is not None:
            # Method 1: set_curriculum_weights (preferred)
            if hasattr(datamodule, "set_curriculum_weights"):
                datamodule.set_curriculum_weights(weights)
            # Method 2: Direct sampler weight update
            elif hasattr(datamodule, "train_sampler") and hasattr(datamodule.train_sampler, "weights"):
                datamodule.train_sampler.weights = torch.from_numpy(weights).float()
            # Method 3: DataModule with set_epoch (some implementations)
            elif hasattr(datamodule, "set_epoch"):
                datamodule.set_epoch(epoch)
        
        self._last_weights = weights
        
        if self.verbose:
            included = mask.sum()
            print(f"[PLCurriculum] Epoch {epoch}: pace={info['pace']:.3f}, "
                  f"included={included}/{len(mask)} ({info['inclusion_rate']:.1%}), "
                  f"weight_mean={info['weights_mean']:.3f}")
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Collect per-sample losses for loss-based curriculum."""
        if self.curriculum.config.use_loss_weighting and isinstance(outputs, dict):
            if "loss_per_sample" in outputs:
                # Store for next epoch
                pl_module.last_train_losses = outputs["loss_per_sample"].detach().cpu().numpy()
    
    def state_dict(self) -> dict[str, Any]:
        """Return callback state for checkpointing."""
        return {
            "curriculum_state": self.curriculum.get_state(),
        }
    
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load callback state from checkpoint."""
        if "curriculum_state" in state_dict:
            self.curriculum.load_state(state_dict["curriculum_state"])


# ════════════════════════════════════════════════════════════════════════════
# MosaicML Composer Callback
# ════════════════════════════════════════════════════════════════════════════

class ComposerCurriculumCallback:
    """
    MosaicML Composer callback for curriculum learning.
    
    Composer uses algorithms for modifying training behavior, not callbacks.
    This class provides both an Algorithm and a Callback interface.
    
    Usage:
        from composer import Trainer
        from training.curriculum_callbacks import ComposerCurriculumCallback
        
        callback = ComposerCurriculumCallback(difficulty_scores, pace_fn="linear")
        trainer = Trainer(
            model=model,
            train_dataloader=train_loader,
            callbacks=[callback],  # or algorithms=[callback] if using Algorithm interface
        )
    """
    
    def __init__(
        self,
        difficulty_scores: np.ndarray,
        pace_function: str = "linear",
        total_epochs: int = 100,
        min_fraction: float = 0.1,
        max_fraction: float = 1.0,
        use_loss_weighting: bool = False,
        loss_temperature: float = 1.0,
        n_steps: int = 10,
        exp_rate: float = 5.0,
        verbose: bool = True,
    ):
        config = CurriculumConfig(
            pace_function=pace_function,
            total_epochs=total_epochs,
            min_fraction=min_fraction,
            max_fraction=max_fraction,
            use_loss_weighting=use_loss_weighting,
            loss_temperature=loss_temperature,
            n_steps=n_steps,
            exp_rate=exp_rate,
        )
        self.curriculum = BaseCurriculum(config, difficulty_scores)
        self.verbose = verbose
        self._event_to_attr = {
            "epoch_start": "on_epoch_start",
            "epoch_end": "on_epoch_end",
            "batch_end": "on_batch_end",
        }
    
    # Composer Algorithm interface
    def match(self, event, state):
        """Composer Algorithm: match events we care about."""
        return event in ["EPOCH_START", "BATCH_END"]
    
    def apply(self, event, state, logger):
        """Composer Algorithm: apply curriculum updates."""
        if event == "EPOCH_START":
            epoch = state.timestamp.epoch.value
            
            # Get losses for loss-based weighting
            losses = None
            if self.curriculum.config.use_loss_weighting:
                # In Composer, losses might be in state.batch or via metrics
                if hasattr(state, "last_losses"):
                    losses = state.last_losses
            
            info = self.curriculum.update(epoch, losses)
            weights = self.curriculum.get_sample_weights()
            
            # Update dataloader sampler if possible
            if hasattr(state, "train_dataloader") and state.train_dataloader is not None:
                sampler = getattr(state.train_dataloader, "sampler", None)
                if sampler is not None and hasattr(sampler, "weights"):
                    sampler.weights = torch.from_numpy(weights).float()
            
            if self.verbose:
                mask = self.curriculum.get_inclusion_mask()
                included = mask.sum()
                print(f"[ComposerCurriculum] Epoch {epoch}: pace={info['pace']:.3f}, "
                      f"included={included}/{len(mask)} ({info['inclusion_rate']:.1%})")
        
        elif event == "BATCH_END" and self.curriculum.config.use_loss_weighting:
            # Collect per-sample losses
            if hasattr(state, "outputs") and isinstance(state.outputs, dict):
                if "loss_per_sample" in state.outputs:
                    state.last_losses = state.outputs["loss_per_sample"].detach().cpu().numpy()
    
    # Composer Callback interface (alternative)
    def on_epoch_start(self, state, logger):
        """Composer Callback: epoch start hook."""
        self.apply("EPOCH_START", state, logger)
    
    def on_batch_end(self, state, logger):
        """Composer Callback: batch end hook."""
        self.apply("BATCH_END", state, logger)
    
    def state_dict(self) -> dict[str, Any]:
        """Return callback state for checkpointing."""
        return {
            "curriculum_state": self.curriculum.get_state(),
        }
    
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load callback state from checkpoint."""
        if "curriculum_state" in state_dict:
            self.curriculum.load_state(state_dict["curriculum_state"])


# ════════════════════════════════════════════════════════════════════════════
# Adapter for existing custom curriculum
# ════════════════════════════════════════════════════════════════════════════

def create_curriculum_callback(
    framework: str = "auto",
    difficulty_scores: Optional[np.ndarray] = None,
    mode: str = "difficulty",
    **kwargs,
) -> Union[PLCurriculumCallback, ComposerCurriculumCallback, "CustomCurriculumAdapter"]:
    """
    Factory function to create curriculum callback for specified framework.
    
    Args:
        framework: "pytorch_lightning", "composer", "custom", or "auto"
        difficulty_scores: Per-sample difficulty scores in [0, 1]
        mode: "difficulty", "self_paced", "loss_weighted", "combined"
        **kwargs: Additional config options
    
    Returns:
        Curriculum callback/adapter for the specified framework.
    """
    if framework == "auto":
        # Try to detect framework
        try:
            import pytorch_lightning
            framework = "pytorch_lightning"
        except ImportError:
            try:
                import composer
                framework = "composer"
            except ImportError:
                framework = "custom"
    
    if framework == "pytorch_lightning":
        return PLCurriculumCallback(difficulty_scores, **kwargs)
    elif framework == "composer":
        return ComposerCurriculumCallback(difficulty_scores, **kwargs)
    elif framework == "custom":
        return CustomCurriculumAdapter(difficulty_scores, mode, **kwargs)
    else:
        raise ValueError(f"Unknown framework: {framework}")


class CustomCurriculumAdapter:
    """
    Adapter to wrap existing custom curriculum implementations
    (DifficultyCurriculum, SelfPacedLearning, etc.) with a unified interface.
    """
    
    def __init__(
        self,
        difficulty_scores: Optional[np.ndarray] = None,
        mode: str = "difficulty",
        **kwargs,
    ):
        self.mode = mode
        self.difficulty_scores = difficulty_scores
        self.kwargs = kwargs
        self._curriculum = None
        self._init_curriculum()
    
    def _init_curriculum(self):
        """Initialize the underlying curriculum implementation."""
        if self.mode == "difficulty":
            from training.curriculum import DifficultyCurriculum, DifficultyCurriculumConfig
            # Filter kwargs to only include valid DifficultyCurriculumConfig fields
            valid_fields = {'n_levels', 'start_level', 'advance_rate', 'min_competence', 
                           'competence_metric', 'pace_function', 'max_level'}
            config_kwargs = {k: v for k, v in self.kwargs.items() if k in valid_fields}
            config = DifficultyCurriculumConfig(**config_kwargs)
            self._curriculum = DifficultyCurriculum(config, self.difficulty_scores)
        elif self.mode == "self_paced":
            from training.curriculum import SelfPacedLearning, SelfPacedConfig
            config = SelfPacedConfig(**self.kwargs)
            self._curriculum = SelfPacedLearning(config, len(self.difficulty_scores))
        elif self.mode == "combined":
            # Combined mode uses both difficulty and self-paced internally
            from training.curriculum import DifficultyCurriculum, DifficultyCurriculumConfig, SelfPacedLearning, SelfPacedConfig
            valid_fields = {'n_levels', 'start_level', 'advance_rate', 'min_competence', 
                           'competence_metric', 'pace_function', 'max_level',
                           'lambda_pace', 'min_fraction', 'use_loss_weighting', 'loss_temp'}
            config_kwargs = {k: v for k, v in self.kwargs.items() if k in valid_fields}
            # Create both curricula
            diff_config = DifficultyCurriculumConfig(**{k: v for k, v in config_kwargs.items() 
                                                        if k in {'n_levels', 'start_level', 'advance_rate', 
                                                               'min_competence', 'competence_metric', 
                                                               'pace_function', 'max_level'}})
            self._diff_curriculum = DifficultyCurriculum(diff_config, self.difficulty_scores)
            sp_config = SelfPacedConfig(**{k: v for k, v in config_kwargs.items() 
                                           if k in {'lambda_pace', 'min_fraction', 'use_loss_weighting', 'loss_temp', 'total_epochs', 'pace'}})
            self._curriculum = SelfPacedLearning(sp_config, len(self.difficulty_scores))
        else:
            raise ValueError(f"Unknown curriculum mode: {self.mode}")
    
    def update(self, epoch: int, losses: Optional[np.ndarray] = None) -> dict[str, Any]:
        """Update curriculum for epoch."""
        if self._curriculum is None:
            self._init_curriculum()
        
        if self.mode == "difficulty":
            self._curriculum.update(epoch)
            mask = self._curriculum.get_inclusion_mask()
            weights = self._curriculum.get_difficulty_weights()
            return {
                "pace": float(mask.mean()),
                "n_included": int(mask.sum()),
                "inclusion_rate": float(mask.mean()),
                "weights_mean": float(weights.mean()),
                "current_epoch": epoch,
            }
        elif self.mode == "self_paced":
            weights = self._curriculum.get_weights(epoch, losses)
            return {
                "pace": float(weights.mean()),
                "n_included": int((weights > 0).sum()),
                "inclusion_rate": float((weights > 0).mean()),
                "weights_mean": float(weights.mean()),
                "current_epoch": epoch,
            }
        elif self.mode == "combined":
            # Combined: use both difficulty and self-paced
            self._diff_curriculum.update(epoch)
            diff_mask = self._diff_curriculum.get_inclusion_mask()
            diff_weights = self._diff_curriculum.get_difficulty_weights()
            sp_weights = self._curriculum.get_weights(epoch, losses)
            # Combine: only include samples in both
            combined_mask = diff_mask & (sp_weights > 0)
            combined_weights = diff_weights * sp_weights
            combined_weights = combined_weights / (combined_weights.max() + 1e-8)
            return {
                "pace": float(combined_mask.mean()),
                "n_included": int(combined_mask.sum()),
                "inclusion_rate": float(combined_mask.mean()),
                "weights_mean": float(combined_weights.mean()),
                "current_epoch": epoch,
            }
        
        return {"current_epoch": epoch}
    
    def get_inclusion_mask(self) -> np.ndarray:
        """Get inclusion mask."""
        if self._curriculum is None:
            self._init_curriculum()
        
        if self.mode == "difficulty":
            return self._curriculum.get_inclusion_mask()
        elif self.mode == "self_paced":
            return self._curriculum.v > 0
        elif self.mode == "combined":
            diff_mask = self._diff_curriculum.get_inclusion_mask()
            sp_weights = self._curriculum.get_weights(
                getattr(self._curriculum, "current_epoch", 0), None
            )
            return diff_mask & (sp_weights > 0)
        return np.ones(len(self.difficulty_scores), dtype=bool)
    
    def get_sample_weights(self) -> np.ndarray:
        """Get sample weights."""
        if self._curriculum is None:
            self._init_curriculum()
        
        if self.mode == "difficulty":
            return self._curriculum.get_difficulty_weights()
        elif self.mode == "self_paced":
            return self._curriculum.v
        elif self.mode == "combined":
            diff_weights = self._diff_curriculum.get_difficulty_weights()
            sp_weights = self._curriculum.get_weights(
                getattr(self._curriculum, "current_epoch", 0), None
            )
            combined = diff_weights * sp_weights
            # Mirror update() normalization: scale to (0, 1] max
            combined = combined / (combined.max() + 1e-8)
            return combined
        return np.ones(len(self.difficulty_scores), dtype=float)
    
    def state_dict(self) -> dict[str, Any]:
        """Get state for checkpointing."""
        return {
            "mode": self.mode,
            "current_epoch": getattr(self._curriculum, "current_epoch", 0),
            "kwargs": self.kwargs,
        }
    
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load state from checkpoint."""
        if self._curriculum is not None:
            self._curriculum.current_epoch = state.get("current_epoch", 0)


# ════════════════════════════════════════════════════════════════════════════
# Example integration helpers
# ════════════════════════════════════════════════════════════════════════════

def integrate_curriculum_with_dataloader(
    dataloader: torch_data.DataLoader,
    weights: np.ndarray,
    replacement: bool = True,
) -> torch_data.DataLoader:
    """
    Create a new DataLoader with WeightedRandomSampler using curriculum weights.
    
    Args:
        dataloader: Original DataLoader
        weights: Sample weights from curriculum
        replacement: Whether to sample with replacement
    
    Returns:
        New DataLoader with weighted sampler.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")
    
    dataset = dataloader.dataset
    sampler = torch_data.WeightedRandomSampler(
        weights=torch.from_numpy(weights).float(),
        num_samples=len(weights),
        replacement=replacement,
    )
    
    return torch_data.DataLoader(
        dataset,
        batch_size=dataloader.batch_size,
        sampler=sampler,
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last,
        collate_fn=dataloader.collate_fn,
    )


def make_curriculum_datamodule(
    base_datamodule,
    curriculum_callback: Union[PLCurriculumCallback, CustomCurriculumAdapter],
) -> type:
    """
    Create a DataModule subclass that integrates curriculum weights.
    
    Usage:
        MyCurriculumDataModule = make_curriculum_datamodule(MyDataModule, callback)
        datamodule = MyCurriculumDataModule(...)
        trainer = Trainer(datamodule=datamodule, callbacks=[callback])
    """
    class CurriculumDataModule(base_datamodule.__class__):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._curriculum_callback = curriculum_callback
            self._curriculum_weights = None
        
        def set_curriculum_weights(self, weights):
            self._curriculum_weights = weights
        
        def train_dataloader(self):
            loader = super().train_dataloader()
            if self._curriculum_weights is not None and TORCH_AVAILABLE:
                return integrate_curriculum_with_dataloader(loader, self._curriculum_weights)
            return loader
    
    CurriculumDataModule.__name__ = f"Curriculum{base_datamodule.__class__.__name__}"
    return CurriculumDataModule


if __name__ == "__main__":
    # Demo usage
    np.random.seed(42)
    difficulty = np.random.rand(1000)
    
    # Test base curriculum
    config = CurriculumConfig(pace_function="linear", total_epochs=10, min_fraction=0.1)
    curriculum = BaseCurriculum(config, difficulty)
    
    for epoch in [0, 3, 5, 9]:
        info = curriculum.update(epoch)
        mask = curriculum.get_inclusion_mask()
        print(f"Epoch {epoch}: pace={info['pace']:.3f}, included={mask.sum()}/{len(mask)}")
    
    # Test PL callback creation
    pl_callback = PLCurriculumCallback(difficulty, pace_function="sqrt", total_epochs=20)
    print(f"\nPL Callback created: {pl_callback}")
    
    # Test Composer callback creation
    composer_callback = ComposerCurriculumCallback(difficulty, pace_function="exp", total_epochs=20)
    print(f"Composer Callback created: {composer_callback}")
    
    # Test adapter
    adapter = CustomCurriculumAdapter(difficulty, mode="difficulty", pace_function="linear", total_epochs=10)
    print(f"Adapter created: {adapter}")