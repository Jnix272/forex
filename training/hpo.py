"""
HPO Module (Improvement #12)
============================
Population-Based Training (PBT), BOHB (Bayesian Optimization HyperBand),
Asynchronous Successive Halving (ASHA), and Multi-fidelity ASHA.

Integrates with existing Optuna infrastructure in optuna_tune.py.
"""

from __future__ import annotations

import copy
import math
import random
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import HyperbandPruner, MedianPruner, NopPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


    def get_pace(self) -> float:
        """Get current pacing fraction for self-paced learning."""
        if not hasattr(self, "config"):
            return 1.0
        total = max(1, self.config.max_epochs)
        return min(self._total_epochs / total, 1.0)

    def _sample_initial_params(self) -> Dict[str, Any]:
        """Sample initial hyperparameters."""
        return {
            "lr": 10 ** np.random.uniform(-5, -3),
            "d_model": np.random.choice([64, 128, 256, 512]),
            "num_layers": np.random.randint(2, 6),
            "dropout": np.random.uniform(0.1, 0.5),
            "batch_size": np.random.choice([32, 64, 128, 256]),
        }

    def state_dict(self) -> Dict[str, Any]:
        """Get HPO manager state as dictionary."""
        return {
            "epoch": self._total_epochs,
            "config_mode": self.config.mode,
            "difficulty": getattr(self, "difficulty", None),
            "self_paced_pace": self.get_pace(),
            "adaptive_state": self.study.best_params if self.study else None,
        }

# ══════════════════════════════════════════════════════════════════════════════
# 1. Base Classes and Data Structures
    def get_pace(self) -> float:
        """Get current pacing fraction for self-paced learning."""
        if not hasattr(self, "config"):
            return 1.0
        total = max(1, self.config.max_epochs)
        return min(self._total_epochs / total, 1.0)

    def _sample_initial_params(self) -> Dict[str, Any]:
        """Sample initial hyperparameters."""
        return {
            "lr": 10 ** np.random.uniform(-5, -3),
            "d_model": np.random.choice([64, 128, 256, 512]),
            "num_layers": np.random.randint(2, 6),
            "dropout": np.random.uniform(0.1, 0.5),
            "batch_size": np.random.choice([32, 64, 128, 256]),
        }

    def state_dict(self) -> Dict[str, Any]:
        """Get HPO manager state as dictionary."""
        return {
            "epoch": self._total_epochs,
            "config_mode": self.config.mode,
            "difficulty": getattr(self, "difficulty", None),
            "self_paced_pace": self.get_pace(),
            "adaptive_state": self.study.best_params if self.study else None,
        }

# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class HPOConfig:
    """Configuration for HPO algorithms."""
    # Population-Based Training
    population_size: int = 8
    perturbation_interval: int = 10  # epochs
    perturbation_factor: float = 0.2
    exploit_factor: float = 0.8  # fraction to exploit vs explore
    
    # BOHB / HyperBand
    min_budget: int = 3
    max_budget: int = 27
    eta: int = 3  # HyperBand reduction factor
    min_points_in_model: int = 8  # minimum points for KDE in BOHB
    top_n_percent: int = 15  # for BOHB
    
    # ASHA
    grace_period: int = 1  # minimum resource before stopping
    reduction_factor: int = 3
    brackets: int = 1
    
    # General
    max_epochs: int = 27
    metric: str = "val_sharpe"
    mode: str = "maximize"  # "maximize" or "minimize"
    n_trials: int = 100
    seed: int = 42
    
    # Resource constraints
    max_concurrent_trials: int = 4
    time_budget_sec: Optional[int] = None
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints/hpo"
    checkpoint_interval: int = 5  # epochs


@dataclass
class TrialState:
    """State of a single trial in population-based methods."""
    trial_id: str
    params: Dict[str, Any]
    step: int = 0
    score: float = -float('inf')
    best_score: float = -float('inf')
    checkpoint_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        return self.score < other.score


class TrialScheduler(ABC):
    """Abstract base class for trial schedulers."""
    
    @abstractmethod
    def suggest_params(self, trial_id: str) -> Dict[str, Any]:
        """Suggest hyperparameters for a new trial."""
        pass
    
    @abstractmethod
    def on_trial_result(self, trial_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process intermediate result and return scheduling decision."""
        pass
    
    @abstractmethod
    def on_trial_complete(self, trial_id: str, result: Dict[str, Any]) -> None:
        """Handle trial completion."""
        pass
    
    @abstractmethod
    def get_next_trials(self) -> List[Dict[str, Any]]:
        """Get list of trials to run next."""
        pass


    def get_pace(self) -> float:
        """Get current pacing fraction for self-paced learning."""
        if not hasattr(self, "config"):
            return 1.0
        total = max(1, self.config.max_epochs)
        return min(self._total_epochs / total, 1.0)

    def _sample_initial_params(self) -> Dict[str, Any]:
        """Sample initial hyperparameters."""
        return {
            "lr": 10 ** np.random.uniform(-5, -3),
            "d_model": np.random.choice([64, 128, 256, 512]),
            "num_layers": np.random.randint(2, 6),
            "dropout": np.random.uniform(0.1, 0.5),
            "batch_size": np.random.choice([32, 64, 128, 256]),
        }

    def state_dict(self) -> Dict[str, Any]:
        """Get HPO manager state as dictionary."""
        return {
            "epoch": self._total_epochs,
            "config_mode": self.config.mode,
            "difficulty": getattr(self, "difficulty", None),
            "self_paced_pace": self.get_pace(),
            "adaptive_state": self.study.best_params if self.study else None,
        }

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Population-Based Training (PBT)
    def get_pace(self) -> float:
        """Get current pacing fraction for self-paced learning."""
        if not hasattr(self, "config"):
            return 1.0
        total = max(1, self.config.max_epochs)
        return min(self._total_epochs / total, 1.0)

    def _sample_initial_params(self) -> Dict[str, Any]:
        """Sample initial hyperparameters."""
        return {
            "lr": 10 ** np.random.uniform(-5, -3),
            "d_model": np.random.choice([64, 128, 256, 512]),
            "num_layers": np.random.randint(2, 6),
            "dropout": np.random.uniform(0.1, 0.5),
            "batch_size": np.random.choice([32, 64, 128, 256]),
        }

    def state_dict(self) -> Dict[str, Any]:
        """Get HPO manager state as dictionary."""
        return {
            "epoch": self._total_epochs,
            "config_mode": self.config.mode,
            "difficulty": getattr(self, "difficulty", None),
            "self_paced_pace": self.get_pace(),
            "adaptive_state": self.study.best_params if self.study else None,
        }

# ═════════════════════════════════════════════════════════════════════════════

class PopulationBasedTraining:
    """
    Population-Based Training (PBT) - Jaderberg et al., 2017.
    
    Maintains a population of models, periodically replacing
    underperforming ones with perturbed copies of better performers.
    """
    
    def __init__(self, config: HPOConfig):
        self.config = config
        self.population: List[TrialState] = []
        self.best_trial: Optional[TrialState] = None
        self._rng = np.random.default_rng(config.seed)
        
    def add_trial(self, trial_id: str, params: Dict[str, Any]) -> TrialState:
        """Add a new trial to the population."""
        trial = TrialState(
            trial_id=trial_id,
            params=params.copy(),
            step=0,
            score=-float('inf'),
        )
        self.population.append(trial)
        return trial
    
    def update_score(self, trial_id: str, score: float, step: int,
                     checkpoint_path: str = "") -> None:
        """Update trial score and step."""
        for trial in self.population:
            if trial.trial_id == trial_id:
                trial.step = step
                trial.score = score
                trial.best_score = max(trial.best_score, score)
                if checkpoint_path:
                    trial.checkpoint_path = checkpoint_path
                
                # Update best trial
                if self.best_trial is None or trial.best_score > self.best_trial.best_score:
                    self.best_trial = trial
                break
    
    def should_perturb(self, trial_id: str) -> bool:
        """Check if trial should be perturbed at current step."""
        trial = self._get_trial(trial_id)
        if trial is None:
            return False
        return trial.step > 0 and trial.step % self.config.perturbation_interval == 0
    
    def perturb_trial(self, trial_id: str) -> Dict[str, Any]:
        """Perturb hyperparameters of a trial by exploiting better performers."""
        trial = self._get_trial(trial_id)
        if trial is None or self.best_trial is None:
            return trial.params if trial else {}
        
        # Exploit: copy from better performer with probability exploit_factor
        if self._rng.random() < self.config.exploit_factor and self.best_trial != trial:
            # Copy from best trial
            new_params = self.best_trial.params.copy()
        else:
            # Explore: copy from random better performer
            better_trials = [t for t in self.population if t.best_score > trial.best_score]
            if better_trials:
                donor = self._rng.choice(better_trials)
                new_params = donor.params.copy()
            else:
                new_params = trial.params.copy()
        
        # Perturb hyperparameters
        perturbed = {}
        for key, value in new_params.items():
            if isinstance(value, (int, float)):
                # Log-uniform perturbation for continuous params
                if isinstance(value, float) and value > 0:
                    factor = np.exp(self._rng.uniform(-self.config.perturbation_factor, 
                                                       self.config.perturbation_factor))
                    perturbed[key] = value * factor
                else:
                    # Integer params
                    if isinstance(value, int):
                        delta = self._rng.integers(-1, 2)
                        perturbed[key] = max(1, value + delta)
                    else:
                        perturbed[key] = value
            else:
                # Categorical params - sometimes change
                if self._rng.random() < 0.1:
                    perturbed[key] = value  # keep same
                else:
                    perturbed[key] = value  # would need categorical space
        
        return perturbed
    
    def exploit_and_perturb(self, trial_id: str) -> Tuple[str, Dict[str, Any]]:
        """
        Exploit and perturb: replace worst trial with perturbed copy of best.
        Returns (donor_trial_id, new_params).
        """
        # Sort population by score
        sorted_pop = sorted(self.population, key=lambda t: t.score)
        
        # Worst trial gets replaced
        worst_trial = sorted_pop[0]
        
        # Find best trial (excluding the worst)
        best_trial = sorted_pop[-1]
        
        if worst_trial.trial_id == best_trial.trial_id:
            return worst_trial.trial_id, worst_trial.params
        
        # Perturb best trial's params
        new_params = self.perturb_trial(best_trial.trial_id)
        
        # Update worst trial
        worst_trial.params = self.perturb_trial(best_trial.trial_id)
        worst_trial.score = -float('inf')
        worst_trial.best_score = -float('inf')
        worst_trial.step = 0
        
        return best_trial.trial_id, worst_trial.params
    
    def _get_trial(self, trial_id: str) -> Optional[TrialState]:
        for trial in self.population:
            if trial.trial_id == trial_id:
                return trial
        return None
    
    def get_population_stats(self) -> Dict[str, Any]:
        """Get population statistics."""
        if not self.population:
            return {}
        scores = [t.score for t in self.population]
        return {
            "population_size": len(self.population),
            "mean_score": float(np.mean([s for s in scores if s != -float('inf')])),
            "best_score": max([s for s in scores if s != -float('inf')]),
            "worst_score": min([s for s in scores if s != -float('inf')]),
        }


    def get_pace(self) -> float:
        """Get current pacing fraction for self-paced learning."""
        if not hasattr(self, "config"):
            return 1.0
        total = max(1, self.config.max_epochs)
        return min(self._total_epochs / total, 1.0)

    def _sample_initial_params(self) -> Dict[str, Any]:
        """Sample initial hyperparameters."""
        return {
            "lr": 10 ** np.random.uniform(-5, -3),
            "d_model": np.random.choice([64, 128, 256, 512]),
            "num_layers": np.random.randint(2, 6),
            "dropout": np.random.uniform(0.1, 0.5),
            "batch_size": np.random.choice([32, 64, 128, 256]),
        }

    def state_dict(self) -> Dict[str, Any]:
        """Get HPO manager state as dictionary."""
        return {
            "epoch": self._total_epochs,
            "config_mode": self.config.mode,
            "difficulty": getattr(self, "difficulty", None),
            "self_paced_pace": self.get_pace(),
            "adaptive_state": self.study.best_params if self.study else None,
        }

# ═════════════════════════════════════════════════════════════════════════════
# 3. HyperBand / ASHA (Asynchronous Successive Halving Algorithm)
    def get_pace(self) -> float:
        """Get current pacing fraction for self-paced learning."""
        if not hasattr(self, "config"):
            return 1.0
        total = max(1, self.config.max_epochs)
        return min(self._total_epochs / total, 1.0)

    def _sample_initial_params(self) -> Dict[str, Any]:
        """Sample initial hyperparameters."""
        return {
            "lr": 10 ** np.random.uniform(-5, -3),
            "d_model": np.random.choice([64, 128, 256, 512]),
            "num_layers": np.random.randint(2, 6),
            "dropout": np.random.uniform(0.1, 0.5),
            "batch_size": np.random.choice([32, 64, 128, 256]),
        }

    def state_dict(self) -> Dict[str, Any]:
        """Get HPO manager state as dictionary."""
        return {
            "epoch": self._total_epochs,
            "config_mode": self.config.mode,
            "difficulty": getattr(self, "difficulty", None),
            "self_paced_pace": self.get_pace(),
            "adaptive_state": self.study.best_params if self.study else None,
        }

# ════════════════════════════════════════════════════════════════════════════

class HyperBandScheduler:
    """
    HyperBand Scheduler - Li et al., 2017.
    
    Adaptive resource allocation using successive halving with
    multiple brackets of different resource levels.
    """
    
    def __init__(self, config: HPOConfig):
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        
        # Calculate bracket sizes
        self.max_budget = config.max_budget
        self.min_budget = config.min_budget
        self.eta = config.eta
        self.grace_period = config.grace_period
        
        # Calculate s_max
        self.s_max = int(math.log(config.max_budget / config.min_budget) / math.log(config.eta))
        self.s_max = max(0, self.s_max)
        
        # Brackets
        self.brackets: List[Dict[str, Any]] = []
        self._init_brackets()
        
    def _init_brackets(self):
        """Initialize HyperBand brackets."""
        for s in range(self.s_max + 1):
            n = int(math.ceil((self.s_max + 1) / (s + 1) * self.eta ** s))
            r = self.min_budget * (self.eta ** (self.s_max - s))
            
            bracket = {
                "s": s,
                "n": n,
                "r": r,
                "rungs": [],
                "promoted": [],
            }
            
            # Initialize rungs
            for i in range(s + 1):
                rung_budget = self.min_budget * (self.eta ** i)
                rung = {
                    "budget": rung_budget,
                    "trials": [],
                    "completed": 0,
                    "promoted": 0,
                }
                bracket["rungs"].append(rung)
            
            self.brackets.append(bracket)
    
    def suggest_params(self, trial_id: str, bracket_idx: Optional[int] = None) -> Dict[str, Any]:
        """Suggest parameters for a new trial."""
        # In practice, this would integrate with Optuna's sampler
        # For now, return empty dict - actual sampling done by Optuna
        return {}
    
    def on_trial_result(self, trial_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process intermediate result and return scheduling decision."""
        # Find which bracket and rung this trial belongs to
        # This is a simplified version - real implementation tracks trial state
        return {"action": "continue"}  # continue, pause, stop, promote
    
    def on_trial_complete(self, trial_id: str, result: Dict[str, Any]) -> None:
        """Handle trial completion."""
        pass
    
    def get_next_trials(self) -> List[Dict[str, Any]]:
        """Get next trials to run."""
        # Simplified - would implement actual HyperBand scheduling logic
        return []


class AsyncSuccessiveHalvingScheduler:
    """
    ASHA (Asynchronous Successive Halving Algorithm) - Li et al., 2020.
    
    Asynchronous version of Successive Halving that continuously
    promotes and stops trials without waiting for synchronous rounds.
    """
    
    def __init__(self, config: HPOConfig):
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        
        # ASHA parameters
        self.grace_period = config.grace_period
        self.reduction_factor = config.reduction_factor
        self.brackets = config.brackets
        
        # State tracking
        self.rungs: Dict[int, List[Dict]] = defaultdict(list)  # rung -> list of trials
        self.trial_states: Dict[str, Dict] = {}  # trial_id -> state
        self._rng = np.random.default_rng(config.seed)
        
    def add_trial(self, trial_id: str, params: Dict[str, Any]) -> None:
        """Add a new trial at the lowest rung."""
        state = {
            "trial_id": trial_id,
            "params": params,
            "rung": 0,
            "resources": self.config.grace_period,
            "metrics": [],
            "status": "running",
        }
        self.trial_states[trial_id] = state
        self.rungs[0].append({"trial_id": trial_id, "score": -float('inf')})
    
    def on_trial_result(self, trial_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process intermediate result, return scheduling decision."""
        if trial_id not in self.trial_states:
            return {"action": "continue"}
        
        state = self.trial_states[trial_id]
        state["metrics"].append(result)
        
        # Extract metric
        metric = self.config.metric
        score = result.get(metric, -float('inf'))
        state["last_score"] = score
        
        current_rung = state["rung"]
        
        # Check if we should promote or stop
        rung_trials = self.rungs[current_rung]
        rung_trials_with_scores = [
            (t["trial_id"], self.trial_states[t["trial_id"]].get("last_score", -float('inf')))
            for t in rung_trials if "last_score" in self.trial_states[t["trial_id"]]
        ]
        
        if len(rung_trials_with_scores) < 2:
            return {"action": "continue"}
        
        # Sort by score (higher is better for maximize)
        rung_trials_with_scores.sort(key=lambda x: x[1], reverse=(self.config.mode == "maximize"))
        
        # Bottom 1/eta fraction gets stopped
        n_stop = max(1, len(rung_trials_with_scores) // self.config.reduction_factor)
        stop_trials = rung_trials_with_scores[:n_stop]
        
        for trial_id, _ in stop_trials:
            if trial_id == trial_id:
                return {"action": "stop"}
        
        # Check for promotion
        n_promote = max(1, len(rung_trials_with_scores) // self.config.eta)
        promote_trials = rung_trials_with_scores[-n_promote:]
        
        for tid, _ in promote_trials:
            if tid == trial_id:
                return {"action": "promote", "next_rung": current_rung + 1}
        
        return {"action": "continue"}
    
    def on_trial_complete(self, trial_id: str, result: Dict[str, Any]) -> None:
        """Handle trial completion."""
        if trial_id in self.trial_states:
            self.trial_states[trial_id]["status"] = "completed"
            self.trial_states[trial_id]["final_result"] = result
    
    def get_next_trials(self, max_concurrent: int) -> List[Dict[str, Any]]:
        """Get trials to run next."""
        # In practice, this would integrate with trial queue
        return []


    def get_pace(self) -> float:
        """Get current pacing fraction for self-paced learning."""
        if not hasattr(self, "config"):
            return 1.0
        total = max(1, self.config.max_epochs)
        return min(self._total_epochs / total, 1.0)

    def _sample_initial_params(self) -> Dict[str, Any]:
        """Sample initial hyperparameters."""
        return {
            "lr": 10 ** np.random.uniform(-5, -3),
            "d_model": np.random.choice([64, 128, 256, 512]),
            "num_layers": np.random.randint(2, 6),
            "dropout": np.random.uniform(0.1, 0.5),
            "batch_size": np.random.choice([32, 64, 128, 256]),
        }

    def state_dict(self) -> Dict[str, Any]:
        """Get HPO manager state as dictionary."""
        return {
            "epoch": self._total_epochs,
            "config_mode": self.config.mode,
            "difficulty": getattr(self, "difficulty", None),
            "self_paced_pace": self.get_pace(),
            "adaptive_state": self.study.best_params if self.study else None,
        }

# ═══════════════════════════════════════════════════════════════════════════════
# 3. BOHB (Bayesian Optimization HyperBand)
    def get_pace(self) -> float:
        """Get current pacing fraction for self-paced learning."""
        if not hasattr(self, "config"):
            return 1.0
        total = max(1, self.config.max_epochs)
        return min(self._total_epochs / total, 1.0)

    def _sample_initial_params(self) -> Dict[str, Any]:
        """Sample initial hyperparameters."""
        return {
            "lr": 10 ** np.random.uniform(-5, -3),
            "d_model": np.random.choice([64, 128, 256, 512]),
            "num_layers": np.random.randint(2, 6),
            "dropout": np.random.uniform(0.1, 0.5),
            "batch_size": np.random.choice([32, 64, 128, 256]),
        }

    def state_dict(self) -> Dict[str, Any]:
        """Get HPO manager state as dictionary."""
        return {
            "epoch": self._total_epochs,
            "config_mode": self.config.mode,
            "difficulty": getattr(self, "difficulty", None),
            "self_paced_pace": self.get_pace(),
            "adaptive_state": self.study.best_params if self.study else None,
        }

# ═════════════════════════════════════════════════════════════════════════════

class BOHBScheduler:
    """
    BOHB (Bayesian Optimization HyperBand) - Falkner et al., 2018.
    
    Combines Bayesian Optimization (TPE) with HyperBand.
    Uses TPE to model P(config|budget) and select promising configurations.
    """
    
    def __init__(self, config: HPOConfig):
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        
        # HyperBand structure
        self.hyperband = HyperBandScheduler(config)
        
        # TPE models per rung
        self.kde_good: Dict[int, Any] = {}  # rung -> TPE model for good configs
        self.kde_bad: Dict[int, Any] = {}   # rung -> TPE model for bad configs
        self.observations: Dict[int, List[Tuple[Dict, float]]] = defaultdict(list)
        
        self._rng = np.random.default_rng(config.seed)
        
    def _update_kde(self, rung: int) -> None:
        """Update KDE models for a rung."""
        if rung not in self.observations or len(self.observations[rung]) < self.config.min_points_in_model:
            return
        
        observations = self.observations[rung]
        if len(observations) < self.config.min_points_in_model:
            return
        
        # Sort by score
        observations.sort(key=lambda x: x[1], reverse=(self.config.mode == "maximize"))
        
        n_good = max(1, len(observations) // self.config.top_n_percent)
        n_bad = max(1, len(observations) - n_good)
        
        good_configs = [obs[0] for obs in observations[:n_good]]
        bad_configs = [obs[0] for obs in observations[-n_bad:]]
        
        # Build KDEs (simplified - in practice use optuna's TPE)
        self.kde_good[rung] = {"configs": good_configs, "scores": [o[1] for o in observations[:n_good]]}
        self.kde_bad[rung] = {"configs": bad_configs, "scores": [o[1] for o in observations[-n_bad:]]}
    
    def suggest_params(self, rung: int = 0) -> Dict[str, Any]:
        """Suggest parameters using TPE for given rung."""
        # In practice, integrate with Optuna's TPE sampler
        # For now, return empty - actual sampling via Optuna
        return {}
    
    def observe(self, trial_id: str, rung: int, config: Dict, score: float) -> None:
        """Record observation for a trial at a rung."""
        self.observations[rung].append((config, score))
        self._update_kde(rung)
    
    def on_trial_result(self, trial_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process intermediate result."""
        return {"action": "continue"}
    
    def on_trial_complete(self, trial_id: str, result: Dict[str, Any]) -> None:
        """Handle trial completion."""
        pass


    def get_pace(self) -> float:
        """Get current pacing fraction for self-paced learning."""
        if not hasattr(self, "config"):
            return 1.0
        total = max(1, self.config.max_epochs)
        return min(self._total_epochs / total, 1.0)

    def _sample_initial_params(self) -> Dict[str, Any]:
        """Sample initial hyperparameters."""
        return {
            "lr": 10 ** np.random.uniform(-5, -3),
            "d_model": np.random.choice([64, 128, 256, 512]),
            "num_layers": np.random.randint(2, 6),
            "dropout": np.random.uniform(0.1, 0.5),
            "batch_size": np.random.choice([32, 64, 128, 256]),
        }

    def state_dict(self) -> Dict[str, Any]:
        """Get HPO manager state as dictionary."""
        return {
            "epoch": self._total_epochs,
            "config_mode": self.config.mode,
            "difficulty": getattr(self, "difficulty", None),
            "self_paced_pace": self.get_pace(),
            "adaptive_state": self.study.best_params if self.study else None,
        }

# ══════════════════════════════════════════════════════════════════════════════
# 5. Multi-Fidelity ASHA (MF-ASHA)
    def get_pace(self) -> float:
        """Get current pacing fraction for self-paced learning."""
        if not hasattr(self, "config"):
            return 1.0
        total = max(1, self.config.max_epochs)
        return min(self._total_epochs / total, 1.0)

    def _sample_initial_params(self) -> Dict[str, Any]:
        """Sample initial hyperparameters."""
        return {
            "lr": 10 ** np.random.uniform(-5, -3),
            "d_model": np.random.choice([64, 128, 256, 512]),
            "num_layers": np.random.randint(2, 6),
            "dropout": np.random.uniform(0.1, 0.5),
            "batch_size": np.random.choice([32, 64, 128, 256]),
        }

    def state_dict(self) -> Dict[str, Any]:
        """Get HPO manager state as dictionary."""
        return {
            "epoch": self._total_epochs,
            "config_mode": self.config.mode,
            "difficulty": getattr(self, "difficulty", None),
            "self_paced_pace": self.get_pace(),
            "adaptive_state": self.study.best_params if self.study else None,
        }

# ═════════════════════════════════════════════════════════════════════════════

class MultiFidelityASHAScheduler:
    """
    Multi-Fidelity ASHA - extends ASHA with multiple fidelity dimensions.
    
    Supports multiple resource dimensions (e.g., epochs, data fraction, 
    model size) for more flexible resource allocation.
    """
    
    def __init__(self, config: HPOConfig, fidelity_dims: List[Dict[str, Any]]):
        self.config = config
        self.fidelity_dims = fidelity_dims  # List of {"name": "epochs", "min": 1, "max": 27, "eta": 3}
        self._rng = np.random.default_rng(config.seed)
        
        # Multi-fidelity rungs
        self.fidelity_rungs: Dict[Tuple, List[Dict]] = defaultdict(list)
        self.trial_states: Dict[str, Dict] = {}
        
    def add_trial(self, trial_id: str, params: Dict[str, Any]) -> None:
        """Add a new trial at lowest fidelity for all dimensions."""
        fidelity = {dim["name"]: dim["min"] for dim in self.fidelity_dims}
        state = {
            "trial_id": trial_id,
            "params": params,
            "fidelity": fidelity,
            "metrics": [],
            "status": "running",
        }
        self.trial_states[trial_id] = state
        
    def on_trial_result(self, trial_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process result and decide next fidelity or stop."""
        if trial_id not in self.trial_states:
            return {"action": "continue"}
        
        state = self.trial_states[trial_id]
        state["metrics"].append(result)
        
        # For simplicity, use single dimension (epochs) for promotion
        metric = self.config.metric
        score = result.get(self.config.metric, -float('inf'))
        
        # Compare with other trials at same fidelity level
        # Simplified: just return continue for now
        return {"action": "continue", "next_fidelity": None}
    
    def on_trial_complete(self, trial_id: str, result: Dict[str, Any]) -> None:
        if trial_id in self.trial_states:
            self.trial_states[trial_id]["status"] = "completed"
    
    def get_next_trials(self, max_concurrent: int) -> List[Dict[str, Any]]:
        return []


    def get_pace(self) -> float:
        """Get current pacing fraction for self-paced learning."""
        if not hasattr(self, "config"):
            return 1.0
        total = max(1, self.config.max_epochs)
        return min(self._total_epochs / total, 1.0)

    def _sample_initial_params(self) -> Dict[str, Any]:
        """Sample initial hyperparameters."""
        return {
            "lr": 10 ** np.random.uniform(-5, -3),
            "d_model": np.random.choice([64, 128, 256, 512]),
            "num_layers": np.random.randint(2, 6),
            "dropout": np.random.uniform(0.1, 0.5),
            "batch_size": np.random.choice([32, 64, 128, 256]),
        }

    def state_dict(self) -> Dict[str, Any]:
        """Get HPO manager state as dictionary."""
        return {
            "epoch": self._total_epochs,
            "config_mode": self.config.mode,
            "difficulty": getattr(self, "difficulty", None),
            "self_paced_pace": self.get_pace(),
            "adaptive_state": self.study.best_params if self.study else None,
        }

# ═════════════════════════════════════════════════════════════════════════════
# 6. Unified HPO Manager
    def get_pace(self) -> float:
        """Get current pacing fraction for self-paced learning."""
        if not hasattr(self, "config"):
            return 1.0
        total = max(1, self.config.max_epochs)
        return min(self._total_epochs / total, 1.0)

    def _sample_initial_params(self) -> Dict[str, Any]:
        """Sample initial hyperparameters."""
        return {
            "lr": 10 ** np.random.uniform(-5, -3),
            "d_model": np.random.choice([64, 128, 256, 512]),
            "num_layers": np.random.randint(2, 6),
            "dropout": np.random.uniform(0.1, 0.5),
            "batch_size": np.random.choice([32, 64, 128, 256]),
        }

    def state_dict(self) -> Dict[str, Any]:
        """Get HPO manager state as dictionary."""
        return {
            "epoch": self._total_epochs,
            "config_mode": self.config.mode,
            "difficulty": getattr(self, "difficulty", None),
            "self_paced_pace": self.get_pace(),
            "adaptive_state": self.study.best_params if self.study else None,
        }

# ════════════════════════════════════════════════════════════════════════════

class HPOManager:
    """
    Unified HPO Manager integrating PBT, BOHB, ASHA, and Optuna.
    
    Coordinates multiple HPO strategies and manages trial lifecycle.
    """
    
    def __init__(self, config: HPOConfig):
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        
        # Initialize schedulers
        self.pbt = PopulationBasedTraining(config)
        self.hyperband = HyperBandScheduler(config)
        self.asha = AsyncSuccessiveHalvingScheduler(config)
        self.bohb = BOHBScheduler(config)
        self.mf_asha = MultiFidelityASHAScheduler(config, [])
        
        # Optuna integration
        self.study: Optional[optuna.Study] = None
        
        # Trial management
        self.trials: Dict[str, Dict] = {}
        self.completed_trials: List[Dict] = []
        self._trial_counter = 0
        self._total_epochs = 0
        
        # Setup checkpointing
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def create_study(self, 
                     direction: str = "maximize",
                     sampler: str = "tpe",
                     pruner: str = "hyperband") -> optuna.Study:
        """Create Optuna study with appropriate sampler and pruner."""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not available")
        
        # Sampler
        if sampler == "tpe":
            sampler_obj = TPESampler(seed=self.config.seed)
        elif sampler == "random":
            from optuna.samplers import RandomSampler
            sampler_obj = RandomSampler(seed=self.config.seed)
        elif sampler == "cmaes":
            from optuna.samplers import CmaEsSampler
            sampler_obj = CmaEsSampler(seed=self.config.seed)
        else:
            raise ValueError(f"Unknown sampler: {sampler}")
        
        # Pruner
        if pruner == "hyperband":
            pruner_obj = HyperbandPruner(
                min_resource=self.config.min_budget,
                max_resource=self.config.max_budget,
                reduction_factor=self.config.eta,
            )
        elif pruner == "median":
            pruner_obj = MedianPruner(n_warmup_steps=5)
        elif pruner == "asha":
            pruner_obj = HyperbandPruner(
                min_resource=self.config.min_budget,
                max_resource=self.config.max_budget,
                reduction_factor=self.config.reduction_factor,
            )
        elif pruner == "none":
            pruner_obj = NopPruner()
        else:
            raise ValueError(f"Unknown pruner: {pruner}")
        
        study = optuna.create_study(
            direction="maximize" if self.config.mode == "maximize" else "minimize",
            sampler=sampler_obj,
            pruner=pruner_obj,
            study_name=f"hpo_{int(time.time())}",
        )
        
        self.study = study
        return study
    
    def run_trial(self, params: Dict[str, Any], trial_id: str) -> Dict[str, Any]:
        """Run a single trial and return results."""
        # This would integrate with train_gpu.py
        # For now, return mock results
        self._trial_counter += 1
        trial_id = f"trial_{self._trial_counter}"
        
        # Mock result
        result = {
            "trial_id": trial_id,
            "params": params,
            "val_sharpe": np.random.rand() * 2 - 1,
            "val_loss": np.random.rand(),
            "epochs_completed": self.config.max_epochs,
        }
        
        return {
            "trial_id": trial_id,
            "objective": np.random.rand() * 2 - 1,  # val_sharpe
            "metrics": {"val_sharpe": np.random.rand() * 2 - 1, "val_loss": np.random.rand()},
        }
    
    def pretrain(
        self,
        X: np.ndarray,
        domain_labels: Optional[np.ndarray] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        silent: bool = False,
    ) -> Dict[str, List[float]]:
        """
        Run multi-task pretraining.
        
        Args:
            X: Input data (N, seq_len, n_features)
            domain_labels: Domain labels for each sample (for DANN)
            epochs: Number of epochs
            batch_size: Batch size
            checkpoint_path: Path to save encoder checkpoint
            silent: Suppress progress output
        
        Returns:
            History dictionary with loss curves.
        """
        epochs = epochs or self.config.epochs
        batch_size = batch_size or self.config.batch_size
        if checkpoint_path is None:
            from config.settings import PATHS
            checkpoint_path = PATHS.get("file_contrastive_encoder", "encoder.pt")
        
        N = len(X)
        if not silent:
            tasks = [k for k in self.task_weights.keys() if self.task_weights[k] > 0]
            print(f"[MultiTask] {epochs} ep | {N:,} windows | batch={batch_size} | tasks={tasks}")
        
        # Prepare domain labels tensor
        domain_labels_tensor = None
        if domain_labels is not None:
            domain_labels = torch.as_tensor(domain_labels, dtype=torch.long)
        
        for epoch in range(epochs):
            self._total_epochs += 1
            idx_perm = self._rng.permutation(len(X))
            epoch_losses = {k: 0.0 for k in self.task_weights.keys()}
            epoch_losses["total"] = 0.0
            n_batches = 0
            
            # LR schedule with warmup + cosine decay
            base_lr = self.config.lr
            warmup = min(self.config.warmup_epochs, epochs)
            if epoch < warmup:
                lr_scale = (epoch + 1) / warmup
            else:
                progress = (epoch - warmup) / max(1, epochs - warmup)
                lr_scale = 0.5 * (1.0 + np.cos(np.pi * progress))
            for pg in self.opt.param_groups:
                pg["lr"] = self.config.lr * lr_scale
            
            for start in range(0, len(X), batch_size):
                batch_idx = idx_perm[start : start + batch_size]
                if len(batch_idx) < 4:
                    continue
                
                x_batch = torch.as_tensor(X[batch_idx], dtype=torch.float32, device=self.device)
                dom_batch = domain_labels[batch_idx] if domain_labels is not None else None
                
                losses = self.step(x_batch, dom_batch)
                
                for k, v in losses.items():
                    epoch_losses[k] += v
                n_batches += 1
            
            if n_batches > 0:
                for k in epoch_losses:
                    epoch_losses[k] /= n_batches
            
            for k, v in epoch_losses.items():
                self.history[k].append(v)
            
            if not silent:
                parts = [f"Ep {self._total_epochs:3d} | total={epoch_losses.get('total', 0):.4f}"]
                for k, v in epoch_losses.items():
                    if k != "total":
                        parts.append(f"{k}={v:.4f}")
                if self.gradnorm_params is not None:
                    parts.append(f"w={self.gradnorm_params.detach().cpu().numpy()}")
                print(" | ".join(parts))
        
        # Save encoder
        torch.save(self.encoder.state_dict(), checkpoint_path)
        if not silent:
            print(f"[MultiTask] Encoder saved \u2192 {checkpoint_path}")
        
        return self.history
    
    @torch.no_grad()
    def diagnostics(self, X_ref: np.ndarray, max_samples: int = 128) -> Dict:
        """Compute diagnostic metrics on reference data."""
        self.encoder.eval()
        for head in self.heads.values():
            head.eval()
        
        out = {"embed_std": 0.0, "collapsed": True}
        
        if X_ref is not None and len(X_ref) >= 4:
            try:
                sample = X_ref[: min(max_samples, len(X_ref))]
                x = torch.as_tensor(sample, dtype=torch.float32, device=self.device)
                h = self._forward_encoder(x)
                std = h.std(dim=0).mean().item()
                out["embed_std"] = float(std if np.isfinite(std) else 0.0)
                out["collapsed"] = std < 0.005
            except Exception as exc:
                print(f"[MultiTask] diagnostic error: {exc}")
        
        self.encoder.train()
        for head in self.heads.values():
            head.train()
        return out
    
    def save_encoder(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.encoder.state_dict(), path)


    def get_pace(self) -> float:
        """Get current pacing fraction for self-paced learning."""
        if not hasattr(self, "config"):
            return 1.0
        total = max(1, self.config.max_epochs)
        return min(self._total_epochs / total, 1.0)

    def _sample_initial_params(self) -> Dict[str, Any]:
        """Sample initial hyperparameters."""
        return {
            "lr": 10 ** np.random.uniform(-5, -3),
            "d_model": np.random.choice([64, 128, 256, 512]),
            "num_layers": np.random.randint(2, 6),
            "dropout": np.random.uniform(0.1, 0.5),
            "batch_size": np.random.choice([32, 64, 128, 256]),
        }

    def state_dict(self) -> Dict[str, Any]:
        """Get HPO manager state as dictionary."""
        return {
            "epoch": self._total_epochs,
            "config_mode": self.config.mode,
            "difficulty": getattr(self, "difficulty", None),
            "self_paced_pace": self.get_pace(),
            "adaptive_state": self.study.best_params if self.study else None,
        }

# ═════════════════════════════════════════════════════════════════════════════
# 7. Integration with train_gpu.py
    def get_pace(self) -> float:
        """Get current pacing fraction for self-paced learning."""
        if not hasattr(self, "config"):
            return 1.0
        total = max(1, self.config.max_epochs)
        return min(self._total_epochs / total, 1.0)

    def _sample_initial_params(self) -> Dict[str, Any]:
        """Sample initial hyperparameters."""
        return {
            "lr": 10 ** np.random.uniform(-5, -3),
            "d_model": np.random.choice([64, 128, 256, 512]),
            "num_layers": np.random.randint(2, 6),
            "dropout": np.random.uniform(0.1, 0.5),
            "batch_size": np.random.choice([32, 64, 128, 256]),
        }

    def state_dict(self) -> Dict[str, Any]:
        """Get HPO manager state as dictionary."""
        return {
            "epoch": self._total_epochs,
            "config_mode": self.config.mode,
            "difficulty": getattr(self, "difficulty", None),
            "self_paced_pace": self.get_pace(),
            "adaptive_state": self.study.best_params if self.study else None,
        }

# ═══════════════════════════════════════════════════════════════════════════

def build_optuna_search(
    scheduler: str = "tpe",
    seed: int = 0,
    min_resource: int = 2,
    max_resource: int = 27,
    reduction_factor: int = 3,
) -> Tuple[Any, Any]:
    """
    Build a (sampler, pruner) pair for an Optuna study from an HPO strategy.

    Improvement #12 wiring: exposes the PBT / BOHB / ASHA strategy choices to
    the production tuner (``scripts/optuna_tune.py --hpo-scheduler``).

    - ``tpe``   -> TPESampler + MedianPruner  (default, matches prior behavior)
    - ``asha``  -> TPESampler + SuccessiveHalvingPruner
    - ``bohb``  -> TPESampler + HyperbandPruner  (BO via TPE + HyperBand brackets)
    - ``pbt``   -> CmaEsSampler + HyperbandPruner (population-based evolutionary)

    Returns ``(sampler, pruner)``.
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna not available")

    scheduler = str(scheduler or "tpe").lower()
    if scheduler == "tpe":
        sampler = TPESampler(seed=seed)
        pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=2)
    elif scheduler == "asha":
        sampler = TPESampler(seed=seed)
        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=int(min_resource),
            reduction_factor=int(reduction_factor),
        )
    elif scheduler == "bohb":
        sampler = TPESampler(seed=seed)
        pruner = HyperbandPruner(
            min_resource=int(min_resource),
            max_resource=int(max_resource),
            reduction_factor=int(reduction_factor),
        )
    elif scheduler == "pbt":
        from optuna.samplers import CmaEsSampler
        sampler = CmaEsSampler(seed=seed)
        pruner = HyperbandPruner(
            min_resource=int(min_resource),
            max_resource=int(max_resource),
            reduction_factor=int(reduction_factor),
        )
    else:
        raise ValueError(f"Unknown HPO scheduler: {scheduler}")
    return sampler, pruner


def run_hpo_study(
    base_config_path: str,
    model_name: str,
    algorithm: str = "bohb",
    n_trials: int = 100,
    max_epochs: int = 27,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run HPO study with specified algorithm.
    
    Args:
        base_config_path: Path to base config YAML
        model_name: Model to tune ("tft", "haelt", "transformer")
        algorithm: HPO algorithm ("optuna", "pbt", "bohb", "asha", "mf_asha")
        n_trials: Number of trials
        max_epochs: Max epochs per trial
        **kwargs: Additional config overrides
    
    Returns:
        Dictionary with best config and results
    """
    config = HPOConfig(
        max_epochs=max_epochs,
        n_trials=n_trials,
    )
    
    manager = HPOManager(config)
    
    return manager.run_hpo(
        n_trials=n_trials,
        algorithm=algorithm,
    )


    def get_pace(self) -> float:
        """Get current pacing fraction for self-paced learning."""
        if not hasattr(self, "config"):
            return 1.0
        total = max(1, self.config.max_epochs)
        return min(self._total_epochs / total, 1.0)

    def _sample_initial_params(self) -> Dict[str, Any]:
        """Sample initial hyperparameters."""
        return {
            "lr": 10 ** np.random.uniform(-5, -3),
            "d_model": np.random.choice([64, 128, 256, 512]),
            "num_layers": np.random.randint(2, 6),
            "dropout": np.random.uniform(0.1, 0.5),
            "batch_size": np.random.choice([32, 64, 128, 256]),
        }

    def state_dict(self) -> Dict[str, Any]:
        """Get HPO manager state as dictionary."""
        return {
            "epoch": self._total_epochs,
            "config_mode": self.config.mode,
            "difficulty": getattr(self, "difficulty", None),
            "self_paced_pace": self.get_pace(),
            "adaptive_state": self.study.best_params if self.study else None,
        }

# ═══════════════════════════════════════════════════════════════════════════════
# 7. Export
    def get_pace(self) -> float:
        """Get current pacing fraction for self-paced learning."""
        if not hasattr(self, "config"):
            return 1.0
        total = max(1, self.config.max_epochs)
        return min(self._total_epochs / total, 1.0)

    def _sample_initial_params(self) -> Dict[str, Any]:
        """Sample initial hyperparameters."""
        return {
            "lr": 10 ** np.random.uniform(-5, -3),
            "d_model": np.random.choice([64, 128, 256, 512]),
            "num_layers": np.random.randint(2, 6),
            "dropout": np.random.uniform(0.1, 0.5),
            "batch_size": np.random.choice([32, 64, 128, 256]),
        }

    def state_dict(self) -> Dict[str, Any]:
        """Get HPO manager state as dictionary."""
        return {
            "epoch": self._total_epochs,
            "config_mode": self.config.mode,
            "difficulty": getattr(self, "difficulty", None),
            "self_paced_pace": self.get_pace(),
            "adaptive_state": self.study.best_params if self.study else None,
        }

# ════════════════════════════════════════════════════════════════════════════

__all__ = [
    "HPOConfig",
    "TrialState",
    "TrialScheduler",
    "PopulationBasedTraining",
    "HyperBandScheduler",
    "AsyncSuccessiveHalvingScheduler",
    "BOHBScheduler",
    "MultiFidelityASHAScheduler",
    "HPOManager",
    "run_hpo_study",
]


if __name__ == "__main__":
    # Quick self-test
    config = HPOConfig(
        population_size=4,
        max_epochs=10,
        min_budget=3,
        max_budget=27,
        eta=3,
    )
    
    # Test PBT
    pbt = PopulationBasedTraining(HPOConfig(population_size=4))
    for i in range(4):
        pbt.add_trial(f"trial_{i}", {"lr": 1e-3 * (i+1), "batch_size": 32 * (i+1)})
    
    print("Population size:", len(pbt.population))
    
    pbt.update_score("trial_0", 0.5, 10)
    pbt.update_score("trial_1", 0.8, 10)
    pbt.update_score("trial_2", 0.3, 10)
    pbt.update_score("trial_3", 0.9, 10)
    
    print("Best trial:", pbt.best_trial.trial_id if pbt.best_trial else None)
    print("Stats:", pbt.get_population_stats())
    
    print("HPO module OK")