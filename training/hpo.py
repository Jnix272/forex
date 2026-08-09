"""
HPO Module (Improvement #12)
============================
Population-Based Training (PBT), BOHB (Bayesian Optimization HyperBand),
Asynchronous Successive Halving (ASHA), and Multi-fidelity ASHA.

Integrates with existing Optuna infrastructure in optuna_tune.py.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import optuna
    from optuna.pruners import HyperbandPruner, MedianPruner, NopPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


@dataclass
class HPOConfig:
    """Configuration for HPO algorithms."""
    population_size: int = 8
    perturbation_interval: int = 10
    perturbation_factor: float = 0.2
    exploit_factor: float = 0.8

    min_budget: int = 3
    max_budget: int = 27
    eta: int = 3
    min_points_in_model: int = 8
    top_n_percent: int = 15

    grace_period: int = 1
    reduction_factor: int = 3
    brackets: int = 1

    max_epochs: int = 27
    metric: str = "val_sharpe"
    mode: str = "maximize"
    n_trials: int = 100
    seed: int = 42

    max_concurrent_trials: int = 4
    time_budget_sec: int | None = None

    checkpoint_dir: str = "checkpoints/hpo"
    checkpoint_interval: int = 5

    # Default search space used by HyperBand / BOHB suggest_params
    search_space: dict[str, Any] = field(default_factory=lambda: {
        "lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
        "dropout": {"type": "uniform", "low": 0.1, "high": 0.5},
        "hidden_size": {"type": "choice", "values": [128, 256, 512]},
        "batch_size": {"type": "choice", "values": [64, 128, 256, 512]},
        "weight_decay": {"type": "loguniform", "low": 1e-6, "high": 1e-2},
    })


def _sample_search_space(rng: np.random.Generator, space: dict[str, Any]) -> dict[str, Any]:
    """Draw one config from a typed search-space dict."""
    params: dict[str, Any] = {}
    for name, spec in (space or {}).items():
        if not isinstance(spec, dict):
            params[name] = spec
            continue
        t = str(spec.get("type", "uniform")).lower()
        if t == "choice":
            values = list(spec.get("values") or [])
            if not values:
                continue
            params[name] = values[int(rng.integers(0, len(values)))]
        elif t == "loguniform":
            low = float(spec["low"])
            high = float(spec["high"])
            params[name] = float(np.exp(rng.uniform(np.log(low), np.log(high))))
        elif t == "int":
            low = int(spec["low"])
            high = int(spec["high"])
            params[name] = int(rng.integers(low, high + 1))
        else:  # uniform
            low = float(spec.get("low", 0.0))
            high = float(spec.get("high", 1.0))
            params[name] = float(rng.uniform(low, high))
    return params


@dataclass
class TrialState:
    """State of a single trial in population-based methods."""
    trial_id: str
    params: dict[str, Any]
    step: int = 0
    score: float = -float("inf")
    best_score: float = -float("inf")
    checkpoint_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: TrialState) -> bool:
        return self.score < other.score


class TrialScheduler(ABC):
    """Abstract base class for trial schedulers."""

    @abstractmethod
    def suggest_params(self, trial_id: str) -> dict[str, Any]:
        pass

    @abstractmethod
    def on_trial_result(self, trial_id: str, result: dict[str, Any]) -> dict[str, Any]:
        pass

    @abstractmethod
    def on_trial_complete(self, trial_id: str, result: dict[str, Any]) -> None:
        pass

    @abstractmethod
    def get_next_trials(self) -> list[dict[str, Any]]:
        pass


class PopulationBasedTraining:
    """Population-Based Training (PBT) — Jaderberg et al., 2017."""

    def __init__(self, config: HPOConfig):
        self.config = config
        self.population: list[TrialState] = []
        self.best_trial: TrialState | None = None
        self._rng = np.random.default_rng(config.seed)

    def add_trial(self, trial_id: str, params: dict[str, Any]) -> TrialState:
        trial = TrialState(trial_id=trial_id, params=params.copy(), step=0, score=-float("inf"))
        self.population.append(trial)
        return trial

    def update_score(
        self,
        trial_id: str,
        score: float,
        step: int,
        checkpoint_path: str = "",
    ) -> None:
        for trial in self.population:
            if trial.trial_id != trial_id:
                continue
            trial.step = step
            trial.score = score
            trial.best_score = max(trial.best_score, score)
            if checkpoint_path:
                trial.checkpoint_path = checkpoint_path
            if self.best_trial is None or trial.best_score > self.best_trial.best_score:
                self.best_trial = trial
            break

    def should_perturb(self, trial_id: str) -> bool:
        trial = self._get_trial(trial_id)
        if trial is None:
            return False
        return trial.step > 0 and trial.step % self.config.perturbation_interval == 0

    def perturb_trial(self, trial_id: str) -> dict[str, Any]:
        trial = self._get_trial(trial_id)
        if trial is None or self.best_trial is None:
            return trial.params if trial else {}

        if self._rng.random() < self.config.exploit_factor and self.best_trial != trial:
            new_params = self.best_trial.params.copy()
        else:
            better = [t for t in self.population if t.best_score > trial.best_score]
            donor = self._rng.choice(better) if better else trial
            new_params = donor.params.copy()

        perturbed: dict[str, Any] = {}
        for key, value in new_params.items():
            if isinstance(value, float) and value > 0:
                factor = np.exp(
                    self._rng.uniform(
                        -self.config.perturbation_factor,
                        self.config.perturbation_factor,
                    )
                )
                perturbed[key] = value * factor
            elif isinstance(value, int):
                # Symmetric ±1 step (np.integers high is exclusive → {-1,0,1})
                delta = int(self._rng.integers(-1, 2))
                perturbed[key] = max(1, int(value) + delta)
            else:
                perturbed[key] = value
        return perturbed

    def exploit_and_perturb(self, trial_id: str) -> tuple[str, dict[str, Any]]:
        sorted_pop = sorted(self.population, key=lambda t: t.score)
        worst_trial = sorted_pop[0]
        best_trial = sorted_pop[-1]
        if worst_trial.trial_id == best_trial.trial_id:
            return worst_trial.trial_id, worst_trial.params

        new_params = self.perturb_trial(best_trial.trial_id)
        worst_trial.params = new_params
        worst_trial.score = -float("inf")
        worst_trial.best_score = -float("inf")
        worst_trial.step = 0
        return best_trial.trial_id, worst_trial.params

    def _get_trial(self, trial_id: str) -> TrialState | None:
        for trial in self.population:
            if trial.trial_id == trial_id:
                return trial
        return None

    def get_population_stats(self) -> dict[str, Any]:
        if not self.population:
            return {}
        scores = [t.score for t in self.population if t.score != -float("inf")]
        if not scores:
            return {"population_size": len(self.population)}
        return {
            "population_size": len(self.population),
            "mean_score": float(np.mean(scores)),
            "best_score": float(max(scores)),
            "worst_score": float(min(scores)),
        }


class HyperBandScheduler:
    """HyperBand Scheduler — Li et al., 2017."""

    def __init__(self, config: HPOConfig):
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        self.max_budget = config.max_budget
        self.min_budget = config.min_budget
        self.eta = config.eta
        self.grace_period = config.grace_period
        self.s_max = max(
            0,
            int(math.log(config.max_budget / config.min_budget) / math.log(config.eta)),
        )
        self.brackets: list[dict[str, Any]] = []
        self._init_brackets()

    def _init_brackets(self) -> None:
        for s in range(self.s_max + 1):
            n = int(math.ceil((self.s_max + 1) / (s + 1) * self.eta ** s))
            r = self.min_budget * (self.eta ** (self.s_max - s))
            bracket: dict[str, Any] = {"s": s, "n": n, "r": r, "rungs": [], "promoted": []}
            for i in range(s + 1):
                bracket["rungs"].append({
                    "budget": self.min_budget * (self.eta ** i),
                    "trials": [],
                    "completed": 0,
                    "promoted": 0,
                })
            self.brackets.append(bracket)

    def suggest_params(self, trial_id: str, bracket_idx: int | None = None) -> dict[str, Any]:
        return _sample_search_space(self._rng, self.config.search_space)

    def on_trial_result(self, trial_id: str, result: dict[str, Any]) -> dict[str, Any]:
        return {"action": "continue"}

    def on_trial_complete(self, trial_id: str, result: dict[str, Any]) -> None:
        return None

    def get_next_trials(self) -> list[dict[str, Any]]:
        out = []
        for i, bracket in enumerate(self.brackets):
            tid = f"hb_s{bracket['s']}_{len(out)}"
            out.append({
                "trial_id": tid,
                "bracket": i,
                "budget": float(bracket["r"]),
                "params": self.suggest_params(tid, bracket_idx=i),
            })
        return out


class AsyncSuccessiveHalvingScheduler:
    """ASHA — Li et al., 2020."""

    def __init__(self, config: HPOConfig):
        self.config = config
        self.grace_period = config.grace_period
        self.reduction_factor = config.reduction_factor
        self.brackets = config.brackets
        self.rungs: dict[int, list[dict]] = defaultdict(list)
        self.trial_states: dict[str, dict] = {}
        self._rng = np.random.default_rng(config.seed)

    def add_trial(self, trial_id: str, params: dict[str, Any]) -> None:
        self.trial_states[trial_id] = {
            "trial_id": trial_id,
            "params": params,
            "rung": 0,
            "resources": self.config.grace_period,
            "metrics": [],
            "status": "running",
        }
        self.rungs[0].append({"trial_id": trial_id, "score": -float("inf")})

    def on_trial_result(self, trial_id: str, result: dict[str, Any]) -> dict[str, Any]:
        if trial_id not in self.trial_states:
            return {"action": "continue"}

        state = self.trial_states[trial_id]
        state["metrics"].append(result)
        score = float(result.get(self.config.metric, -float("inf")))
        state["last_score"] = score
        current_rung = int(state["rung"])

        for entry in self.rungs[current_rung]:
            if entry["trial_id"] == trial_id:
                entry["score"] = score
                break

        scored = [
            # Use the rung-level score (stored in rungs[current_rung] entry) rather than
            # last_score, which may reflect a higher rung and cause unfair cross-rung comparisons.
            (t["trial_id"], float(t.get("score", -float("inf"))))
            for t in self.rungs[current_rung]
            if t.get("score", -float("inf")) > -float("inf")
        ]
        # Need a full η-cohort before eliminating; otherwise every early report
        # would kill peers and waste the remaining budget.
        if len(scored) < self.reduction_factor:
            return {"action": "continue"}

        maximize = self.config.mode == "maximize"
        scored.sort(key=lambda x: x[1], reverse=maximize)

        # SHA: keep top 1/η, stop everyone else (not only the bottom 1/η).
        # Stopping only the bottom slice left mediocre trials running → wasted compute.
        n_keep = max(1, len(scored) // self.reduction_factor)
        keep_ids = {tid for tid, _ in scored[:n_keep]}
        if trial_id not in keep_ids:
            state["status"] = "stopped"
            return {"action": "stop"}

        # Survivors advance to the next rung.
        state["rung"] = current_rung + 1
        self.rungs[current_rung + 1].append({"trial_id": trial_id, "score": score})
        return {"action": "promote", "next_rung": current_rung + 1}

    def on_trial_complete(self, trial_id: str, result: dict[str, Any]) -> None:
        if trial_id in self.trial_states:
            self.trial_states[trial_id]["status"] = "completed"
            self.trial_states[trial_id]["final_result"] = result

    def get_next_trials(self, max_concurrent: int = 4) -> list[dict[str, Any]]:
        return []


class BOHBScheduler:
    """BOHB — Falkner et al., 2018."""

    def __init__(self, config: HPOConfig):
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        self.hyperband = HyperBandScheduler(config)
        self.kde_good: dict[int, Any] = {}
        self.kde_bad: dict[int, Any] = {}
        self.observations: dict[int, list[tuple[dict, float]]] = defaultdict(list)

    def _update_kde(self, rung: int) -> None:
        obs = self.observations.get(rung, [])
        if len(obs) < self.config.min_points_in_model:
            return
        obs = sorted(obs, key=lambda x: x[1], reverse=(self.config.mode == "maximize"))
        n_good = max(1, len(obs) // max(1, self.config.top_n_percent))
        n_bad = max(1, len(obs) - n_good)
        self.kde_good[rung] = {
            "configs": [o[0] for o in obs[:n_good]],
            "scores": [o[1] for o in obs[:n_good]],
        }
        self.kde_bad[rung] = {
            "configs": [o[0] for o in obs[-n_bad:]],
            "scores": [o[1] for o in obs[-n_bad:]],
        }

    def suggest_params(self, rung: int = 0) -> dict[str, Any]:
        """Sample from KDE of good configs when available, else search_space prior."""
        good = self.kde_good.get(rung)
        if good and good.get("configs"):
            # Perturb a randomly chosen good config
            base = dict(good["configs"][int(self._rng.integers(0, len(good["configs"])))])
            prior = _sample_search_space(self._rng, self.config.search_space)
            out = {}
            for k, v in prior.items():
                if k in base and isinstance(base[k], (int, float)) and isinstance(v, (int, float)):
                    # Mix: 70% good + 30% prior noise
                    out[k] = float(base[k]) * 0.7 + float(v) * 0.3
                    if isinstance(base[k], int):
                        out[k] = int(round(out[k]))
                else:
                    out[k] = base.get(k, v)
            return out
        return _sample_search_space(self._rng, self.config.search_space)

    def observe(self, trial_id: str, rung: int, config: dict, score: float) -> None:
        self.observations[rung].append((config, score))
        self._update_kde(rung)

    def on_trial_result(self, trial_id: str, result: dict[str, Any]) -> dict[str, Any]:
        return {"action": "continue"}

    def on_trial_complete(self, trial_id: str, result: dict[str, Any]) -> None:
        return None


class MultiFidelityASHAScheduler:
    """Multi-fidelity ASHA over several resource dimensions."""

    def __init__(self, config: HPOConfig, fidelity_dims: list[dict[str, Any]]):
        self.config = config
        self.fidelity_dims = fidelity_dims
        self._rng = np.random.default_rng(config.seed)
        self.fidelity_rungs: dict[tuple, list[dict]] = defaultdict(list)
        self.trial_states: dict[str, dict] = {}

    def add_trial(self, trial_id: str, params: dict[str, Any]) -> None:
        fidelity = {dim["name"]: dim["min"] for dim in self.fidelity_dims}
        self.trial_states[trial_id] = {
            "trial_id": trial_id,
            "params": params,
            "fidelity": fidelity,
            "metrics": [],
            "status": "running",
        }

    def on_trial_result(self, trial_id: str, result: dict[str, Any]) -> dict[str, Any]:
        if trial_id not in self.trial_states:
            return {"action": "continue"}
        self.trial_states[trial_id]["metrics"].append(result)
        return {"action": "continue", "next_fidelity": None}

    def on_trial_complete(self, trial_id: str, result: dict[str, Any]) -> None:
        if trial_id in self.trial_states:
            self.trial_states[trial_id]["status"] = "completed"

    def get_next_trials(self, max_concurrent: int = 4) -> list[dict[str, Any]]:
        return []


class HPOManager:
    """Unified HPO manager for PBT / BOHB / ASHA / Optuna wiring."""

    def __init__(self, config: HPOConfig):
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        self.pbt = PopulationBasedTraining(config)
        self.hyperband = HyperBandScheduler(config)
        self.asha = AsyncSuccessiveHalvingScheduler(config)
        self.bohb = BOHBScheduler(config)
        self.mf_asha = MultiFidelityASHAScheduler(config, [])
        self.study: Any | None = None
        self.trials: dict[str, dict] = {}
        self.completed_trials: list[dict] = []
        self._trial_counter = 0
        self._total_epochs = 0
        self.difficulty = None
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _sample_initial_params(self) -> dict[str, Any]:
        """Sample a random hyperparameter set for PBT / dry-run studies."""
        return {
            "lr": float(10 ** self._rng.uniform(-5, -3)),
            "d_model": int(self._rng.choice([64, 128, 256, 512])),
            "num_layers": int(self._rng.integers(2, 6)),
            "dropout": float(self._rng.uniform(0.1, 0.5)),
            "batch_size": int(self._rng.choice([32, 64, 128, 256])),
        }

    def get_pace(self) -> float:
        total = max(1, self.config.max_epochs)
        return min(self._total_epochs / total, 1.0)

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self._total_epochs,
            "config_mode": self.config.mode,
            "difficulty": self.difficulty,
            "self_paced_pace": self.get_pace(),
            "adaptive_state": (
                self.study.best_params if self.study is not None and self.study.best_trial else None
            ),
        }

    def create_study(
        self,
        direction: str = "maximize",
        sampler: str = "tpe",
        pruner: str = "hyperband",
    ) -> Any:
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not available")

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

    def run_trial(self, params: dict[str, Any], trial_id: str | None = None) -> dict[str, Any]:
        """Dry-run trial hook (real training is wired via optuna_tune / train_gpu)."""
        self._trial_counter += 1
        tid = trial_id or f"trial_{self._trial_counter}"
        score = float(self._rng.random() * 2 - 1)
        self._total_epochs += 1
        result = {
            "trial_id": tid,
            "params": params,
            "objective": score,
            "metrics": {
                self.config.metric: score,
                "val_loss": float(self._rng.random()),
            },
        }
        self.completed_trials.append(result)
        return result

    def run_hpo(
        self,
        n_trials: int | None = None,
        algorithm: str = "asha",
        **_kwargs: Any,
    ) -> dict[str, Any]:
        """
        Run a lightweight HPO loop over the selected scheduler.

        Real model training is expected to plug into ``run_trial`` / Optuna; this
        method registers sampled configs, executes dry-run trials, and returns
        the best params by ``config.metric``.
        """
        n = int(n_trials or self.config.n_trials)
        algorithm = str(algorithm or "asha").lower()
        if algorithm == "optuna":
            self.create_study()

        best_score = -float("inf") if self.config.mode == "maximize" else float("inf")
        best_params: dict[str, Any] = {}
        best_trial_id: str | None = None

        for i in range(n):
            params = self._sample_initial_params()
            trial_id = f"{algorithm}_{i}"
            if algorithm == "pbt":
                self.pbt.add_trial(trial_id, params)
            elif algorithm in ("asha", "mf_asha"):
                self.asha.add_trial(trial_id, params)
            elif algorithm == "bohb":
                self.bohb.observe(trial_id, 0, params, 0.0)

            result = self.run_trial(params, trial_id=trial_id)
            score = float(result["metrics"].get(self.config.metric, result["objective"]))

            if algorithm == "pbt":
                self.pbt.update_score(trial_id, score, step=self.config.max_epochs)
            elif algorithm in ("asha", "mf_asha"):
                self.asha.on_trial_result(trial_id, {self.config.metric: score})
                self.asha.on_trial_complete(trial_id, result)
            elif algorithm == "bohb":
                self.bohb.observe(trial_id, 0, params, score)

            improved = (
                score > best_score if self.config.mode == "maximize" else score < best_score
            )
            if improved:
                best_score = score
                best_params = dict(params)
                best_trial_id = trial_id

        return {
            "algorithm": algorithm,
            "n_trials": n,
            "best_trial_id": best_trial_id,
            "best_params": best_params,
            "best_score": best_score if best_params else None,
            "metric": self.config.metric,
        }


def build_optuna_search(
    scheduler: str = "tpe",
    seed: int = 0,
    min_resource: int = 2,
    max_resource: int = 27,
    reduction_factor: int = 3,
) -> tuple[Any, Any]:
    """Build a (sampler, pruner) pair for an Optuna study from an HPO strategy."""
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
    **kwargs: Any,
) -> dict[str, Any]:
    """Run an HPO study with the specified algorithm via ``HPOManager.run_hpo``."""
    config = HPOConfig(max_epochs=max_epochs, n_trials=n_trials, **{
        k: v for k, v in kwargs.items() if k in HPOConfig.__dataclass_fields__
    })
    manager = HPOManager(config)
    result = manager.run_hpo(n_trials=n_trials, algorithm=algorithm)
    result["base_config_path"] = base_config_path
    result["model_name"] = model_name
    return result


__all__ = [
    "AsyncSuccessiveHalvingScheduler",
    "BOHBScheduler",
    "HPOConfig",
    "HPOManager",
    "HyperBandScheduler",
    "MultiFidelityASHAScheduler",
    "PopulationBasedTraining",
    "TrialScheduler",
    "TrialState",
    "build_optuna_search",
    "run_hpo_study",
]


if __name__ == "__main__":
    cfg = HPOConfig(population_size=4, max_epochs=10, min_budget=3, max_budget=27, eta=3)
    pbt = PopulationBasedTraining(HPOConfig(population_size=4))
    for i in range(4):
        pbt.add_trial(f"trial_{i}", {"lr": 1e-3 * (i + 1), "batch_size": 32 * (i + 1)})
    pbt.update_score("trial_0", 0.5, 10)
    pbt.update_score("trial_1", 0.8, 10)
    pbt.update_score("trial_2", 0.3, 10)
    pbt.update_score("trial_3", 0.9, 10)
    print("Best trial:", pbt.best_trial.trial_id if pbt.best_trial else None)
    print("Stats:", pbt.get_population_stats())
    mgr = HPOManager(cfg)
    print("run_hpo:", mgr.run_hpo(n_trials=3, algorithm="asha"))
    print("HPO module OK")
