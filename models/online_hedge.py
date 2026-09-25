"""
models/online_hedge.py
======================
Online Adaptive Ensemble Weighting using the Hedge / Multiplicative Weights (Exp3) algorithm.

In live execution, deep neural network weights cannot be safely updated via gradient
descent inside the live loop without risking catastrophic forgetting, latency jitter,
and violation of risk constraints.

Instead, the Hedge algorithm dynamically adjusts the mixture / voting weights of
different base models (e.g. HAELT, TFT, Mamba, GNN, XGBoost, or RL agents) on every
completed bar based on realized returns and market regime alignment.

Mathematical Formulation:
-------------------------
For K models, weights w_i are initialized uniformly:
    w_{i, 0} = 1 / K

On bar t, each model produces a predicted signal \\hat{y}_{i, t} \\in [-1, 1].
The ensemble consensus is:
    \\hat{Y}_t = \\sum_{i=1}^K w_{i, t} \\cdot \\hat{y}_{i, t}

When bar t+1 completes with price return R_{t+1} and volatility \\sigma_{t+1} (ATR):
1. Normalized Reward per model:
    r_{i, t} = sign(\\hat{y}_{i, t}) \\cdot \\frac{R_{t+1}}{\\sigma_{t+1} + \\epsilon}
2. Discounted Cumulative Score:
    S_{i, t+1} = \\gamma \\cdot S_{i, t} + r_{i, t}
3. Softmax Weight Update with Exploration Floor \\epsilon_{floor}:
    w_{i, t+1} = (1 - \\epsilon_{floor}) \\frac{\\exp(\\eta S_{i, t+1})}{\\sum_{j=1}^K \\exp(\\eta S_{j, t+1})} + \\frac{\\epsilon_{floor}}{K}

Properties:
-----------
- Zero gradient computation (<0.1 ms execution time).
- No catastrophic forgetting of base representations.
- Mathematical no-regret guarantee against the best single expert in hindsight.
- Persistent state across engine restarts via atomic JSON serialization.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class OnlineHedgeEnsemble:
    """
    Adaptive online ensemble meta-learner implementing the Hedge / Exp3 algorithm.
    """

    def __init__(
        self,
        model_names: list[str],
        learning_rate: float = 0.1,
        discount_factor: float = 0.98,
        min_weight_floor: float = 0.05,
        state_path: Path | str | None = None,
        initial_sharpes: dict[str, float] | None = None,
    ):
        """
        Parameters
        ----------
        model_names : list[str]
            Names/IDs of the base models or agents participating in the ensemble.
        learning_rate : float
            eta parameter governing responsiveness of weight shifts. Higher values
            shift weights more aggressively to recent winners. Default 0.1.
        discount_factor : float
            gamma parameter in (0, 1] governing memory decay of past rewards.
            Default 0.98 (approx 50-bar half life on 5-minute candles).
        min_weight_floor : float
            epsilon parameter ensuring each model maintains a minimum voting floor
            (min_weight_floor / K). Prevents models from being permanently zeroed out.
        state_path : Path | str, optional
            Path to persist weight state to disk for zero-loss engine restarts.
        initial_sharpes : dict[str, float], optional
            Out-of-sample Sharpe ratios used to seed initial model scores and weights,
            biasing the ensemble towards empirically superior policies from bar 1.
        """
        if not model_names:
            raise ValueError("model_names list cannot be empty")

        self.model_names = list(model_names)
        self.k = len(self.model_names)
        self.eta = float(learning_rate)
        self.gamma = float(discount_factor)
        self.epsilon = float(min_weight_floor)
        self.state_path = Path(state_path) if state_path else None
        self.initial_sharpes = dict(initial_sharpes) if initial_sharpes else None

        # Cumulative performance scores S_i and normalized mixture weights w_i
        if self.initial_sharpes:
            self.scores: dict[str, float] = {
                name: float(self.initial_sharpes.get(name, 1.0)) for name in self.model_names
            }
            score_arr = np.array([self.scores[name] for name in self.model_names], dtype=np.float64)
            scaled = self.eta * score_arr
            scaled -= np.max(scaled)
            exp_s = np.exp(scaled)
            base_p = exp_s / np.sum(exp_s)
            init_w = (1.0 - self.epsilon) * base_p + (self.epsilon / self.k)
            init_w = init_w / np.sum(init_w)
            self.weights: dict[str, float] = {
                name: float(init_w[i]) for i, name in enumerate(self.model_names)
            }
        else:
            self.scores: dict[str, float] = {name: 0.0 for name in self.model_names}
            self.weights: dict[str, float] = {name: 1.0 / self.k for name in self.model_names}

        # History tracking for diagnostics
        self.update_count = 0
        self.last_rewards: dict[str, float] = {name: 0.0 for name in self.model_names}

        # Attempt to restore state from disk if exists
        if self.state_path and self.state_path.is_file():
            self.load_state(self.state_path)

    def get_weights(self) -> dict[str, float]:
        """Return a copy of the current normalized model weights."""
        return dict(self.weights)

    def get_scores(self) -> dict[str, float]:
        """Return a copy of current cumulative discounted scores."""
        return dict(self.scores)

    def predict(self, model_predictions: dict[str, float]) -> tuple[float, dict[str, float]]:
        """
        Compute weighted consensus prediction across base models.

        Parameters
        ----------
        model_predictions : dict[str, float]
            Map of model_name -> scalar prediction (e.g. continuous score or signed direction).

        Returns
        -------
        tuple[float, dict[str, float]]
            (weighted_prediction, current_weights)
        """
        if not model_predictions:
            return 0.0, self.get_weights()

        total_weight = 0.0
        weighted_sum = 0.0

        for name, pred in model_predictions.items():
            if name in self.weights:
                w = self.weights[name]
                weighted_sum += w * float(pred)
                total_weight += w

        if total_weight > 1e-9:
            consensus = weighted_sum / total_weight
        else:
            consensus = float(np.mean(list(model_predictions.values())))

        return float(consensus), self.get_weights()

    def update(
        self,
        model_predictions: dict[str, float],
        realized_return: float,
        current_atr: float = 1.0,
    ) -> dict[str, float]:
        """
        Update model weights using the Hedge multiplicative exponential update rule.

        Parameters
        ----------
        model_predictions : dict[str, float]
            Predictions emitted by models on the prior bar that generated this return.
        realized_return : float
            Price return over the bar: (Close_t - Close_{t-1}) / Close_{t-1}.
        current_atr : float
            Current Average True Range / volatility measure used to normalize return.

        Returns
        -------
        dict[str, float]
            Updated model weights.
        """
        if not model_predictions:
            return self.get_weights()

        # Volatility normalization guard
        norm_vol = max(float(current_atr), 1e-6)

        # 1. Compute realized reward per model
        for name in self.model_names:
            if name in model_predictions:
                pred = float(model_predictions[name])
                # Reward: positive when model direction aligns with price movement
                # Scaled by signal magnitude and inverse volatility
                dir_sign = float(np.sign(pred)) if abs(pred) > 1e-4 else 0.0
                reward = dir_sign * (realized_return / norm_vol)
            else:
                reward = 0.0

            self.last_rewards[name] = float(reward)
            # 2. Discounted cumulative score update: S_{t+1} = gamma * S_t + r_t
            self.scores[name] = self.gamma * self.scores[name] + reward

        # 3. Softmax with numerical stability (subtract max score)
        score_arr = np.array([self.scores[name] for name in self.model_names], dtype=np.float64)
        scaled_scores = self.eta * score_arr
        scaled_scores -= np.max(scaled_scores)  # prevent exp overflow
        exp_scores = np.exp(scaled_scores)
        sum_exp = np.sum(exp_scores)

        if sum_exp > 1e-12:
            base_probs = exp_scores / sum_exp
        else:
            base_probs = np.ones(self.k) / self.k

        # 4. Mix with uniform exploration floor: w = (1 - eps) * base_probs + eps / K
        final_weights = (1.0 - self.epsilon) * base_probs + (self.epsilon / self.k)
        # Ensure exact sum to 1.0
        final_weights = final_weights / np.sum(final_weights)

        for i, name in enumerate(self.model_names):
            self.weights[name] = float(final_weights[i])

        self.update_count += 1

        # Periodically or on each update persist to disk
        if self.state_path is not None:
            try:
                self.save_state()
            except Exception as e:
                logger.warning(f"Failed to auto-save Hedge state: {e}")

        return self.get_weights()

    def save_state(self, path: Path | str | None = None) -> None:
        """Atomically persist state to JSON."""
        target_path = Path(path) if path else self.state_path
        if not target_path:
            return

        payload = {
            "model_names": self.model_names,
            "weights": self.weights,
            "scores": self.scores,
            "update_count": self.update_count,
            "eta": self.eta,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
        }

        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(target_path)

    def load_state(self, path: Path | str | None = None) -> bool:
        """Restore weights and scores from JSON file."""
        src_path = Path(path) if path else self.state_path
        if not src_path or not src_path.is_file():
            return False

        try:
            data = json.loads(src_path.read_text(encoding="utf-8"))
            loaded_weights = data.get("weights", {})
            loaded_scores = data.get("scores", {})

            # Match with currently registered model names
            for name in self.model_names:
                if name in loaded_weights:
                    self.weights[name] = float(loaded_weights[name])
                if name in loaded_scores:
                    self.scores[name] = float(loaded_scores[name])

            # Re-normalize in case model list was altered
            total_w = sum(self.weights.values())
            if total_w > 1e-9:
                for name in self.weights:
                    self.weights[name] /= total_w

            self.update_count = int(data.get("update_count", 0))
            return True
        except Exception as exc:
            logger.warning(f"Could not load Hedge state from {src_path}: {exc}")
            return False

    def reset(self) -> None:
        """Reset scores and weights to initial distribution (using initial_sharpes if provided)."""
        if self.initial_sharpes:
            self.scores = {name: float(self.initial_sharpes.get(name, 1.0)) for name in self.model_names}
            score_arr = np.array([self.scores[name] for name in self.model_names], dtype=np.float64)
            scaled = self.eta * score_arr
            scaled -= np.max(scaled)
            exp_s = np.exp(scaled)
            base_p = exp_s / np.sum(exp_s)
            init_w = (1.0 - self.epsilon) * base_p + (self.epsilon / self.k)
            init_w = init_w / np.sum(init_w)
            self.weights = {name: float(init_w[i]) for i, name in enumerate(self.model_names)}
        else:
            self.scores = {name: 0.0 for name in self.model_names}
            self.weights = {name: 1.0 / self.k for name in self.model_names}
        self.update_count = 0
        if self.state_path:
            self.save_state()
