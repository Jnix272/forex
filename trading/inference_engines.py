import abc

import numpy as np


class BaseInferenceEngine(abc.ABC):
    """
    Abstract base class for all inference engines used in the live trading pipeline.
    Enforces a consistent interface across PyTorch, ONNX/DirectML, and RL agents.
    """

    @abc.abstractmethod
    def select_action(self, obs: np.ndarray) -> int:
        """
        Takes a single bar's feature vector (1D array), updates internal rolling
        buffers or state, and returns an action.

        Returns:
            int: 0=Buy, 1=Hold, 2=Sell
        """
        pass

    @abc.abstractmethod
    def reset_buffer(self) -> None:
        """
        Clears any internal rolling observation buffers or recurrent state.
        Called on session restarts or large time gaps.
        """
        pass


class StationaryEnsembleInferenceEngine(BaseInferenceEngine):
    """
    Production inference engine for the 4-Hour Stationary Feature Ensemble.
    Uses trained LightGBM boosters on curated stationary features with conviction hurdles.
    """

    def __init__(
        self,
        pair: str = "EURUSD",
        model_dir: str = "checkpoints/stationary_ensemble",
        override_threshold: float | None = None,
    ):
        import json
        from pathlib import Path
        import lightgbm as lgb

        self.pair = pair.upper()
        self.model_dir = Path(model_dir)
        meta_path = self.model_dir / "metadata.json"

        if not meta_path.exists():
            raise FileNotFoundError(f"Stationary ensemble metadata not found at {meta_path}")

        with open(meta_path) as f:
            self.metadata = json.load(f)

        self.curated_indices = self.metadata["curated_feature_indices"]
        default_thresh = self.metadata["thresholds_bps"].get(self.pair, 5.0)
        self.threshold_bps = float(override_threshold if override_threshold is not None else default_thresh)

        model_path = self.model_dir / f"{self.pair.lower()}_4h_lgb.txt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file for {self.pair} not found at {model_path}")

        self.booster = lgb.Booster(model_file=str(model_path))

    def predict_return(self, obs: np.ndarray) -> float:
        """Predict expected 4-hour forward return in basis points."""
        obs_1d = np.asarray(obs).ravel()
        if len(obs_1d) == len(self.curated_indices):
            feats = obs_1d.reshape(1, -1)
        elif len(obs_1d) >= max(self.curated_indices) + 1:
            feats = obs_1d[self.curated_indices].reshape(1, -1)
        else:
            raise ValueError(
                f"Observation length {len(obs_1d)} insufficient for feature indices (max={max(self.curated_indices)})"
            )
        pred = float(self.booster.predict(feats)[0])
        return pred

    def select_action(self, obs: np.ndarray) -> int:
        """
        Select trade action using conviction hurdle.
        Returns:
            int: 0=Buy, 1=Hold, 2=Sell
        """
        pred_bps = self.predict_return(obs)
        if pred_bps > self.threshold_bps:
            return 0  # Buy
        elif pred_bps < -self.threshold_bps:
            return 2  # Sell
        return 1  # Hold

    def reset_buffer(self) -> None:
        """Stateless tree ensemble has no internal temporal buffer to clear."""
        pass

