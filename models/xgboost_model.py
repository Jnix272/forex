"""
models/xgboost_model.py
========================
Wrapper for XGBoost to make it compatible with the PyTorch-based
(Batch, Time, Features) sequence data loader and inference pipeline.

Temporal Feature Extraction
-----------------------------
Instead of naively using only the last bar (x[:, -1, :]) or doing a
memory-exploding full flatten, the model computes summary statistics over
the time dimension for each feature:
  mean, std, min, max, last, range  -> 6 x F input features

This gives XGBoost temporal awareness while keeping the input size manageable.
"""
import warnings
from typing import Union

import numpy as np
import xgboost as xgb

try:
    import torch
    import torch.nn as nn
    TORCH = True
except ImportError:
    TORCH = False
    nn = object


class XGBoostForecaster(nn.Module if TORCH else object):
    """
    Wraps an XGBoost model so it can be evaluated in PyTorch inference pipelines.

    Input handling modes (set via ``sequence_mode``):
      - "last_bar" : take only x[:, -1, :] -- fastest but no temporal context.
      - "temporal" : compute [mean, std, min, max, last, range] per feature -- default.
      - "flatten"  : full TxF flatten -- keeps all info but huge feature space.
    """

    def __init__(
        self,
        num_classes: int = 1,
        flatten_sequence: bool = False,
        seq_len: int = 60,
        sequence_mode: str = "temporal",
        **xgb_kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.seq_len = seq_len
        # Back-compat: flatten_sequence=True maps to flatten mode
        if flatten_sequence:
            sequence_mode = "flatten"
        self.sequence_mode = sequence_mode

        if self.num_classes == 1:
            self.model = xgb.XGBRegressor(**xgb_kwargs)
            self._is_classifier = False
        else:
            self.model = xgb.XGBClassifier(
                num_class=self.num_classes if self.num_classes > 2 else None,
                **xgb_kwargs,
            )
            self._is_classifier = True

    # -- feature extraction ---------------------------------------------------

    @staticmethod
    def _extract_temporal_features(x: np.ndarray) -> np.ndarray:
        """
        Convert (B, T, F) array -> (B, 6*F) array of per-feature summary stats.

        Statistics per feature across the T time steps:
          mean, std, min, max, last-bar value, range (max-min)
        """
        mean = x.mean(axis=1)       # (B, F)
        std  = x.std(axis=1)        # (B, F)
        xmin = x.min(axis=1)        # (B, F)
        xmax = x.max(axis=1)        # (B, F)
        last = x[:, -1, :]          # (B, F)
        rng  = xmax - xmin          # (B, F)
        return np.concatenate([mean, std, xmin, xmax, last, rng], axis=1)

    def _prepare_inputs(self, x: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
        """Convert any input to a 2-D numpy array suitable for XGBoost."""
        if TORCH and isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        if x.ndim == 3:
            if self.sequence_mode == "flatten":
                B, T, F = x.shape
                return x.reshape(B, T * F)
            elif self.sequence_mode == "temporal":
                return self._extract_temporal_features(x)
            else:  # "last_bar" or unknown
                return x[:, -1, :]
        elif x.ndim == 2:
            return x
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.ndim}D")

    # -- forward / fit / save -------------------------------------------------

    def forward(self, x: Union[np.ndarray, "torch.Tensor"]) -> Union[np.ndarray, "torch.Tensor"]:
        """Forward pass matching PyTorch signature."""
        x_np = self._prepare_inputs(x)

        try:
            if self._is_classifier:
                preds = self.model.predict_proba(x_np)
            else:
                preds = self.model.predict(x_np)
        except xgb.core.XGBoostError as e:
            warnings.warn(
                f"XGBoost predict failed (model might not be fitted). "
                f"Returning zeros. Error: {e}"
            )
            if self._is_classifier:
                preds = np.zeros((x_np.shape[0], self.num_classes))
                preds[:, 1] = 1.0  # Default to "Flat/Hold"
            else:
                preds = np.zeros(x_np.shape[0])

        if TORCH and isinstance(x, torch.Tensor):
            res = torch.tensor(preds, dtype=torch.float32, device=x.device)
            if self.num_classes == 1:
                return res.squeeze(-1) if res.ndim > 1 else res
            return res
        return preds

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Scikit-learn compatible fit."""
        X_np = self._prepare_inputs(X)
        self.model.fit(X_np, y, **kwargs)
        return self

    def save_model(self, filepath: str):
        self.model.save_model(filepath)

    def load_model(self, filepath: str):
        self.model.load_model(filepath)
