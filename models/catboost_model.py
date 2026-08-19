"""
models/catboost_model.py
========================
Wrapper for CatBoost to make it compatible with the PyTorch-based
(Batch, Time, Features) sequence data loader and inference pipeline.

Temporal Feature Extraction
-----------------------------
Instead of naively using only the last bar (x[:, -1, :]) or doing a
memory-exploding full flatten, the model computes summary statistics over
the time dimension for each feature:
  mean, std, min, max, last, range  -> 6 x F input features

This gives CatBoost temporal awareness while keeping the input size manageable,
while maintaining CatBoost's robust protection against target leakage.
"""

import warnings
from typing import Union

import catboost as cb
import numpy as np

try:
    import torch
    import torch.nn as nn

    TORCH = True
except ImportError:
    TORCH = False
    nn = object


class CatBoostForecaster(nn.Module if TORCH else object):
    """
    Wraps a CatBoost model so it can be evaluated in PyTorch inference pipelines.

    Input handling modes (set via ``sequence_mode``):
      - "last_bar" : take only x[:, -1, :] -- fastest but no temporal context.
      - "temporal" : compute [mean, std, min, max, last, range, skew, kurtosis,
                     trend, early_mean, mid_mean, late_mean, vol] per feature.
      - "flatten"  : full TxF flatten -- keeps all info but huge feature space.
    """

    def __init__(
        self,
        num_classes: int = 1,
        flatten_sequence: bool = False,
        seq_len: int = 60,
        sequence_mode: str = "temporal",
        **cb_kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.seq_len = seq_len
        # Back-compat: flatten_sequence=True maps to flatten mode
        if flatten_sequence:
            sequence_mode = "flatten"
        self.sequence_mode = sequence_mode

        if self.num_classes == 1:
            self.model = cb.CatBoostRegressor(verbose=False, **cb_kwargs)
            self._is_classifier = False
        else:
            self.model = cb.CatBoostClassifier(
                classes_count=self.num_classes if self.num_classes > 2 else None,
                verbose=False,
                **cb_kwargs,
            )
            self._is_classifier = True

    # -- feature extraction ---------------------------------------------------

    @staticmethod
    def _extract_temporal_features(x: np.ndarray) -> np.ndarray:
        """
        Convert (B, T, F) array -> (B, 13*F) array of per-feature summary stats.

        Statistics per feature across the T time steps:
          Basic:    mean, std, min, max, last, range (max-min)
          Shape:    skew, kurtosis
          Trend:    linear slope, acceleration (change in slope)
          Window:   early_mean (first 25%), mid_mean (middle 50%), late_mean (last 25%)
          Volatility: vol (std of first differences)
        """
        B, T, F = x.shape

        # Basic statistics
        mean = x.mean(axis=1)  # (B, F)
        std = x.std(axis=1)  # (B, F)
        xmin = x.min(axis=1)  # (B, F)
        xmax = x.max(axis=1)  # (B, F)
        last = x[:, -1, :]  # (B, F)
        rng = xmax - xmin  # (B, F)

        # Shape statistics (skewness, kurtosis)
        from scipy import stats as scipy_stats

        skew = scipy_stats.skew(x, axis=1, nan_policy="omit").astype(np.float32)  # (B, F)
        kurt = scipy_stats.kurtosis(x, axis=1, nan_policy="omit").astype(np.float32)  # (B, F)

        # Trend: linear slope via least-squares
        t_idx = np.arange(T, dtype=np.float32)
        t_mean = t_idx.mean()
        t_var = ((t_idx - t_mean) ** 2).sum()
        if t_var > 0:
            x_mean = x.mean(axis=1, keepdims=True)  # (B, 1, F)
            t_centered = (t_idx - t_mean).reshape(1, -1, 1)  # (1, T, 1)
            slope = ((x - x_mean) * t_centered).sum(axis=1) / t_var  # (B, F)
            # Acceleration: slope of the slope (second derivative proxy)
            half = T // 2
            x_first_mean = x[:, :half, :].mean(axis=1, keepdims=True)  # (B, 1, F)
            x_second_mean = x[:, half:, :].mean(axis=1, keepdims=True)  # (B, 1, F)
            t_first_centered = (t_idx[:half] - t_idx[:half].mean()).reshape(1, -1, 1)  # (1, half, 1)
            t_second_centered = (t_idx[half:] - t_idx[half:].mean()).reshape(1, -1, 1)  # (1, T-half, 1)
            slope_first = ((x[:, :half, :] - x_first_mean) * t_first_centered).sum(axis=1) / max(1, ((t_idx[:half] - t_idx[:half].mean()) ** 2).sum())  # (B, F)
            slope_second = ((x[:, half:, :] - x_second_mean) * t_second_centered).sum(axis=1) / max(1, ((t_idx[half:] - t_idx[half:].mean()) ** 2).sum())  # (B, F)
            accel = slope_second - slope_first  # (B, F)
        else:
            slope = np.zeros((B, F), dtype=np.float32)
            accel = np.zeros((B, F), dtype=np.float32)

        # Multi-scale window means
        q1 = max(1, T // 4)
        q3 = T - max(1, T // 4)
        early_mean = x[:, :q1, :].mean(axis=1)  # (B, F) first 25%
        mid_mean = x[:, q1:q3, :].mean(axis=1)  # (B, F) middle 50%
        late_mean = x[:, q3:, :].mean(axis=1)  # (B, F) last 25%

        # Volatility: std of first differences
        diffs = np.diff(x, axis=1)  # (B, T-1, F)
        vol = diffs.std(axis=1)  # (B, F)

        return np.concatenate([
            mean, std, xmin, xmax, last, rng,  # 6F
            skew, kurt,  # 2F
            slope, accel,  # 2F
            early_mean, mid_mean, late_mean,  # 3F
            vol,  # 1F
        ], axis=1)  # Total: 14F

    def _prepare_inputs(self, x: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
        """Convert any input to a 2-D numpy array suitable for CatBoost."""
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
        except cb.CatBoostError as e:
            warnings.warn(f"CatBoost predict failed (model might not be fitted). Returning zeros. Error: {e}", stacklevel=2)
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
