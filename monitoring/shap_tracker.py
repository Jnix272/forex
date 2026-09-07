"""SHAP feature importance tracker — logs to W&B and alerts on drift."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from config.settings import PATHS

try:
    import torch.nn as nn
    TORCH = True
except ImportError:
    TORCH = False

try:
    import shap  # noqa: F401
    SHAP = True
except ImportError:
    SHAP = False

__all__ = ["SHAPFeatureTracker"]


class SHAPFeatureTracker:
    """
    Computes SHAP values after each retraining cycle and:
      - Logs feature importance to Weights & Biases
      - Alerts if top-N features shift significantly (early drift signal)
      - Saves CSV summaries under log_dir
    """

    def __init__(
        self,
        feature_names: list[str],
        n_background: int = 500,
        n_explain: int = 200,
        log_dir: str | None = None,
        alert_threshold: float = 0.3,
    ):
        if log_dir is None:
            log_dir = PATHS["logs_shap"]
        self.feature_names = feature_names
        self.n_bg = n_background
        self.n_explain = n_explain
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.alert_thresh = alert_threshold
        self._prev_top5: list[str] | None = None

    def compute(
        self,
        model: "nn.Module",
        X: np.ndarray,
        device: str = "cpu",
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """Compute SHAP values. Returns (shap_values, importance_df)."""
        if not SHAP:
            print("[SHAP] shap not installed - pip install shap")
            return np.array([]), pd.DataFrame()
        if not TORCH:
            print("[SHAP] torch required for DeepExplainer")
            return np.array([]), pd.DataFrame()

        import shap as shap_lib
        import torch

        model.eval()
        idx_bg = np.random.choice(len(X), min(self.n_bg, len(X)), replace=False)
        idx_exp = np.random.choice(len(X), min(self.n_explain, len(X)), replace=False)
        X_bg = torch.tensor(X[idx_bg], dtype=torch.float32).to(device)
        X_explain = torch.tensor(X[idx_exp], dtype=torch.float32).to(device)

        try:
            explainer = shap_lib.DeepExplainer(model, X_bg)
            shap_vals = explainer.shap_values(X_explain)
            shap_abs = np.abs(shap_vals).mean(axis=(0, 1))
        except Exception as e:
            print(f"[SHAP] DeepExplainer failed: {e}. Trying KernelExplainer...")
            X_bg_flat = X_bg[:, -1, :].cpu().numpy()
            X_exp_flat = X_explain[:, -1, :].cpu().numpy()

            def model_fn(x):
                t = torch.tensor(x, dtype=torch.float32).unsqueeze(1).expand(-1, X.shape[1], -1)
                with torch.no_grad():
                    return model(t.to(device)).cpu().numpy()

            explainer = shap_lib.KernelExplainer(model_fn, X_bg_flat[:50])
            sv = explainer.shap_values(X_exp_flat[:50])
            shap_abs = np.abs(sv).mean(axis=0)

        imp_df = (
            pd.DataFrame({"feature": self.feature_names[: len(shap_abs)], "mean_shap": shap_abs})
            .sort_values("mean_shap", ascending=False)
            .reset_index(drop=True)
        )

        top5 = imp_df["feature"].head(5).tolist()
        if self._prev_top5 is not None:
            overlap = len(set(top5) & set(self._prev_top5)) / 5
            if overlap < (1 - self.alert_thresh):
                print(f"[SHAP] ⚠ Feature importance shift! Top-5 changed {(1 - overlap) * 100:.0f}%")
        self._prev_top5 = top5

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        imp_df.to_csv(self.log_dir / f"shap_importance_{timestamp}.csv", index=False)
        print(f"[SHAP] Top-5 features: {top5}")
        return shap_abs, imp_df

    def log_to_wandb(self, importance_df: pd.DataFrame, run: Any = None):
        try:
            import wandb

            wb_run: Any = run if run is not None else cast(Any, wandb.run)
            if wb_run is None:
                return
            top20 = importance_df.head(20)
            table = wandb.Table(
                data=list(zip(top20["feature"], top20["mean_shap"], strict=False)),
                columns=["feature", "mean_shap"],
            )
            wb_run.log({"feature_importance": wandb.plot.bar(table, "feature", "mean_shap", title="SHAP Feature Importance")})
        except Exception as e:
            print(f"[SHAP] W&B log failed: {e}")
