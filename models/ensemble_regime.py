"""
Regime-weighted ensemble meta-learner.
Routes to best base model per detected market regime.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RegimeRouter(nn.Module):
    """Lightweight regime classifier → model weights."""
    def __init__(self, n_models: int, n_regimes: int = 6, hidden: int = 32):
        super().__init__()
        self.n_models = n_models
        self.n_regimes = n_regimes
        # Learnable model weights per regime
        self.regime_weights = nn.Parameter(torch.ones(n_regimes, n_models) / n_models)

    def forward(self, regime_id: torch.Tensor) -> torch.Tensor:
        """
        regime_id: (B,) long tensor with values 0..n_regimes-1
        returns: (B, n_models) softmax weights
        """
        weights = F.softmax(self.regime_weights[regime_id], dim=-1)
        return weights


class RegimeEnsembleMetaLearner(nn.Module):
    """
    Ensemble that weights base models by detected regime.
    
    Regime encoding (6 regimes = 3 vol × 2 trend):
    0: low_vol + trending
    1: low_vol + ranging  
    2: normal_vol + trending
    3: normal_vol + ranging
    4: high_vol + trending
    5: high_vol + ranging
    """
    def __init__(
        self,
        base_models: list[nn.Module],
        regime_features: list[str],  # e.g., ["realized_vol_regime", "trend_quality"]
        n_regimes: int = 6,
    ):
        super().__init__()
        self.base_models = nn.ModuleList(base_models)
        self.regime_features = regime_features
        self.n_models = len(base_models)
        self.n_regimes = n_regimes

        # Regime classifier from features
        self.regime_classifier = nn.Sequential(
            nn.Linear(len(regime_features), 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_regimes),
        )
        self.router = RegimeRouter(self.n_models, n_regimes)

        # Freeze base models
        for model in self.base_models:
            for p in model.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor, regime_feats: torch.Tensor) -> tuple:
        """
        x: (B, T, F) - input features
        regime_feats: (B, len(regime_features)) - regime indicators at last timestep
        Returns: (ensemble_pred, model_weights, regime_probs)
        """
        x.shape[0]

        # Get base model predictions (no grad)
        with torch.no_grad():
            base_preds = []
            for model in self.base_models:
                pred = model(x)  # (B, 1) or (B, 3) for direction
                if pred.dim() == 2 and pred.shape[1] == 3:
                    pred = pred[:, 2] - pred[:, 0]  # buy - sell logit
                elif pred.dim() == 2 and pred.shape[1] == 1:
                    pred = pred.squeeze(-1)
                base_preds.append(pred)
            base_preds = torch.stack(base_preds, dim=1)  # (B, n_models)

        # Classify regime
        regime_logits = self.regime_classifier(regime_feats)  # (B, n_regimes)
        regime_probs = F.softmax(regime_logits, dim=-1)

        # Get model weights per regime
        model_weights_per_regime = F.softmax(self.router.regime_weights, dim=-1)  # (n_regimes, n_models)

        # Expected weights = sum_p(regime) * weights(regime)
        weights = regime_probs @ model_weights_per_regime  # (B, n_models)

        # Weighted ensemble prediction
        ensemble_pred = (base_preds * weights).sum(dim=1)  # (B,)

        return ensemble_pred, weights, regime_probs

    def get_regime_weights(self) -> torch.Tensor:
        """Return learned model weights per regime for inspection."""
        return F.softmax(self.router.regime_weights, dim=-1).detach().cpu().numpy()


def train_regime_ensemble(
    base_models: list[nn.Module],
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    regime_features: list[str],
    epochs: int = 10,
    device: str = "cuda",
    lr: float = 1e-3,
) -> RegimeEnsembleMetaLearner:
    """Train regime-weighted ensemble."""
    model = RegimeEnsembleMetaLearner(base_models, regime_features).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for xb, yb, *extra in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            # Extract regime features from last timestep
            regime_feats = xb[:, -1, -len(regime_features):].to(device)

            pred, weights, regime_probs = model(xb, regime_feats)

            # MSE loss on continuous reward
            loss = F.mse_loss(pred, yb)

            # Diversity regularization: encourage different weights per regime
            regime_weights = model.get_regime_weights()
            weight_entropy = -np.sum(regime_weights * np.log(regime_weights + 1e-8), axis=1).mean()
            loss = loss - 0.01 * weight_entropy

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb, *extra in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                regime_feats = xb[:, -1, -len(regime_features):].to(device)
                pred, _, _ = model(xb, regime_feats)
                val_losses.append(F.mse_loss(pred, yb).item())

        print(f"RegimeEnsemble Epoch {epoch}: train_loss={np.mean(train_losses):.4f} val_loss={np.mean(val_losses):.4f}")

    return model


def export_regime_ensemble_onnx(
    model: RegimeEnsembleMetaLearner,
    output_path: str,
    seq_len: int = 80,
    n_features: int = 227,
    n_regime_features: int = 2,
    opset_version: int = 17,
) -> None:
    """
    Export regime ensemble to ONNX.
    
    Note: ONNX export requires fixed input shapes. The base models are embedded
    in the forward pass, so we export the routing logic separately.
    """
    import torch

    model.eval()

    # Create dummy inputs
    torch.randn(1, seq_len, n_features)
    dummy_regime = torch.randn(1, n_regime_features)

    # Export just the regime classifier + router (base models need separate export)
    class EnsembleRouter(nn.Module):
        def __init__(self, regime_classifier, regime_weights):
            super().__init__()
            self.regime_classifier = regime_classifier
            self.regime_weights = regime_weights

        def forward(self, regime_feats):
            regime_logits = self.regime_classifier(regime_feats)
            regime_probs = torch.softmax(regime_logits, dim=-1)
            model_weights = torch.softmax(self.regime_weights, dim=-1)
            weights = regime_probs @ model_weights
            return weights, regime_probs

    router = EnsembleRouter(model.regime_classifier, model.router.regime_weights)
    router.eval()

    torch.onnx.export(
        router,
        dummy_regime,
        output_path.replace('.onnx', '_router.onnx'),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['regime_features'],
        output_names=['model_weights', 'regime_probs'],
        dynamic_axes={
            'regime_features': {0: 'batch'},
            'model_weights': {0: 'batch'},
            'regime_probs': {0: 'batch'},
        }
    )
    print(f"Exported regime router to {output_path.replace('.onnx', '_router.onnx')}")
