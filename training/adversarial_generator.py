"""
Adversarial Market Generation
Injects synthetic flash crashes, spread blowouts, and sentiment shocks
during training to bulletproof the model against broker manipulation
and extreme market regimes.
"""
from __future__ import annotations

import random

import torch
import torch.nn as nn


class AdversarialGenerator(nn.Module):
    def __init__(
        self,
        whipsaw_prob: float = 0.05,
        spread_blowout_prob: float = 0.05,
        sentiment_shock_prob: float = 0.05,
        whipsaw_magnitude: float = 5.0,
        spread_multiplier: float = 10.0,
        probability: float | None = None,
        feature_names: list[str] | None = None,
    ):
        super().__init__()
        # ``probability`` is a convenience override applied to all event types.
        if probability is not None:
            p = float(probability)
            whipsaw_prob = p
            spread_blowout_prob = p
            sentiment_shock_prob = p
        self.whipsaw_prob = whipsaw_prob
        self.spread_blowout_prob = spread_blowout_prob
        self.sentiment_shock_prob = sentiment_shock_prob
        self.whipsaw_magnitude = whipsaw_magnitude
        self.spread_multiplier = spread_multiplier
        self.feature_names = list(feature_names) if feature_names else []

    def set_feature_names(self, feature_names: list[str] | None) -> None:
        self.feature_names = list(feature_names) if feature_names else []

    def forward(
        self,
        features: torch.Tensor,
        feature_names: list[str] | None = None,
    ) -> torch.Tensor:
        """
        Inject adversarial noise into the batch.
        features: (Batch, Seq, Features)
        """
        if not self.training:
            return features

        names = feature_names if feature_names is not None else self.feature_names
        if not names:
            # Without a schema we cannot target columns safely — no-op with warning once.
            if not getattr(self, "_warned_no_schema", False):
                print("[Adversarial] No feature_names provided; skipping injections")
                self._warned_no_schema = True
            return features

        adv_features = features.clone()
        batch_size, seq_len, _num_features = adv_features.shape

        close_idx = names.index("close") if "close" in names else -1
        spread_idx = names.index("spread_mean") if "spread_mean" in names else -1

        if close_idx >= 0 and random.random() < self.whipsaw_prob and seq_len >= 2:
            shock_seq_idx = random.randint(max(0, seq_len - 10), seq_len - 1)
            n_hit = max(1, batch_size // 4)
            batch_indices = torch.randint(0, batch_size, (n_hit,), device=features.device)
            adv_features[batch_indices, shock_seq_idx, close_idx] -= self.whipsaw_magnitude
            if shock_seq_idx + 1 < seq_len:
                adv_features[batch_indices, shock_seq_idx + 1, close_idx] += self.whipsaw_magnitude

        if spread_idx >= 0 and random.random() < self.spread_blowout_prob:
            n_hit = max(1, batch_size // 2)
            batch_indices = torch.randint(0, batch_size, (n_hit,), device=features.device)
            adv_features[batch_indices, -5:, spread_idx] *= self.spread_multiplier

        if random.random() < self.sentiment_shock_prob:
            fb_indices = [i for i, name in enumerate(names) if name.startswith("fb_")]
            if fb_indices:
                n_hit = max(1, batch_size // 4)
                batch_indices = torch.randint(0, batch_size, (n_hit,), device=features.device)
                for idx in fb_indices:
                    adv_features[batch_indices, -1, idx] *= -1.0

        return adv_features
