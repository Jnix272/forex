"""
Adversarial Market Generation
Injects synthetic flash crashes, spread blowouts, and sentiment shocks
during training to bulletproof the model against broker manipulation
and extreme market regimes.
"""
import torch
import torch.nn as nn
import random

class AdversarialGenerator(nn.Module):
    def __init__(self, 
                 whipsaw_prob: float = 0.05,
                 spread_blowout_prob: float = 0.05,
                 sentiment_shock_prob: float = 0.05,
                 whipsaw_magnitude: float = 5.0,
                 spread_multiplier: float = 10.0):
        super().__init__()
        self.whipsaw_prob = whipsaw_prob
        self.spread_blowout_prob = spread_blowout_prob
        self.sentiment_shock_prob = sentiment_shock_prob
        self.whipsaw_magnitude = whipsaw_magnitude
        self.spread_multiplier = spread_multiplier
        
    def forward(self, features: torch.Tensor, feature_names: list) -> torch.Tensor:
        """
        Inject adversarial noise into the batch.
        features: (Batch, Seq, Features)
        """
        if not self.training:
            return features
            
        adv_features = features.clone()
        batch_size, seq_len, num_features = adv_features.shape
        
        # Determine column indices
        close_idx = feature_names.index('close') if 'close' in feature_names else -1
        spread_idx = feature_names.index('spread_mean') if 'spread_mean' in feature_names else -1
        
        # 1. Whipsaw Injector (Flash Crash)
        if close_idx >= 0 and random.random() < self.whipsaw_prob:
            # Inject a sudden V-shape reversal at the end of the sequence
            shock_seq_idx = random.randint(seq_len - 10, seq_len - 1)
            batch_indices = torch.randint(0, batch_size, (batch_size // 4,))
            # Create a massive downward spike, then immediate recovery
            adv_features[batch_indices, shock_seq_idx, close_idx] -= self.whipsaw_magnitude
            if shock_seq_idx + 1 < seq_len:
                adv_features[batch_indices, shock_seq_idx+1, close_idx] += self.whipsaw_magnitude
                
        # 2. Spread Blowout
        if spread_idx >= 0 and random.random() < self.spread_blowout_prob:
            # Simulate an illiquid period where spread jumps 10x
            batch_indices = torch.randint(0, batch_size, (batch_size // 2,))
            adv_features[batch_indices, -5:, spread_idx] *= self.spread_multiplier
            
        # 3. Sentiment Shock
        if random.random() < self.sentiment_shock_prob:
            # Find FinBERT PCA columns and invert them
            fb_indices = [i for i, name in enumerate(feature_names) if name.startswith('fb_')]
            if fb_indices:
                batch_indices = torch.randint(0, batch_size, (batch_size // 4,))
                for idx in fb_indices:
                    # Invert sentiment to simulate fake news or contradictory headlines
                    adv_features[batch_indices, -1, idx] *= -1.0
                    
        return adv_features
