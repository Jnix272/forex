"""
pretrain/hard_example_mining.py
===============================
Extends hard-example mining into the unsupervised phase by targeting
confident-wrong windows, missed opportunities, and regime transitions.
"""

import json
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np


class PretrainHardExampleMiner:
    def __init__(self, max_hard_ratio: float = 0.25):
        """
        max_hard_ratio ensures that hard examples never dominate the batch,
        preventing the pretrainer from overfitting to rare noisy data.
        """
        self.max_hard_ratio = max_hard_ratio
        self.logger = logging.getLogger(__name__)
        self.hard_examples = []

    def ingest_validation_failures(self, confident_wrong: list, missed_opportunities: list, regime_transitions: list):
        """
        Ingests indices of sequences that the supervised model struggled with during validation,
        so the next pretraining cycle can focus on building better representations for them.
        """
        self.hard_examples.extend(confident_wrong)
        self.hard_examples.extend(missed_opportunities)
        self.hard_examples.extend(regime_transitions)

        # Remove duplicates
        self.hard_examples = list(set(self.hard_examples))
        self.logger.info(f"Ingested {len(self.hard_examples)} unique hard examples for future pretraining.")

    def sample_batch_indices(self, total_batch_size: int, all_indices: list) -> list:
        """
        Constructs a batch that mixes normal uniform sampling with a capped ratio of hard examples.
        """
        if not self.hard_examples:
            return random.sample(all_indices, total_batch_size)

        num_hard = min(int(total_batch_size * self.max_hard_ratio), len(self.hard_examples))
        num_normal = total_batch_size - num_hard

        batch = random.sample(self.hard_examples, num_hard)
        batch.extend(random.sample(all_indices, num_normal))

        random.shuffle(batch)
        return batch

    def compute_feature_vulnerability(
        self,
        X: np.ndarray,
        hard_indices: list,
        method: str = "gradient_norm",
    ) -> np.ndarray:
        """
        Compute per-dimension vulnerability scores from hard examples.
        
        Args:
            X: Full training data array of shape (n_samples, seq_len, n_features)
            hard_indices: Indices of hard examples
            method: "gradient_norm" | "variance" | "range" - how to compute vulnerability
            
        Returns:
            Per-feature vulnerability scores (n_features,) - higher = more vulnerable
        """
        if not hard_indices or len(hard_indices) == 0:
            return np.ones(X.shape[-1], dtype=np.float32)
        
        # Extract hard example windows
        hard_data = X[hard_indices]  # (n_hard, seq_len, n_features)
        
        if method == "gradient_norm":
            # Compute gradient of loss w.r.t input (approximate via temporal gradient)
            # For each feature, compute temporal gradient magnitude across hard examples
            grad_magnitudes = []
            for t in range(1, hard_data.shape[1]):
                diff = hard_data[:, t, :] - hard_data[:, t-1, :]  # (n_hard, n_features)
                grad_magnitudes.append(np.abs(diff).mean(axis=0))
            vulnerability = np.mean(grad_magnitudes, axis=0) if grad_magnitudes else np.zeros(X.shape[-1])
            
        elif method == "variance":
            # High variance in hard examples = model uncertain about this feature
            vulnerability = hard_data.var(axis=(0, 1))  # (n_features,)
            
        elif method == "range":
            # Large range in hard examples = feature varies widely on difficult samples
            vulnerability = hard_data.max(axis=(0, 1)) - hard_data.min(axis=(0, 1))
            
        else:
            raise ValueError(f"Unknown vulnerability method: {method}")
        
        # Normalize to [0.1, 2.0] range for epsilon multipliers
        # 0.1 = less attack, 2.0 = double attack on this dimension
        vuln_min, vuln_max = vulnerability.min(), vulnerability.max()
        if vuln_max > vuln_min:
            normalized = 0.1 + 1.9 * (vulnerability - vuln_min) / (vuln_max - vuln_min)
        else:
            normalized = np.ones_like(vulnerability)
        
        return normalized.astype(np.float32)

    def save_vulnerability_scores(self, vulnerability: np.ndarray, path: str = "logs/hard_feature_dims.json"):
        """Save per-feature vulnerability scores for adversarial training."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vulnerability_scores": vulnerability.tolist(),
            "n_features": len(vulnerability),
            "method": "gradient_norm",
        }
        with open(path, "w") as f:
            json.dump(data, f)
        self.logger.info(f"Saved feature vulnerability scores to {path}")

    @staticmethod
    def load_vulnerability_scores(path: str = "logs/hard_feature_dims.json") -> Optional[np.ndarray]:
        """Load per-feature vulnerability scores for adversarial training."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return np.array(data.get("vulnerability_scores", []), dtype=np.float32)
        except Exception:
            return None
