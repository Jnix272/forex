"""
pretrain/hard_example_mining.py
===============================
Extends hard-example mining into the unsupervised phase by targeting
confident-wrong windows, missed opportunities, and regime transitions.
"""

import logging
import random

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
