"""
pretrain/handoff_logic.py
=========================
Evaluates if a pretrained embedding space is ready for supervised learning.
Automatically discards bad pretrained weights to prevent model collapse.
"""

import logging

class PretrainHandoffGate:
    def __init__(self, std_threshold: float = 0.05, max_uniformity: float = 0.8):
        self.std_threshold = std_threshold
        self.max_uniformity = max_uniformity
        self.logger = logging.getLogger(__name__)

    def evaluate_representation_quality(self, embedding_std: float, uniformity: float = None) -> bool:
        """
        Determines if the pretraining loop actually learned a diverse representation.
        A low embedding std means the model collapsed to predicting the same constant vector.
        """
        if embedding_std < self.std_threshold:
            self.logger.error(f"Pretraining Failed: Representation collapsed! (std={embedding_std:.4f} < {self.std_threshold})")
            return False
            
        if uniformity is not None and uniformity > self.max_uniformity:
            self.logger.error(f"Pretraining Failed: Contrastive uniformity too high! (uniformity={uniformity:.4f} > {self.max_uniformity})")
            return False
            
        self.logger.info("Pretrain representation quality is acceptable. Proceeding to handoff.")
        return True
        
    def evaluate_warm_start_shock(self, baseline_val_loss: float, epoch1_val_loss: float) -> bool:
        """
        If the first supervised epoch explodes the validation loss compared to a no-pretrain baseline,
        the pretrained weights are actively fighting the downstream task and must be discarded.
        """
        if epoch1_val_loss > baseline_val_loss * 1.5:
            self.logger.error(f"Warm-Start Shock Detected! Epoch 1 loss ({epoch1_val_loss:.4f}) is vastly worse than baseline ({baseline_val_loss:.4f}).")
            self.logger.error("The pretrained weights are destructive. Discarding pretrain.")
            return False
            
        self.logger.info("Warm-start shock evaluation passed. Pretrained weights are stable.")
        return True
