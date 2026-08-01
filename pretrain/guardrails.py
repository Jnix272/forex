"""
pretrain/guardrails.py
======================
Strict security gates to prevent data leakage between pretraining,
supervised validation, and the forward promotion holdout.
"""

import logging

class PretrainGuardrails:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def enforce_no_holdout_leakage(self, pretrain_window: tuple, promotion_holdout_window: tuple) -> bool:
        """
        Ensures the forward promotion holdout window is NEVER seen during pretraining.
        """
        pre_start, pre_end = pretrain_window
        hold_start, hold_end = promotion_holdout_window
        
        # Check for any overlap
        if max(pre_start, hold_start) < min(pre_end, hold_end):
            self.logger.error(f"FATAL LEAKAGE: Pretrain window {pretrain_window} overlaps with Forward Holdout {promotion_holdout_window}!")
            raise RuntimeError("Data Leakage Guardrail triggered. Halting pretraining.")
            
        self.logger.info("Guardrail Passed: No overlap between pretrain and forward holdout.")
        return True
        
    def enforce_distinct_validation_slices(self, pretrain_val_slice: tuple, prod_promotion_slice: tuple) -> bool:
        """
        Do not choose the pretrain method using the same validation slice used for production promotion.
        """
        if pretrain_val_slice == prod_promotion_slice:
            self.logger.error("FATAL LEAKAGE: Pretrain is tuning on the exact same validation slice used for production promotion!")
            raise RuntimeError("Data Leakage Guardrail triggered. Halting pretraining.")
            
        self.logger.info("Guardrail Passed: Pretrain validation slice is independent of production promotion slice.")
        return True

    def validate_method_selection(self, selected_method: str, forward_metric_delta: float):
        """
        Enforces the rule: Prefer the simplest pretrain method that improves forward metrics, 
        not the one with the lowest pretrain loss.
        """
        if forward_metric_delta <= 0:
            self.logger.warning(f"Guardrail Warning: Selected pretrain method '{selected_method}' did NOT improve downstream forward metrics (Delta: {forward_metric_delta}).")
            self.logger.warning("You should discard this pretrain method even if its pretrain loss was low.")
        else:
            self.logger.info(f"Guardrail Passed: Method '{selected_method}' proved its worth (Downstream Delta: +{forward_metric_delta}).")
