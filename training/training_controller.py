"""
training/training_controller.py
===============================
Master training controller that watches for overfitting patterns (e.g. 
train loss dropping while val loss rising) and generates the compact
training_control_report.json.
"""

import json
from pathlib import Path
import logging

class TrainingController:
    def __init__(self, report_dir: str = "logs"):
        self.logger = logging.getLogger(__name__)
        self.report_path = Path(report_dir) / "training_control_report.json"
        
        self.train_loss_history = []
        self.val_loss_history = []
        self.val_sharpe_history = []
        
        self.report_data = {
            "model_recipe_used": "unknown",
            "memory_suggestions": [],
            "overfitting_signals_detected": [],
            "curriculum_changes": [],
            "lr_changes": [],
            "best_epoch": None,
            "restore_decision": False,
            "final_promotion_recommendation": "pending"
        }
        
    def set_recipe(self, recipe_name: str):
        self.report_data["model_recipe_used"] = recipe_name
        
    def log_memory_suggestion(self, suggestion: str):
        self.report_data["memory_suggestions"].append(suggestion)
        
    def log_curriculum_change(self, change: str):
        self.report_data["curriculum_changes"].append(change)
        
    def log_lr_change(self, change: str):
        self.report_data["lr_changes"].append(change)

    def evaluate_epoch(self, epoch: int, train_loss: float, val_loss: float, val_sharpe: float) -> dict:
        """
        Detects overfitting signals at the end of every epoch.
        Returns suggested hyperparameter responses.
        """
        self.train_loss_history.append(train_loss)
        self.val_loss_history.append(val_loss)
        self.val_sharpe_history.append(val_sharpe)
        
        responses = {
            "lower_lr": False,
            "increase_dropout": False,
            "hold_curriculum": False,
            "stop_early": False
        }
        
        # 1. Detect if train loss keeps improving while val loss rises
        if len(self.train_loss_history) >= 3:
            train_improving = self.train_loss_history[-1] < self.train_loss_history[-2] < self.train_loss_history[-3]
            val_rising = self.val_loss_history[-1] > self.val_loss_history[-2] > self.val_loss_history[-3]
            
            if train_improving and val_rising:
                msg = f"Epoch {epoch}: Overfitting detected (Train loss dropping, Val loss rising)."
                self.logger.warning(msg)
                self.report_data["overfitting_signals_detected"].append(msg)
                
                responses["increase_dropout"] = True
                responses["hold_curriculum"] = True
                
        # 2. Detect if Sharpe spikes once and then collapses
        if len(self.val_sharpe_history) >= 4:
            peak_sharpe = max(self.val_sharpe_history)
            current_sharpe = self.val_sharpe_history[-1]
            
            if peak_sharpe > 0.5 and current_sharpe < (peak_sharpe * 0.5):
                msg = f"Epoch {epoch}: Sharpe collapse detected (Peak {peak_sharpe:.2f} -> Current {current_sharpe:.2f})."
                self.logger.warning(msg)
                self.report_data["overfitting_signals_detected"].append(msg)
                
                responses["lower_lr"] = True
                responses["stop_early"] = True
                
        return responses

    def finalize_training(self, best_epoch: int, promoted: bool):
        self.report_data["best_epoch"] = best_epoch
        self.report_data["restore_decision"] = True
        self.report_data["final_promotion_recommendation"] = "promoted" if promoted else "demoted"
        self._save_report()
        
    def _save_report(self):
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.report_path, "w", encoding="utf-8") as f:
            json.dump(self.report_data, f, indent=2)
