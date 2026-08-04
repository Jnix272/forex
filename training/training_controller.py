"""
training/training_controller.py
===============================
Master training controller that watches for overfitting patterns (e.g.
train loss dropping while val loss rising) and generates the compact
training_control_report.json.

``evaluate_epoch`` returns actionable flags; callers (``supervised_loop``)
must apply them via ``apply_responses``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


class TrainingController:
    def __init__(self, report_dir: str = "logs"):
        self.logger = logging.getLogger(__name__)
        self.report_path = Path(report_dir) / "training_control_report.json"

        self.train_loss_history: list[float] = []
        self.val_loss_history: list[float] = []
        self.val_sharpe_history: list[float] = []

        self.report_data: dict[str, Any] = {
            "model_recipe_used": "unknown",
            "memory_suggestions": [],
            "overfitting_signals_detected": [],
            "curriculum_changes": [],
            "lr_changes": [],
            "actions_applied": [],
            "best_epoch": None,
            "restore_decision": False,
            "final_promotion_recommendation": "pending",
        }

    def set_recipe(self, recipe_name: str) -> None:
        self.report_data["model_recipe_used"] = recipe_name

    def log_memory_suggestion(self, suggestion: str) -> None:
        self.report_data["memory_suggestions"].append(suggestion)

    def log_curriculum_change(self, change: str) -> None:
        self.report_data["curriculum_changes"].append(change)

    def log_lr_change(self, change: str) -> None:
        self.report_data["lr_changes"].append(change)

    def evaluate_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_sharpe: float,
    ) -> dict[str, bool]:
        """
        Detect overfitting signals at the end of every epoch.
        Returns suggested hyperparameter responses for ``apply_responses``.
        """
        self.train_loss_history.append(train_loss)
        self.val_loss_history.append(val_loss)
        self.val_sharpe_history.append(val_sharpe)

        responses = {
            "lower_lr": False,
            "increase_dropout": False,
            "hold_curriculum": False,
            "stop_early": False,
        }

        if len(self.train_loss_history) >= 3:
            train_improving = (
                self.train_loss_history[-1]
                < self.train_loss_history[-2]
                < self.train_loss_history[-3]
            )
            val_rising = (
                self.val_loss_history[-1]
                > self.val_loss_history[-2]
                > self.val_loss_history[-3]
            )
            if train_improving and val_rising:
                msg = (
                    f"Epoch {epoch}: Overfitting detected "
                    "(Train loss dropping, Val loss rising)."
                )
                self.logger.warning(msg)
                self.report_data["overfitting_signals_detected"].append(msg)
                responses["increase_dropout"] = True
                responses["hold_curriculum"] = True

        if len(self.val_sharpe_history) >= 4:
            peak_sharpe = max(self.val_sharpe_history)
            current_sharpe = self.val_sharpe_history[-1]
            if peak_sharpe > 0.5 and current_sharpe < (peak_sharpe * 0.5):
                msg = (
                    f"Epoch {epoch}: Sharpe collapse detected "
                    f"(Peak {peak_sharpe:.2f} -> Current {current_sharpe:.2f})."
                )
                self.logger.warning(msg)
                self.report_data["overfitting_signals_detected"].append(msg)
                responses["lower_lr"] = True
                responses["stop_early"] = True

        return responses

    def apply_responses(
        self,
        responses: dict[str, bool],
        *,
        model=None,
        optimizer=None,
        scheduler=None,
        curriculum_state: dict[str, Any] | None = None,
        lr_mult: float = 0.5,
        dropout_bump: float = 0.05,
        dropout_ceil: float = 0.50,
        epoch: int | None = None,
    ) -> dict[str, Any]:
        """
        Apply evaluate_epoch flags to live training objects.

        Returns a summary of what changed (for logs / early-stop decisions).
        """
        applied: dict[str, Any] = {
            "lower_lr": False,
            "increase_dropout": False,
            "hold_curriculum": False,
            "stop_early": bool(responses.get("stop_early")),
            "new_lr": None,
            "dropout_modules_bumped": 0,
        }
        ep = epoch if epoch is not None else len(self.train_loss_history)

        if responses.get("lower_lr") and optimizer is not None:
            for pg in optimizer.param_groups:
                old = float(pg["lr"])
                pg["lr"] = max(1e-7, old * lr_mult)
                applied["new_lr"] = float(pg["lr"])
            if scheduler is not None:
                for attr in ("base_lrs", "_last_lr"):
                    vals = getattr(scheduler, attr, None)
                    if isinstance(vals, list):
                        setattr(scheduler, attr, [v * lr_mult for v in vals])
            applied["lower_lr"] = True
            msg = f"Epoch {ep}: LR ×{lr_mult} → {applied['new_lr']}"
            self.log_lr_change(msg)
            self.logger.info("[TrainingController] %s", msg)

        if responses.get("increase_dropout") and model is not None:
            import torch.nn as nn
            bumped = 0
            for mod in model.modules():
                if isinstance(mod, nn.Dropout):
                    mod.p = min(dropout_ceil, float(mod.p) + dropout_bump)
                    bumped += 1
            applied["increase_dropout"] = bumped > 0
            applied["dropout_modules_bumped"] = bumped
            if bumped:
                msg = f"Epoch {ep}: bumped Dropout.p by +{dropout_bump} on {bumped} module(s)"
                self.report_data["actions_applied"].append(msg)
                self.logger.info("[TrainingController] %s", msg)

        if responses.get("hold_curriculum") and curriculum_state is not None:
            curriculum_state["seq_frozen"] = True
            curriculum_state["hold_requested"] = True
            applied["hold_curriculum"] = True
            msg = f"Epoch {ep}: curriculum advance held (seq_frozen=True)"
            self.log_curriculum_change(msg)
            self.logger.info("[TrainingController] %s", msg)

        if applied["stop_early"]:
            msg = f"Epoch {ep}: early-stop recommended after Sharpe collapse"
            self.report_data["actions_applied"].append(msg)

        self.report_data["actions_applied"].append(
            {k: v for k, v in applied.items() if v not in (False, 0, None)}
        )
        return applied

    def finalize_training(
        self,
        best_epoch: int,
        promoted: bool,
        *,
        restored_best: bool = False,
    ) -> None:
        """
        Persist the control report.

        ``restored_best`` must reflect whether best-epoch weights were actually
        reloaded into the live model (not assumed True).
        """
        self.report_data["best_epoch"] = best_epoch
        self.report_data["restore_decision"] = bool(restored_best)
        self.report_data["final_promotion_recommendation"] = (
            "promoted" if promoted else "demoted"
        )
        self._save_report()

    def _save_report(self) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.report_path, "w", encoding="utf-8") as f:
            json.dump(self.report_data, f, indent=2)
