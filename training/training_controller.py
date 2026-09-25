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
    def __init__(self, report_dir: str = "logs", adaptation: dict | None = None):
        self.logger = logging.getLogger(__name__)
        self.report_path = Path(report_dir) / "training_control_report.json"

        self.train_loss_history: list[float] = []
        self.val_loss_history: list[float] = []
        self.val_sharpe_history: list[float] = []
        self.val_sharpe_ema_history: list[float] = []
        self.val_sharpe_ema: float | None = None
        self.dir_acc_history: list[float] = []
        _adap = adaptation or {}
        # How many consecutive epochs dir_acc must stay below 0.50 before acting.
        self.dir_acc_below_random_window: int = int(_adap.get("dir_acc_below_random_window", 2))
        self.dir_acc_random_threshold: float = float(_adap.get("dir_acc_random_threshold", 0.50))
        self.collapse_drop: float = float(_adap.get("collapse_drop", 0.50))
        self.collapse_min_peak: float = float(_adap.get("collapse_min_peak", 0.50))
        self.collapse_lr_mult: float = float(_adap.get("collapse_lr_mult", 0.50))
        self.stable_window: int = int(_adap.get("stable_window", 3))
        self.recovery_window: int = int(_adap.get("recovery_window", 2))
        self.min_epochs_per_stage: int = int(_adap.get("min_epochs_per_stage", 1))
        self.advance_lr_mult: float = float(_adap.get("advance_lr_mult", 1.0))
        self.collapse_reversal_threshold: float = float(_adap.get("collapse_reversal_threshold", -0.10))
        # sharpe_ema_alpha smooths the noisy per-epoch val Sharpe before it feeds the
        # collapse detector, so a single lucky epoch can't set an unreachable "peak".
        self.sharpe_ema_alpha: float = float(_adap.get("sharpe_ema_alpha", 0.30))
        self.min_stable_sharpe: float = float(_adap.get("min_stable_sharpe", 0.05))

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
        dir_acc: float | None = None,
    ) -> dict[str, bool]:
        """
        Detect overfitting signals at the end of every epoch.
        Returns suggested hyperparameter responses for ``apply_responses``.
        """
        self.train_loss_history.append(train_loss)
        self.val_loss_history.append(val_loss)
        self.val_sharpe_history.append(val_sharpe)
        self.val_sharpe_ema = (
            val_sharpe
            if self.val_sharpe_ema is None
            else self.sharpe_ema_alpha * val_sharpe + (1.0 - self.sharpe_ema_alpha) * self.val_sharpe_ema
        )
        self.val_sharpe_ema_history.append(self.val_sharpe_ema)
        if dir_acc is not None:
            self.dir_acc_history.append(float(dir_acc))

        responses = {
            "lower_lr": False,
            "increase_dropout": False,
            "hold_curriculum": False,
            "stop_early": False,
        }

        if len(self.train_loss_history) >= 3:
            train_improving = self.train_loss_history[-1] < self.train_loss_history[-2] < self.train_loss_history[-3]
            val_rising = self.val_loss_history[-1] > self.val_loss_history[-2] > self.val_loss_history[-3]
            if train_improving and val_rising:
                msg = f"Epoch {epoch}: Overfitting detected (Train loss dropping, Val loss rising)."
                self.logger.warning(msg)
                self.report_data["overfitting_signals_detected"].append(msg)
                responses["increase_dropout"] = True
                responses["hold_curriculum"] = True

        if len(self.val_sharpe_ema_history) >= 4:
            # Use the EMA-smoothed stream (not the raw per-epoch Sharpe) so a single
            # noisy epoch can't set an unreachable "peak" that every later epoch is
            # then compared against.
            peak_sharpe = max(self.val_sharpe_ema_history)
            current_sharpe = self.val_sharpe_ema_history[-1]
            collapse_threshold = peak_sharpe * (1.0 - self.collapse_drop)
            if (
                peak_sharpe > self.collapse_min_peak
                and peak_sharpe >= self.min_stable_sharpe
                and current_sharpe < collapse_threshold
            ):
                msg = (
                    f"Epoch {epoch}: Sharpe collapse detected (EMA peak {peak_sharpe:.2f} -> "
                    f"EMA current {current_sharpe:.2f}, threshold {collapse_threshold:.2f}; "
                    f"raw={val_sharpe:.2f})."
                )
                self.logger.warning(msg)
                self.report_data["overfitting_signals_detected"].append(msg)
                responses["lower_lr"] = True
                responses["stop_early"] = True

        # Directional accuracy gate: if the model predicts the wrong direction
        # more often than random for N consecutive epochs, lower LR and bump
        # dropout. Unlike the Sharpe collapse gate this fires even when Sharpe
        # is near zero (i.e. when the model is mostly predicting HOLD/zero).
        if (
            len(self.dir_acc_history) >= self.dir_acc_below_random_window
            and all(
                a < self.dir_acc_random_threshold
                for a in self.dir_acc_history[-self.dir_acc_below_random_window :]
            )
        ):
            msg = (
                f"Epoch {epoch}: dir_acc below random for "
                f"{self.dir_acc_below_random_window} consecutive epochs "
                f"(last={self.dir_acc_history[-1]:.3f} < {self.dir_acc_random_threshold:.2f}). "
                "Lowering LR and bumping dropout."
            )
            self.logger.warning(msg)
            self.report_data["overfitting_signals_detected"].append(msg)
            responses["lower_lr"] = True
            responses["increase_dropout"] = True
            responses["hold_curriculum"] = True

        return responses

    def apply_responses(
        self,
        responses: dict[str, bool],
        *,
        model=None,
        optimizer=None,
        scheduler=None,
        curriculum_state: dict[str, Any] | None = None,
        lr_mult: float | None = None,
        dropout_bump: float = 0.05,
        dropout_ceil: float = 0.50,
        epoch: int | None = None,
    ) -> dict[str, Any]:
        """
        Apply evaluate_epoch flags to live training objects.

        Returns a summary of what changed (for logs / early-stop decisions).
        """
        # Use adaptation-configured multiplier; fall back to legacy default only if unconfigured
        _lr_mult = lr_mult if lr_mult is not None else self.collapse_lr_mult
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
                pg["lr"] = max(1e-7, old * _lr_mult)
                applied["new_lr"] = float(pg["lr"])
            if scheduler is not None:
                for attr in ("base_lrs", "_last_lr", "initial_lrs", "max_lrs", "min_lrs"):
                    vals = getattr(scheduler, attr, None)
                    if isinstance(vals, list):
                        setattr(scheduler, attr, [v * _lr_mult for v in vals])
            applied["lower_lr"] = True
            msg = f"Epoch {ep}: LR x{_lr_mult} → {applied['new_lr']}"
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

        self.report_data["actions_applied"].append({k: v for k, v in applied.items() if v not in (False, 0, None)})
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
        self.report_data["final_promotion_recommendation"] = "promoted" if promoted else "demoted"
        self._save_report()

    def _save_report(self) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.report_path, "w", encoding="utf-8") as f:
            json.dump(self.report_data, f, indent=2)
