"""Adaptive curriculum controller for model training.

The controller keeps curriculum movement tied to validation behavior instead of
fixed epochs: rolling Sharpe must be stable before difficulty or sequence length
can advance, sharp collapses freeze or revert sequence length, and stage changes
emit auditable events.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AdaptiveCurriculumConfig:
    stages: tuple[str, ...] = ("easy", "medium", "hard")
    seq_lens: tuple[int, ...] = (30, 60, 90, 120)
    stable_epochs_required: int = 3
    stability_window: int = 3
    min_stable_sharpe: float = 0.50
    max_sharpe_std: float = 0.15
    collapse_ratio: float = 0.50
    collapse_floor: float = 0.20
    recovery_epochs: int = 2
    lr_reduction_mult: float = 0.85
    allow_seq_revert: bool = True


@dataclass
class CurriculumEvent:
    epoch: int
    event: str
    reason: str
    stage: str
    seq_len: int
    lr_multiplier: float = 1.0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "event": self.event,
            "reason": self.reason,
            "stage": self.stage,
            "seq_len": self.seq_len,
            "lr_multiplier": self.lr_multiplier,
            "details": dict(self.details),
        }


class CurriculumController:
    """Performance-gated curriculum controller.

    `evaluate_epoch` returns an action dict suitable for training loops and
    appends every meaningful decision to `events` for W&B/console/report logging.
    """

    def __init__(
        self,
        start_stage: str = "easy",
        start_seq_len: int | None = None,
        config: AdaptiveCurriculumConfig | None = None,
    ):
        self.logger = logging.getLogger(__name__)
        self.config = config or AdaptiveCurriculumConfig()
        if start_stage not in self.config.stages:
            raise ValueError(f"Unknown curriculum stage: {start_stage}")

        self.current_stage = start_stage
        self.current_seq_len = int(start_seq_len or self.config.seq_lens[0])
        self.stable_epochs = 0
        self.recovery_left = 0
        self.sharpe_history: list[float] = []
        self.events: list[CurriculumEvent] = []

    def _rolling_stats(self) -> dict[str, float]:
        window = self.sharpe_history[-self.config.stability_window :]
        if not window:
            return {"mean": 0.0, "std": 0.0, "peak": 0.0, "current": 0.0}
        mean = sum(window) / len(window)
        variance = sum((x - mean) ** 2 for x in window) / len(window)
        return {
            "mean": mean,
            "std": variance ** 0.5,
            "peak": max(self.sharpe_history),
            "current": self.sharpe_history[-1],
        }

    def _next_stage(self) -> str | None:
        idx = self.config.stages.index(self.current_stage)
        if idx >= len(self.config.stages) - 1:
            return None
        return self.config.stages[idx + 1]

    def _next_seq_len(self) -> int | None:
        ordered = sorted(set(int(x) for x in self.config.seq_lens))
        for seq_len in ordered:
            if seq_len > self.current_seq_len:
                return seq_len
        return None

    def _previous_seq_len(self) -> int | None:
        ordered = sorted(set(int(x) for x in self.config.seq_lens))
        prev = [seq_len for seq_len in ordered if seq_len < self.current_seq_len]
        return prev[-1] if prev else None

    def _append_event(
        self,
        epoch: int,
        event: str,
        reason: str,
        lr_multiplier: float = 1.0,
        **details: Any,
    ) -> None:
        item = CurriculumEvent(
            epoch=epoch,
            event=event,
            reason=reason,
            stage=self.current_stage,
            seq_len=self.current_seq_len,
            lr_multiplier=lr_multiplier,
            details=details,
        )
        self.events.append(item)
        self.logger.info("[Curriculum] %s", item.to_dict())

    def evaluate_epoch(self, epoch: int, val_sharpe: float, val_loss: float = 0.0) -> dict[str, Any]:
        """Evaluate validation metrics and return curriculum actions."""
        val_sharpe = float(val_sharpe)
        val_loss = float(val_loss)
        self.sharpe_history.append(val_sharpe)
        stats = self._rolling_stats()

        actions: dict[str, Any] = {
            "advance_difficulty": False,
            "delay_seq_len_increase": False,
            "revert_seq_len": False,
            "advance_seq_len": False,
            "reduce_lr": False,
            "lr_multiplier": 1.0,
            "new_stage": self.current_stage,
            "new_seq_len": self.current_seq_len,
            "stable_epochs": self.stable_epochs,
            "recovery_left": self.recovery_left,
            "rolling_sharpe_mean": stats["mean"],
            "rolling_sharpe_std": stats["std"],
            "event": None,
        }

        peak = stats["peak"]
        collapsed_from_peak = peak > 0 and val_sharpe <= peak * self.config.collapse_ratio
        collapsed_absolute = val_sharpe < self.config.collapse_floor
        if collapsed_from_peak or collapsed_absolute:
            self.stable_epochs = 0
            self.recovery_left = self.config.recovery_epochs
            actions["delay_seq_len_increase"] = True
            actions["reduce_lr"] = True
            actions["lr_multiplier"] = self.config.lr_reduction_mult
            reason = "sharpe_collapse"

            previous_seq = self._previous_seq_len()
            if self.config.allow_seq_revert and previous_seq is not None:
                old_seq = self.current_seq_len
                self.current_seq_len = previous_seq
                actions["revert_seq_len"] = True
                actions["new_seq_len"] = self.current_seq_len
                self._append_event(
                    epoch,
                    "seq_len_revert",
                    reason,
                    self.config.lr_reduction_mult,
                    from_seq_len=old_seq,
                    to_seq_len=previous_seq,
                    val_sharpe=val_sharpe,
                    peak_sharpe=peak,
                    val_loss=val_loss,
                )
                actions["event"] = "seq_len_revert"
            else:
                self._append_event(
                    epoch,
                    "curriculum_freeze",
                    reason,
                    self.config.lr_reduction_mult,
                    val_sharpe=val_sharpe,
                    peak_sharpe=peak,
                    val_loss=val_loss,
                )
                actions["event"] = "curriculum_freeze"
            actions["stable_epochs"] = self.stable_epochs
            actions["recovery_left"] = self.recovery_left
            return actions

        if self.recovery_left > 0:
            self.recovery_left -= 1
            self.stable_epochs = 0
            actions["delay_seq_len_increase"] = True
            actions["recovery_left"] = self.recovery_left
            self._append_event(
                epoch,
                "recovery_hold",
                "post_collapse_recovery",
                val_sharpe=val_sharpe,
                val_loss=val_loss,
            )
            actions["event"] = "recovery_hold"
            return actions

        stable = (
            len(self.sharpe_history) >= self.config.stability_window
            and stats["mean"] >= self.config.min_stable_sharpe
            and stats["std"] <= self.config.max_sharpe_std
        )
        self.stable_epochs = self.stable_epochs + 1 if stable else 0
        actions["stable_epochs"] = self.stable_epochs

        if self.stable_epochs < self.config.stable_epochs_required:
            return actions

        next_stage = self._next_stage()
        next_seq = self._next_seq_len()
        if next_stage is not None:
            old_stage = self.current_stage
            self.current_stage = next_stage
            self.stable_epochs = 0
            actions.update(
                {
                    "advance_difficulty": True,
                    "reduce_lr": True,
                    "lr_multiplier": self.config.lr_reduction_mult,
                    "new_stage": self.current_stage,
                    "stable_epochs": 0,
                    "event": "difficulty_advance",
                }
            )
            self._append_event(
                epoch,
                "difficulty_advance",
                "rolling_sharpe_stable",
                self.config.lr_reduction_mult,
                from_stage=old_stage,
                to_stage=self.current_stage,
                rolling_sharpe_mean=stats["mean"],
                rolling_sharpe_std=stats["std"],
            )
        elif next_seq is not None:
            old_seq = self.current_seq_len
            self.current_seq_len = next_seq
            self.stable_epochs = 0
            actions.update(
                {
                    "advance_seq_len": True,
                    "new_seq_len": self.current_seq_len,
                    "stable_epochs": 0,
                    "event": "seq_len_advance",
                }
            )
            self._append_event(
                epoch,
                "seq_len_advance",
                "rolling_sharpe_stable",
                from_seq_len=old_seq,
                to_seq_len=self.current_seq_len,
                rolling_sharpe_mean=stats["mean"],
                rolling_sharpe_std=stats["std"],
            )

        return actions

    def event_log(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.events]

    def state_dict(self) -> dict[str, Any]:
        return {
            "current_stage": self.current_stage,
            "current_seq_len": self.current_seq_len,
            "stable_epochs": self.stable_epochs,
            "recovery_left": self.recovery_left,
            "sharpe_history": list(self.sharpe_history),
            "events": self.event_log(),
        }

    def write_report(self, output_path: str | Path) -> dict[str, Any]:
        report = self.state_dict()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return report
