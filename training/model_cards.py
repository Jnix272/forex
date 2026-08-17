"""
training/model_cards.py
=======================
Generates JSON model cards for exported model weights to document
architecture, performance, data windows, and known weaknesses.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REQUIRED_KEYS = (
    "model_name",
    "architecture",
    "created_at",
    "training_data",
    "performance",
    "known_weaknesses",
    "promotion_status",
)


def _finite_float(value: Any, *, name: str, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        print(f"[ModelCard] WARN: {name}={value!r} not numeric; using {default}")
        return default
    if not math.isfinite(v):
        print(f"[ModelCard] WARN: {name}={value!r} non-finite; using {default}")
        return default
    return v


class ModelCardGenerator:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        model_name: str,
        architecture: str,
        data_window: str,
        pairs: list[str],
        features: list[str],
        label_method: str,
        val_sharpe: float,
        val_loss: float,
        forward_holdout_pnl: float,
        weaknesses: list[str],
        promotion_status: str,
        filename: str = "model_card.json",
    ) -> Path:
        """
        Creates a JSON model card with all metadata requested in Priority 5.
        Non-finite performance fields are replaced with 0.0 and logged.
        """
        card_data = {
            "model_name": str(model_name),
            "architecture": str(architecture),
            "created_at": datetime.now(UTC).isoformat(),
            "training_data": {
                "window": str(data_window),
                "pairs": list(pairs or []),
                "features_count": len(features or []),
                "features_list": list(features or []),
                "label_method": str(label_method),
            },
            "performance": {
                "validation_sharpe": round(_finite_float(val_sharpe, name="val_sharpe"), 4),
                "validation_loss": round(_finite_float(val_loss, name="val_loss"), 4),
                "forward_holdout_pnl": round(_finite_float(forward_holdout_pnl, name="forward_holdout_pnl"), 2),
            },
            "known_weaknesses": list(weaknesses or []),
            "promotion_status": str(promotion_status),
        }
        missing = [k for k in _REQUIRED_KEYS if k not in card_data]
        if missing:
            raise ValueError(f"Model card missing required keys: {missing}")

        target_path = self.checkpoint_dir / filename
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(card_data, f, indent=2)

        return target_path
