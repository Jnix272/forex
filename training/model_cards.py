"""
training/model_cards.py
=======================
Generates JSON model cards for exported model weights to document
architecture, performance, data windows, and known weaknesses.
"""

import json
from pathlib import Path
from datetime import datetime, timezone

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
        filename: str = "model_card.json"
    ) -> Path:
        """
        Creates a JSON model card with all metadata requested in Priority 5.
        """
        card_data = {
            "model_name": model_name,
            "architecture": architecture,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "training_data": {
                "window": data_window,
                "pairs": pairs,
                "features_count": len(features),
                "features_list": features,
                "label_method": label_method
            },
            "performance": {
                "validation_sharpe": round(val_sharpe, 4),
                "validation_loss": round(val_loss, 4),
                "forward_holdout_pnl": round(forward_holdout_pnl, 2)
            },
            "known_weaknesses": weaknesses,
            "promotion_status": promotion_status
        }
        
        target_path = self.checkpoint_dir / filename
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(card_data, f, indent=2)
            
        return target_path
