"""
data/feature_schema.py
======================
Generates and enforces SHA-256 hashes for feature schema ordering to prevent
catastrophic column-swaps during live trading.
"""

import hashlib
import json
import logging
from pathlib import Path


class FeatureSchemaEnforcer:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)

    def generate_hash(self, ordered_features: list[str]) -> str:
        """Generates a strict SHA-256 hash of the feature names."""
        feature_string = ",".join(ordered_features)
        return hashlib.sha256(feature_string.encode("utf-8")).hexdigest()

    def export_schema(self, ordered_features: list[str], filename: str = "production_schema.json"):
        """Exports the exact ordered list of features and its hash."""
        schema_hash = self.generate_hash(ordered_features)

        schema_data = {
            "schema_hash": schema_hash,
            "feature_count": len(ordered_features),
            "ordered_features": ordered_features,
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        schema_path = self.output_dir / filename
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(schema_data, f, indent=2)

        self.logger.info(f"Exported feature schema (Hash: {schema_hash[:8]}...)")
        return schema_hash

    def enforce_live_schema(self, expected_hash: str, live_features: list[str]):
        """Intercepts a live data feed and fails fast if the features are misaligned."""
        live_hash = self.generate_hash(live_features)

        if live_hash != expected_hash:
            self.logger.error("FATAL: Live inference feature schema does NOT match training schema.")
            self.logger.error(f"Expected Hash: {expected_hash}")
            self.logger.error(f"Live Hash: {live_hash}")
            raise RuntimeError("Live feature order mismatch. Halting execution to prevent bad trades.")

        self.logger.info("Live schema matches training schema perfectly.")
