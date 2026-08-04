"""
retraining/live_feedback.py
============================
INF-012/013/014: Live trading feedback pipeline.

Connects live execution outcomes back to the training pipeline:
- Ingests hard examples from execution audit logs
- Computes slippage and fill quality metrics
- Provides a priority sampler for retraining that oversamples mistakes
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class LiveFeedbackStore:
    """Persistent store connecting live trade outcomes to retraining."""

    def __init__(self, store_dir: str = "logs/feedback"):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._hard_examples: list[dict] = []
        self._metrics: list[dict] = []

    def ingest_audit_log(self, audit_path: str) -> dict:
        """Parse execution audit JSONL and extract retraining-relevant signals.

        Returns summary of ingested data.
        """
        path = Path(audit_path)
        if not path.exists():
            logger.warning(f"[LiveFeedback] Audit log not found: {audit_path}")
            return {"error": "file_not_found"}

        hard_examples = []
        fills = []
        signals = []
        rejections = []

        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                event = record.get("event", "")
                if event == "TRADE_CLOSED" and record.get("is_hard_example"):
                    hard_examples.append(record)
                elif event == "ORDER_FILLED":
                    fills.append(record)
                elif event == "SIGNAL":
                    signals.append(record)
                elif event == "ORDER_REJECTED":
                    rejections.append(record)

        self._hard_examples.extend(hard_examples)

        # Compute execution quality metrics
        slippage_pips = [f.get("slippage_pips", 0) for f in fills if f.get("slippage_pips") is not None]
        metrics = {
            "audit_file": str(path.name),
            "ingested_at": datetime.now(UTC).isoformat(),
            "total_signals": len(signals),
            "total_fills": len(fills),
            "total_rejections": len(rejections),
            "hard_examples_found": len(hard_examples),
            "mean_slippage_pips": float(np.mean(slippage_pips)) if slippage_pips else 0.0,
            "max_slippage_pips": float(np.max(slippage_pips)) if slippage_pips else 0.0,
            "rejection_rate": len(rejections) / max(1, len(signals)),
        }
        self._metrics.append(metrics)

        # Persist hard examples for the training pipeline
        self._save_hard_examples()
        self._save_metrics(metrics)

        logger.info(
            f"[LiveFeedback] Ingested {path.name}: "
            f"{len(hard_examples)} hard examples, "
            f"{len(fills)} fills, {len(rejections)} rejections"
        )
        return metrics

    def get_hard_example_timestamps(self) -> list[str]:
        """Return timestamps of hard examples for priority sampling."""
        return [ex.get("ts", "") for ex in self._hard_examples]

    def get_priority_weights(self, n_samples: int, base_timestamps: np.ndarray,
                             hard_boost: float = 3.0) -> np.ndarray:
        """Generate sampling weights that oversample hard examples.

        Args:
            n_samples: total number of training samples
            base_timestamps: timestamps of training data (int64 epoch)
            hard_boost: multiplier for samples near hard example timestamps

        Returns:
            Weight array of shape [n_samples] for use with WeightedRandomSampler.
        """
        weights = np.ones(n_samples, dtype=np.float64)

        if not self._hard_examples:
            return weights

        # Map hard example timestamps to training indices
        hard_ts_list = []
        for ex in self._hard_examples:
            ts_str = ex.get("ts", "")
            if ts_str:
                try:
                    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    hard_ts_list.append(int(dt.timestamp()))
                except (ValueError, TypeError):
                    continue

        if not hard_ts_list:
            return weights

        hard_ts = np.array(hard_ts_list, dtype=np.int64)

        for ht in hard_ts:
            distances = np.abs(base_timestamps - ht)
            # Boost samples within 1 hour of a hard example
            nearby = distances < 3600
            weights[nearby] *= hard_boost

        # Normalize to mean=1
        weights /= weights.mean()
        return weights

    def get_feedback_summary(self) -> dict:
        """Return aggregated feedback metrics for training config decisions."""
        if not self._metrics:
            return {"status": "no_data"}

        return {
            "total_ingestions": len(self._metrics),
            "total_hard_examples": len(self._hard_examples),
            "avg_slippage_pips": float(np.mean([m.get("mean_slippage_pips", 0) for m in self._metrics])),
            "avg_rejection_rate": float(np.mean([m.get("rejection_rate", 0) for m in self._metrics])),
            "last_ingestion": self._metrics[-1].get("ingested_at", ""),
        }

    def _save_hard_examples(self):
        path = self.store_dir / "hard_examples.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._hard_examples[-1000:], f, indent=2, default=str)
        except OSError as e:
            logger.warning(f"[LiveFeedback] Could not save hard examples: {e}")

    def _save_metrics(self, metrics: dict):
        path = self.store_dir / "feedback_metrics.jsonl"
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, default=str) + "\n")
        except OSError as e:
            logger.warning(f"[LiveFeedback] Could not save metrics: {e}")

    def load_stored_hard_examples(self) -> list[dict]:
        """Load previously stored hard examples from disk."""
        path = self.store_dir / "hard_examples.json"
        if not path.exists():
            return []
        try:
            with open(path, encoding="utf-8") as f:
                self._hard_examples = json.load(f)
            return self._hard_examples
        except (json.JSONDecodeError, OSError):
            return []
