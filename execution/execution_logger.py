"""
execution/execution_logger.py
==============================
INF-008: Persistent JSONL audit trail for the full signal→order→fill chain.

Every live trading decision is logged to a date-partitioned JSONL file for:
- Post-mortem debugging after unexpected losses
- Regulatory compliance and audit requirements
- Feeding live mistakes back into retraining (INF-012/013/014)
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ExecutionLogger:
    """Append-only JSONL logger for trade execution events."""

    def __init__(self, log_dir: str = "logs/execution"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._current_date: str | None = None
        self._file_path: Path | None = None

    @property
    def audit_file(self) -> Path:
        today = datetime.now(UTC).strftime("%Y%m%d")
        if today != self._current_date:
            self._current_date = today
            self._file_path = self.log_dir / f"audit_{today}.jsonl"
        return self._file_path

    def log_signal(self, pair: str, direction: str, confidence: float, model_name: str = "", features_hash: str = ""):
        self._write(
            {
                "event": "SIGNAL",
                "pair": pair,
                "direction": direction,
                "confidence": confidence,
                "model_name": model_name,
                "features_hash": features_hash,
            }
        )

    def log_order(
        self,
        order_id: str,
        pair: str,
        side: str,
        size: float,
        price: float,
        sl: float,
        tp: float,
        order_type: str = "MARKET",
    ):
        self._write(
            {
                "event": "ORDER_PLACED",
                "order_id": order_id,
                "pair": pair,
                "side": side,
                "size": size,
                "price": price,
                "sl": sl,
                "tp": tp,
                "order_type": order_type,
            }
        )

    def log_fill(self, order_id: str, fill_price: float, slippage_pips: float, fill_size: float = 0.0):
        self._write(
            {
                "event": "ORDER_FILLED",
                "order_id": order_id,
                "fill_price": fill_price,
                "fill_size": fill_size,
                "slippage_pips": slippage_pips,
            }
        )

    def log_rejection(self, order_id: str, reason: str):
        self._write(
            {
                "event": "ORDER_REJECTED",
                "order_id": order_id,
                "reason": reason,
            }
        )

    def log_trade_close(
        self,
        order_id: str,
        entry_price: float,
        exit_price: float,
        pnl_pips: float,
        hit_sl: bool,
        slippage_pips: float,
        hold_duration_s: float = 0.0,
    ):
        record = {
            "event": "TRADE_CLOSED",
            "order_id": order_id,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_pips": pnl_pips,
            "hit_sl": hit_sl,
            "slippage_pips": slippage_pips,
            "hold_duration_s": hold_duration_s,
            "is_hard_example": hit_sl or abs(slippage_pips) > 2.0,
        }
        self._write(record)
        if record["is_hard_example"]:
            self._flag_for_retraining(order_id, pnl_pips, hit_sl)

    def log_error(self, context: str, error: str, order_id: str = ""):
        self._write(
            {
                "event": "ERROR",
                "context": context,
                "error": error,
                "order_id": order_id,
            }
        )

    def _flag_for_retraining(self, order_id: str, pnl_pips: float, hit_sl: bool):
        """Mark trade as a hard example for the retraining pipeline."""
        hard_path = self.log_dir / "hard_examples.jsonl"
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "order_id": order_id,
            "pnl_pips": pnl_pips,
            "hit_sl": hit_sl,
        }
        try:
            with open(hard_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as e:
            logger.warning(f"[ExecutionLogger] Could not write hard example: {e}")

    def _write(self, record: dict):
        record["ts"] = datetime.now(UTC).isoformat()
        try:
            with open(self.audit_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError as e:
            logger.error(f"[ExecutionLogger] Write failed: {e}")
