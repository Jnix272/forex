"""
execution/order_manager.py
==========================
INF-007 FIX: OrderManager now persists state to disk.

Manages order states, trailing stops, and take-profit logic dynamically.
State is persisted to a JSON file after every mutation so it survives restarts.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class OrderManager:
    def __init__(self, state_file: str = "logs/order_state.json"):
        self.state_file = Path(state_file)
        self.active_orders: dict = {}
        self.closed_orders: list = []
        self._load_state()

    def _load_state(self):
        """Restore state from disk on startup."""
        if self.state_file.exists():
            try:
                with open(self.state_file, encoding="utf-8") as f:
                    data = json.load(f)
                self.active_orders = data.get("active_orders", {})
                self.closed_orders = data.get("closed_orders", [])
                logger.info(
                    f"[OrderManager] Restored {len(self.active_orders)} active, "
                    f"{len(self.closed_orders)} closed orders from {self.state_file}"
                )
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"[OrderManager] Could not restore state: {e}")
                self.active_orders = {}
                self.closed_orders = []
        else:
            logger.info("[OrderManager] No existing state file - starting fresh")

    def _persist_state(self):
        """Atomically write current state to disk."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.state_file.with_suffix(".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "active_orders": self.active_orders,
                        "closed_orders": self.closed_orders[-500:],
                        "last_updated": datetime.now(UTC).isoformat(),
                    },
                    f,
                    indent=2,
                    default=str,
                )
            tmp_path.replace(self.state_file)
        except OSError as e:
            logger.error(f"[OrderManager] State persist failed: {e}")

    def register_trade(self, trade_id: str, symbol: str, side: str, entry_price: float, sl: float, tp: float):
        self.active_orders[trade_id] = {
            "symbol": symbol,
            "side": side,
            "entry": entry_price,
            "sl": sl,
            "tp": tp,
            "status": "OPEN",
            "opened_at": datetime.now(UTC).isoformat(),
        }
        self._persist_state()

    def close_trade(self, trade_id: str, exit_price: float, reason: str = "manual"):
        if trade_id not in self.active_orders:
            logger.warning(f"[OrderManager] Cannot close unknown trade: {trade_id}")
            return

        order = self.active_orders.pop(trade_id)
        order["status"] = "CLOSED"
        order["exit_price"] = exit_price
        order["close_reason"] = reason
        order["closed_at"] = datetime.now(UTC).isoformat()

        pip_mult = 0.0001 if "JPY" not in order["symbol"] else 0.01
        if order["side"] == "BUY":
            order["pnl_pips"] = (exit_price - order["entry"]) / pip_mult
        else:
            order["pnl_pips"] = (order["entry"] - exit_price) / pip_mult

        self.closed_orders.append(order)
        self._persist_state()
        return order

    def update_trailing_stop(self, trade_id: str, current_price: float, trail_pips: float = 10.0):
        if trade_id not in self.active_orders:
            return

        order = self.active_orders[trade_id]
        pip_mult = 0.0001 if "JPY" not in order["symbol"] else 0.01

        if order["side"] == "BUY":
            new_sl = current_price - (trail_pips * pip_mult)
            if new_sl > order["sl"]:
                order["sl"] = new_sl
                self._persist_state()
        elif order["side"] == "SELL":
            new_sl = current_price + (trail_pips * pip_mult)
            if new_sl < order["sl"]:
                order["sl"] = new_sl
                self._persist_state()

    def check_stops(self, trade_id: str, current_price: float) -> str | None:
        """Check if SL or TP hit. Returns 'sl', 'tp', or None."""
        if trade_id not in self.active_orders:
            return None

        order = self.active_orders[trade_id]
        if order["side"] == "BUY":
            if current_price <= order["sl"]:
                return "sl"
            if order["tp"] and current_price >= order["tp"]:
                return "tp"
        elif order["side"] == "SELL":
            if current_price >= order["sl"]:
                return "sl"
            if order["tp"] and current_price <= order["tp"]:
                return "tp"
        return None

    def get_active_orders(self, symbol: str | None = None) -> dict:
        if symbol is None:
            return dict(self.active_orders)
        return {k: v for k, v in self.active_orders.items() if v["symbol"] == symbol}

    def get_exposure(self, symbol: str | None = None) -> int:
        """Net exposure: +1 per BUY, -1 per SELL."""
        orders = self.get_active_orders(symbol)
        return sum(1 if v["side"] == "BUY" else -1 for v in orders.values())
