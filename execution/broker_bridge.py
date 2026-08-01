"""
execution/broker_bridge.py
==========================
INF-006 FIX: BrokerBridge no longer logs fake executions.

A generalized bridge for routing AI signals to live execution venues.
Supports Interactive Brokers (IBKR), MetaTrader 5 (MT5) via ZeroMQ, and CCXT for Crypto.

This module raises NotImplementedError for unimplemented brokers without
emitting misleading log messages that pollute the audit trail.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class BrokerNotImplementedError(NotImplementedError):
    """Raised when attempting to use an unimplemented broker method."""
    pass


class BrokerBridge:
    def __init__(self, broker: str = "MT5", config: Optional[Dict[str, Any]] = None):
        self.broker = broker.upper()
        self.config = config or {}
        self.connected = False

    def connect(self) -> bool:
        """Establish connection to the selected broker API."""
        if self.broker == "MT5":
            raise BrokerNotImplementedError(
                "MT5 broker bridge is not yet implemented. "
                "See trading/live_engine.py for the OANDA execution path."
            )
        elif self.broker == "IBKR":
            raise BrokerNotImplementedError(
                "IBKR broker bridge is not yet implemented."
            )
        elif self.broker == "OANDA":
            raise BrokerNotImplementedError(
                "Use the OANDA client in trading/live_engine.py directly."
            )
        else:
            raise ValueError(f"Unsupported broker: {self.broker}")

    def execute_order(self, symbol: str, side: str, lot_size: float,
                      limit_price: float = None) -> bool:
        """Route an order to the execution engine.

        NOTE: This stub does NOT log execution messages to avoid polluting
        the audit trail with fake fill records (INF-006).
        """
        if not self.connected:
            raise BrokerNotImplementedError(
                "BrokerBridge.execute_order() called but broker is not connected. "
                "This bridge is a stub — use the OANDA path in live_engine.py."
            )
        raise BrokerNotImplementedError(
            f"BrokerBridge.execute_order() is not implemented for {self.broker}. "
            f"Order: {side} {lot_size} {symbol}"
        )

    def get_positions(self) -> list:
        """Get current open positions from broker."""
        raise BrokerNotImplementedError(
            f"BrokerBridge.get_positions() is not implemented for {self.broker}."
        )

    def get_latency(self) -> Optional[int]:
        """Ping the broker to monitor execution latency."""
        raise BrokerNotImplementedError(
            f"BrokerBridge.get_latency() is not implemented for {self.broker}."
        )

    def is_connected(self) -> bool:
        return self.connected
