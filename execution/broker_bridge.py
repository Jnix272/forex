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

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

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
            if mt5 is None:
                raise ImportError("MetaTrader5 package is not installed.")
            
            # Initialize connection to MT5 terminal
            if not mt5.initialize():
                logger.error(f"mt5.initialize() failed, error code: {mt5.last_error()}")
                return False
            
            # Optional: Login to specific account if config is provided
            login = self.config.get("login")
            password = self.config.get("password")
            server = self.config.get("server")
            
            if login and password and server:
                authorized = mt5.login(login=login, password=password, server=server)
                if not authorized:
                    logger.error(f"mt5.login() failed, error code: {mt5.last_error()}")
                    mt5.shutdown()
                    return False
                    
            self.connected = True
            logger.info("Successfully connected to MetaTrader5.")
            return True
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
        """Route an order to the execution engine."""
        if not self.connected:
            raise BrokerNotImplementedError(
                "BrokerBridge.execute_order() called but broker is not connected. "
            )
            
        if self.broker == "MT5":
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Symbol {symbol} not found.")
                return False
                
            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    logger.error(f"symbol_select({symbol}) failed.")
                    return False
                    
            order_type = mt5.ORDER_TYPE_BUY if side.upper() == "BUY" else mt5.ORDER_TYPE_SELL
            
            if limit_price is not None:
                order_type = mt5.ORDER_TYPE_BUY_LIMIT if side.upper() == "BUY" else mt5.ORDER_TYPE_SELL_LIMIT
                price = limit_price
            else:
                price = mt5.symbol_info_tick(symbol).ask if side.upper() == "BUY" else mt5.symbol_info_tick(symbol).bid
                
            request = {
                "action": mt5.TRADE_ACTION_PENDING if limit_price else mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(lot_size),
                "type": order_type,
                "price": price,
                "deviation": 20,
                "magic": 234000,
                "comment": "AI Execution",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order failed, retcode={result.retcode}")
                return False
                
            logger.info(f"Order executed: {result.order}")
            return True
            
        raise BrokerNotImplementedError(
            f"BrokerBridge.execute_order() is not implemented for {self.broker}. "
            f"Order: {side} {lot_size} {symbol}"
        )
        
    def modify_order(self, ticket: int, stop_loss: float = None, take_profit: float = None) -> bool:
        """Modify an existing order or position in MT5."""
        if not self.connected:
            return False
            
        if self.broker == "MT5":
            position = mt5.positions_get(ticket=ticket)
            if not position:
                logger.error(f"Position {ticket} not found.")
                return False
                
            pos = position[0]
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "symbol": pos.symbol,
            }
            if stop_loss is not None:
                request["sl"] = stop_loss
            if take_profit is not None:
                request["tp"] = take_profit
                
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Modify order failed, retcode={result.retcode}")
                return False
            return True
            
        raise BrokerNotImplementedError(f"modify_order() not implemented for {self.broker}")

    def close_position(self, ticket: int) -> bool:
        """Close an existing position in MT5."""
        if not self.connected:
            return False
            
        if self.broker == "MT5":
            position = mt5.positions_get(ticket=ticket)
            if not position:
                logger.error(f"Position {ticket} not found.")
                return False
                
            pos = position[0]
            tick = mt5.symbol_info_tick(pos.symbol)
            
            order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": order_type,
                "position": pos.ticket,
                "price": price,
                "deviation": 20,
                "magic": 234000,
                "comment": "Close Position",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Close position failed, retcode={result.retcode}")
                return False
            return True
            
        raise BrokerNotImplementedError(f"close_position() not implemented for {self.broker}")

    def get_positions(self) -> list:
        """Get current open positions from broker."""
        if not self.connected:
            return []
            
        if self.broker == "MT5":
            positions = mt5.positions_get()
            if positions is None:
                return []
            
            return [
                {
                    "ticket": p.ticket,
                    "symbol": p.symbol,
                    "volume": p.volume,
                    "type": "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
                    "price_open": p.price_open,
                    "sl": p.sl,
                    "tp": p.tp,
                    "profit": p.profit,
                }
                for p in positions
            ]
            
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
