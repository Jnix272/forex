"""
execution/broker_bridge.py
==========================
INF-006 FIX: BrokerBridge no longer logs fake executions.

Venues:
  - MT5: MetaTrader5 package (orders, positions, latency)
  - IBKR: optional ``ib_insync`` (connect/orders/positions/latency)
  - OANDA: use trading/live_engine.py (not this bridge)

Disconnected calls fail closed (raise), never return empty/False silently.
"""

from __future__ import annotations

import logging
import time
from typing import Any

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

try:
    from ib_insync import IB, Forex, LimitOrder, MarketOrder, StopOrder

    _IB_INSYNC = True
except ImportError:
    IB = None  # type: ignore[misc, assignment]
    Forex = None  # type: ignore[misc, assignment]
    MarketOrder = None  # type: ignore[misc, assignment]
    LimitOrder = None  # type: ignore[misc, assignment]
    StopOrder = None  # type: ignore[misc, assignment]
    _IB_INSYNC = False

logger = logging.getLogger(__name__)


class BrokerNotImplementedError(NotImplementedError):
    """Raised when attempting to use an unimplemented broker method."""


class BrokerNotConnectedError(RuntimeError):
    """Raised when a live broker call is made without an active connection."""


class BrokerBridge:
    def __init__(self, broker: str = "MT5", config: dict[str, Any] | None = None):
        self.broker = broker.upper()
        self.config = config or {}
        self.connected = False
        self._ib: Any = None

    def _require_connected(self, op: str) -> None:
        if not self.connected:
            raise BrokerNotConnectedError(
                f"BrokerBridge.{op}() requires an active connection (broker={self.broker}). Call connect() first."
            )

    def connect(self) -> bool:
        """Establish connection to the selected broker API."""
        if self.broker == "MT5":
            if mt5 is None:
                raise ImportError("MetaTrader5 package is not installed.")

            if not mt5.initialize():
                logger.error(f"mt5.initialize() failed, error code: {mt5.last_error()}")
                return False

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

        if self.broker == "IBKR":
            if not _IB_INSYNC:
                raise ImportError("IBKR requires ib_insync. Install with: pip install ib_insync")
            host = str(self.config.get("host", "127.0.0.1"))
            port = int(self.config.get("port", 7497))  # 7497 paper, 7496 live TWS
            client_id = int(self.config.get("client_id", 1))
            self._ib = IB()
            try:
                self._ib.connect(host, port, clientId=client_id, timeout=10)
            except Exception as e:
                self._ib = None
                logger.error(f"IBKR connect failed: {e}")
                return False
            if not self._ib.isConnected():
                self._ib = None
                return False
            self.connected = True
            logger.info("Successfully connected to IBKR via ib_insync (%s:%s).", host, port)
            return True

        if self.broker == "OANDA":
            raise BrokerNotImplementedError("Use the OANDA client in trading/live_engine.py directly.")
        raise ValueError(f"Unsupported broker: {self.broker}")

    def disconnect(self) -> None:
        """Tear down broker connection."""
        if self.broker == "MT5" and mt5 is not None and self.connected:
            try:
                mt5.shutdown()
            except Exception:
                pass
        if self.broker == "IBKR" and self._ib is not None:
            try:
                self._ib.disconnect()
            except Exception:
                pass
            self._ib = None
        self.connected = False

    def execute_order(
        self,
        symbol,
        side: str,
        lot_size: float,
        limit_price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> bool:
        """Route an order to the execution engine."""
        self._require_connected("execute_order")

        if self.broker == "MT5":
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Symbol {symbol} not found.")
                return False

            if not symbol_info.visible and not mt5.symbol_select(symbol, True):
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
            if stop_loss is not None:
                request["sl"] = float(stop_loss)
            if take_profit is not None:
                request["tp"] = float(take_profit)

            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order failed, retcode={result.retcode}")
                return False

            logger.info(f"Order executed: {result.order}")
            return True

        if self.broker == "IBKR":
            contract = self._ibkr_fx_contract(symbol)
            action = "BUY" if side.upper() == "BUY" else "SELL"
            # IB FX size is usually in base-currency units (not MT5 lots).
            qty = float(lot_size)
            if limit_price is not None:
                order = LimitOrder(action, qty, float(limit_price))
            else:
                order = MarketOrder(action, qty)
            trade = self._ib.placeOrder(contract, order)
            if stop_loss is not None or take_profit is not None:
                # Deliver SL/TP as child orders attached to the parent, so they
                # survive engine restarts (fail-closed protection on IBKR).
                opp = "SELL" if action == "BUY" else "BUY"
                child_orders = []
                if take_profit is not None:
                    tp = LimitOrder(opp, qty, float(take_profit))
                    tp.parent = order
                    child_orders.append(tp)
                if stop_loss is not None:
                    sl = StopOrder(opp, qty, float(stop_loss))
                    sl.parent = order
                    child_orders.append(sl)
                for child in child_orders:
                    self._ib.placeOrder(contract, child)
                    self._ib.sleep(0.2)
            self._ib.sleep(0.5)
            status = str(getattr(trade.orderStatus, "status", "") or "")
            if status in {"Cancelled", "Inactive", "ApiCancelled"}:
                logger.error(f"IBKR order failed: status={status}")
                return False
            logger.info(f"IBKR order placed: {action} {qty} {symbol} status={status or 'Submitted'}")
            return True

        raise BrokerNotImplementedError(
            f"BrokerBridge.execute_order() is not implemented for {self.broker}. Order: {side} {lot_size} {symbol}"
        )

    def modify_order(self, ticket: int, stop_loss: float | None = None, take_profit: float | None = None) -> bool:
        """Modify an existing order or position."""
        self._require_connected("modify_order")

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

        if self.broker == "IBKR":
            trade = self._ibkr_trade_by_id(ticket)
            if trade is None:
                logger.error(f"IBKR trade/order {ticket} not found.")
                return False
            order = trade.order
            if stop_loss is not None:
                # Attach/replace stop via order attributes when supported.
                order.auxPrice = float(stop_loss)
            if take_profit is not None:
                order.lmtPrice = float(take_profit)
            self._ib.placeOrder(trade.contract, order)
            self._ib.sleep(0.3)
            return True

        raise BrokerNotImplementedError(f"modify_order() not implemented for {self.broker}")

    def close_position(self, ticket: int) -> bool:
        """Close an existing position."""
        self._require_connected("close_position")

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

        if self.broker == "IBKR":
            for pos in self._ib.positions():
                con_id = int(getattr(pos.contract, "conId", 0) or 0)
                if con_id != int(ticket):
                    continue
                qty = float(pos.position)
                if qty == 0:
                    continue
                action = "SELL" if qty > 0 else "BUY"
                order = MarketOrder(action, abs(qty))
                self._ib.placeOrder(pos.contract, order)
                self._ib.sleep(0.5)
                return True
            logger.error(f"IBKR position {ticket} not found.")
            return False

        raise BrokerNotImplementedError(f"close_position() not implemented for {self.broker}")

    def get_positions(self) -> list:
        """Get current open positions from broker."""
        self._require_connected("get_positions")

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

        if self.broker == "IBKR":
            out = []
            for pos in self._ib.positions():
                qty = float(pos.position)
                if qty == 0:
                    continue
                sym = getattr(pos.contract, "symbol", "") or getattr(pos.contract, "localSymbol", "")
                out.append(
                    {
                        "ticket": int(getattr(pos.contract, "conId", 0) or 0),
                        "symbol": str(sym).replace(".", ""),
                        "volume": abs(qty),
                        "type": "BUY" if qty > 0 else "SELL",
                        "price_open": float(getattr(pos, "avgCost", 0.0) or 0.0),
                        "sl": None,
                        "tp": None,
                        "profit": None,
                    }
                )
            return out

        raise BrokerNotImplementedError(f"BrokerBridge.get_positions() is not implemented for {self.broker}.")

    def get_latency(self) -> int | None:
        """Round-trip latency to the broker in milliseconds."""
        self._require_connected("get_latency")

        if self.broker == "MT5":
            symbol = str(self.config.get("latency_symbol", "EURUSD"))
            # Prefer terminal-reported ping when present.
            try:
                info = mt5.terminal_info()
                for attr in ("ping", "pinglast", "ping_last"):
                    ping = getattr(info, attr, None) if info is not None else None
                    if ping is not None and int(ping) >= 0:
                        return int(ping)
            except Exception:
                pass
            t0 = time.perf_counter()
            tick = mt5.symbol_info_tick(symbol)
            dt_ms = int((time.perf_counter() - t0) * 1000)
            if tick is None:
                return None
            return max(0, dt_ms)

        if self.broker == "IBKR":
            t0 = time.perf_counter()
            try:
                # reqCurrentTime is a lightweight server round-trip.
                self._ib.reqCurrentTime()
                self._ib.sleep(0.05)
            except Exception as e:
                logger.error(f"IBKR latency probe failed: {e}")
                return None
            return max(0, int((time.perf_counter() - t0) * 1000))

        raise BrokerNotImplementedError(f"BrokerBridge.get_latency() is not implemented for {self.broker}.")

    def get_bid_ask(self, symbol: str) -> tuple[float, float]:
        """Return (bid, ask) for ``symbol``. Fail-closed when disconnected."""
        self._require_connected("get_bid_ask")

        if self.broker == "MT5":
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                raise RuntimeError(f"MT5: no tick for {symbol}")
            return float(tick.bid), float(tick.ask)

        if self.broker == "IBKR":
            contract = self._ibkr_fx_contract(symbol)
            self._ib.qualifyContracts(contract)
            tickers = self._ib.reqTickers(contract)
            self._ib.sleep(0.3)
            if not tickers:
                raise RuntimeError(f"IBKR: no ticker for {symbol}")
            t = tickers[0]
            bid = float(getattr(t, "bid", 0) or 0)
            ask = float(getattr(t, "ask", 0) or 0)
            if bid <= 0 or ask <= 0:
                mid = float(getattr(t, "close", 0) or getattr(t, "last", 0) or 0)
                if mid <= 0:
                    raise RuntimeError(f"IBKR: empty bid/ask for {symbol}")
                return mid - 1e-5, mid + 1e-5
            return bid, ask

        raise BrokerNotImplementedError(f"BrokerBridge.get_bid_ask() is not implemented for {self.broker}.")

    def get_account_equity(self) -> float:
        """Account equity / NAV when the venue exposes it."""
        self._require_connected("get_account_equity")
        if self.broker == "MT5":
            info = mt5.account_info()
            if info is None:
                raise RuntimeError("MT5: account_info() returned None")
            return float(getattr(info, "equity", 0) or 0)
        if self.broker == "IBKR":
            vals = self._ib.accountValues()
            for v in vals:
                if str(getattr(v, "tag", "")) == "NetLiquidation" and str(getattr(v, "currency", "")) in (
                    "",
                    "USD",
                    "BASE",
                ):
                    try:
                        return float(v.value)
                    except Exception:
                        continue
            raise RuntimeError("IBKR: NetLiquidation not found in accountValues")
        raise BrokerNotImplementedError(f"BrokerBridge.get_account_equity() is not implemented for {self.broker}.")

    def is_connected(self) -> bool:
        if self.broker == "IBKR" and self._ib is not None:
            try:
                self.connected = bool(self._ib.isConnected())
            except Exception:
                self.connected = False
        return self.connected

    @staticmethod
    def _ibkr_fx_contract(symbol: str):
        """Map EURUSD / EUR.USD / EUR_USD → ib_insync Forex contract."""
        raw = str(symbol).upper().replace(".", "").replace("_", "").replace("/", "")
        if len(raw) != 6:
            raise ValueError(f"IBKR FX symbol must be 6 letters, got {symbol!r}")
        return Forex(raw)

    def _ibkr_trade_by_id(self, ticket: int):
        for trade in list(self._ib.openTrades()) + list(self._ib.trades()):
            oid = int(getattr(trade.order, "orderId", -1) or -1)
            if oid == int(ticket):
                return trade
        return None
