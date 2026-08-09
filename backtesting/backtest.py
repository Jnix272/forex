"""
backtesting/backtest.py
=======================
Production-grade backtesting engine for forex scaling models.

Accounts for:
  - Bid-ask spread (the Golden Rule)
  - Commission per lot
  - Slippage (execution lag)
  - Market impact (Square Root Law for large orders)
  - Scaling in/out (partial fills and average entry tracking)
  - Dynamic stop-loss management

The primary reason most scalping models fail in production is that they
don't account for execution lag and real transaction costs. This engine
enforces realistic execution at every step.
"""

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
import pandas as pd

try:
    from numba import njit
    _NUMBA_OK = True
except ImportError:  # pragma: no cover - optional / version-skew fallback
    _NUMBA_OK = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        """No-op decorator when Numba is unavailable (e.g. NumPy too new)."""
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def _wrap(fn):
            return fn

        return _wrap


class ScalingAction(IntEnum):
    HOLD = 0
    OPEN_LONG = 1
    OPEN_SHORT = 2
    SCALE_IN_25 = 3
    SCALE_IN_50 = 4
    SCALE_IN_100 = 5
    SCALE_OUT_25 = 6
    SCALE_OUT_50 = 7
    SCALE_OUT_100 = 8
    CLOSE_ALL = 9


# ─────────────────────────────────────────────────────────────────────────────
# TRADE RECORDS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    """Record of a single trade execution."""
    trade_id: int
    entry_time: pd.Timestamp
    entry_price: float
    entry_lots: float
    direction: int           # +1 = long, -1 = short
    stop_loss: float
    take_profit: float
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_lots: float | None = None
    pnl_pips: float = 0.0
    gross_pnl_usd: float = 0.0
    pnl_usd: float = 0.0
    commission: float = 0.0
    slippage_pips: float = 0.0
    exit_reason: str = ""    # 'stop_loss', 'take_profit', 'signal', 'scale_out', 'eod'
    scale_additions: list[dict] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# BACKTESTING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class ForexScalingBacktest:
    """
    Event-driven backtesting engine for forex scaling strategies.

    Supports:
      - Scale-in (pyramiding and martingale) at multiple price levels
      - Scale-out (partial profit taking) at predefined targets
      - Dynamic stop-loss trailing
      - Realistic transaction cost modeling

    Usage
    -----
        bt = ForexScalingBacktest(bars_df, signals_df)
        results = bt.run()
        bt.print_performance()
        equity_curve = bt.get_equity_curve()
    """

    def __init__(
        self,
        bars: pd.DataFrame,
        signals: pd.DataFrame,
        initial_equity: float = 10_000.0,
        lot_size: float = 10_000.0,
        commission_per_lot: float = 3.5,
        slippage_pips: float = 0.5,
        pip_size: float = 0.0001,
        pip_value_per_lot: float = 1.0,
        max_lots: float = 3.0,
        execution_delay_bars: int = 1,    # Simulate 1-bar execution delay
        use_bid_ask: bool = True,          # Golden Rule: trade on bid/ask, not mid
        daily_volume_lots: float = 500.0,
        apply_market_impact: bool = True,
        max_drawdown_limit: float = 0.20,
        min_spread_clamp: float = 0.00005,
        max_spread_clamp: float = 0.0050,
    ):
        """
        Parameters
        ----------
        bars     : OHLCV bars with 'open','high','low','close','bid_close','ask_close','spread_avg'
        signals  : DataFrame with columns: 'action' (ScalingAction int), 'stop_loss', 'take_profit'
        """
        # Validation
        required_cols = ["open", "high", "low", "close"]
        missing = [c for c in required_cols if c not in bars.columns]
        if missing:
            raise ValueError(f"bars DataFrame missing required columns: {missing}")

        self.bars = bars
        self.signals = signals.reindex(bars.index)
        # Actions are events, not state. Do not forward-fill them, or one model
        # signal can reopen a new trade every bar after a stop/TP closes.
        if "action" in self.signals.columns:
            self.signals["action"] = self.signals["action"].fillna(ScalingAction.HOLD.value)
        # PIPE-009: Only forward-fill SL/TP within continuous position windows.
        # Reset to NaN whenever direction changes to prevent stale values bleeding.
        for col in ("lots", "stop_loss", "take_profit"):
            if col in self.signals.columns:
                if "action" in self.signals.columns:
                    # Detect direction changes: wherever action flips sign, reset ffill
                    action_series = self.signals["action"]
                    direction_change = (action_series.diff().abs() > 0) & (action_series != 0)
                    # Set values to NaN at direction change points before ffill
                    masked = self.signals[col].copy()
                    masked[direction_change] = float("nan")
                    self.signals[col] = masked.ffill()
                else:
                    self.signals[col] = self.signals[col].ffill()

        self.initial_equity = initial_equity
        self.equity = initial_equity
        self.peak_equity = initial_equity
        self.lot_size = lot_size
        self.commission_per_lot = commission_per_lot
        self.slippage_pips = slippage_pips
        self.pip_size = pip_size
        self.pip_value_per_lot = pip_value_per_lot
        self.max_lots = max_lots
        self.execution_delay = execution_delay_bars
        self.use_bid_ask = use_bid_ask
        self.daily_volume_lots = daily_volume_lots
        self.apply_market_impact = apply_market_impact
        self.max_drawdown_limit = max_drawdown_limit
        self.min_spread_clamp = min_spread_clamp
        self.max_spread_clamp = max_spread_clamp

        # State
        self.position: float = 0.0         # Current net lots
        self.avg_entry_price: float = 0.0   # VWAP entry price
        self.current_stop: float = 0.0
        self.current_tp: float = 0.0
        self.holding_bars: int = 0

        # Records
        self.trades: list[Trade] = []
        self.equity_curve: list[float] = []
        self.daily_pnl: list[float] = []
        self._trade_counter: int = 0
        self._open_trade: Trade | None = None

        # Pre-extract arrays for fast lookup
        self._arr_open = self.bars["open"].values
        self._arr_high = self.bars["high"].values
        self._arr_low = self.bars["low"].values
        self._arr_close = self.bars["close"].values
        self._arr_ts = self.bars.index.values
        if self.use_bid_ask and "bid_close" in self.bars.columns and "ask_close" in self.bars.columns:
            self._arr_bid_close = self.bars["bid_close"].values
            self._arr_ask_close = self.bars["ask_close"].values
        else:
            self._arr_bid_close = None
            self._arr_ask_close = None
        if "spread_avg" in self.bars.columns:
            self._arr_spread = self.bars["spread_avg"].values
        else:
            self._arr_spread = None

        self._sig_action = self.signals["action"].values if "action" in self.signals.columns else np.zeros(len(self.bars))
        self._sig_sl = self.signals["stop_loss"].values if "stop_loss" in self.signals.columns else np.zeros(len(self.bars))
        self._sig_tp = self.signals["take_profit"].values if "take_profit" in self.signals.columns else np.zeros(len(self.bars))
        self._sig_lots = self.signals["lots"].values if "lots" in self.signals.columns else np.full(len(self.bars), 0.1)

    # ── Price helpers ────────────────────────────────────────────────────────

    def _session_spread_mult(self, idx: int) -> float:
        """Shared session→spread mult (LABEL_REGIME / session_utils SoT)."""
        try:
            from trading.session_utils import session_spread_mult
        except Exception:
            return 1.0
        # Prefer explicit session columns when present on bars
        sess = None
        asia_london = False
        london_ny = False
        try:
            if "session_label" in self.bars.columns:
                sess = self.bars["session_label"].iloc[idx]
            if "asia_london" in self.bars.columns:
                asia_london = bool(float(self.bars["asia_london"].iloc[idx]) > 0)
            if "london_ny" in self.bars.columns:
                london_ny = bool(float(self.bars["london_ny"].iloc[idx]) > 0)
        except Exception:
            pass
        now = None
        if sess is None and not asia_london and not london_ny:
            try:
                ts = self.bars.index[idx]
                if hasattr(ts, "to_pydatetime"):
                    now = ts.to_pydatetime()
            except Exception:
                now = None
        return float(session_spread_mult(
            sess, asia_london=asia_london, london_ny=london_ny, now=now,
        ))

    def _get_execution_price(self, idx: int, direction: int, lots: float, is_close: bool = False) -> float:
        """
        Realistic execution price including:
          - Spread: buys at ask, sells at bid
          - Slippage: additional friction from execution lag
          - Market impact: Square Root Law for large orders
        """
        # Use bid/ask if available (Golden Rule)
        if self._arr_bid_close is not None and self._arr_ask_close is not None:
            base_price = self._arr_ask_close[idx] if direction > 0 else self._arr_bid_close[idx]
            spread = self._arr_spread[idx] if self._arr_spread is not None else (self._arr_ask_close[idx] - self._arr_bid_close[idx])
        else:
            # Synthetic / missing book: scale flat default by shared session→spread mult
            if self._arr_spread is not None:
                spread = self._arr_spread[idx]
            else:
                spread = 0.0001 * self._session_spread_mult(idx)
            spread_half = spread / 2
            base_price = self._arr_close[idx] + direction * spread_half

        # Slippage (session-aware when using shared calibrator vocabulary)
        slip_pips = self.slippage_pips * self._session_spread_mult(idx)
        slippage = direction * slip_pips * self.pip_size
        price = base_price + slippage

        # Market impact (Square Root Law: impact ∝ spread × √(lots / ADV))
        if self.apply_market_impact and lots > 0 and not is_close:
            # Clamp spread between 0.5 pips and 50 pips to avoid extreme impact values
            spread = min(max(float(spread), self.min_spread_clamp), self.max_spread_clamp)
            impact_fraction = spread * np.sqrt(abs(lots) / max(self.daily_volume_lots, 1.0))
            price += direction * impact_fraction

        return float(price)

    def _compute_cost(self, lots: float) -> float:
        """Commission per round-turn."""
        return abs(lots) * self.commission_per_lot

    # ── Trade execution ──────────────────────────────────────────────────────

    def _open_position(self, idx: int, direction: int, lots: float,
                       stop_loss: float, take_profit: float) -> Trade:
        """Open a new position."""
        if lots > self.max_lots:
            lots = self.max_lots

        exec_price = self._get_execution_price(idx, direction, lots)
        cost = self._compute_cost(lots)
        self.equity -= cost

        self.position = direction * lots
        self.avg_entry_price = exec_price
        self.current_stop = stop_loss
        self.current_tp = take_profit
        self.holding_bars = 0

        self._trade_counter += 1
        trade = Trade(
            trade_id=self._trade_counter,
            entry_time=self.bars.index[idx],
            entry_price=exec_price,
            entry_lots=lots,
            direction=direction,
            stop_loss=stop_loss,
            take_profit=take_profit,
            commission=cost,
            slippage_pips=self.slippage_pips,
        )
        self._open_trade = trade
        return trade

    def _scale_in(self, idx: int, lots: float):
        """Add to existing position (pyramid or martingale)."""
        if self.position == 0:
            return  # no open position to scale into
        if abs(self.position) + lots > self.max_lots:
            lots = self.max_lots - abs(self.position)
        if lots <= 0:
            return

        direction = int(np.sign(self.position))
        exec_price = self._get_execution_price(idx, direction, lots)
        cost = self._compute_cost(lots)
        self.equity -= cost

        # Update weighted average entry
        total_lots = abs(self.position) + lots
        self.avg_entry_price = (
            abs(self.position) * self.avg_entry_price + lots * exec_price
        ) / total_lots

        self.position += direction * lots

        if self._open_trade:
            self._open_trade.scale_additions.append({
                "time": self.bars.index[idx],
                "price": exec_price,
                "lots": lots,
                "cost": cost,
            })
            self._open_trade.commission += cost

    def _close_position(self, idx: int, fraction: float = 1.0,
                        exit_reason: str = "signal", override_price: float = None) -> float:
        """Close all or part of position. Returns realised P&L in USD."""
        if self.position == 0:
            return 0.0

        close_lots = abs(self.position) * fraction
        direction = int(np.sign(self.position))
        
        # Use override_price if provided, otherwise compute execution price
        if override_price is not None:
            exec_price = override_price
        else:
            exec_price = self._get_execution_price(idx, -direction, close_lots, is_close=True)
        
        cost = self._compute_cost(close_lots)

        pnl_pips = direction * (exec_price - self.avg_entry_price) / self.pip_size
        gross_pnl_usd = pnl_pips * self.pip_value_per_lot * close_lots
        net_cash_pnl = gross_pnl_usd - cost
        self.equity += net_cash_pnl

        self.position -= direction * close_lots
        if self._open_trade:
            self._open_trade.gross_pnl_usd += gross_pnl_usd
            self._open_trade.commission += cost
            self._open_trade.pnl_usd = self._open_trade.gross_pnl_usd - self._open_trade.commission
        if abs(self.position) < 0.001:
            self.position = 0.0
            if self._open_trade:
                self._open_trade.exit_time = self.bars.index[idx]
                self._open_trade.exit_price = exec_price
                self._open_trade.exit_lots = close_lots
                self._open_trade.pnl_pips = pnl_pips
                self._open_trade.exit_reason = exit_reason
                self.trades.append(self._open_trade)
                self._open_trade = None
                self.holding_bars = 0

        return net_cash_pnl

    # ── Stop/TP checks ───────────────────────────────────────────────────────

    def _check_stops(self, idx: int) -> bool:
        """
        Check if price has hit stop-loss or take-profit during current bar.
        Returns True if position was closed.
        """
        if self.position == 0:
            return False

        direction = np.sign(self.position)

        # Stop-loss check
        if direction > 0 and self._arr_low[idx] <= self.current_stop:
            exec_px = self.current_stop - self.slippage_pips * self.pip_size
            self._close_position(idx, fraction=1.0, exit_reason="stop_loss", override_price=exec_px)
            return True
        if direction < 0 and self._arr_high[idx] >= self.current_stop:
            exec_px = self.current_stop + self.slippage_pips * self.pip_size
            self._close_position(idx, fraction=1.0, exit_reason="stop_loss", override_price=exec_px)
            return True

        # Take-profit check
        if direction > 0 and self._arr_high[idx] >= self.current_tp:
            exec_px = self.current_tp - self.slippage_pips * self.pip_size
            self._close_position(idx, fraction=0.5, exit_reason="scale_out_tp", override_price=exec_px)
            # Trail stop to breakeven after partial profit
            if self.position != 0:
                self.current_stop = max(self.current_stop, self.avg_entry_price)
            return False  # Position still open (partially)

        if direction < 0 and self._arr_low[idx] <= self.current_tp:
            exec_px = self.current_tp + self.slippage_pips * self.pip_size
            self._close_position(idx, fraction=0.5, exit_reason="scale_out_tp", override_price=exec_px)
            if self.position != 0:
                self.current_stop = min(self.current_stop, self.avg_entry_price)
            return False

        return False

    # ── Numba-accelerated core loop ────────────────────────────────────────────

    @staticmethod
    @njit(cache=True)
    def _run_core_numba(
        arr_close, arr_high, arr_low, arr_bid, arr_ask, arr_spread, arr_ts,
        sig_action, sig_sl, sig_tp, sig_lots,
        n_bars, execution_delay, lot_size, commission_per_lot, slippage_pips,
        pip_size, pip_value_per_lot, max_lots, max_drawdown_limit,
        initial_equity, min_spread_clamp, max_spread_clamp, daily_volume_lots,
        apply_market_impact, use_bid_ask,
    ):
        """Numba-accelerated backtest core. Returns pre-allocated result arrays + equity_curve."""
        res_eq = np.empty(n_bars, dtype=np.float64)
        res_unreal = np.empty(n_bars, dtype=np.float64)
        res_total = np.empty(n_bars, dtype=np.float64)
        res_pos = np.empty(n_bars, dtype=np.float64)
        res_dd = np.empty(n_bars, dtype=np.float64)
        res_hold = np.empty(n_bars, dtype=np.int32)
        equity_curve = np.empty(n_bars, dtype=np.float64)

        equity = initial_equity
        peak_equity = initial_equity
        position = 0.0
        avg_entry_price = 0.0
        current_stop = 0.0
        current_tp = 0.0
        holding_bars = 0
        n_valid = 0

        for i in range(n_bars):
            signal_idx = i - execution_delay
            if signal_idx >= len(sig_action):
                break

            if signal_idx < 0:
                action = 0
                stop_loss = 0.0
                take_profit = 0.0
                lots_to_trade = 0.0
            else:
                action = int(sig_action[signal_idx])
                stop_loss = float(sig_sl[signal_idx])
                take_profit = float(sig_tp[signal_idx])
                lots_to_trade = float(sig_lots[signal_idx])

            # Check stops
            if position != 0:
                direction = 0.0
                if position > 0:
                    direction = 1.0
                elif position < 0:
                    direction = -1.0

                # Only check stops if current_stop is not NaN (matches original behavior)
                if not np.isnan(current_stop):
                    if direction > 0 and arr_low[i] <= current_stop:
                        # Close on stop loss — use bid price (selling long)
                        if avg_entry_price != 0.0:
                            close_lots = abs(position)
                            exec_price = current_stop - slippage_pips * pip_size
                            pnl_pips = direction * (exec_price - avg_entry_price) / pip_size
                            gross_pnl = pnl_pips * pip_value_per_lot * close_lots
                            cost = close_lots * commission_per_lot
                            equity += gross_pnl - cost
                        position = 0.0
                        avg_entry_price = 0.0
                        holding_bars = 0

                    elif direction < 0 and arr_high[i] >= current_stop:
                        # Close on stop loss — use ask price (buying short)
                        if avg_entry_price != 0.0:
                            close_lots = abs(position)
                            exec_price = current_stop + slippage_pips * pip_size
                            pnl_pips = direction * (exec_price - avg_entry_price) / pip_size
                            gross_pnl = pnl_pips * pip_value_per_lot * close_lots
                            cost = close_lots * commission_per_lot
                            equity += gross_pnl - cost
                        position = 0.0
                        avg_entry_price = 0.0
                        holding_bars = 0

                # Only check TP if current_tp is not NaN
                if position != 0 and not np.isnan(current_tp):
                    if direction > 0 and arr_high[i] >= current_tp:
                        close_lots = abs(position) * 0.5
                        # Partial close at TP — use bid price (selling long)
                        if avg_entry_price != 0.0:
                            exec_price = current_tp - slippage_pips * pip_size
                            pnl_pips = direction * (exec_price - avg_entry_price) / pip_size
                            gross_pnl = pnl_pips * pip_value_per_lot * close_lots
                            cost = close_lots * commission_per_lot
                            equity += gross_pnl - cost
                        position -= direction * close_lots
                        current_stop = max(current_stop, avg_entry_price)
                        if abs(position) < 0.001:
                            position = 0.0
                            holding_bars = 0

                    elif direction < 0 and arr_low[i] <= current_tp:
                        close_lots = abs(position) * 0.5
                        if avg_entry_price != 0.0:
                            exec_price = current_tp + slippage_pips * pip_size
                            pnl_pips = direction * (exec_price - avg_entry_price) / pip_size
                            gross_pnl = pnl_pips * pip_value_per_lot * close_lots
                            cost = close_lots * commission_per_lot
                            equity += gross_pnl - cost
                        position -= direction * close_lots
                        current_stop = min(current_stop, avg_entry_price)
                        if abs(position) < 0.001:
                            position = 0.0
                            holding_bars = 0

            # Execute signal
            if position == 0:
                if action == 1:  # OPEN_LONG
                    if lots_to_trade > max_lots:
                        lots_to_trade = max_lots
                    # Execution price
                    if use_bid_ask:
                        base_price = arr_ask[i]
                        spread = arr_spread[i] if arr_spread[i] > 0 else (arr_ask[i] - arr_bid[i])
                    else:
                        spread = arr_spread[i] if arr_spread[i] > 0 else 0.0001
                        base_price = arr_close[i] + 0.5 * spread
                    slippage = 1.0 * slippage_pips * pip_size
                    price = base_price + slippage
                    impact = 0.0
                    if apply_market_impact and lots_to_trade > 0:
                        spread_c = min(max(spread, min_spread_clamp), max_spread_clamp)
                        impact = spread_c * np.sqrt(lots_to_trade / max(daily_volume_lots, 1.0))
                        price += 1.0 * impact
                    cost = lots_to_trade * commission_per_lot
                    equity -= cost
                    position = 1.0 * lots_to_trade
                    avg_entry_price = price
                    current_stop = stop_loss
                    current_tp = take_profit
                    holding_bars = 0

                elif action == 2:  # OPEN_SHORT
                    if lots_to_trade > max_lots:
                        lots_to_trade = max_lots
                    if use_bid_ask:
                        base_price = arr_bid[i]
                        spread = arr_spread[i] if arr_spread[i] > 0 else (arr_ask[i] - arr_bid[i])
                    else:
                        spread = arr_spread[i] if arr_spread[i] > 0 else 0.0001
                        base_price = arr_close[i] - 0.5 * spread
                    slippage = -1.0 * slippage_pips * pip_size
                    price = base_price + slippage
                    impact = 0.0
                    if apply_market_impact and lots_to_trade > 0:
                        spread_c = min(max(spread, min_spread_clamp), max_spread_clamp)
                        impact = spread_c * np.sqrt(lots_to_trade / max(daily_volume_lots, 1.0))
                        price += -1.0 * impact
                    cost = lots_to_trade * commission_per_lot
                    equity -= cost
                    position = -1.0 * lots_to_trade
                    avg_entry_price = price
                    current_stop = stop_loss
                    current_tp = take_profit
                    holding_bars = 0

            elif position != 0:
                direction = 1.0 if position > 0 else -1.0

                if action == 3:  # SCALE_IN_25
                    lots = lots_to_trade * 0.25
                    if abs(position) + lots > max_lots:
                        lots = max_lots - abs(position)
                    if lots > 0:
                        if use_bid_ask:
                            base_price = arr_ask[i] if direction > 0 else arr_bid[i]
                            spread = arr_spread[i] if arr_spread[i] > 0 else (arr_ask[i] - arr_bid[i])
                        else:
                            spread = arr_spread[i] if arr_spread[i] > 0 else 0.0001
                            base_price = arr_close[i] + direction * 0.5 * spread
                        slippage = direction * slippage_pips * pip_size
                        price = base_price + slippage

                        if apply_market_impact:
                            price += direction * (min(max(arr_spread[i] if arr_spread[i] > 0 else 0.0001, min_spread_clamp), max_spread_clamp) * np.sqrt(lots / max(daily_volume_lots, 1.0)))
                        cost = lots * commission_per_lot
                        equity -= cost
                        total_lots = abs(position) + lots
                        avg_entry_price = (abs(position) * avg_entry_price + lots * price) / total_lots
                        position += direction * lots

                elif action == 4:  # SCALE_IN_50
                    lots = lots_to_trade * 0.50
                    if abs(position) + lots > max_lots:
                        lots = max_lots - abs(position)
                    if lots > 0:
                        if use_bid_ask:
                            base_price = arr_ask[i] if direction > 0 else arr_bid[i]
                            spread = arr_spread[i] if arr_spread[i] > 0 else (arr_ask[i] - arr_bid[i])
                        else:
                            spread = arr_spread[i] if arr_spread[i] > 0 else 0.0001
                            base_price = arr_close[i] + direction * 0.5 * spread
                        slippage = direction * slippage_pips * pip_size
                        price = base_price + slippage

                        if apply_market_impact:
                            price += direction * (min(max(arr_spread[i] if arr_spread[i] > 0 else 0.0001, min_spread_clamp), max_spread_clamp) * np.sqrt(lots / max(daily_volume_lots, 1.0)))
                        cost = lots * commission_per_lot
                        equity -= cost
                        total_lots = abs(position) + lots
                        avg_entry_price = (abs(position) * avg_entry_price + lots * price) / total_lots
                        position += direction * lots

                elif action == 5:  # SCALE_IN_100
                    lots = lots_to_trade * 1.0
                    if abs(position) + lots > max_lots:
                        lots = max_lots - abs(position)
                    if lots > 0:
                        if use_bid_ask:
                            base_price = arr_ask[i] if direction > 0 else arr_bid[i]
                            spread = arr_spread[i] if arr_spread[i] > 0 else (arr_ask[i] - arr_bid[i])
                        else:
                            spread = arr_spread[i] if arr_spread[i] > 0 else 0.0001
                            base_price = arr_close[i] + direction * 0.5 * spread
                        slippage = direction * slippage_pips * pip_size
                        price = base_price + slippage

                        if apply_market_impact:
                            price += direction * (min(max(arr_spread[i] if arr_spread[i] > 0 else 0.0001, min_spread_clamp), max_spread_clamp) * np.sqrt(lots / max(daily_volume_lots, 1.0)))
                        cost = lots * commission_per_lot
                        equity -= cost
                        total_lots = abs(position) + lots
                        avg_entry_price = (abs(position) * avg_entry_price + lots * price) / total_lots
                        position += direction * lots

                elif action == 6:  # SCALE_OUT_25
                    close_lots = abs(position) * 0.25
                    if use_bid_ask:
                        base_price = arr_bid[i] if direction > 0 else arr_ask[i]
                    else:
                        spread = arr_spread[i] if arr_spread[i] > 0 else 0.0001
                        base_price = arr_close[i] - direction * 0.5 * spread
                    slippage = -direction * slippage_pips * pip_size
                    price = base_price + slippage

                    cost = close_lots * commission_per_lot
                    if avg_entry_price != 0.0:
                        pnl_pips = direction * (price - avg_entry_price) / pip_size
                        gross_pnl = pnl_pips * pip_value_per_lot * close_lots
                        equity += gross_pnl - cost
                    position -= direction * close_lots
                    if abs(position) < 0.001:
                        position = 0.0
                        avg_entry_price = 0.0
                        holding_bars = 0

                elif action == 7:  # SCALE_OUT_50
                    close_lots = abs(position) * 0.5
                    if use_bid_ask:
                        base_price = arr_bid[i] if direction > 0 else arr_ask[i]
                    else:
                        spread = arr_spread[i] if arr_spread[i] > 0 else 0.0001
                        base_price = arr_close[i] - direction * 0.5 * spread
                    slippage = -direction * slippage_pips * pip_size
                    price = base_price + slippage

                    cost = close_lots * commission_per_lot
                    if avg_entry_price != 0.0:
                        pnl_pips = direction * (price - avg_entry_price) / pip_size
                        gross_pnl = pnl_pips * pip_value_per_lot * close_lots
                        equity += gross_pnl - cost
                    position -= direction * close_lots
                    if abs(position) < 0.001:
                        position = 0.0
                        avg_entry_price = 0.0
                        holding_bars = 0

                elif action == 8:  # SCALE_OUT_100
                    close_lots = abs(position) * 1.0
                    if use_bid_ask:
                        base_price = arr_bid[i] if direction > 0 else arr_ask[i]
                    else:
                        spread = arr_spread[i] if arr_spread[i] > 0 else 0.0001
                        base_price = arr_close[i] - direction * 0.5 * spread
                    slippage = -direction * slippage_pips * pip_size
                    price = base_price + slippage

                    cost = close_lots * commission_per_lot
                    if avg_entry_price != 0.0:
                        pnl_pips = direction * (price - avg_entry_price) / pip_size
                        gross_pnl = pnl_pips * pip_value_per_lot * close_lots
                        equity += gross_pnl - cost
                    position = 0.0
                    avg_entry_price = 0.0
                    holding_bars = 0

                elif action == 9:  # CLOSE_ALL
                    close_lots = abs(position)
                    if use_bid_ask:
                        base_price = arr_bid[i] if direction > 0 else arr_ask[i]
                    else:
                        spread = arr_spread[i] if arr_spread[i] > 0 else 0.0001
                        base_price = arr_close[i] - direction * 0.5 * spread
                    slippage = -direction * slippage_pips * pip_size
                    price = base_price + slippage

                    cost = close_lots * commission_per_lot
                    if avg_entry_price != 0.0:
                        pnl_pips = direction * (price - avg_entry_price) / pip_size
                        gross_pnl = pnl_pips * pip_value_per_lot * close_lots
                        equity += gross_pnl - cost
                    position = 0.0
                    avg_entry_price = 0.0
                    holding_bars = 0

            if position != 0:
                holding_bars += 1

            direction = 1.0 if position > 0 else (-1.0 if position < 0 else 0.0)
            if direction != 0.0 and avg_entry_price != 0.0:
                if use_bid_ask:
                    mark_px = arr_bid[i] if direction > 0 else arr_ask[i]
                else:
                    mark_px = arr_close[i]
                unrealised = ((mark_px - avg_entry_price) / pip_size) * direction * abs(position) * pip_value_per_lot
            else:
                unrealised = 0.0

            peak_equity = max(peak_equity, equity + unrealised)
            total_value = equity + unrealised
            if peak_equity > 0:
                drawdown = max(0.0, (peak_equity - total_value) / peak_equity)
            else:
                drawdown = 0.0

            equity_curve[i] = equity
            res_eq[i] = equity
            res_unreal[i] = unrealised
            res_total[i] = total_value
            res_pos[i] = position
            res_dd[i] = drawdown
            res_hold[i] = holding_bars
            n_valid = i + 1

            if drawdown > max_drawdown_limit:
                if position != 0:
                    close_lots = abs(position)
                    direction = 1.0 if position > 0 else -1.0
                    if use_bid_ask:
                        base_price = arr_bid[i] if direction > 0 else arr_ask[i]
                    else:
                        spread = arr_spread[i] if arr_spread[i] > 0 else 0.0001
                        base_price = arr_close[i] - direction * 0.5 * spread
                    slippage = -direction * slippage_pips * pip_size
                    price = base_price + slippage

                    cost = close_lots * commission_per_lot
                    if avg_entry_price != 0.0:
                        pnl_pips = direction * (price - avg_entry_price) / pip_size
                        gross_pnl = pnl_pips * pip_value_per_lot * close_lots
                        equity += gross_pnl - cost
                    position = 0.0
                    avg_entry_price = 0.0
                    holding_bars = 0
                    equity_curve[i] = equity
                    res_eq[i] = equity
                    res_unreal[i] = 0.0
                    res_total[i] = equity
                    res_pos[i] = 0.0
                break

        # Force-close at end
        if position != 0 and n_valid > 0:
            i = n_valid - 1
            close_lots = abs(position)
            direction = 1.0 if position > 0 else -1.0
            if use_bid_ask:
                base_price = arr_bid[i] if direction > 0 else arr_ask[i]
            else:
                spread = arr_spread[i] if arr_spread[i] > 0 else 0.0001
                base_price = arr_close[i] - direction * 0.5 * spread
            slippage = -direction * slippage_pips * pip_size
            price = base_price + slippage

            cost = close_lots * commission_per_lot
            if avg_entry_price != 0.0:
                pnl_pips = direction * (price - avg_entry_price) / pip_size
                gross_pnl = pnl_pips * pip_value_per_lot * close_lots
                equity += gross_pnl - cost
            position = 0.0
            equity_curve[i] = equity
            res_eq[i] = equity
            res_unreal[i] = 0.0
            res_total[i] = equity
            res_pos[i] = 0.0

        # Truncate unused tail
        if n_valid < n_bars:
            res_eq = res_eq[:n_valid]
            res_unreal = res_unreal[:n_valid]
            res_total = res_total[:n_valid]
            res_pos = res_pos[:n_valid]
            res_dd = res_dd[:n_valid]
            res_hold = res_hold[:n_valid]
            equity_curve = equity_curve[:n_valid]

        return (res_eq, res_unreal, res_total, res_pos, res_dd, res_hold, equity_curve, n_valid)

    # ── Main loop ────────────────────────────────────────────────────────────

    def run(self, use_numba: bool = True, return_trades: bool = False) -> pd.DataFrame:
        """
        Execute the backtest bar by bar.

        When ``use_numba`` is True (default) and Numba is available, large runs
        use the JIT equity-curve core. Trade-level ``self.trades`` is only
        populated on the Python path; ``performance_metrics`` falls back to
        equity-curve stats when trades are empty.

        Returns
        -------
        pd.DataFrame: Bar-by-bar equity and P&L records
        """
        print(f"[Backtest] Running {len(self.bars):,} bars | "
              f"Initial equity: ${self.initial_equity:,.2f}")

        n_bars = len(self.bars)
        # Prefer Python when trade records are needed and the book is small;
        # Numba shines on large equity-only sweeps (metrics use equity fallback).
        _NUMBA_MIN_BARS = 50_000
        if use_numba and _NUMBA_OK and n_bars >= _NUMBA_MIN_BARS and not return_trades:
            return self._run_numba_path(n_bars)
        if use_numba and not _NUMBA_OK:
            print("[Backtest] Numba unavailable — using Python path")

        return self._run_python_path(n_bars)

    def _run_numba_path(self, n_bars: int) -> pd.DataFrame:
        """JIT-accelerated backtest core."""
        arr_bid = self._arr_bid_close if self._arr_bid_close is not None else self._arr_close
        arr_ask = self._arr_ask_close if self._arr_ask_close is not None else self._arr_close
        arr_spread = self._arr_spread if self._arr_spread is not None else np.full(n_bars, 0.0001)

        sig_action_f = self._sig_action.astype(np.float64)
        sig_sl_f = self._sig_sl.astype(np.float64)
        sig_tp_f = self._sig_tp.astype(np.float64)
        sig_lots_f = self._sig_lots.astype(np.float64)

        res_eq, res_unreal, res_total, res_pos, res_dd, res_hold, equity_curve, n_valid = \
            self._run_core_numba(
                self._arr_close, self._arr_high, self._arr_low,
                arr_bid, arr_ask, arr_spread, self._arr_ts,
                sig_action_f, sig_sl_f, sig_tp_f, sig_lots_f,
                n_bars, self.execution_delay, self.lot_size,
                self.commission_per_lot, self.slippage_pips, self.pip_size,
                self.pip_value_per_lot, self.max_lots,
                self.max_drawdown_limit, self.initial_equity,
                self.min_spread_clamp, self.max_spread_clamp,
                self.daily_volume_lots, self.apply_market_impact, self.use_bid_ask,
            )

        # Update Python-level state for API compat
        if n_valid > 0:
            self.equity = float(res_eq[-1])
            self.position = float(res_pos[-1])
        self.equity_curve = equity_curve.tolist()

        res_ts = self._arr_ts[:n_valid]
        self.results_df = pd.DataFrame({
            "timestamp": res_ts,
            "equity": res_eq,
            "unrealised_pnl": res_unreal,
            "total_value": res_total,
            "position": res_pos,
            "drawdown": res_dd,
            "holding_bars": res_hold,
        }).set_index("timestamp")
        return self.results_df

    def _run_python_path(self, n_bars: int) -> pd.DataFrame:
        """Pure-Python fallback backtest core (original logic)."""
        res_ts = self._arr_ts.copy()
        res_equity = np.empty(n_bars, dtype=np.float64)
        res_unrealised = np.empty(n_bars, dtype=np.float64)
        res_total = np.empty(n_bars, dtype=np.float64)
        res_position = np.empty(n_bars, dtype=np.float64)
        res_drawdown = np.empty(n_bars, dtype=np.float64)
        res_holding = np.empty(n_bars, dtype=np.int32)

        for i in range(n_bars):
            ts = self._arr_ts[i]
            signal_idx = i - self.execution_delay
            if signal_idx >= len(self.signals):
                break

            if signal_idx < 0:
                action = 0
                stop_loss = 0.0
                take_profit = 0.0
                lots_to_trade = 0.0
            else:
                _raw_action = self._sig_action[signal_idx]
                try:
                    _a = float(_raw_action)
                except (TypeError, ValueError):
                    action = 0
                else:
                    action = int(_a) if np.isfinite(_a) else 0
                stop_loss = float(self._sig_sl[signal_idx])
                take_profit = float(self._sig_tp[signal_idx])
                lots_to_trade = float(self._sig_lots[signal_idx])

            stopped = self._check_stops(i)

            if not stopped:
                if action == ScalingAction.OPEN_LONG and self.position == 0:
                    self._open_position(i, +1, lots_to_trade, stop_loss, take_profit)
                elif action == ScalingAction.OPEN_SHORT and self.position == 0:
                    self._open_position(i, -1, lots_to_trade, stop_loss, take_profit)
                elif action == ScalingAction.SCALE_IN_25 and self.position != 0:
                    self._scale_in(i, lots_to_trade * 0.25)
                elif action == ScalingAction.SCALE_IN_50 and self.position != 0:
                    self._scale_in(i, lots_to_trade * 0.50)
                elif action == ScalingAction.SCALE_OUT_25 and self.position != 0:
                    self._close_position(i, 0.25, "scale_out_25")
                elif action == ScalingAction.SCALE_OUT_50 and self.position != 0:
                    self._close_position(i, 0.50, "scale_out_50")
                elif action == ScalingAction.CLOSE_ALL and self.position != 0:
                    self._close_position(i, 1.0, "signal_exit")

            if self.position != 0:
                self.holding_bars += 1

            if self.position != 0:
                direction = np.sign(self.position)
                if self.use_bid_ask:
                    mark_px = self._arr_bid_close[i] if direction > 0 else self._arr_ask_close[i]
                else:
                    mark_px = self._arr_close[i]
                unrealised = ((mark_px - self.avg_entry_price) / self.pip_size) * direction * abs(self.position) * self.pip_value_per_lot
            else:
                unrealised = 0.0

            self.peak_equity = max(self.peak_equity, self.equity + unrealised)
            drawdown = max(0, (self.peak_equity - (self.equity + unrealised)) / self.peak_equity)

            self.equity_curve.append(self.equity)

            res_equity[i] = self.equity
            res_unrealised[i] = unrealised
            res_total[i] = self.equity + unrealised
            res_position[i] = self.position
            res_drawdown[i] = drawdown
            res_holding[i] = self.holding_bars

            if drawdown > self.max_drawdown_limit:
                self._close_position(i, 1.0, "circuit_breaker")
                i += 1
                res_ts = res_ts[:i]
                res_equity = res_equity[:i]
                res_unrealised = res_unrealised[:i]
                res_total = res_total[:i]
                res_position = res_position[:i]
                res_drawdown = res_drawdown[:i]
                res_holding = res_holding[:i]
                break

        if self.position != 0:
            self._close_position(len(res_ts) - 1, 1.0, "end_of_data")

        self.results_df = pd.DataFrame({
            "timestamp": res_ts[:len(res_equity)],
            "equity": res_equity,
            "unrealised_pnl": res_unrealised,
            "total_value": res_total,
            "position": res_position,
            "drawdown": res_drawdown,
            "holding_bars": res_holding,
        }).set_index("timestamp")
        return self.results_df

    # ── Performance reporting ────────────────────────────────────────────────

    def performance_metrics(self) -> dict:
        """Compute comprehensive performance statistics."""
        if not self.trades:
            if self.results_df is not None and len(self.results_df) > 0:
                raise RuntimeError(
                    "Trade metrics requested but trade log is empty! "
                    "Use run(return_trades=True) to populate trades instead of "
                    "silently falling back to equity-curve metrics."
                )
            return {"error": "No trades executed"}

        gross_pnl = sum(t.gross_pnl_usd for t in self.trades)
        net_pnl = self.equity - self.initial_equity
        if net_pnl == 0.0:
            print("WARNING: Total PnL is exactly 0.0. No profitable or losing trades executed.")

        total_cost = sum(t.commission for t in self.trades)
        winning_trades = [t for t in self.trades if t.pnl_usd > 0]
        losing_trades = [t for t in self.trades if t.pnl_usd < 0]

        # Use mark-to-market total_value (updates every bar) instead of equity
        # (which only updates on trade close), to avoid artificially deflating std
        mtm_col = "total_value" if "total_value" in self.results_df.columns else "equity"
        returns = self.results_df[mtm_col].pct_change().dropna()
        # Assume ~2% annual risk-free rate
        rf_annual = 0.02
        rf_per_bar = rf_annual / (252 * 24 * 60)
        excess_returns = returns - rf_per_bar

        sharpe = 0.0
        sortino = 0.0
        ann_factor = np.sqrt(252 * 24 * 60)  # 1-min bars

        if len(excess_returns) > 1 and excess_returns.std(ddof=1) > 1e-12:
            sharpe = (excess_returns.mean() / excess_returns.std(ddof=1)) * ann_factor

        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) > 1 and downside_returns.std(ddof=1) > 1e-12:
            sortino = (excess_returns.mean() / downside_returns.std(ddof=1)) * ann_factor

        equity_arr = self.results_df["total_value"].values
        rolling_max = np.maximum.accumulate(equity_arr)
        drawdowns = (rolling_max - equity_arr) / (rolling_max + 1e-9)
        max_dd = drawdowns.max()

        avg_bars_held = np.mean([
            (t.exit_time - t.entry_time).total_seconds() / 60
            if t.exit_time else 0 for t in self.trades
        ])

        long_trades = [t for t in self.trades if t.direction > 0]
        short_trades = [t for t in self.trades if t.direction < 0]
        long_wins = [t for t in long_trades if t.pnl_usd > 0]
        short_wins = [t for t in short_trades if t.pnl_usd > 0]

        return {
            "total_return_pct": (self.equity / self.initial_equity - 1) * 100,
            "gross_pnl_usd": gross_pnl,
            "total_pnl_usd": net_pnl,
            "total_commission_usd": total_cost,
            "net_pnl_usd": net_pnl,
            "n_trades": len(self.trades),
            "win_rate_pct": len(winning_trades) / len(self.trades) * 100,
            "long_win_rate_pct": len(long_wins) / len(long_trades) * 100 if long_trades else 0,
            "short_win_rate_pct": len(short_wins) / len(short_trades) * 100 if short_trades else 0,
            "avg_win_usd": np.mean([t.pnl_usd for t in winning_trades]) if winning_trades else 0,
            "avg_loss_usd": np.mean([t.pnl_usd for t in losing_trades]) if losing_trades else 0,
            "win_loss_ratio": (
                abs(np.mean([t.pnl_usd for t in winning_trades]))
                / max(abs(np.mean([t.pnl_usd for t in losing_trades])), 0.01)
                if winning_trades and losing_trades else 0
            ),
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown_pct": max_dd * 100,
            "avg_holding_minutes": avg_bars_held,
            "profit_factor": (
                sum(t.pnl_usd for t in winning_trades)
                / max(abs(sum(t.pnl_usd for t in losing_trades)), 0.01)
                if losing_trades else float("inf")
            ),
        }

    def _equity_curve_metrics(self) -> dict:
        """Metrics from results_df when Trade records are unavailable (Numba path)."""
        mtm_col = "total_value" if "total_value" in self.results_df.columns else "equity"
        equity_arr = self.results_df[mtm_col].values.astype(np.float64)
        net_pnl = float(self.equity - self.initial_equity)
        returns = self.results_df[mtm_col].pct_change().dropna()
        rf_per_bar = 0.02 / (252 * 24 * 60)
        excess_returns = returns - rf_per_bar
        sharpe = 0.0
        sortino = 0.0
        ann_factor = np.sqrt(252 * 24 * 60)
        if len(excess_returns) > 1 and excess_returns.std(ddof=1) > 1e-12:
            sharpe = float((excess_returns.mean() / excess_returns.std(ddof=1)) * ann_factor)
        downside = excess_returns[excess_returns < 0]
        if len(downside) > 1 and downside.std(ddof=1) > 1e-12:
            sortino = float((excess_returns.mean() / downside.std(ddof=1)) * ann_factor)
        rolling_max = np.maximum.accumulate(equity_arr)
        max_dd = float(((rolling_max - equity_arr) / (rolling_max + 1e-9)).max())
        return {
            "total_return_pct": (self.equity / self.initial_equity - 1) * 100,
            "gross_pnl_usd": net_pnl,
            "total_pnl_usd": net_pnl,
            "total_commission_usd": 0.0,
            "net_pnl_usd": net_pnl,
            "n_trades": 0,
            "win_rate_pct": 0.0,
            "long_win_rate_pct": 0.0,
            "short_win_rate_pct": 0.0,
            "avg_win_usd": 0.0,
            "avg_loss_usd": 0.0,
            "win_loss_ratio": 0.0,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown_pct": max_dd * 100,
            "avg_holding_minutes": 0.0,
            "profit_factor": 0.0,
            "metrics_source": "equity_curve",
        }

    def print_performance(self):
        """Print formatted performance report."""
        m = self.performance_metrics()
        print("\n" + "=" * 55)
        print("BACKTEST PERFORMANCE REPORT")
        print("=" * 55)
        for k, v in m.items():
            if k == "error":
                print(f"  ERROR: {v}")
            elif isinstance(v, float):
                print(f"  {k:<30} {v:>10.4f}")
            else:
                print(f"  {k:<30} {v:>10}")
        print("=" * 55)

    def get_equity_curve(self) -> pd.Series:
        """Return the total equity curve (including unrealised P&L)."""
        if hasattr(self, "results_df"):
            return self.results_df["total_value"]
        return pd.Series(self.equity_curve)

    def get_trade_log(self) -> pd.DataFrame:
        """Return all trades as a DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([{
            "trade_id": t.trade_id,
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "direction": "Long" if t.direction > 0 else "Short",
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "lots": t.entry_lots,
            "pnl_pips": t.pnl_pips,
            "gross_pnl_usd": t.gross_pnl_usd,
            "pnl_usd": t.pnl_usd,
            "commission": t.commission,
            "exit_reason": t.exit_reason,
            "n_scale_adds": len(t.scale_additions),
        } for t in self.trades])

    def generate_tear_sheet(self):
        """Generate pyfolio tear sheet safely without crashing on bad runs."""
        try:
            import warnings

            import pyfolio as pf

            if not hasattr(self, "results_df") or self.results_df.empty:
                print("WARNING: No results to generate tear sheet.")
                return

            returns = self.results_df["total_value"].pct_change().dropna()

            # Metrics Check: Warn on NaN or 0 variance
            if len(returns) < 2:
                print("WARNING: Not enough return data points to generate tear sheet.")
                return
            if returns.std() == 0 or returns.isna().any():
                print("WARNING: Returns have 0 variance or NaN values. Tear sheet aborted.")
                return

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pf.create_simple_tear_sheet(returns)
        except ImportError:
            print("WARNING: pyfolio is not installed. Cannot generate tear sheet.")
        except Exception as e:
            print(f"WARNING: pyfolio tear sheet generation failed: {e}")


if __name__ == "__main__":
    # Smoke test with synthetic data
    import sys
    from pathlib import Path

    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    from data.data_ingestion import ForexDataPipeline, load_or_generate

    ticks = load_or_generate(n_rows=20_000)
    pipeline = ForexDataPipeline(bar_freq="5min")
    bars = pipeline.run(ticks)

    # Create dummy signals (random strategy for testing)
    rng = np.random.default_rng(42)
    signals = pd.DataFrame(index=bars.index)
    signals["action"] = rng.choice(
        [ScalingAction.HOLD, ScalingAction.OPEN_LONG, ScalingAction.OPEN_SHORT, ScalingAction.CLOSE_ALL],
        size=len(bars), p=[0.7, 0.1, 0.1, 0.1]
    )
    signals["lots"] = 0.1
    signals["stop_loss"] = bars["close"] - 0.0010
    signals["take_profit"] = bars["close"] + 0.0015

    bt = ForexScalingBacktest(bars=bars, signals=signals, initial_equity=10_000)
    results = bt.run()
    bt.print_performance()

    trades = bt.get_trade_log()
    if len(trades) > 0:
        print(f"\nSample trades:\n{trades.head(5).to_string()}")
