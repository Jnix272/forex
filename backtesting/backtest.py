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
from datetime import datetime
from enum import IntEnum

import numpy as np

try:
    import polars as pl
    _POLARS_OK = True
except ImportError:  # pragma: no cover
    _POLARS_OK = False
    pl = None  # type: ignore[assignment]

try:
    import pandas as pd
    _PANDAS_OK = True
except ImportError:  # pragma: no cover
    _PANDAS_OK = False
    pd = None  # type: ignore[assignment]

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


# -----------------------------------------------------------------------------
# HELPERS: accept both Polars and pandas DataFrames
# -----------------------------------------------------------------------------

def _df_columns(df) -> list:
    """Return list of column names."""
    return list(df.columns)


def _get_column_numpy(df, col: str) -> np.ndarray:
    """Extract column as numpy array from Polars or pandas DataFrame."""
    if _POLARS_OK and pl is not None and isinstance(df, pl.DataFrame):
        return df[col].to_numpy()
    if _PANDAS_OK and pd is not None and isinstance(df, pd.DataFrame):
        return df[col].values
    raise TypeError(f"Unsupported DataFrame type: {type(df)}")


def _get_index_numpy(df) -> np.ndarray:
    """Extract the index/timestamp column as numpy array."""
    if _POLARS_OK and pl is not None and isinstance(df, pl.DataFrame):
        for cand in ("timestamp_utc", "timestamp", "time", "date"):
            if cand in df.columns:
                return df[cand].to_numpy()
        return np.arange(len(df), dtype=np.int64)
    if _PANDAS_OK and pd is not None and isinstance(df, pd.DataFrame):
        return df.index.values
    raise TypeError(f"Unsupported DataFrame type: {type(df)}")


# -----------------------------------------------------------------------------
# TRADE RECORDS
# -----------------------------------------------------------------------------

@dataclass
class Trade:
    """Record of a single trade execution."""
    trade_id: int
    entry_time: datetime
    entry_price: float
    entry_lots: float
    direction: int           # +1 = long, -1 = short
    stop_loss: float
    take_profit: float
    exit_time: datetime | None = None
    exit_price: float | None = None
    exit_lots: float | None = None
    pnl_pips: float = 0.0
    gross_pnl_usd: float = 0.0
    pnl_usd: float = 0.0
    commission: float = 0.0
    slippage_pips: float = 0.0
    exit_reason: str = ""    # 'stop_loss', 'take_profit', 'signal', 'scale_out', 'eod'
    scale_additions: list[dict] = field(default_factory=list)


# -----------------------------------------------------------------------------
# BACKTESTING ENGINE
# -----------------------------------------------------------------------------

class ForexScalingBacktest:
    """
    Event-driven backtesting engine for forex scaling strategies.

    Supports:
      - Scale-in (pyramiding and martingale) at multiple price levels
      - Scale-out (partial profit taking) at predefined targets
      - Dynamic stop-loss trailing
      - Realistic transaction cost modeling

    Accepts both polars and pandas DataFrames for bars and signals.

    Usage
    -----
        bt = ForexScalingBacktest(bars_df, signals_df)
        results = bt.run()
        bt.print_performance()
        equity_curve = bt.get_equity_curve()
    """

    def __init__(
        self,
        bars,
        signals,
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
        bars_per_year: float | None = None,
    ):
        """
        Parameters
        ----------
        bars     : OHLCV bars with 'open','high','low','close','bid_close','ask_close','spread_avg'
                   Accepts polars or pandas DataFrame.
        signals  : DataFrame with columns: 'action' (ScalingAction int), 'stop_loss', 'take_profit'
                   Accepts polars or pandas DataFrame.
        """
        # Validate required columns
        required_cols = ["open", "high", "low", "close"]
        bar_cols = _df_columns(bars)
        missing = [c for c in required_cols if c not in bar_cols]
        if missing:
            raise ValueError(f"bars DataFrame missing required columns: {missing}")

        self._bars_polars = _POLARS_OK and pl is not None and isinstance(bars, pl.DataFrame)
        self._bars_pandas = _PANDAS_OK and pd is not None and isinstance(bars, pd.DataFrame)

        self.bars = bars
        self._bar_len = len(bars)

        # ── Signals alignment ────────────────────────────────────────────────
        sig_cols = _df_columns(signals)

        if self._bars_pandas and pd is not None and isinstance(signals, pd.DataFrame):
            self.signals = signals.reindex(bars.index)
            # Actions are events, not state. Do not forward-fill them.
            if "action" in self.signals.columns:
                self.signals["action"] = self.signals["action"].fillna(ScalingAction.HOLD.value)
            # Forward-fill stops/TP/lots (gaps only; never overwrite signal rows)
            for col in ("lots", "stop_loss", "take_profit"):
                if col in self.signals.columns:
                    self.signals[col] = self.signals[col].ffill()
        elif _POLARS_OK and pl is not None and isinstance(signals, pl.DataFrame):
            sig = signals
            if "action" in sig.columns:
                sig = sig.with_columns(
                    pl.col("action").fill_null(ScalingAction.HOLD.value)
                )
            for col in ("lots", "stop_loss", "take_profit"):
                if col in sig.columns:
                    sig = sig.with_columns(pl.col(col).forward_fill())
            self.signals = sig
        else:
            raise TypeError(
                f"signals must be a polars or pandas DataFrame, got {type(signals)}"
            )

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
        self.bars_per_year = float(bars_per_year) if bars_per_year else (252.0 * 24.0 * 60.0)
        self._rf_per_bar = 0.02 / self.bars_per_year
        self.numba_min_bars = 50_000

        # State
        self.position: float = 0.0
        self.avg_entry_price: float = 0.0
        self.current_stop: float = 0.0
        self.current_tp: float = 0.0
        self.holding_bars: int = 0

        # Records
        self.trades: list[Trade] = []
        self.equity_curve: list[float] = []
        self.daily_pnl: list[float] = []
        self._trade_counter: int = 0
        self._open_trade: Trade | None = None
        self.results_df = None

        # Pre-extract numpy arrays for fast lookup (works for both Polars and pandas)
        self._arr_open = _get_column_numpy(bars, "open")
        self._arr_high = _get_column_numpy(bars, "high")
        self._arr_low = _get_column_numpy(bars, "low")
        self._arr_close = _get_column_numpy(bars, "close")
        self._arr_ts = _get_index_numpy(bars)

        if self.use_bid_ask and "bid_close" in bar_cols and "ask_close" in bar_cols:
            self._arr_bid_close = _get_column_numpy(bars, "bid_close")
            self._arr_ask_close = _get_column_numpy(bars, "ask_close")
        else:
            self._arr_bid_close = None
            self._arr_ask_close = None

        if "spread_avg" in bar_cols:
            self._arr_spread = _get_column_numpy(bars, "spread_avg")
        else:
            self._arr_spread = None

        # Pre-extract signal arrays
        sig_len = len(self.signals)
        self._sig_action = (
            _get_column_numpy(self.signals, "action").astype(np.float64)
            if "action" in sig_cols else np.zeros(sig_len, dtype=np.float64)
        )
        self._sig_sl = (
            _get_column_numpy(self.signals, "stop_loss").astype(np.float64)
            if "stop_loss" in sig_cols else np.zeros(sig_len, dtype=np.float64)
        )
        self._sig_tp = (
            _get_column_numpy(self.signals, "take_profit").astype(np.float64)
            if "take_profit" in sig_cols else np.zeros(sig_len, dtype=np.float64)
        )
        self._sig_lots = (
            _get_column_numpy(self.signals, "lots").astype(np.float64)
            if "lots" in sig_cols else np.full(sig_len, 0.1, dtype=np.float64)
        )

    # ── Timestamp helper ─────────────────────────────────────────────────────

    def _bar_timestamp(self, idx: int):
        """Return a timestamp object for bar at position idx."""
        ts = self._arr_ts[idx]
        if hasattr(ts, "astype"):
            try:
                import datetime as _dt
                ts_ms = int(np.datetime64(ts, "ms").view(np.int64))
                return _dt.datetime.utcfromtimestamp(ts_ms / 1000.0)
            except Exception:
                pass
        return ts

    # ── Price helpers ────────────────────────────────────────────────────────

    def _session_spread_mult(self, idx: int) -> float:
        """Shared session→spread mult (LABEL_REGIME / session_utils SoT)."""
        try:
            from trading.session_utils import session_spread_mult
        except Exception:
            return 1.0
        sess = None
        asia_london = False
        london_ny = False
        bar_cols = _df_columns(self.bars)
        try:
            if "session_label" in bar_cols:
                sess = _get_column_numpy(self.bars, "session_label")[idx]
            if "asia_london" in bar_cols:
                asia_london = bool(float(_get_column_numpy(self.bars, "asia_london")[idx]) > 0)
            if "london_ny" in bar_cols:
                london_ny = bool(float(_get_column_numpy(self.bars, "london_ny")[idx]) > 0)
        except Exception:
            pass
        now = None
        if sess is None and not asia_london and not london_ny:
            try:
                ts = self._arr_ts[idx]
                if hasattr(ts, "to_pydatetime"):
                    now = ts.to_pydatetime()
                elif hasattr(ts, "astype"):
                    import datetime as _dt
                    ts_ms = int(np.datetime64(ts, "ms").view(np.int64))
                    now = _dt.datetime.utcfromtimestamp(ts_ms / 1000.0)
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
        if self._arr_bid_close is not None and self._arr_ask_close is not None:
            base_price = self._arr_ask_close[idx] if direction > 0 else self._arr_bid_close[idx]
            spread = self._arr_spread[idx] if self._arr_spread is not None else (self._arr_ask_close[idx] - self._arr_bid_close[idx])
        else:
            if self._arr_spread is not None:
                spread = self._arr_spread[idx]
            else:
                spread = 0.0001 * self._session_spread_mult(idx)
            spread_half = spread / 2
            base_price = self._arr_close[idx] + direction * spread_half

        slip_pips = self.slippage_pips
        slippage = direction * slip_pips * self.pip_size
        price = base_price + slippage

        if self.apply_market_impact and lots > 0 and not is_close:
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
            entry_time=self._bar_timestamp(idx),
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
            return
        if abs(self.position) + lots > self.max_lots:
            lots = self.max_lots - abs(self.position)
        if lots <= 0:
            return

        direction = int(np.sign(self.position))
        exec_price = self._get_execution_price(idx, direction, lots)
        cost = self._compute_cost(lots)
        self.equity -= cost

        total_lots = abs(self.position) + lots
        self.avg_entry_price = (
            abs(self.position) * self.avg_entry_price + lots * exec_price
        ) / total_lots

        self.position += direction * lots

        if self._open_trade:
            self._open_trade.scale_additions.append({
                "time": self._bar_timestamp(idx),
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
                self._open_trade.exit_time = self._bar_timestamp(idx)
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

        stop_ok = (self.current_stop is not None
                   and np.isfinite(float(self.current_stop))
                   and float(self.current_stop) > 0.0)
        tp_ok = (self.current_tp is not None
                 and np.isfinite(float(self.current_tp))
                 and float(self.current_tp) > 0.0)

        if stop_ok and direction > 0 and self._arr_low[idx] <= self.current_stop:
            exec_px = self.current_stop - self.slippage_pips * self.pip_size
            self._close_position(idx, fraction=1.0, exit_reason="stop_loss", override_price=exec_px)
            return True
        if stop_ok and direction < 0 and self._arr_high[idx] >= self.current_stop:
            exec_px = self.current_stop + self.slippage_pips * self.pip_size
            self._close_position(idx, fraction=1.0, exit_reason="stop_loss", override_price=exec_px)
            return True

        if tp_ok and direction > 0 and self._arr_high[idx] >= self.current_tp:
            exec_px = self.current_tp - self.slippage_pips * self.pip_size
            self._close_position(idx, fraction=0.5, exit_reason="scale_out_tp", override_price=exec_px)
            if self.position != 0:
                self.current_stop = max(self.current_stop, self.avg_entry_price)
            return False

        if tp_ok and direction < 0 and self._arr_low[idx] <= self.current_tp:
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

                if not np.isnan(current_stop) and current_stop > 0.0:
                    if direction > 0 and arr_low[i] <= current_stop:
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

                if position != 0 and not np.isnan(current_tp) and current_tp > 0.0:
                    if direction > 0 and arr_high[i] >= current_tp:
                        close_lots = abs(position) * 0.5
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

    def run(self, use_numba: bool = True, return_trades: bool = False):
        """
        Execute the backtest bar by bar.

        When ``use_numba`` is True (default) and Numba is available, large runs
        use the JIT equity-curve core. Trade-level ``self.trades`` is only
        populated on the Python path; ``performance_metrics`` falls back to
        equity-curve stats when trades are empty.

        Returns
        -------
        pl.DataFrame (or pd.DataFrame if polars is unavailable):
            Bar-by-bar equity and P&L records.
        """
        print(f"[Backtest] Running {len(self.bars):,} bars | "
              f"Initial equity: ${self.initial_equity:,.2f}")

        # Reset simulation state so run() is idempotent
        self.position = 0.0
        self.avg_entry_price = 0.0
        self.current_stop = 0.0
        self.current_tp = 0.0
        self.holding_bars = 0
        self.equity = self.initial_equity
        self.peak_equity = self.initial_equity
        self.trades = []
        self.equity_curve = []
        self.daily_pnl = []
        self._trade_counter = 0
        self._open_trade = None
        self.results_df = None

        n_bars = len(self.bars)
        if use_numba and _NUMBA_OK and n_bars >= self.numba_min_bars and not return_trades:
            return self._run_numba_path(n_bars)
        if use_numba and not _NUMBA_OK:
            print("[Backtest] Numba unavailable — using Python path")

        return self._run_python_path(n_bars)

    def _build_results_df(
        self,
        res_ts: np.ndarray,
        res_equity: np.ndarray,
        res_unrealised: np.ndarray,
        res_total: np.ndarray,
        res_position: np.ndarray,
        res_drawdown: np.ndarray,
        res_holding: np.ndarray,
    ):
        """Build a Polars DataFrame (or pandas fallback) from result arrays."""
        n = len(res_equity)
        ts_slice = res_ts[:n]

        if _POLARS_OK and pl is not None:
            try:
                ts_list = ts_slice.tolist()
            except Exception:
                ts_list = list(ts_slice)
            return pl.DataFrame({
                "timestamp": ts_list,
                "equity": res_equity.tolist(),
                "unrealised_pnl": res_unrealised.tolist(),
                "total_value": res_total.tolist(),
                "position": res_position.tolist(),
                "drawdown": res_drawdown.tolist(),
                "holding_bars": res_holding.tolist(),
            })

        if _PANDAS_OK and pd is not None:
            return pd.DataFrame({
                "timestamp": ts_slice,
                "equity": res_equity,
                "unrealised_pnl": res_unrealised,
                "total_value": res_total,
                "position": res_position,
                "drawdown": res_drawdown,
                "holding_bars": res_holding,
            }).set_index("timestamp")

        raise RuntimeError("Neither polars nor pandas is available.")

    def _run_numba_path(self, n_bars: int):
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

        if n_valid > 0:
            self.equity = float(res_eq[-1])
            self.position = float(res_pos[-1])
        self.equity_curve = equity_curve.tolist()

        res_ts = self._arr_ts[:n_valid]
        self.results_df = self._build_results_df(
            res_ts, res_eq, res_unreal, res_total, res_pos, res_dd, res_hold
        )
        return self.results_df

    def _run_python_path(self, n_bars: int):
        """Pure-Python fallback backtest core."""
        res_ts = self._arr_ts.copy()
        res_equity = np.empty(n_bars, dtype=np.float64)
        res_unrealised = np.empty(n_bars, dtype=np.float64)
        res_total = np.empty(n_bars, dtype=np.float64)
        res_position = np.empty(n_bars, dtype=np.float64)
        res_drawdown = np.empty(n_bars, dtype=np.float64)
        res_holding = np.empty(n_bars, dtype=np.int32)

        # FIX B7: track the slice endpoint explicitly
        # (i += 1 inside a for loop is a Python no-op; loop variable is reassigned each iteration)
        end_idx = n_bars

        for i in range(n_bars):
            signal_idx = i - self.execution_delay
            if signal_idx >= len(self._sig_action):
                end_idx = i
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
                # FIX B8: SCALE_IN_100 was missing from Python path
                elif action == ScalingAction.SCALE_IN_100 and self.position != 0:
                    self._scale_in(i, lots_to_trade * 1.0)
                elif action == ScalingAction.SCALE_OUT_25 and self.position != 0:
                    self._close_position(i, 0.25, "scale_out_25")
                elif action == ScalingAction.SCALE_OUT_50 and self.position != 0:
                    self._close_position(i, 0.50, "scale_out_50")
                # FIX B8: SCALE_OUT_100 was missing from Python path
                elif action == ScalingAction.SCALE_OUT_100 and self.position != 0:
                    self._close_position(i, 1.0, "scale_out_100")
                elif action == ScalingAction.CLOSE_ALL and self.position != 0:
                    self._close_position(i, 1.0, "signal_exit")

            if self.position != 0:
                self.holding_bars += 1

            if self.position != 0:
                direction = np.sign(self.position)
                if self.use_bid_ask and self._arr_bid_close is not None and self._arr_ask_close is not None:
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
                res_equity[i] = self.equity
                res_unrealised[i] = 0.0
                res_total[i] = self.equity
                res_position[i] = 0.0
                res_drawdown[i] = 0.0
                # FIX B7: correctly set end_idx BEFORE break
                end_idx = i + 1
                break

        # Slice result arrays to actual processed length
        res_ts = res_ts[:end_idx]
        res_equity = res_equity[:end_idx]
        res_unrealised = res_unrealised[:end_idx]
        res_total = res_total[:end_idx]
        res_position = res_position[:end_idx]
        res_drawdown = res_drawdown[:end_idx]
        res_holding = res_holding[:end_idx]

        if self.position != 0:
            self._close_position(len(res_ts) - 1, 1.0, "end_of_data")
        # Reflect the end-of-data force close on the last reported record.
        if len(res_equity):
            res_equity[-1] = self.equity
            res_unrealised[-1] = 0.0
            res_total[-1] = self.equity
            res_position[-1] = 0.0
            res_drawdown[-1] = 0.0

        self.results_df = self._build_results_df(
            res_ts, res_equity, res_unrealised, res_total,
            res_position, res_drawdown, res_holding,
        )
        return self.results_df

    # ── Performance reporting ────────────────────────────────────────────────

    def _results_col_numpy(self, col: str) -> np.ndarray:
        """Extract a column from results_df as numpy regardless of backend."""
        if self.results_df is None:
            return np.array([])
        if _POLARS_OK and pl is not None and isinstance(self.results_df, pl.DataFrame):
            return self.results_df[col].to_numpy().astype(np.float64)
        if _PANDAS_OK and pd is not None and isinstance(self.results_df, pd.DataFrame):
            return self.results_df[col].values.astype(np.float64)
        return np.array([])

    def _has_results_col(self, col: str) -> bool:
        """Check if results_df has the given column."""
        if self.results_df is None:
            return False
        return col in self.results_df.columns

    def performance_metrics(self) -> dict:
        """Compute comprehensive performance statistics."""
        if not self.trades:
            if self.results_df is not None and len(self.results_df) > 0:
                m = self._equity_curve_metrics()
                m["warning"] = (m.get("warning", "") + " "
                                "Trade-level metrics unavailable (empty trade log); "
                                "using equity-curve fallback.").strip()
                return m
            return {"error": "No trades executed"}

        gross_pnl = sum(t.gross_pnl_usd for t in self.trades)
        net_pnl = self.equity - self.initial_equity
        if net_pnl == 0.0:
            print("WARNING: Total PnL is exactly 0.0. No profitable or losing trades executed.")

        total_cost = sum(t.commission for t in self.trades)
        winning_trades = [t for t in self.trades if t.pnl_usd > 0]
        losing_trades = [t for t in self.trades if t.pnl_usd < 0]

        mtm_col = "total_value" if self._has_results_col("total_value") else "equity"
        equity_arr = self._results_col_numpy(mtm_col)

        # pct_change equivalent via numpy (avoids pandas dependency)
        returns = np.diff(equity_arr) / (equity_arr[:-1] + 1e-12)
        excess_returns = returns - self._rf_per_bar

        sharpe = 0.0
        sortino = 0.0
        ann_factor = np.sqrt(self.bars_per_year)

        if len(excess_returns) > 1:
            er_std = float(np.std(excess_returns, ddof=1))
            if er_std > 1e-12:
                sharpe = float(np.mean(excess_returns) / er_std) * ann_factor

        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) > 1:
            ds_std = float(np.std(downside_returns, ddof=1))
            if ds_std > 1e-12:
                sortino = float(np.mean(excess_returns) / ds_std) * ann_factor

        rolling_max = np.maximum.accumulate(equity_arr)
        drawdowns = (rolling_max - equity_arr) / (rolling_max + 1e-9)
        max_dd = float(drawdowns.max()) if len(drawdowns) else 0.0

        avg_bars_held = float(np.mean([
            (t.exit_time - t.entry_time).total_seconds() / 60
            if t.exit_time and t.entry_time else 0
            for t in self.trades
        ]))

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
            "avg_win_usd": float(np.mean([t.pnl_usd for t in winning_trades])) if winning_trades else 0,
            "avg_loss_usd": float(np.mean([t.pnl_usd for t in losing_trades])) if losing_trades else 0,
            "win_loss_ratio": (
                abs(float(np.mean([t.pnl_usd for t in winning_trades])))
                / max(abs(float(np.mean([t.pnl_usd for t in losing_trades]))), 0.01)
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
        mtm_col = "total_value" if self._has_results_col("total_value") else "equity"
        equity_arr = self._results_col_numpy(mtm_col)
        net_pnl = float(self.equity - self.initial_equity)

        returns = np.diff(equity_arr) / (equity_arr[:-1] + 1e-12)
        excess_returns = returns - self._rf_per_bar
        sharpe = 0.0
        sortino = 0.0
        ann_factor = np.sqrt(self.bars_per_year)

        if len(excess_returns) > 1:
            er_std = float(np.std(excess_returns, ddof=1))
            if er_std > 1e-12:
                sharpe = float(np.mean(excess_returns) / er_std * ann_factor)

        downside = excess_returns[excess_returns < 0]
        if len(downside) > 1:
            ds_std = float(np.std(downside, ddof=1))
            if ds_std > 1e-12:
                sortino = float(np.mean(excess_returns) / ds_std * ann_factor)

        rolling_max = np.maximum.accumulate(equity_arr)
        max_dd = float(((rolling_max - equity_arr) / (rolling_max + 1e-9)).max()) if len(equity_arr) else 0.0

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

    def get_equity_curve(self):
        """Return the total equity curve (including unrealised P&L)."""
        if self.results_df is not None and self._has_results_col("total_value"):
            return self.results_df["total_value"]
        if _POLARS_OK and pl is not None:
            return pl.Series("equity", self.equity_curve)
        if _PANDAS_OK and pd is not None:
            return pd.Series(self.equity_curve)
        return self.equity_curve

    def get_trade_log(self):
        """Return all trades as a Polars DataFrame (or pandas fallback)."""
        if not self.trades:
            if _POLARS_OK and pl is not None:
                return pl.DataFrame()
            if _PANDAS_OK and pd is not None:
                return pd.DataFrame()
            return []

        rows = [{
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
        } for t in self.trades]

        if _POLARS_OK and pl is not None:
            return pl.DataFrame(rows)
        if _PANDAS_OK and pd is not None:
            return pd.DataFrame(rows)
        return rows

    def generate_tear_sheet(self):
        """Generate pyfolio tear sheet safely without crashing on bad runs."""
        try:
            import warnings
            import pyfolio as pf

            if self.results_df is None or len(self.results_df) == 0:
                print("WARNING: No results to generate tear sheet.")
                return

            equity_arr = self._results_col_numpy("total_value")
            returns_np = np.diff(equity_arr) / (equity_arr[:-1] + 1e-12)

            if len(returns_np) < 2:
                print("WARNING: Not enough return data points to generate tear sheet.")
                return
            if np.std(returns_np) == 0 or not np.all(np.isfinite(returns_np)):
                print("WARNING: Returns have 0 variance or NaN values. Tear sheet aborted.")
                return

            if _PANDAS_OK and pd is not None:
                returns_series = pd.Series(returns_np)
            else:
                print("WARNING: pyfolio requires pandas. Cannot generate tear sheet.")
                return

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pf.create_simple_tear_sheet(returns_series)
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
    if hasattr(bars, "to_pandas"):  # Polars pipeline output → pandas bar schema
        bars = bars.to_pandas()
        if "timestamp_utc" in bars.columns:
            bars = bars.set_index("timestamp_utc")

    # Create dummy signals (random strategy for testing)
    rng = np.random.default_rng(42)
    if _POLARS_OK and pl is not None and isinstance(bars, pl.DataFrame):
        n = len(bars)
        signals = pl.DataFrame({
            "action": rng.choice(
                [ScalingAction.HOLD, ScalingAction.OPEN_LONG,
                 ScalingAction.OPEN_SHORT, ScalingAction.CLOSE_ALL],
                size=n, p=[0.7, 0.1, 0.1, 0.1],
            ).astype(np.int64).tolist(),
            "lots": [0.1] * n,
            "stop_loss": (bars["close"].to_numpy() - 0.0010).tolist(),
            "take_profit": (bars["close"].to_numpy() + 0.0015).tolist(),
        })
    elif _PANDAS_OK and pd is not None:
        signals = pd.DataFrame(index=bars.index)
        signals["action"] = rng.choice(
            [ScalingAction.HOLD, ScalingAction.OPEN_LONG,
             ScalingAction.OPEN_SHORT, ScalingAction.CLOSE_ALL],
            size=len(bars), p=[0.7, 0.1, 0.1, 0.1]
        )
        signals["lots"] = 0.1
        signals["stop_loss"] = bars["close"] - 0.0010
        signals["take_profit"] = bars["close"] + 0.0015
    else:
        raise RuntimeError("Need polars or pandas for the smoke test")

    bt = ForexScalingBacktest(bars=bars, signals=signals, initial_equity=10_000)
    results = bt.run()
    bt.print_performance()

    trades = bt.get_trade_log()
    if len(trades) > 0:
        print(f"\nSample trades:\n{trades.head(5)}")
