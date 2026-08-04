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

    # ── Price helpers ────────────────────────────────────────────────────────

    def _get_execution_price(self, idx: int, direction: int, lots: float) -> float:
        """
        Realistic execution price including:
          - Spread: buys at ask, sells at bid
          - Slippage: additional friction from execution lag
          - Market impact: Square Root Law for large orders
        """
        row = self.bars.iloc[idx]

        # Use bid/ask if available (Golden Rule)
        if self.use_bid_ask and "bid_close" in self.bars.columns and "ask_close" in self.bars.columns:
            base_price = row["ask_close"] if direction > 0 else row["bid_close"]
        else:
            spread_half = row.get("spread_avg", 0.0001) / 2
            base_price = row["close"] + direction * spread_half

        # Slippage
        slippage = direction * self.slippage_pips * self.pip_size
        price = base_price + slippage

        # Market impact (Square Root Law: impact ∝ spread × √(lots / ADV))
        if self.apply_market_impact and lots > 0:
            spread = row.get("spread_avg", row.get("ask_close", row.get("close", 1.0))
                             - row.get("bid_close", row.get("close", 1.0) - 0.0001))
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
                        exit_reason: str = "signal") -> float:
        """Close all or part of position. Returns realised P&L in USD."""
        if self.position == 0:
            return 0.0

        close_lots = abs(self.position) * fraction
        direction = int(np.sign(self.position))
        exec_price = self._get_execution_price(idx, -direction, close_lots)
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

        row = self.bars.iloc[idx]
        direction = np.sign(self.position)

        # Stop-loss check
        if direction > 0 and row["low"] <= self.current_stop:
            self._close_position(idx, fraction=1.0, exit_reason="stop_loss")
            return True
        if direction < 0 and row["high"] >= self.current_stop:
            self._close_position(idx, fraction=1.0, exit_reason="stop_loss")
            return True

        # Take-profit check
        if direction > 0 and row["high"] >= self.current_tp:
            self._close_position(idx, fraction=0.5, exit_reason="scale_out_tp")
            # Trail stop to breakeven after partial profit
            if self.position != 0:
                self.current_stop = max(self.current_stop, self.avg_entry_price)
            return False  # Position still open (partially)

        if direction < 0 and row["low"] <= self.current_tp:
            self._close_position(idx, fraction=0.5, exit_reason="scale_out_tp")
            if self.position != 0:
                self.current_stop = min(self.current_stop, self.avg_entry_price)
            return False

        return False

    # ── Main loop ────────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """
        Execute the backtest bar by bar.

        Returns
        -------
        pd.DataFrame: Bar-by-bar equity and P&L records
        """
        print(f"[Backtest] Running {len(self.bars):,} bars | "
              f"Initial equity: ${self.initial_equity:,.2f}")

        records = []

        for i, (ts, row) in enumerate(self.bars.iterrows()):
            # Apply execution delay (signals from bar i arrive at bar i+delay)
            signal_idx = max(0, i - self.execution_delay)
            if signal_idx >= len(self.signals):
                break

            sig = self.signals.iloc[signal_idx]
            _raw_action = sig.get("action", 0)
            try:
                _a = float(_raw_action)
            except (TypeError, ValueError):
                action = 0
            else:
                action = int(_a) if np.isfinite(_a) else 0
            stop_loss = float(sig.get("stop_loss", 0))
            take_profit = float(sig.get("take_profit", 0))
            lots_to_trade = float(sig.get("lots", 0.1))

            # 1. Check stops/take-profits first
            stopped = self._check_stops(i)

            if not stopped:
                # 2. Execute signal
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

            # Update holding time
            if self.position != 0:
                self.holding_bars += 1

            # Mark-to-market unrealised P&L
            if self.position != 0:
                direction = np.sign(self.position)
                unrealised = (row["close"] - self.avg_entry_price) * direction * abs(self.position) * self.lot_size
            else:
                unrealised = 0.0

            self.peak_equity = max(self.peak_equity, self.equity + unrealised)
            drawdown = max(0, (self.peak_equity - (self.equity + unrealised)) / self.peak_equity)

            self.equity_curve.append(self.equity)
            records.append({
                "timestamp": ts,
                "equity": self.equity,
                "unrealised_pnl": unrealised,
                "total_value": self.equity + unrealised,
                "position": self.position,
                "drawdown": drawdown,
                "holding_bars": self.holding_bars,
            })

            # Risk Integrity: Max Drawdown Circuit Breaker
            if drawdown > self.max_drawdown_limit:
                print(f"[Backtest] Max drawdown circuit breaker triggered at {ts}: {drawdown:.2%} (Limit: {self.max_drawdown_limit:.2%})")
                self._close_position(i, 1.0, "circuit_breaker")
                break

        # Force-close at end
        if self.position != 0:
            self._close_position(len(self.bars) - 1, 1.0, "end_of_data")

        self.results_df = pd.DataFrame(records).set_index("timestamp")
        return self.results_df

    # ── Performance reporting ────────────────────────────────────────────────

    def performance_metrics(self) -> dict:
        """Compute comprehensive performance statistics."""
        if not self.trades:
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

            returns = self.results_df["equity"].pct_change().dropna()

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
