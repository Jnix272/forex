"""
backtesting/improvements.py
============================
Backtesting additions:
  1. MonteCarloBacktest   - randomize trade order 1000x for confidence intervals
  2. SlippageCalibrator   - fit power-law slippage model to real fill data
  3. LockboxTest          - held-out 2024 data, evaluated only once before live
"""

import json
import warnings
from datetime import UTC, datetime
from pathlib import Path

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

from config.settings import PATHS

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1. MONTE CARLO BACKTEST
# ─────────────────────────────────────────────────────────────────────────────


class MonteCarloBacktest:
    """
    Runs 1000 bootstrap resamples of the trade sequence to build confidence
    intervals on Sharpe ratio and max drawdown.

    Why this matters:
      A single backtest Sharpe of 1.8 could be luck - maybe you happened
      to catch 3 big trend days in the right order. Monte Carlo shows
      whether the strategy is robust to different trade orderings or if it
      depends on a specific sequence of lucky events.

    Method: bootstrap trade P&Ls with replacement, compute Sharpe + maxDD for
    each resample, report [5th, 50th, 95th] percentiles.
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        confidence: float = 0.95,
        initial_equity: float = 10_000.0,
        random_seed: int = 42,
    ):
        self.n_sims = n_simulations
        self.conf = confidence
        self.equity = initial_equity
        self.seed = random_seed

    def run(
        self,
        trade_pnls: np.ndarray,  # Array of per-trade P&L in USD
        annual_factor: float | None = None,
        trades_per_year: float | None = None,
    ) -> dict:
        """
        Resample trades N times, compute Sharpe + max drawdown per sample.

        Parameters
        ----------
        trade_pnls : array of realized P&L per trade (not per bar)
        annual_factor : explicit annualization multiplier under the square root.
            Prefer ``trades_per_year`` for trade-based P&Ls. When both are
            None, Sharpe is left unannualized (factor=1) rather than pretending
            each trade is a trading day (legacy ``annual_factor=252``).
        trades_per_year : estimated closed trades per year; used when
            ``annual_factor`` is None.

        Returns
        -------
        dict with Sharpe and max_drawdown at [5th, median, 95th] percentiles
        """
        from evaluation.monte_carlo import block_bootstrap_indices

        trade_pnls = np.asarray(trade_pnls, dtype=np.float64)
        trade_pnls = trade_pnls[np.isfinite(trade_pnls)]
        n = len(trade_pnls)
        if n < 2:
            return self._empty_result(n, "Need at least 2 closed trades for Monte Carlo")

        if annual_factor is not None:
            ann_factor = float(annual_factor)
            ann_note = "annual_factor"
        elif trades_per_year is not None:
            ann_factor = float(trades_per_year)
            ann_note = "trades_per_year"
        else:
            # Unannualized trade Sharpe - honest default for per-trade P&L.
            ann_factor = 1.0
            ann_note = "unannualized"

        # Facade: resample via evaluation.monte_carlo (Improvement #3).
        # block_length=1 reduces the block bootstrap to the historical
        # i.i.d. bootstrap-with-replacement over trades.
        idx = block_bootstrap_indices(n, block_length=1, n_bootstraps=self.n_sims, seed=self.seed)

        # Vectorized 2D processing for all simulations at once
        idx = np.asarray(idx)  # shape: (n_sims, n)
        pnls_2d = trade_pnls[idx]  # shape: (n_sims, n)

        # Cumulative equity
        eq = np.empty((self.n_sims, n + 1), dtype=np.float64)
        eq[:, 0] = self.equity
        eq[:, 1:] = self.equity + np.cumsum(pnls_2d, axis=1)

        # Sharpe ratio
        rets = pnls_2d / self.equity
        rets_mean = rets.mean(axis=1)
        rets_std = rets.std(ddof=1, axis=1)
        valid_std = rets_std > 1e-12
        sharpes = np.zeros(self.n_sims, dtype=np.float64)
        sharpes[valid_std] = (rets_mean[valid_std] / rets_std[valid_std]) * np.sqrt(ann_factor)

        # Max drawdown
        peaks = np.maximum.accumulate(eq, axis=1)
        dds = np.where(peaks > 0, (peaks - eq) / peaks, 1.0)
        max_dds = dds.max(axis=1)

        lo = (1 - self.conf) / 2
        hi = 1.0 - lo
        pct_lo = 100.0 * lo
        pct_hi = 100.0 * hi

        # Original (unshuffled) stats
        orig_rets = trade_pnls / self.equity
        orig_sharpe = self._safe_sharpe(orig_rets, ann_factor)
        orig_eq = self.equity + np.cumsum(trade_pnls)
        orig_eq = np.concatenate([[self.equity], orig_eq])
        orig_peak = np.maximum.accumulate(orig_eq)
        orig_dd = float(np.where(orig_peak > 0, (orig_peak - orig_eq) / orig_peak, 1.0).max())

        # Percentile that original Sharpe falls in -> robustness indicator
        sharpe_pct = float(np.mean(sharpes < orig_sharpe))

        result = {
            "n_trades": n,
            "n_simulations": self.n_sims,
            "original_sharpe": round(orig_sharpe, 4),
            "original_max_dd": round(orig_dd, 4),
            "sharpe_5th": round(float(np.percentile(sharpes, pct_lo)), 4),
            "sharpe_median": round(float(np.percentile(sharpes, 50)), 4),
            "sharpe_95th": round(float(np.percentile(sharpes, pct_hi)), 4),
            "sharpe_percentile": round(sharpe_pct, 4),
            "max_dd_5th": round(float(np.percentile(max_dds, pct_lo)), 4),
            "max_dd_median": round(float(np.percentile(max_dds, 50)), 4),
            "max_dd_95th": round(float(np.percentile(max_dds, pct_hi)), 4),
            "prob_sharpe_above_1": round(float(np.mean(sharpes > 1.0)), 4),
            "prob_sharpe_above_0": round(float(np.mean(sharpes > 0.0)), 4),
            "robust": bool(sharpe_pct > 0.75 and np.percentile(sharpes, pct_lo) > 0.0),
            "method": "bootstrap_with_replacement",
            "annualization": ann_note,
            "annual_factor": ann_factor,
            "warning": ""
            if ann_note != "unannualized"
            else ("Sharpe is unannualized (per-trade). Pass trades_per_year to annualize."),
        }

        self._print_report(result)
        return result

    def _safe_sharpe(self, returns: np.ndarray, annual_factor: float) -> float:
        returns = np.asarray(returns, dtype=np.float64)
        returns = returns[np.isfinite(returns)]
        if len(returns) < 2:
            return 0.0
        std = float(returns.std(ddof=1))
        if std <= 1e-12:
            return 0.0
        return float(returns.mean() / std * np.sqrt(annual_factor))

    def _empty_result(self, n_trades: int, warning: str) -> dict:
        result = {
            "n_trades": n_trades,
            "n_simulations": self.n_sims,
            "original_sharpe": 0.0,
            "original_max_dd": 0.0,
            "sharpe_5th": 0.0,
            "sharpe_median": 0.0,
            "sharpe_95th": 0.0,
            "sharpe_percentile": 0.0,
            "max_dd_5th": 0.0,
            "max_dd_median": 0.0,
            "max_dd_95th": 0.0,
            "prob_sharpe_above_1": 0.0,
            "prob_sharpe_above_0": 0.0,
            "robust": False,
            "method": "bootstrap_with_replacement",
            "warning": warning,
        }
        self._print_report(result)
        return result

    def _print_report(self, r: dict):
        robust = "ROBUST ✓" if r["robust"] else "NOT ROBUST ✗"
        print(f"\n[MC Backtest] {r['n_simulations']:,} simulations | {r['n_trades']} trades")
        print(f"  Status: {robust}")
        print(f"  Original Sharpe:  {r['original_sharpe']:.3f} (beats {r['sharpe_percentile']:.0%} of permutations)")
        print(f"  Sharpe CI [{r['sharpe_5th']:.2f}, {r['sharpe_median']:.2f}, {r['sharpe_95th']:.2f}]")
        print(f"  Max DD  CI [{r['max_dd_5th']:.2%}, {r['max_dd_median']:.2%}, {r['max_dd_95th']:.2%}]")
        print(f"  P(Sharpe > 1.0): {r['prob_sharpe_above_1']:.1%}")

    def run_from_backtest(self, backtest_obj) -> dict:
        """
        Convenience: pass the ForexScalingBacktest object directly.
        Extracts trade P&L automatically.
        """
        if hasattr(backtest_obj, "_trade_pnls"):
            return self.run(np.array(backtest_obj._trade_pnls))
        elif hasattr(backtest_obj, "trades"):
            pnls = [
                (
                    t.get("pnl_usd", t.get("pnl", 0))
                    if isinstance(t, dict)
                    else getattr(t, "pnl_usd", getattr(t, "pnl", 0))
                )
                for t in backtest_obj.trades
            ]
            return self.run(np.array(pnls))
        else:
            raise ValueError("Backtest object must have ._trade_pnls or .trades attribute")


# ─────────────────────────────────────────────────────────────────────────────
# 2. SLIPPAGE CALIBRATOR
# ─────────────────────────────────────────────────────────────────────────────


class SlippageCalibrator:
    """
    Fits a power-law slippage model to real fill data from LMAX or a live broker.

    Standard slippage model (Almgren et al.):
      slippage = α x (order_size / ADV)^β

    Where:
      α   = impact coefficient (fitted)
      β   = exponent (typically 0.5 for square-root impact, fitted from data)
      ADV = average daily volume

    Without calibration, most backtests assume a fixed slippage (e.g. 1 pip).
    Real slippage is:
      - Near-zero for small orders in liquid session
      - 2–5x higher during news events
      - Correlated with the current spread

    Usage:
        cal = SlippageCalibrator()
        cal.fit(real_fills_df)          # From LMAX fill export
        slip = cal.predict(lots=2.0, spread_pips=1.2, session="london_ny")
    """  # noqa: RUF002

    def __init__(self, adv_lots: float = 5000.0):
        self.adv = adv_lots
        self.alpha_: float = 0.5  # Default impact coefficient
        self.beta_: float = 0.5  # Default exponent (square-root)
        # Production keys only (asia/london/ny/asia_london/london_ny/off).
        # Relative to london_ny via shared session_spread_mult / LABEL_REGIME.
        try:
            from trading.session_utils import default_session_slip_factors

            self.session_factors: dict[str, float] = default_session_slip_factors()
        except Exception:
            self.session_factors = {
                "london_ny": 1.0,
                "london": 1.06,
                "ny": 1.06,
                "asia_london": 1.18,
                "asia": 1.41,
                "off": 1.76,
            }
        self._fitted = False

    def fit(
        self,
        fills_df,
    ) -> dict:
        """
        Fit slippage model to real fill data.

        Accepts polars or pandas DataFrame.

        fills_df columns:
          lots          : order size in lots
          requested_price : requested entry price
          fill_price    : actual fill price
          direction     : +1 long / -1 short
          session       : production key or legacy alias (tokyo→asia, overnight→off)

        Slippage = (fill_price - requested_price) x direction (negative = adverse)
        """
        from trading.session_utils import normalize_session_name

        if _POLARS_OK and pl is not None and isinstance(fills_df, pl.DataFrame):
            if pd is None:
                raise TypeError("pandas is required to normalize a polars fills DataFrame")
            df = fills_df.to_pandas()
        elif _PANDAS_OK and pd is not None and isinstance(fills_df, pd.DataFrame):
            df = fills_df.copy()
        else:
            raise TypeError(f"fills_df must be a polars or pandas DataFrame, got {type(fills_df)}")

        assert pd is not None
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"fills_df did not normalize to pandas DataFrame, got {type(df)}")

        slip = ((df["fill_price"] - df["requested_price"]) * df["direction"] / 0.0001).to_numpy(dtype=np.float64)
        df["slip_pips"] = np.maximum(slip, 0.0)  # Adverse slippage only
        df = df[df["lots"] > 0.001]

        if len(df) < 10:
            print("[Slippage] Insufficient data (<10 fills) - using defaults")
            return {"alpha": self.alpha_, "beta": self.beta_}

        # Fit power law: log(slip) = log(α) + β x log(lots/ADV)  # noqa: RUF003
        x = np.log(df["lots"] / self.adv + 1e-9)
        y = np.log(np.maximum(np.asarray(df["slip_pips"], dtype=np.float64), 0.01))

        try:
            beta, log_alpha = np.polyfit(x, y, 1)
            self.alpha_ = float(np.exp(log_alpha))
            self.beta_ = float(np.clip(beta, 0.3, 1.0))

            # Fit session multipliers (normalize legacy names onto production keys)
            if "session" in df.columns:
                df = df.copy()
                df["session"] = df["session"].map(normalize_session_name)
                ref_slip = self.alpha_ * (1.0 / self.adv) ** self.beta_
                for sess in df["session"].dropna().unique():
                    sk = normalize_session_name(sess)
                    if sk not in self.session_factors:
                        self.session_factors[sk] = 1.5
                    sess_slip = df[df["session"] == sess]["slip_pips"].mean()
                    if sess_slip > 0 and ref_slip > 0:
                        self.session_factors[sk] = float(sess_slip / ref_slip)

            self._fitted = True
            print(f"[Slippage] Fitted | α={self.alpha_:.4f} | β={self.beta_:.3f} | n_fills={len(df)}")  # noqa: RUF001
        except Exception as e:
            print(f"[Slippage] Fit failed: {e} - using defaults")

        return {"alpha": self.alpha_, "beta": self.beta_, "session_factors": self.session_factors, "n_fills": len(df)}

    def predict(
        self,
        lots: float,
        spread_pips: float = 1.0,
        session: str = "london_ny",
        urgency: float = 1.0,  # 1.0 = market order, 0.5 = limit order
    ) -> float:
        """
        Predict slippage in pips for a given order.

        Returns expected adverse slippage in pips.
        """
        from trading.session_utils import normalize_session_name, session_spread_mult

        # Base impact
        impact = self.alpha_ * (lots / self.adv) ** self.beta_

        # Session multiplier (production keys + legacy aliases)
        sk = normalize_session_name(session)
        if sk in self.session_factors:
            sess_mult = self.session_factors[sk]
        else:
            # Shared SoT relative to london_ny when factor table lacks the key
            ref = float(session_spread_mult("london_ny") or 0.85)
            sess_mult = float(session_spread_mult(sk)) / ref if ref else 1.5

        # Spread component (wider spread -> more slippage risk)
        spread_mult = 1.0 + 0.3 * max(spread_pips - 1.0, 0)

        # Urgency (limit orders have lower impact)
        total = impact * sess_mult * spread_mult * urgency

        return float(np.clip(total, 0.0, 10.0))

    def calibrate_from_lmax(self, fill_csv_path: str) -> dict:
        """
        Parse LMAX fill export CSV and fit model.

        Expected LMAX fill report columns (adjust as needed):
          DateTime, InstrumentId, Side, Quantity, RequestedPrice, FillPrice
        """
        try:
            # Prefer pandas for CSV parsing with datetime support
            if _PANDAS_OK and pd is not None:
                df = pd.read_csv(
                    fill_csv_path,
                    parse_dates=["DateTime"],
                    iterator=False,
                )
                df.columns = df.columns.str.lower().str.strip()
                col_map = {}
                for c in df.columns:
                    if "quantity" in c or "size" in c or "lot" in c:
                        col_map[c] = "lots"
                    elif "request" in c:
                        col_map[c] = "requested_price"
                    elif "fill" in c and "price" in c:
                        col_map[c] = "fill_price"
                    elif "side" in c or "dir" in c:
                        col_map[c] = "direction"
                df = df.rename(columns=col_map)
                if "direction" in df.columns:
                    _dir = {"buy": 1, "sell": -1, "Buy": 1, "Sell": -1}
                    df["direction"] = df["direction"].map(_dir)  # type: ignore[arg-type]
            elif _POLARS_OK and pl is not None:
                # Polars fallback
                df = pl.read_csv(fill_csv_path, try_parse_dates=True)
                df = df.rename({c: c.lower().strip() for c in df.columns})
                col_map = {}
                for c in df.columns:
                    if "quantity" in c or "size" in c or "lot" in c:
                        col_map[c] = "lots"
                    elif "request" in c:
                        col_map[c] = "requested_price"
                    elif "fill" in c and "price" in c:
                        col_map[c] = "fill_price"
                    elif "side" in c or "dir" in c:
                        col_map[c] = "direction"
                df = df.rename(col_map)
                if "direction" in df.columns:
                    df = df.with_columns(
                        pl.col("direction").map_elements(
                            lambda v: {"buy": 1, "sell": -1, "Buy": 1, "Sell": -1}.get(str(v), 0),
                            return_dtype=pl.Int8,
                        )
                    )
            else:
                raise RuntimeError("Neither pandas nor polars is available to read CSV.")

            return self.fit(df)
        except Exception as e:
            print(f"[Slippage] LMAX CSV parse error: {e}")
            return {}

    def save(self, path: str):
        json.dump(
            {"alpha": self.alpha_, "beta": self.beta_, "session_factors": self.session_factors},
            open(path, "w"),  # noqa: SIM115
            indent=2,
        )

    def load(self, path: str):
        d = json.load(open(path))  # noqa: SIM115
        self.alpha_ = d["alpha"]
        self.beta_ = d["beta"]
        self.session_factors = d["session_factors"]
        self._fitted = True


# ─────────────────────────────────────────────────────────────────────────────
# 3. LOCKBOX TEST
# ─────────────────────────────────────────────────────────────────────────────


class LockboxTest:
    """
    Holds out a final out-of-sample test period that is NEVER used during
    development, hyperparameter search, or walk-forward retraining.

    The lockbox is only opened ONCE - when you're ready to commit the model
    to live trading. This prevents inadvertent overfit to the test period
    through repeated evaluation.

    Recommended lockbox: most recent 6 months of data.

    Usage:
        lb = LockboxTest(start="2024-01-01", end="2024-06-30")
        lb.register_model("haelt_v3", model_description="100ep, 20M ticks")
        # ... do ALL development with data before 2024-01-01 ...
        # Only once, when ready to go live:
        result = lb.evaluate(model, test_features, test_labels, test_bars)
        lb.seal()   # Prevents re-evaluation
    """

    def __init__(
        self,
        start: str = "2024-01-01",
        end: str = "2024-12-31",
        log_path: str | None = None,
        max_evals: int = 1,  # Only 1 evaluation allowed
    ):
        if log_path is None:
            log_path = PATHS["file_lockbox_log"]
        self.start = start
        self.end = end
        self.log_path = Path(log_path)
        self.max_evals = max_evals
        self._evals: list[dict] = []
        self._sealed = False
        self._model_registry: list[dict] = []
        self._load_log()

    def _load_log(self):
        if self.log_path.exists():
            data = json.load(open(self.log_path))  # noqa: SIM115
            self._evals = data.get("evals", [])
            self._sealed = data.get("sealed", False)
            self._model_registry = data.get("models", [])
            if self._evals:
                print(f"[Lockbox] LOADED - {len(self._evals)} prior evaluation(s)")
                if self._sealed:
                    print("[Lockbox] SEALED - no further evaluations allowed")

    def _save_log(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(
            {
                "sealed": self._sealed,
                "evals": self._evals,
                "models": self._model_registry,
                "period": {"start": self.start, "end": self.end},
            },
            open(self.log_path, "w"),  # noqa: SIM115
            indent=2,
        )

    def register_model(self, model_name: str, description: str = ""):
        """
        Register a model version before development begins.
        Establishes audit trail of what was developed before opening the lockbox.
        """
        entry = {
            "name": model_name,
            "description": description,
            "registered_at": datetime.now(UTC).isoformat(),
        }
        self._model_registry.append(entry)
        self._save_log()
        print(f"[Lockbox] Registered model: {model_name}")

    def check_data_leak(self, df) -> bool:
        """
        Verify that a DataFrame does NOT contain lockbox period data.
        Returns True if data is clean (no leak).
        Accepts polars or pandas DataFrames.
        """
        if df is None or len(df) == 0:
            return True

        # FIX I2: accept Polars DataFrames (no .index or pd.to_datetime available)
        if _POLARS_OK and pl is not None and isinstance(df, pl.DataFrame):
            # Try common timestamp column names
            ts_col = None
            for cand in ("timestamp_utc", "timestamp", "time", "date"):
                if cand in df.columns:
                    ts_col = cand
                    break
            if ts_col is None:
                # No timestamp column found - assume clean
                return True
            import datetime as _dt

            start_dt = _dt.datetime.fromisoformat(self.start).replace(tzinfo=_dt.UTC)
            end_dt = _dt.datetime.fromisoformat(self.end).replace(tzinfo=_dt.UTC)
            # Convert column to python datetimes for comparison
            ts_series = df[ts_col].cast(pl.Datetime("us", "UTC"), strict=False)
            overlap = int(df.filter((ts_series >= pl.lit(start_dt)) & (ts_series <= pl.lit(end_dt))).height)
            if overlap > 0:
                print(f"[Lockbox] ⚠ DATA LEAK DETECTED: {overlap} rows from lockbox period!")
                return False
            return True

        # Pandas path
        if _PANDAS_OK and pd is not None and isinstance(df, pd.DataFrame):
            if df.empty:
                return True
            idx = pd.to_datetime(df.index, utc=True)
            start_ts = pd.Timestamp(self.start, tz="UTC")
            end_ts = pd.Timestamp(self.end, tz="UTC")
            overlap = int(((idx >= start_ts) & (idx <= end_ts)).sum())
            if overlap > 0:
                print(f"[Lockbox] ⚠ DATA LEAK DETECTED: {overlap} rows from lockbox period!")
                return False
            return True

        raise TypeError(f"check_data_leak requires polars or pandas DataFrame, got {type(df)}")

    def evaluate(
        self,
        model_name: str,
        predictions: np.ndarray,  # Model signals: +1, 0, -1
        returns: np.ndarray,  # Actual forward returns
        trade_pnls: np.ndarray | None = None,
        annual_factor: float = 252 * 24 * 60,  # Default to 1-min bars
        notes: str = "",
    ) -> dict:
        """
        Evaluate model on the lockbox test period.
        CAN ONLY BE CALLED ONCE (or max_evals times).
        """
        if self._sealed:
            raise RuntimeError(
                "[Lockbox] SEALED - this test period has already been evaluated. "
                "Opening the lockbox a second time invalidates out-of-sample integrity."
            )

        if len(self._evals) >= self.max_evals:
            raise RuntimeError(f"[Lockbox] Maximum evaluations ({self.max_evals}) reached. The lockbox is exhausted.")

        print(f"\n[Lockbox] {'═' * 50}")
        print(f"[Lockbox] OPENING LOCKBOX - model: {model_name}")
        print(f"[Lockbox] Period: {self.start} -> {self.end}")
        print(f"[Lockbox] {'═' * 50}")

        # Compute metrics
        directional_acc = float(np.mean(np.sign(predictions) == np.sign(returns)))
        strategy_rets = predictions * returns
        std = strategy_rets.std(ddof=1) if len(strategy_rets) > 1 else 0.0
        sharpe = float(strategy_rets.mean() / (std + 1e-9) * np.sqrt(annual_factor)) if std > 1e-12 else 0.0

        # Max drawdown
        cum_eq = 10000 * np.cumprod(1 + strategy_rets)
        peak = np.maximum.accumulate(cum_eq)
        max_dd = float(((peak - cum_eq) / peak).max())

        # Monte Carlo on lockbox trades
        mc_result = {}
        if trade_pnls is not None and len(trade_pnls) > 0:
            mc = MonteCarloBacktest(n_simulations=500)
            mc_result = mc.run(trade_pnls)

        result = {
            "model": model_name,
            "evaluated_at": datetime.now(UTC).isoformat(),
            "lockbox_period": {"start": self.start, "end": self.end},
            "n_predictions": len(predictions),
            "directional_acc": round(directional_acc, 4),
            "sharpe": round(sharpe, 4),
            "max_drawdown": round(max_dd, 4),
            "total_return": round(float(strategy_rets.sum()), 6),
            "monte_carlo": mc_result,
            "notes": notes,
            "registered_models": [m["name"] for m in self._model_registry],
        }

        self._evals.append(result)

        # Auto-seal after first evaluation
        if len(self._evals) >= self.max_evals:
            self.seal()

        self._save_log()

        print("\n[Lockbox] RESULTS:")
        print(f"  Directional accuracy: {directional_acc:.1%}")
        print(f"  Sharpe ratio:         {sharpe:.3f}")
        print(f"  Max drawdown:         {max_dd:.1%}")
        if mc_result:
            print(f"  MC robust:            {mc_result.get('robust', '?')}")
        print(f"[Lockbox] {'═' * 50}\n")

        return result

    def seal(self):
        """Permanently seal the lockbox - no further evaluations."""
        self._sealed = True
        self._save_log()
        print("[Lockbox] SEALED - out-of-sample integrity preserved ✓")

    @property
    def is_sealed(self) -> bool:
        return self._sealed

    @property
    def n_evaluations(self) -> int:
        return len(self._evals)


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TESTS
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    if pd is None:
        raise RuntimeError("pandas is required for the backtesting smoke tests")

    print("Backtesting improvements - smoke tests")
    print("=" * 50)

    # Monte Carlo
    rng = np.random.default_rng(42)
    pnls = rng.normal(5, 30, 300)  # 300 trades
    mc = MonteCarloBacktest(n_simulations=500)
    result = mc.run(pnls)
    print(f"\n  MC: robust={result['robust']} | Sharpe CI: [{result['sharpe_5th']:.2f}, {result['sharpe_95th']:.2f}]")

    # Slippage calibrator
    print()
    sc = SlippageCalibrator()
    # Synthetic fill data
    fills = pd.DataFrame(
        {
            "lots": rng.uniform(0.1, 5.0, 200),
            "requested_price": rng.uniform(1.085, 1.090, 200),
            "fill_price": rng.uniform(1.085, 1.090, 200),
            "direction": rng.choice([1, -1], 200),
            "session": rng.choice(["london_ny", "london", "ny", "asia"], 200),
        }
    )
    fills["fill_price"] += fills["direction"] * rng.uniform(0, 0.0002, 200)
    sc.fit(fills)
    for lots, sess in [(0.1, "london_ny"), (1.0, "london_ny"), (3.0, "tokyo")]:
        slip = sc.predict(lots, spread_pips=1.1, session=sess)
        print(f"  Slippage {lots:.1f}L {sess}: {slip:.4f} pips")
    # Legacy tokyo alias → asia production key
    assert "tokyo" not in sc.session_factors or "asia" in sc.session_factors
    assert sc.predict(1.0, session="tokyo") == sc.predict(1.0, session="asia")

    # Lockbox
    print()
    with tempfile.TemporaryDirectory() as td:
        log_path = f"{td}/lockbox.json"
        lb = LockboxTest(start="2024-01-01", end="2024-06-30", log_path=log_path, max_evals=1)
        lb.register_model("haelt_v3", "100 epochs, 20M ticks, Huber loss")

        # Check data leak detection
        clean_df = pd.DataFrame(index=pd.date_range("2023-01-01", "2023-12-31", freq="1D", tz="UTC"))
        dirty_df = pd.DataFrame(index=pd.date_range("2024-03-01", "2024-04-01", freq="1D", tz="UTC"))
        print(f"\n  Data leak check (clean): {lb.check_data_leak(clean_df)}")
        print(f"  Data leak check (dirty): {lb.check_data_leak(dirty_df)}")

        # Evaluate
        n = 1000
        preds = rng.choice([-1, 0, 1], n)
        rets = rng.normal(0.001, 0.003, n)
        ev = lb.evaluate("haelt_v3", preds, rets, notes="First and only evaluation")
        print(f"\n  Lockbox sealed: {lb.is_sealed}")
        print(f"  Sharpe: {ev['sharpe']:.3f} | Dir acc: {ev['directional_acc']:.1%}")

    print("\nAll backtesting improvement tests passed ✓")
