"""
trading/live_engine.py
=======================
Live trading engine - connects all model components to broker execution.

Architecture:
  Kafka (live ticks) -> Feature pipeline -> TIP-Search inference
  -> UQ confidence filter -> Portfolio VaR check
  -> Regime Kelly sizing -> Almgren-Chriss execution
  -> DrawdownAwareExit guard -> Broker order submission

Supported brokers (via abstract interface):
  - LMAX FIX 4.4 (institutional, recommended)
  - OANDA v20 REST API
  - Interactive Brokers TWS

Run:
  python trading/live_engine.py --broker lmax --pair EURUSD --model haelt
"""

import json
import os
import signal
import subprocess
import sys
import time
import warnings

import numpy as np

try:
    import polars as pl

    _POLARS = True
except ImportError:
    pl = None
    _POLARS = False
import threading
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class _LazyPandas:
    """Import pandas on first real use to keep smoke imports lightweight."""

    _module = None

    def _load(self):
        if self._module is None:
            import pandas as pandas_module

            self._module = pandas_module
        return self._module

    def __getattr__(self, name):
        return getattr(self._load(), name)


pd = _LazyPandas()


class _LazySymbol:
    """Resolve a heavy class/function only when it is called or inspected."""

    def __init__(self, module_name: str, symbol_name: str):
        self.module_name = module_name
        self.symbol_name = symbol_name
        self._value = None

    def _load(self):
        if self._value is None:
            module = __import__(self.module_name, fromlist=[self.symbol_name])
            self._value = getattr(module, self.symbol_name)
        return self._value

    def __call__(self, *args, **kwargs):
        return self._load()(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._load(), name)


FeatureEngineer = _LazySymbol("features.feature_engineering", "FeatureEngineer")
AdvancedFeatureBuilder = _LazySymbol("features.advanced_features", "AdvancedFeatureBuilder")
MacroYieldFeatureBuilder = _LazySymbol("features.macro_features", "MacroYieldFeatureBuilder")
SentimentPipeline = _LazySymbol("features.finbert_sentiment", "SentimentPipeline")
fetch_oanda = _LazySymbol("data.fetch_oanda_sentiment", "run_collector")
DualStreamSentiment = _LazySymbol("pretrain.contrastive", "DualStreamSentiment")
TIPSearchManager = _LazySymbol("pretrain.contrastive", "TIPSearchManager")
DriftDetector = _LazySymbol("pretrain.contrastive", "DriftDetector")
RegimeConditionalKelly = _LazySymbol("risk.execution", "RegimeConditionalKelly")
AlmgrenChrissExecutor = _LazySymbol("risk.execution", "AlmgrenChrissExecutor")
SessionLimitsEnforcer = _LazySymbol("risk.execution", "SessionLimitsEnforcer")
DrawdownAwareExitPolicy = _LazySymbol("risk.execution", "DrawdownAwareExitPolicy")
PortfolioVaR = _LazySymbol("risk.execution", "PortfolioVaR")
ShadowModeDeployer = _LazySymbol("monitoring.infra_tools", "ShadowModeDeployer")
SHAPFeatureTracker = _LazySymbol("monitoring.infra_tools", "SHAPFeatureTracker")
DemotionMonitor = _LazySymbol("monitoring.demotion_monitor", "DemotionMonitor")
ForexPrometheusExporter = _LazySymbol("monitoring.prometheus_exporter", "ForexPrometheusExporter")
LiveLogger = _LazySymbol("monitoring.live_logger", "LiveLogger")
from config.settings import (  # noqa: E402
    ALERTS,
    FEATURES,
    GOVERNANCE,
    MONITORING,
    PATHS,
    RELOAD_MODEL_FLAG,
    active_checkpoint_dir,
    price_to_pips,
    resolve_checkpoint_paths,
)
from config.strategy_profiles import STRATEGY_PROFILES, strategy_profile  # noqa: E402

load_cross_asset_panel = _LazySymbol("data.cross_asset", "load_cross_asset_panel")
get_latest_headlines = _LazySymbol("data.news_feed", "get_latest_headlines")
from trading.live_actions import LiveAction, model_class_to_live_action  # noqa: E402

DisagreementGate = _LazySymbol("trading.live_guards", "DisagreementGate")
EconomicCalendarGuard = _LazySymbol("trading.live_guards", "EconomicCalendarGuard")
NoTradeZoneGate = _LazySymbol("trading.live_guards", "NoTradeZoneGate")
RegimeRouter = _LazySymbol("trading.live_guards", "RegimeRouter")
SpreadVolatilityGuard = _LazySymbol("trading.live_guards", "SpreadVolatilityGuard")
TradeJournal = _LazySymbol("trading.live_guards", "TradeJournal")


def consume_reload_flag(flag_path) -> bool:
    """Return True if reload_model.flag was present and atomically cleared."""
    flag_path = Path(flag_path)
    if not flag_path.is_file():
        return False
    try:
        flag_path.unlink(missing_ok=True)
        return True
    except Exception as exc:
        print(f"[Live] WARN: could not clear reload flag {flag_path}: {exc}")
        return False


def _is_numeric_dtype(dtype) -> bool:
    method = getattr(dtype, "is_numeric", None)
    if method is not None:
        return bool(method() if callable(method) else method)
    try:
        import numpy as _np

        return bool(_np.issubdtype(dtype, _np.number))
    except Exception:
        return str(dtype).lower().startswith(("float", "int", "uint", "double"))


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT LOADING + HOT RELOAD
# ─────────────────────────────────────────────────────────────────────────────


class _DemoAgent:
    """Random-action placeholder - only when --demo or no checkpoint."""

    returns_live_actions = True

    def __init__(self, label: str = "demo"):
        import random

        self._r = random
        self.label = label

    def select_action(self, obs):
        return self._r.randint(0, 2)

    def reset_buffer(self):
        pass


def _read_sidecar_model_name(pt_path: Path, fallback: str) -> str:
    try:
        from inference.onnx_inference import _read_training_config

        cfg = _read_training_config(pt_path, fallback)
        return str(cfg.get("model") or fallback).lower().strip()
    except Exception:
        return fallback


def _read_schema_n_features(paths) -> int | None:
    for candidate in (
        paths.onnx_path.with_suffix(".schema.json") if paths.onnx_path else None,
        paths.pt_path.with_suffix(".schema.json") if paths.pt_path else None,
        paths.checkpoint_dir / "production_best.schema.json",
    ):
        if candidate is None or not Path(candidate).is_file():
            continue
        try:
            data = json.loads(Path(candidate).read_text(encoding="utf-8"))
            value = data.get("n_features")
            if value:
                return int(value)
        except Exception:
            pass
    return None


def _ensemble_export_checkpoint(paths) -> Path:
    original = paths.checkpoint_dir / "ensemble" / "ensemble_meta_best.pt"
    if original.is_file() and original.with_suffix(original.suffix + ".json").is_file():
        return original
    return paths.pt_path


def build_inference_agents(
    model_name: str,
    runtime: str = "pytorch",
    demo: bool = False,
    seq_len: int = 60,
    n_features: int | None = None,
    checkpoint_dir: Path | None = None,
    use_rl_fast: bool = True,
    rl_algo: str = "dqn",
) -> tuple[object, object, dict]:
    """Load fast/slow agents from active checkpoint dir or fall back to demo."""
    paths = resolve_checkpoint_paths(model_name, checkpoint_dir)
    meta = {
        "checkpoint_dir": str(paths.checkpoint_dir),
        "pt_path": str(paths.pt_path) if paths.pt_path else None,
        "onnx_path": str(paths.onnx_path) if paths.onnx_path else None,
        "source": paths.source,
        "reload_flag": str(paths.reload_flag),
        "runtime": runtime,
        "demo": demo,
        "seq_len": seq_len,
        "n_features": n_features,
        "model_name": model_name,
    }

    if demo:
        print("[Live] WARN: --demo flag set - using random DemoAgent (no trained weights)")
        agent = _DemoAgent("demo")
        return agent, agent, meta

    arch_name = model_name
    if paths.pt_path is not None:
        arch_name = _read_sidecar_model_name(paths.pt_path, model_name)
    meta["arch_name"] = arch_name

    if str(arch_name).lower() == "ensemble" and runtime == "onnx" and paths.onnx_path is None:
        raise RuntimeError(
            "[Live] ONNX runtime requires an exported ensemble ONNX model. "
            "Export it via export-ensemble or use --runtime pytorch."
        )

    slow_engine = None
    if runtime == "onnx":
        ckpt_onnx = paths.onnx_path
        if ckpt_onnx is None and paths.pt_path is not None:
            print("[Live] ONNX not found - exporting from PyTorch checkpoint...")
            from inference.onnx_inference import export_to_onnx

            ckpt_onnx = Path(
                export_to_onnx(
                    checkpoint_path=str(paths.pt_path),
                    model_name=arch_name,
                    seq_len=seq_len,
                    n_features=n_features,
                )
            )
        if ckpt_onnx is None or not Path(ckpt_onnx).is_file():
            raise RuntimeError(
                f"[Live] No ONNX or PyTorch checkpoint under {paths.checkpoint_dir}. "
                "Train and promote a model, or pass --demo for paper testing."
            )
        from inference.onnx_inference import DirectMLInferenceEngine

        slow_engine = DirectMLInferenceEngine(
            onnx_path=str(ckpt_onnx),
            seq_len=seq_len,
            n_features=n_features,
        )
    else:
        if paths.pt_path is None or not Path(paths.pt_path).is_file():
            raise RuntimeError(
                f"[Live] No checkpoint found in {paths.checkpoint_dir} "
                f"(tried production_best.pt and {model_name}_best.pt). "
                "Train and promote a model, or pass --demo."
            )
        from inference.pytorch_inference import PyTorchInferenceEngine

        slow_engine = PyTorchInferenceEngine(
            checkpoint_path=str(paths.pt_path),
            model_name=arch_name,
            seq_len=seq_len,
            n_features=n_features,
        )

    fast_agent = slow_engine
    meta["rl_fast"] = False
    meta["rl_algo"] = rl_algo
    if use_rl_fast:
        try:
            from inference.rl_inference import build_rl_fast_agent

            rl_agent = build_rl_fast_agent(
                checkpoint_dir=str(paths.checkpoint_dir),
                model_name=arch_name,
                algo=rl_algo,
                seq_len=seq_len,
                n_features=n_features or getattr(slow_engine, "n_features", None),
            )
            if rl_agent is not None:
                fast_agent = rl_agent
                meta["rl_fast"] = True
                print("[Live] Fast agent: RL policy (TIP fast path)")
        except Exception as exc:
            print(f"[Live] RL fast agent unavailable ({exc}); using supervised for both paths")

    return fast_agent, slow_engine, meta


def _pandas_freq_to_polars(freq: str) -> str:
    mapping = {
        "1min": "1m",
        "1t": "1m",
        "min": "1m",
        "5min": "5m",
        "15min": "15m",
        "30min": "30m",
        "1h": "1h",
        "1H": "1h",
        "1d": "1d",
    }
    return mapping.get(str(freq), str(freq).replace("min", "m"))


def _ensure_polars_frame(df):
    if not _POLARS:
        return df
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    return df


def _last_float(features, col: str, default: float = 0.0) -> float:
    if col not in getattr(features, "columns", []):
        return float(default)
    try:
        if _POLARS and pl is not None and isinstance(features, pl.DataFrame):
            return float(features.select(pl.col(col).tail(1)).item())
        return float(pd.to_numeric(features[col], errors="coerce").iloc[-1])
    except Exception:
        return float(default)


def _tail_mean(features, col: str, n: int = 20, default: float = 0.0) -> float:
    if col not in getattr(features, "columns", []):
        return float(default)
    try:
        if _POLARS and pl is not None and isinstance(features, pl.DataFrame):
            return float(features.select(pl.col(col).tail(int(n)).mean()).item())
        return float(features[col].tail(int(n)).mean())
    except Exception:
        return float(default)


class LiveTickBuffer:
    """In-memory tick → OHLCV bar aggregator used by LiveTradingEngine."""

    def __init__(self, pair: str, bar_freq: str = "1min", max_bars: int = 500):
        self.pair = pair
        self.freq = pd.tseries.frequencies.to_offset(bar_freq)
        self.freq_str = str(bar_freq)
        self.max_bars = int(max_bars)
        self._ticks: deque = deque(maxlen=50_000)
        self._lock = threading.Lock()

    def push_tick(self, bid: float, ask: float, volume: float = 1.0, ts=None) -> None:
        ts = pd.Timestamp(ts or pd.Timestamp.utcnow())
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        with self._lock:
            self._ticks.append(
                {
                    "timestamp": ts,
                    "bid": float(bid),
                    "ask": float(ask),
                    "mid": (float(bid) + float(ask)) / 2.0,
                    "volume": float(volume),
                }
            )

    def seed_bars(self, ohlcv_df: pd.DataFrame) -> None:
        """Seed the buffer with historical OHLCV bars so the engine starts warm."""
        if ohlcv_df is None or ohlcv_df.empty:
            return
        with self._lock:
            self._seeded_bars = ohlcv_df.copy()

    def get_bars(self):
        with self._lock:
            ticks = list(self._ticks)
            seeded = getattr(self, "_seeded_bars", None)
        live_bars = None
        if len(ticks) >= 2:
            live_bars = self._aggregate_ticks(ticks)
        if seeded is not None and not seeded.empty:
            if live_bars is not None and not live_bars.empty:
                combined = pd.concat([seeded[~seeded.index.isin(live_bars.index)], live_bars]).sort_index()
                return combined.tail(self.max_bars)
            return seeded.tail(self.max_bars)
        if live_bars is not None and len(live_bars) > 0:
            return live_bars.tail(self.max_bars) if hasattr(live_bars, "tail") else live_bars
        return None

    def _aggregate_ticks(self, ticks):
        if _POLARS and pl is not None:
            try:
                return self._aggregate_ticks_polars(ticks)
            except Exception:
                pass
        return self._aggregate_ticks_pandas(ticks)

    def _aggregate_ticks_pandas(self, ticks):
        df = pd.DataFrame(ticks)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        ohlcv = df["mid"].resample(self.freq).agg(["first", "max", "min", "last"])
        ohlcv.columns = ["open", "high", "low", "close"]
        ohlcv["volume"] = df["volume"].resample(self.freq).sum()
        ohlcv["bid_close"] = df["bid"].resample(self.freq).last()
        ohlcv["ask_close"] = df["ask"].resample(self.freq).last()
        return ohlcv.dropna()

    def _aggregate_ticks_polars(self, ticks):
        frame = pl.from_dicts(ticks)
        every = _pandas_freq_to_polars(self.freq_str)
        bars = (
            frame.with_columns(pl.col("timestamp").cast(pl.Datetime(time_zone="UTC")))
            .sort("timestamp")
            .group_by_dynamic("timestamp", every=every)
            .agg(
                [
                    pl.col("mid").first().alias("open"),
                    pl.col("mid").max().alias("high"),
                    pl.col("mid").min().alias("low"),
                    pl.col("mid").last().alias("close"),
                    pl.col("volume").sum().alias("volume"),
                    pl.col("bid").last().alias("bid_close"),
                    pl.col("ask").last().alias("ask_close"),
                ]
            )
            .drop_nulls()
        )
        return bars


from dataclasses import dataclass  # noqa: E402


class BrokerInterface:
    def connect(self) -> bool:
        raise NotImplementedError

    def disconnect(self) -> None:
        raise NotImplementedError

    def get_bid_ask(self, pair: str) -> tuple[float, float]:
        raise NotImplementedError

    def market_order(
        self, pair: str, side: str, lots: float, *, stop_loss: float | None = None, take_profit: float | None = None
    ) -> dict:
        raise NotImplementedError(
            f"{type(self).__name__}.market_order is not implemented - "
            "refusing fake fills. Use PaperBroker or a real broker override."
        )

    def close_position(self, pair: str) -> dict:
        raise NotImplementedError(
            f"{type(self).__name__}.close_position is not implemented - "
            "refusing fake closes. Use PaperBroker or a real broker override."
        )

    def get_account(self) -> dict:
        return {}

    def get_positions(self) -> dict[str, float]:
        return {}


@dataclass
class LiveSafetyConfig:
    """Hard limits evaluated before every live order submission."""

    max_spread_pips: float = 2.5
    max_daily_loss_pct: float = 0.05
    max_orders_per_minute: int = 30


class LiveSafetyGate:
    """Deterministic pre-trade safety checks (spread, daily loss, rate limit)."""

    def __init__(self, config: LiveSafetyConfig, starting_equity: float):
        self.config = config
        self.starting_equity = float(starting_equity)
        self.halted = False
        self._order_times: deque = deque()
        self._current_day: int | None = None

    def new_day(self, equity: float) -> None:
        """Reset daily loss tracking at the start of a new trading day."""
        self.starting_equity = float(equity)
        self.halted = False
        self._current_day = datetime.now(UTC).timetuple().tm_yday

    def allow_order(
        self,
        pair: str,
        side: str,
        lots: float,
        bid: float,
        ask: float,
        equity: float,
        now: float | None = None,
    ) -> dict[str, object]:
        if self.halted:
            return {"ok": False, "reason": "halted"}

        ts = float(time.time() if now is None else now)
        spread_pips = max(0.0, price_to_pips(float(ask) - float(bid), pair))
        if spread_pips > float(self.config.max_spread_pips):
            return {
                "ok": False,
                "reason": f"spread_too_wide:{spread_pips:.2f}>{self.config.max_spread_pips}",
            }

        if self.starting_equity > 0:
            loss_pct = (self.starting_equity - float(equity)) / self.starting_equity
            if loss_pct >= float(self.config.max_daily_loss_pct):
                self.halted = True
                return {
                    "ok": False,
                    "reason": f"daily_loss_limit:{loss_pct:.4f}>={self.config.max_daily_loss_pct}",
                }

        window_start = ts - 60.0
        while self._order_times and self._order_times[0] < window_start:
            self._order_times.popleft()
        if len(self._order_times) >= int(self.config.max_orders_per_minute):
            return {"ok": False, "reason": "order_rate_limit"}

        self._order_times.append(ts)
        return {"ok": True, "reason": "", "pair": pair, "side": side, "lots": lots}


class PaperBroker(BrokerInterface):
    """
    In-memory broker for paper trading and unit tests.
    Tracks mark-to-market positions, fills orders at current bid/ask,
    and updates balance and equity on price changes and position closes.
    """

    def __init__(self, initial_equity: float = 10_000.0):
        self.initial_equity = float(initial_equity)
        self.balance = float(initial_equity)
        self.equity = float(initial_equity)
        self._bid = 1.10000
        self._ask = 1.10005
        self._quotes: dict[str, tuple[float, float]] = {}
        self._connected = False
        self._positions: dict[str, float] = {}  # pair -> signed lots
        self._position_entry: dict[str, float] = {}  # pair -> average entry price

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def update_quote(self, bid: float, ask: float, pair: str | None = None) -> None:
        self._bid = float(bid)
        self._ask = float(ask)
        if pair:
            p = str(pair).upper()
            self._quotes[p] = (float(bid), float(ask))
        self._update_equity()

    def get_bid_ask(self, pair: str) -> tuple[float, float]:
        p = str(pair).upper()
        if p in self._quotes:
            return self._quotes[p]
        return self._bid, self._ask

    def get_positions(self) -> dict[str, float]:
        return {p: float(v) for p, v in self._positions.items() if abs(v) > 1e-12}

    def _get_multiplier(self, pair: str) -> float:
        p = str(pair).upper()
        if "JPY" in p:
            return 1000.0
        return 100_000.0

    def _update_equity(self) -> None:
        unrealized = 0.0
        for pair, lots in self._positions.items():
            if abs(lots) <= 1e-12:
                continue
            entry = self._position_entry.get(pair, 0.0)
            if entry <= 0:
                continue
            bid, ask = self.get_bid_ask(pair)
            mult = self._get_multiplier(pair)
            if lots > 0:
                unrealized += (bid - entry) * mult * lots
            else:
                unrealized += (entry - ask) * mult * abs(lots)
        self.equity = round(self.balance + unrealized, 2)

    def get_account(self) -> dict:
        self._update_equity()
        return {"equity": self.equity, "balance": self.balance}

    def market_order(
        self, pair: str, side: str, lots: float, *, stop_loss: float | None = None, take_profit: float | None = None
    ) -> dict:
        pair_key = str(pair).upper()
        bid, ask = self.get_bid_ask(pair_key)
        is_buy = str(side).lower() in ("buy", "long")
        fill_price = ask if is_buy else bid
        order_lots = float(lots)
        signed_order = order_lots if is_buy else -order_lots

        curr_pos = self._positions.get(pair_key, 0.0)
        mult = self._get_multiplier(pair_key)

        if curr_pos != 0 and ((curr_pos > 0 and not is_buy) or (curr_pos < 0 and is_buy)):
            entry = self._position_entry.get(pair_key, fill_price)
            closing_lots = min(abs(curr_pos), order_lots)
            if curr_pos > 0:
                realized = (fill_price - entry) * mult * closing_lots
            else:
                realized = (entry - fill_price) * mult * closing_lots
            self.balance += realized
            new_pos = curr_pos + signed_order
            self._positions[pair_key] = new_pos
            if abs(new_pos) <= 1e-12:
                self._positions.pop(pair_key, None)
                self._position_entry.pop(pair_key, None)
            elif (curr_pos > 0 and new_pos < 0) or (curr_pos < 0 and new_pos > 0):
                self._position_entry[pair_key] = fill_price
        else:
            total_lots = abs(curr_pos) + order_lots
            prev_entry = self._position_entry.get(pair_key, fill_price)
            avg_entry = ((prev_entry * abs(curr_pos)) + (fill_price * order_lots)) / max(total_lots, 1e-12)
            self._positions[pair_key] = curr_pos + signed_order
            self._position_entry[pair_key] = avg_entry

        self._update_equity()
        return {
            "ok": True,
            "pair": pair,
            "side": side,
            "lots": lots,
            "fill_price": fill_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

    def close_position(self, pair: str) -> dict:
        pair_key = str(pair).upper()
        curr_pos = self._positions.pop(pair_key, 0.0)
        entry = self._position_entry.pop(pair_key, 0.0)
        if abs(curr_pos) > 1e-12 and entry > 0:
            bid, ask = self.get_bid_ask(pair_key)
            fill_price = bid if curr_pos > 0 else ask
            mult = self._get_multiplier(pair_key)
            if curr_pos > 0:
                realized = (fill_price - entry) * mult * curr_pos
            else:
                realized = (entry - fill_price) * mult * abs(curr_pos)
            self.balance += realized
            self._update_equity()
        return {"ok": True, "pair": pair}


class LMAXBroker(BrokerInterface):
    """
    LMAX live broker adapter.

    Pricing path uses the LMAX REST API when ``LMAX_USERNAME`` /
    ``LMAX_PASSWORD`` are set (via ``data.sources.LmaxDataSource``).
    Order routing still requires a FIX 4.4 session - ``market_order`` /
    ``close_position`` refuse fake fills until FIX is configured.
    """

    def __init__(self):
        self._client = None
        self._connected = False
        self._fix_initiator = None
        self._fix_app = None
        # Local position tracking (signed lots). LMAX REST/FIX position
        # queries are venue-specific; we mirror the effect of our own orders
        # so close_position can flatten without an external query.
        self._positions: dict[str, float] = {}

    def get_positions(self) -> dict[str, float]:
        return dict(self._positions)

    def connect(self) -> bool:
        import os

        user = os.getenv("LMAX_USERNAME")
        password = os.getenv("LMAX_PASSWORD")
        if not user or not password:
            print(
                "[Live] LMAXBroker: set LMAX_USERNAME and LMAX_PASSWORD for REST "
                "pricing; FIX credentials still required for orders"
            )
            self._connected = False
            return False
        try:
            from data.sources import LMAXLoader

            self._client = LMAXLoader(username=user, password=password, verbose=True)
            ok = bool(self._client.login())
            self._connected = ok
            if ok:
                print("[Live] LMAXBroker: REST session OK.")

                # FIX Init
                fix_cfg = os.getenv("LMAX_FIX_CONFIG")
                if fix_cfg:
                    try:
                        import quickfix as fix

                        from execution.lmax_fix_app import LMAXFixApp

                        settings = fix.SessionSettings(fix_cfg)
                        self._fix_app = LMAXFixApp(username=user, password=password)
                        storeFactory = fix.FileStoreFactory(settings)
                        logFactory = fix.ScreenLogFactory(settings)

                        self._fix_initiator = fix.SocketInitiator(self._fix_app, storeFactory, settings, logFactory)
                        self._fix_initiator.start()
                        print(f"[Live] LMAXBroker: FIX session started via {fix_cfg}")
                    except Exception as fix_e:
                        print(f"[Live] LMAXBroker: FIX init failed: {fix_e}")
                else:
                    print("[Live] LMAXBroker: LMAX_FIX_CONFIG not set; orders will be rejected.")

            return ok
        except Exception as e:
            print(f"[Live] LMAXBroker: REST connect failed ({e})")
            self._connected = False
            return False

    def disconnect(self) -> None:
        self._connected = False
        self._client = None
        if self._fix_initiator is not None:
            self._fix_initiator.stop()
            self._fix_initiator = None

    def get_bid_ask(self, pair: str) -> tuple[float, float]:
        if not self._connected or self._client is None:
            raise RuntimeError("LMAXBroker not connected - call connect() with LMAX_USERNAME/PASSWORD")
        book = self._client.fetch_orderbook(pair)
        if not book or book.get("best_bid") is None or book.get("best_ask") is None:
            raise RuntimeError(f"LMAXBroker: no book for {pair}")
        return float(book["best_bid"]), float(book["best_ask"])

    def market_order(
        self, pair: str, side: str, lots: float, *, stop_loss: float | None = None, take_profit: float | None = None
    ) -> dict:
        if self._fix_initiator is None or not getattr(self._fix_app, "connected", False):
            raise RuntimeError("LMAXBroker.market_order: FIX session not connected. Set LMAX_FIX_CONFIG.")

        import time
        import uuid

        import quickfix as fix

        msg = fix.Message()
        msg.getHeader().setField(fix.MsgType(fix.MsgType_NewOrderSingle))

        cl_ord_id = str(uuid.uuid4())[:16]
        msg.setField(fix.ClOrdID(cl_ord_id))
        msg.setField(fix.Symbol(pair))

        fix_side = fix.Side_BUY if side.upper() == "BUY" else fix.Side_SELL
        msg.setField(fix.Side(fix_side))

        msg.setField(fix.TransactTime(datetime.now(UTC).strftime("%Y%m%d-%H:%M:%S")))
        msg.setField(fix.OrderQty(float(lots)))
        msg.setField(fix.OrdType(fix.OrdType_MARKET))

        try:
            fix.Session.sendToTarget(msg, self._fix_app.session_id)
            print(f"[LMAX FIX] Sent NewOrderSingle: {cl_ord_id}")
            signed = float(lots) if side.upper() == "BUY" else -float(lots)
            self._positions[pair] = self._positions.get(pair, 0.0) + signed
            return {"ticket": cl_ord_id, "status": "sent"}
        except fix.SessionNotFound as e:
            raise RuntimeError(f"LMAXBroker FIX send failed: {e}")

    def close_position(self, pair: str) -> dict:
        if self._fix_initiator is None or not getattr(self._fix_app, "connected", False):
            raise RuntimeError("LMAXBroker.close_position: FIX session not connected.")
        net = float(self._positions.get(pair, 0.0))
        if net == 0.0:
            return {"ticket": None, "status": "no_position"}
        # Flatten by sending an opposite market order for the tracked size.
        flat_side = "SELL" if net > 0 else "BUY"
        flat_lots = abs(net)
        result = self.market_order(pair, flat_side, flat_lots)
        # Clear local tracking regardless - the flatten order is in flight.
        self._positions.pop(pair, None)
        return result


class BridgeBrokerAdapter(BrokerInterface):
    """Adapt ``execution.broker_bridge.BrokerBridge`` (MT5 / IBKR) to BrokerInterface."""

    def __init__(self, venue: str = "MT5", config: dict | None = None):
        from execution.broker_bridge import BrokerBridge

        self._bridge = BrokerBridge(broker=str(venue).upper(), config=config or {})
        self.venue = str(venue).upper()

    def connect(self) -> bool:
        return bool(self._bridge.connect())

    def disconnect(self) -> None:
        self._bridge.disconnect()

    def get_bid_ask(self, pair: str) -> tuple[float, float]:
        return self._bridge.get_bid_ask(pair)

    def market_order(
        self, pair: str, side: str, lots: float, *, stop_loss: float | None = None, take_profit: float | None = None
    ) -> dict:
        ok = self._bridge.execute_order(
            pair,
            side=str(side).upper(),
            lot_size=float(lots),
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        return {"ok": bool(ok), "pair": pair, "side": side, "lots": lots, "venue": self.venue}

    def close_position(self, pair: str) -> dict:
        positions = self._bridge.get_positions() or []
        closed = 0
        for pos in positions:
            sym = str(pos.get("symbol", "")).upper().replace("/", "").replace("_", "").replace(".", "")
            target = str(pair).upper().replace("/", "").replace("_", "").replace(".", "")
            if sym != target and not sym.startswith(target[:6]):
                continue
            ticket = pos.get("ticket")
            if ticket is None:
                continue
            if self._bridge.close_position(int(ticket)):
                closed += 1
        return {"ok": closed > 0, "pair": pair, "closed": closed, "venue": self.venue}

    def get_account(self) -> dict:
        try:
            equity = float(self._bridge.get_account_equity())
        except Exception as exc:
            raise RuntimeError(f"BridgeBrokerAdapter failed to fetch equity: {exc}") from exc
        return {"equity": equity}

    def get_positions(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for pos in self._bridge.get_positions() or []:
            sym = str(pos.get("symbol", "")).upper().replace(".", "")
            vol = float(pos.get("volume", 0) or 0)
            signed = vol if str(pos.get("type", "BUY")).upper() == "BUY" else -vol
            out[sym] = out.get(sym, 0.0) + signed
        return out


class OANDABroker(BrokerInterface):
    """OANDA v20 REST broker (practice/live via env host override).

    Tick feed: if OANDA_ZMQ_ENDPOINT is set (e.g. "tcp://127.0.0.1:5557"),
    get_bid_ask() reads from the C++ oanda_stream ZMQ PUB socket instead of
    making a REST call, giving sub-millisecond tick access with no per-call
    HTTP overhead. The C++ process must be running separately.
    Orders still go through the v20 REST API (OANDA has no FIX for retail).
    """

    # Binary tick frame layout published by oanda_stream.cpp
    # magic(4) + ts_us(8) + bid(8) + ask(8) + instrument(16) = 44 bytes
    _TICK_MAGIC  = 0x4F414E44  # "OAND"
    _TICK_STRUCT = None  # struct.Struct, lazily built

    def __init__(self):
        self._token = (
            os.environ.get("OANDA_BEARER_TOKEN")
            or os.environ.get("OANDA_API_TOKEN")
            or os.environ.get("OANDA_API_KEY")
        )
        self._account_id = os.environ.get("OANDA_ACCOUNT_ID")
        env = (os.environ.get("OANDA_ENV") or "practice").strip().lower()
        default_host = (
            "https://api-fxtrade.oanda.com"
            if env in ("live", "prod", "production")
            else "https://api-fxpractice.oanda.com"
        )
        self._host = (
            os.environ.get("OANDA_API_URL") or os.environ.get("OANDA_API_HOST") or default_host
        ).rstrip("/")
        self.units_per_lot = float(os.environ.get("OANDA_UNITS_PER_LOT", 10_000.0))
        self._quotes: dict[str, tuple[float, float]] = {}

        # Optional ZMQ tick cache (populated by C++ oanda_stream process)
        self._zmq_endpoint: str | None = os.environ.get("OANDA_ZMQ_ENDPOINT")
        self._zmq_ctx = None
        self._zmq_sub = None
        self._zmq_lock = None
        self._zmq_running = False
        if self._zmq_endpoint:
            self._init_zmq_subscriber()

    @property
    def _bid(self) -> float | None:
        return next((v[0] for v in self._quotes.values()), None)

    @property
    def _ask(self) -> float | None:
        return next((v[1] for v in self._quotes.values()), None)

    def _init_zmq_subscriber(self) -> None:
        """Connect to the C++ oanda_stream ZMQ PUB socket."""
        import threading
        try:
            import zmq as _zmq  # type: ignore[import]
            self._zmq_ctx = _zmq.Context.instance()
            self._zmq_sub = self._zmq_ctx.socket(_zmq.SUB)
            self._zmq_sub.setsockopt(_zmq.RCVTIMEO, 200)   # 200 ms timeout
            self._zmq_sub.setsockopt(_zmq.RCVHWM, 100)
            self._zmq_sub.setsockopt(_zmq.LINGER, 0)
            self._zmq_sub.setsockopt_string(_zmq.SUBSCRIBE, "")  # all topics
            self._zmq_sub.connect(self._zmq_endpoint)
            self._zmq_lock = threading.Lock()
            self._zmq_running = True
            # Cache: instrument -> (bid, ask, ts_us)
            self._zmq_cache: dict = {}
            # Background drain thread keeps cache fresh
            self._zmq_thread = threading.Thread(
                target=self._zmq_drain_loop, daemon=True, name="oanda-zmq-drain"
            )
            self._zmq_thread.start()
            print(f"[OANDABroker] ZMQ tick cache connected to {self._zmq_endpoint}")
        except Exception as exc:
            print(f"[OANDABroker] ZMQ init failed ({exc}); falling back to REST polling")
            self._zmq_sub = None
            self._zmq_endpoint = None
            self._zmq_running = False

    def _zmq_drain_loop(self) -> None:
        """Drain the ZMQ PUB socket and update the bid/ask cache."""
        import struct
        # magic(I) + ts_us(Q) + bid(d) + ask(d) + instrument(16s) = 44 bytes
        fmt = struct.Struct("<IQdd16s")
        while getattr(self, "_zmq_running", True):
            try:
                if self._zmq_sub is None:
                    break
                # Multipart: [topic_bytes, data_bytes]
                parts = self._zmq_sub.recv_multipart(flags=0)
                if len(parts) < 2:
                    continue
                data = parts[1]
                if len(data) < fmt.size:
                    continue
                magic, ts_us, bid, ask, inst_raw = fmt.unpack_from(data)
                if magic != self._TICK_MAGIC:
                    continue
                inst = inst_raw.rstrip(b"\x00").decode("ascii", errors="ignore")
                with self._zmq_lock:
                    self._zmq_cache[inst] = (bid, ask, ts_us)
            except Exception:
                if not getattr(self, "_zmq_running", True):
                    break

    def _zmq_bid_ask(self, pair: str) -> tuple[float, float] | None:
        """Return (bid, ask) from ZMQ cache if available and fresh (< 5 s)."""
        if self._zmq_sub is None or self._zmq_lock is None:
            return None
        import time as _time
        key = self._instrument(pair)
        with self._zmq_lock:
            entry = self._zmq_cache.get(key)
        if entry is None:
            return None
        bid, ask, ts_us = entry
        age_s = (_time.time() * 1e6 - ts_us) / 1e6
        if age_s > 5.0 or age_s < -5.0:
            return None  # stale or desynced — fall back to REST
        return float(bid), float(ask)

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "RFC3339",
        }

    @staticmethod
    def _instrument(pair: str) -> str:
        p = str(pair).upper().replace("/", "").replace("_", "")
        if "_" in str(pair):
            return str(pair).upper()
        if len(p) == 6:
            return f"{p[:3]}_{p[3:]}"
        return p

    def connect(self) -> bool:
        return bool(self._token and self._account_id)

    def disconnect(self) -> None:
        if self._zmq_sub is not None:
            try:
                self._zmq_running = False
                self._zmq_sub.close(linger=0)
            except Exception:
                pass
            self._zmq_sub = None
        return None

    def get_bid_ask(self, pair: str) -> tuple[float, float]:
        inst = self._instrument(pair)
        # Fast path: ZMQ cache populated by C++ oanda_stream process
        cached = self._zmq_bid_ask(pair)
        if cached is not None:
            self._quotes[inst] = cached
            return cached

        import json as _json
        import urllib.request
        import urllib.error

        url = f"{self._host}/v3/accounts/{self._account_id}/pricing?instruments={inst}"
        req = urllib.request.Request(url, headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = _json.loads(resp.read())
        except Exception as exc:
            if inst in self._quotes:
                return self._quotes[inst]
            raise RuntimeError(f"OANDA pricing fetch failed for {inst}: {exc}") from exc

        prices = data.get("prices") or []
        if not prices:
            if inst in self._quotes:
                return self._quotes[inst]
            raise RuntimeError(f"OANDA pricing empty for {inst}")
        px = prices[0]
        bids = px.get("bids") or [{"price": px.get("closeoutBid")}]
        asks = px.get("asks") or [{"price": px.get("closeoutAsk")}]
        b = float(bids[0]["price"])
        a = float(asks[0]["price"])
        self._quotes[inst] = (b, a)
        return b, a

    def get_account(self) -> dict:
        import json as _json
        import urllib.request
        import urllib.error

        url = f"{self._host}/v3/accounts/{self._account_id}/summary"
        req = urllib.request.Request(url, headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = _json.loads(resp.read())
            acct = data.get("account") or {}
            equity = float(acct.get("NAV") or acct.get("balance") or 0.0)
            if equity <= 0:
                raise RuntimeError("OANDA account summary returned empty/zero equity")
            return {
                "equity": equity,
                "balance": float(acct.get("balance") or 0.0),
            }
        except Exception as exc:
            raise RuntimeError(f"OANDA get_account failed: {exc}") from exc

    def get_candles(self, pair: str, count: int = 120, granularity: str = "M5") -> pd.DataFrame | None:
        """Fetch historical candles from OANDA v20 REST for immediate buffer warmup."""
        import json as _json
        import urllib.request
        import urllib.error

        inst = self._instrument(pair)
        gran = str(granularity).upper()
        if gran in ("1MIN", "1M", "M1"):
            gran = "M1"
        elif gran in ("5MIN", "5M", "M5"):
            gran = "M5"
        elif gran in ("15MIN", "15M", "M15"):
            gran = "M15"
        elif gran in ("1H", "H1", "60MIN"):
            gran = "H1"
        elif gran in ("1D", "D"):
            gran = "D"

        url = f"{self._host}/v3/instruments/{inst}/candles?count={int(count)}&granularity={gran}&price=MBA"
        req = urllib.request.Request(url, headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = _json.loads(resp.read())
            candles = data.get("candles") or []
            if not candles:
                return None
            records = []
            for c in candles:
                ts = pd.Timestamp(c["time"])
                mid = c.get("mid") or {}
                bid = c.get("bid") or mid
                ask = c.get("ask") or mid
                records.append(
                    {
                        "timestamp": ts,
                        "open": float(mid.get("o", 0.0)),
                        "high": float(mid.get("h", 0.0)),
                        "low": float(mid.get("l", 0.0)),
                        "close": float(mid.get("c", 0.0)),
                        "volume": float(c.get("volume", 1.0)),
                        "bid_close": float(bid.get("c", mid.get("c", 0.0))),
                        "ask_close": float(ask.get("c", mid.get("c", 0.0))),
                    }
                )
            df = pd.DataFrame(records).set_index("timestamp").sort_index()
            return df
        except Exception as exc:
            print(f"[OANDABroker] Warning: historical candles fetch failed for {inst} ({exc})")
            return None

    def market_order(
        self, pair: str, side: str, lots: float, *, stop_loss: float | None = None, take_profit: float | None = None
    ) -> dict:
        import json as _json
        import urllib.request
        import urllib.error

        units = round(float(lots) * self.units_per_lot)
        if str(side).lower() in ("sell", "short"):
            units = -abs(units)
        else:
            units = abs(units)
        order_body = {
            "type": "MARKET",
            "instrument": self._instrument(pair),
            "units": str(units),
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
        }
        if stop_loss is not None:
            order_body["stopLossOnFill"] = {"price": f"{float(stop_loss):.5f}"}
        if take_profit is not None:
            order_body["takeProfitOnFill"] = {"price": f"{float(take_profit):.5f}"}
        body = _json.dumps({"order": order_body}).encode("utf-8")
        url = f"{self._host}/v3/accounts/{self._account_id}/orders"
        req = urllib.request.Request(url, data=body, headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = _json.loads(resp.read())

            # OANDA v20 returns 201 Created with orderFillTransaction or orderCancelTransaction
            if "orderFillTransaction" in data:
                fill = data["orderFillTransaction"]
                return {
                    "ok": True,
                    "fill": fill,
                    "price": float(fill.get("price", 0.0) or 0.0),
                    "units": float(fill.get("units", 0.0) or 0.0),
                    "raw": data,
                }
            if "orderCancelTransaction" in data:
                cancel = data["orderCancelTransaction"]
                return {
                    "ok": False,
                    "reason": cancel.get("reason", "ORDER_CANCELLED"),
                    "details": cancel,
                    "raw": data,
                }
            if "orderRejectTransaction" in data:
                reject = data["orderRejectTransaction"]
                return {
                    "ok": False,
                    "reason": reject.get("reason", "ORDER_REJECTED"),
                    "details": reject,
                    "raw": data,
                }
            return {"ok": True, "raw": data}
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="ignore")
            try:
                err_json = _json.loads(err_body)
            except Exception:
                err_json = {"raw": err_body}
            return {
                "ok": False,
                "reason": f"http_error_{exc.code}",
                "error": err_json,
                "status_code": exc.code,
            }
        except Exception as exc:
            return {"ok": False, "reason": "network_error", "error": str(exc)}

    def close_position(self, pair: str) -> dict:
        import json as _json
        import urllib.request
        import urllib.error

        url = f"{self._host}/v3/accounts/{self._account_id}/positions/{self._instrument(pair)}/close"
        body = _json.dumps({"longUnits": "ALL", "shortUnits": "ALL"}).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=self._headers(), method="PUT")
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = _json.loads(resp.read())
            if "longOrderCancelTransaction" in data and "shortOrderCancelTransaction" in data:
                return {"ok": False, "reason": "close_cancelled", "details": data}
            return {"ok": True, "raw": data}
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="ignore")
            return {"ok": False, "reason": f"http_error_{exc.code}", "error": err_body}
        except Exception as exc:
            return {"ok": False, "reason": "network_error", "error": str(exc)}

    def get_positions(self):
        import json as _json
        import urllib.request
        import urllib.error

        req = urllib.request.Request(
            f"{self._host}/v3/accounts/{self._account_id}/positions",
            headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = _json.loads(resp.read())
        except Exception:
            return {}

        pos = {}
        for p in data.get("positions", []):
            inst = p["instrument"].replace("_", "")
            long_u = float(p.get("long", {}).get("units", 0))
            short_u = float(p.get("short", {}).get("units", 0))
            # OANDA usually returns short units as negative; abs() also covers
            # feeds/fixtures that report short size as a positive magnitude.
            net = long_u - abs(short_u)
            pos[inst] = net / self.units_per_lot
        return pos


def _align_next_bar(freq_str: str, now: datetime | None = None) -> datetime:
    if now is None:
        now = datetime.now(UTC)
    now_clean = now.replace(microsecond=0)
    try:
        offset = pd.tseries.frequencies.to_offset(freq_str)
        ts = pd.Timestamp(now_clean)
        next_ts = ts.floor(offset) + offset
        if next_ts <= ts:
            next_ts = ts + offset
        return next_ts.to_pydatetime()
    except Exception:
        return now_clean.replace(second=0) + pd.Timedelta(minutes=1)


class LiveTradingEngine:
    """Single-pair live loop: ticks → features → guards → size → broker."""

    def __init__(
        self,
        broker: BrokerInterface,
        fast_agent,
        slow_model,
        pair: str,
        equity: float,
        max_lots: float,
        confidence_thresh: float = 0.45,
        log_dir: str | None = None,
        sentiment_mode: str = "auto",
        prometheus_enabled: bool = True,
        calendar_file: str | None = None,
        journal_path: str | None = None,
        max_spread_pips: float = 2.5,
        guard_min_confidence: float = 0.45,
        bar_freq: str = "1min",
        inference_meta: dict | None = None,
        stop_loss_atr: float = 1.5,
        take_profit_atr: float = 1.5,
        no_trade_gate_enabled: bool = False,
        no_trade_threshold: float = 0.70,
        allow_paper_fallback: bool = False,
        risk_engine=None,
        cross_asset=None,
    ):
        self.broker = broker
        self.pair = str(pair).upper()
        self.equity = float(equity)
        self.max_lots = float(max_lots)
        self.conf_thr = float(confidence_thresh if confidence_thresh is not None else guard_min_confidence)
        self.stop_loss_atr = float(stop_loss_atr)
        self.take_profit_atr = float(take_profit_atr)
        self.allow_paper_fallback = bool(allow_paper_fallback)
        self.bar_freq = str(bar_freq)
        self._inference_meta = dict(inference_meta or {})
        self._equity_fetch_failures = 0

        try:
            from risk.risk_engine import RiskEngine

            self.risk_engine = risk_engine if risk_engine is not None else RiskEngine(equity=self.equity)
        except Exception as e:
            print(f"[Live] RiskEngine unavailable ({e}); continuing with legacy guards only")
            self.risk_engine = None

        self.log_dir = Path(log_dir or PATHS.get("logs_live", "logs/live"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S") + "_" + self.pair.lower()
        self.logger = LiveLogger(log_dir=str(self.log_dir), run_id=self.run_id, component="live_engine", verbose=True)
        self.logger.setup()

        # INF-008: Persistent execution audit trail
        try:
            from execution.execution_logger import ExecutionLogger

            self.exec_logger = ExecutionLogger(log_dir=str(self.log_dir / "execution"))
        except ImportError:
            self.exec_logger = None

        self.fe = FeatureEngineer(
            atr_window=FEATURES.get("atr_window", 14),
            lag_windows=FEATURES.get("lag_windows", [1, 5, 10]),
        )
        self.afb = AdvancedFeatureBuilder(hurst_windows=[30, 60])
        self.macro = MacroYieldFeatureBuilder()
        self.cross_asset = cross_asset
        if self.cross_asset is None:
            try:
                end = pd.Timestamp.utcnow()
                start = end - pd.Timedelta(days=45)
                self.cross_asset = load_cross_asset_panel(
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    cache_dir=str(Path(PATHS.get("data_processed", "data/processed")) / "cross_asset"),
                    source=os.getenv("CROSS_ASSET_SOURCE", "auto").strip() or "auto",
                )
                print(f"[Live] Cross-asset loaded: {len(self.cross_asset)} series")
            except Exception as e:
                print(f"[Live] Cross-asset unavailable ({e}); continuing without it")

        self.rck = RegimeConditionalKelly()
        self.ac = AlmgrenChrissExecutor()
        self.dae = DrawdownAwareExitPolicy()
        self.pvar = PortfolioVaR()
        try:
            from config.settings import LIVE_RISK as _LR

            self.session_limits = SessionLimitsEnforcer(
                session_limits=_LR.get("session_limits"),
            )
        except Exception:
            self.session_limits = SessionLimitsEnforcer()

        mode = (sentiment_mode or os.getenv("LIVE_SENTIMENT_MODE", "auto")).lower()
        if mode in ("off", "none", "neutral"):
            self.sentiment = None
            self.finbert = None
            self._sent_backend = mode
        else:
            self.sentiment = DualStreamSentiment(prefer_backend=mode, use_cache=True)
            self.finbert = SentimentPipeline()
            self._sent_backend = mode
        self.logger.event(
            "INFO", "sentiment_backend", f"[Live] Sentiment mode={mode}", pair=self.pair, mode=mode, backend=mode
        )

        flatten = os.getenv("LIVE_CALENDAR_FLATTEN", "0") not in ("0", "false", "False", "")
        self.calendar_guard = EconomicCalendarGuard(
            pair=self.pair,
            calendar_file=calendar_file,
            flatten_before_event=flatten,
        )
        self.spread_vol_guard = SpreadVolatilityGuard(max_spread_pips=max_spread_pips, pair=self.pair)
        self.regime_router = RegimeRouter()
        self.disagreement_gate = DisagreementGate(min_confidence=guard_min_confidence)
        self.no_trade_zone_gate = NoTradeZoneGate(threshold=no_trade_threshold, enabled=no_trade_gate_enabled)
        journal = journal_path or str(self.log_dir / f"trade_journal_{self.pair.lower()}.jsonl")
        self.trade_journal = TradeJournal(journal)

        class _Wrap:
            def __init__(self, m, action_adapter):
                self.m = m
                self._action_adapter = action_adapter

            def select_action(self, o):
                return self._action_adapter(self.m.select_action(o))

            def set_model(self, m, action_adapter):
                self.m = m
                self._action_adapter = action_adapter

            def set_agent_state(self, *args, **kwargs):
                if hasattr(self.m, "set_agent_state"):
                    return self.m.set_agent_state(*args, **kwargs)
                return None

            def __getattr__(self, item):
                return getattr(self.m, item)

        def _live_action_adapter(agent):
            def _adapt(action):
                action = int(action[0] if isinstance(action, tuple) else action)
                if bool(getattr(agent, "returns_live_actions", False)):
                    return action
                return model_class_to_live_action(action)

            return _adapt

        self._live_action_adapter = _live_action_adapter
        self._agent_wrap_fast = _Wrap(fast_agent, _live_action_adapter(fast_agent))
        self._agent_wrap_slow = _Wrap(slow_model, _live_action_adapter(slow_model))
        self.fast = self._agent_wrap_fast
        self.slow = self._agent_wrap_slow
        self.tip = TIPSearchManager(fast_agent=self.fast, slow_agent=self.slow)

        self.buf = LiveTickBuffer(self.pair, bar_freq=bar_freq)
        self.safety = LiveSafetyGate(LiveSafetyConfig(max_spread_pips=max_spread_pips), starting_equity=self.equity)
        self.drift = DriftDetector()
        self.shadow = ShadowModeDeployer()
        self.demotion = DemotionMonitor(
            sharpe_floor=float(GOVERNANCE.get("demotion_sharpe_floor", 0.5)),
            winrate_floor=float(GOVERNANCE.get("demotion_winrate_floor", 0.45)),
            window_trades=int(GOVERNANCE.get("demotion_window_trades", 300)),
            auto_rollback=True,
            verbose=True,
        )
        self.prom = None
        if prometheus_enabled and bool(ALERTS.get("prometheus_enabled", True)):
            self.prom = ForexPrometheusExporter(
                port=int(ALERTS.get("prometheus_port", 8000)),
                initial_equity=float(equity),
            )

        self._model_name = str(self._inference_meta.get("model_name") or "haelt")
        self._runtime = str(self._inference_meta.get("runtime") or "pytorch")
        self._checkpoint_dir = Path(self._inference_meta.get("checkpoint_dir") or active_checkpoint_dir())
        self._reload_flag = Path(self._inference_meta.get("reload_flag") or (self._checkpoint_dir / RELOAD_MODEL_FLAG))

        self._discord = None
        if bool(ALERTS.get("alert_on_demotion", True)):
            try:
                from monitoring.discord_alerts import DiscordAlerter

                self._discord = DiscordAlerter(
                    min_interval_s=float(ALERTS.get("discord_min_interval_s", 300)),
                    environment=str(ALERTS.get("discord_environment", "production")),
                    verbose=False,
                )
            except Exception as e:
                print(f"[Live] Discord alerter unavailable ({e})")

        self._expected_features = None
        _schema_path = self._checkpoint_dir / "production_best.schema.json"
        try:
            if _schema_path.is_file():
                with open(_schema_path, encoding="utf-8") as f:
                    _sc = json.load(f)
                names = _sc.get("feature_names") or _sc.get("features") or _sc.get("columns")
                if isinstance(names, list) and names:
                    self._expected_features = [str(c) for c in names]
                    print(f"[Live] Loaded feature schema: {len(self._expected_features)} expected features.")
        except Exception as e:
            print(f"[Live] WARN: Failed to load feature schema: {e}")

        self._running = False
        self._position = 0.0
        self._entry_price = 0.0
        self._holding_bars = 0
        self._halt_new_orders = False
        self._bar_log: list = []
        self._baseline_fitted = False
        self._retrain_lock = Path(PATHS.get("checkpoints", "checkpoints")) / "retrain_in_progress.lock"
        self._retrain_cooldown_s = float(MONITORING.get("retrain_cooldown_sec", 21600))
        self._last_retrain_ts = 0.0

    def start(self, max_bars: int | None = None):
        if not self.broker.connect():
            if not self.allow_paper_fallback and not isinstance(self.broker, PaperBroker):
                raise RuntimeError(
                    "[Live] Broker connection failed. Pass allow_paper_fallback=True "
                    "or --allow-paper-fallback only for intentional paper testing."
                )
            print("[Live] Broker connection failed - using PaperBroker (explicit fallback)")
            self.broker = PaperBroker(initial_equity=self.equity)
            if not self.broker.connect():
                raise RuntimeError("PaperBroker fallback failed to connect")
        else:
            try:
                self.broker.get_bid_ask(self.pair)
            except Exception as e:
                if not self.allow_paper_fallback and not isinstance(self.broker, PaperBroker):
                    raise RuntimeError(
                        f"[Live] Broker pricing probe failed ({e}). "
                        "Pass --allow-paper-fallback to continue on PaperBroker."
                    ) from e
                print(f"[Live] Broker pricing probe failed ({e}) - falling back to PaperBroker")
                self.broker = PaperBroker(initial_equity=self.equity)
                self.broker.connect()

        if hasattr(self.broker, "get_candles"):
            try:
                hist_df = self.broker.get_candles(self.pair, count=120, granularity=self.bar_freq)
                if hist_df is not None and not hist_df.empty:
                    self.buf.seed_bars(hist_df)
                    print(f"[Live] Preloaded {len(hist_df)} historical bars for {self.pair} buffer warmup")
            except Exception as exc:
                print(f"[Live] Warning: Historical candle preload failed for {self.pair} ({exc})")

        self._running = True
        signal.signal(signal.SIGINT, lambda *_: self.stop())
        try:
            signal.signal(signal.SIGTERM, lambda *_: self.stop())
        except Exception:
            pass
        self._start_sentiment_loop()
        if self.prom is not None:
            self.prom.start()
        print(f"[Live] Engine started for {self.pair}")
        bar_count = 0
        next_bar_time = self._next_bar()
        while self._running:
            if max_bars is not None and bar_count >= max_bars:
                break
            now = datetime.now(UTC)
            if now < next_bar_time:
                try:
                    bid, ask = self.broker.get_bid_ask(self.pair)
                    if bid and ask:
                        self.buf.push_tick(bid, ask)
                except Exception:
                    pass
                time.sleep(0.1)
                continue
            bar_ts = next_bar_time
            bars = self.buf.get_bars()
            if bars is not None and len(bars) >= 70:
                self._on_new_bar(bars, bar_count)
            bar_count += 1
            # Advance by fixed interval so slow processing never skips bars
            import pandas as _pd_bar
            _interval = _pd_bar.Timedelta(self.bar_freq)
            next_bar_time = bar_ts + _interval
            while next_bar_time <= datetime.now(UTC):
                next_bar_time += _interval
        self.stop()

    def _risk_trade_closed(self, mid: float, reason: str) -> None:
        """Feed a realised closed trade into RiskEngine so its daily-loss,
        consecutive-loss and return-series gates actually fire on the live path.

        Called at every position-flatten site (ATR stop, circuit breaker,
        drawdown guard, calendar flatten). Conventions match the engine:
        ``self._position`` is in mini-lots (10k units), ``pip_value_per_lot``
        is USD per pip per lot.
        """
        if self.risk_engine is None:
            return
        pos = float(abs(self._position))
        if pos <= 1e-12 or self._entry_price <= 0:
            return
        try:
            from config.settings import get_pip_size

            pip = get_pip_size(self.pair)
            pnl = (
                (float(mid) - self._entry_price)
                * (
                    float(self._position) / pos  # sign: +1 long, -1 short
                )
                / max(pip, 1e-12)
                * pos
            )
            direction = "long" if self._position > 0 else "short"
            self.risk_engine.close_position(self.pair)
            self.risk_engine.on_trade_closed(
                pnl=float(pnl),
                equity=self.equity,
                pair=self.pair,
                lots=pos,
                direction=direction,
            )
            self.trade_journal.record(
                {
                    "event": "trade_closed",
                    "reason": reason,
                    "pnl_usd": round(float(pnl), 4),
                    "position": self._position,
                    "entry": self._entry_price,
                    "exit": float(mid),
                }
            )
        except Exception:
            pass

    def _on_new_bar(self, bars, bar_idx: int):
        self._maybe_hot_reload()
        today = datetime.now(UTC).timetuple().tm_yday
        if getattr(self, "_last_trading_day", None) != today:
            self._last_trading_day = today
            self.safety.new_day(self.equity)
            self.dae.new_day()
            if self.risk_engine is not None:
                try:
                    self.risk_engine.new_day(self.equity)
                except Exception:
                    pass
        t0 = time.perf_counter()
        bars = _ensure_polars_frame(bars)
        try:
            features = _ensure_polars_frame(self.fe.build(bars, cross_asset=self.cross_asset))
            macro_df = self.macro.build(bars)
            if macro_df is not None and len(macro_df) > 0:
                macro_df = _ensure_polars_frame(macro_df)
                if "timestamp_utc" in macro_df.columns:
                    macro_df = macro_df.drop("timestamp_utc")
                if macro_df is not None and len(macro_df) == len(features):
                    macro_cols = [c for c in macro_df.columns if c not in features.columns]
                    if macro_cols:
                        if _POLARS and isinstance(features, pl.DataFrame):
                            features = features.hstack(macro_df.select(macro_cols))
                        else:
                            features = pd.concat([features, macro_df[macro_cols]], axis=1)
            if self.afb is not None:
                try:
                    adv_df = self.afb.build(bars, features)
                    if adv_df is not None and len(adv_df) == len(features):
                        adv_df = _ensure_polars_frame(adv_df)
                        adv_cols = [c for c in adv_df.columns if c not in features.columns and c != "timestamp_utc"]
                        if adv_cols:
                            if _POLARS and isinstance(features, pl.DataFrame):
                                features = features.hstack(adv_df.select(adv_cols))
                            else:
                                features = pd.concat([features, adv_df[adv_cols]], axis=1)
                except Exception as e:
                    self.logger.event("WARN", "afb_warning", f"[Live] AFB build skipped: {e}", pair=self.pair)
            bias = 0.0
            if self._sent_backend not in ("off", "none", "neutral"):
                try:
                    if self._sent_backend == "ollama":
                        bias = float(self.sentiment.get_bias())
                    else:
                        headlines = get_latest_headlines(limit=12) or ["Market update"]
                        bias = float(self.finbert.score_headlines(headlines))
                except Exception:
                    try:
                        headlines = get_latest_headlines(limit=12) or ["Market update"]
                        bias = float(self.finbert.score_headlines(headlines))
                        self._sent_backend = "finbert"
                        self.logger.event(
                            "WARN",
                            "sentiment_fallback",
                            "[Live] Sentiment fallback -> finbert",
                            pair=self.pair,
                            backend="finbert",
                        )
                    except Exception as e:
                        self.logger.event("ERROR", "feature_error", f"[Live] Feature error: {e}", pair=self.pair)
                        bias = 0.0
            if _POLARS and isinstance(features, pl.DataFrame):
                features = features.with_columns(pl.lit(bias).cast(pl.Float64).alias("finbert_sentiment"))
            else:
                features["finbert_sentiment"] = bias
        except Exception as e:
            self.logger.event("ERROR", "feature_error", f"[Live] Feature error: {e}", pair=self.pair)
            return

        if int(bar_idx) % int(MONITORING.get("check_freq_bars", 500)) == 0:
            try:
                self._check_drift(features)
            except Exception as e:
                self.logger.event("ERROR", "drift", f"[Live] Drift check failed: {e}", pair=self.pair)

        feature_cols = self._feature_columns(features)
        try:
            if _POLARS and isinstance(features, pl.DataFrame):
                obs = features.select(feature_cols).tail(1).to_numpy().reshape(-1).astype(np.float32)
            else:
                obs = features[feature_cols].tail(1).to_numpy().reshape(-1).astype(np.float32)
        except Exception as e:
            self.logger.event("ERROR", "feature_error", f"[Live] Feature error: {e}", pair=self.pair)
            return

        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        atr = _last_float(features, "atr_6", _last_float(features, f"atr_{FEATURES.get('atr_window', 14)}", 0.0005))
        bid, ask = self.broker.get_bid_ask(self.pair)
        mid = (float(bid) + float(ask)) / 2.0 if bid and ask else _last_float(features, "close", 0.0)

        # ATR stop-loss: flatten when adverse move exceeds stop_loss_atr * ATR
        if abs(self._position) > 0 and atr > 0 and self._entry_price > 0:
            stop_dist = float(self.stop_loss_atr) * float(atr)
            hit_sl = (self._position > 0 and mid <= self._entry_price - stop_dist) or (
                self._position < 0 and mid >= self._entry_price + stop_dist
            )
            if hit_sl:
                _close_ok = self.broker.close_position(self.pair)
                self.trade_journal.record(
                    {
                        "event": "stop_loss",
                        "reason": "atr_stop",
                        "stop_loss_atr": self.stop_loss_atr,
                        "atr": atr,
                        "entry": self._entry_price,
                        "mid": mid,
                        "position": self._position,
                    }
                )
                if not (isinstance(_close_ok, dict) and _close_ok.get("ok") is False):
                    self._risk_trade_closed(mid, "atr_stop")
                    self._position = 0.0
                self._holding_bars = 0
                self._entry_price = 0.0
                return

        state_kw = {
            "position_lots": self._position,
            "entry_price": self._entry_price,
            "equity": self.equity,
            "holding_bars": self._holding_bars,
            "current_price": mid,
        }
        self.fast.set_agent_state(**state_kw)
        self.slow.set_agent_state(**state_kw)

        prev_equity = self.equity
        try:
            acct = self.broker.get_account()
            if not acct or "equity" not in acct:
                raise RuntimeError("broker get_account returned no equity")
            self.equity = float(acct["equity"])
            self._equity_fetch_failures = 0
            if self.risk_engine is not None:
                _risk_mon = self.risk_engine.update_equity(self.equity)
                if _risk_mon.get("circuit_breaker") and abs(self._position) > 0:
                    _cb_ok = self.broker.close_position(self.pair)
                    if not (isinstance(_cb_ok, dict) and _cb_ok.get("ok") is False):
                        self._risk_trade_closed(mid, "risk_circuit_breaker")
                        self._position = 0.0
                        self._entry_price = 0.0
                        self._holding_bars = 0
                    self.trade_journal.record(
                        {
                            "event": "blocked",
                            "reason": "risk_circuit_breaker",
                            "details": _risk_mon.get("breach_reasons"),
                        }
                    )
                    return
        except Exception as e:
            self._equity_fetch_failures = getattr(self, "_equity_fetch_failures", 0) + 1
            self.logger.event(
                "WARN",
                "equity_fetch_failed",
                f"[Live] Broker equity fetch failed ({self._equity_fetch_failures}x): {e}",
                pair=self.pair,
            )
            if self._equity_fetch_failures >= 5:
                self.logger.event(
                    "ERROR", "equity_stale_halt", "[Live] 5 consecutive equity fetch failures - halting", pair=self.pair
                )
                self._running = False
                return
        pnl = self.equity - prev_equity
        if self.prom is not None:
            self.prom.update_equity(self.equity)
        if abs(pnl) > 0:
            self.demotion.on_trade_closed(pnl=pnl, equity=self.equity)
            if self.prom is not None:
                self.prom.update_trade(pnl, won=pnl > 0)
        demotion_alert = self.demotion.on_bar(self.equity)
        if demotion_alert and demotion_alert.get("demoted"):
            triggers = demotion_alert.get("triggers") or ["unknown"]
            self.logger.event(
                "WARN",
                "demotion",
                f"[Live] DEMOTION triggered: {triggers}",
                pair=self.pair,
                triggers=triggers,
            )
            if self._discord is not None:
                try:
                    status = demotion_alert.get("status") or {}
                    self._discord.send(
                        "model_demoted",
                        {
                            "Pair": self.pair,
                            "Triggers": ", ".join(str(t) for t in triggers),
                            "Equity": f"${self.equity:,.2f}",
                            "Sharpe": str(status.get("sharpe", "n/a")),
                            "WinRate": str(status.get("win_rate", "n/a")),
                        },
                    )
                except Exception as exc:
                    self.logger.event(
                        "ERROR",
                        "discord_alert_failed",
                        f"[Live] Discord demotion alert failed: {exc}",
                        pair=self.pair,
                    )
            self._trigger_retrain("demotion", details={"triggers": triggers})

        dae = self.dae.update(self.equity, pnl)
        if str(dae.get("action", "")).upper() in ("FLATTEN", "HALT", "CLOSE_ALL"):
            self._halt_new_orders = True
            if abs(self._position) > 0:
                self._risk_trade_closed(mid, "drawdown_guard")
                self.broker.close_position(self.pair)
                self._position = 0.0
                self._entry_price = 0.0
                self._holding_bars = 0
            self.trade_journal.record(
                {
                    "event": "blocked",
                    "reason": "drawdown_guard",
                    "time": datetime.now(UTC).isoformat(),
                    "bar": int(bar_idx),
                    "final_action": "HOLD",
                }
            )
            return

        calendar_result = self.calendar_guard.check(now=datetime.now(UTC))
        if calendar_result.blocked:
            self.logger.event("WARN", "calendar_guard", "[Live] Economic calendar block -> HOLD", pair=self.pair)
            self.trade_journal.record(
                {"event": "blocked", "reason": calendar_result.reason, "details": calendar_result.to_dict()}
            )
            if (
                calendar_result.details
                and calendar_result.details.get("flatten_before_event")
                and abs(self._position) > 0
            ):
                self._risk_trade_closed(mid, "calendar_flatten")
                self.broker.close_position(self.pair)
                self._position = 0.0
            return

        tip_out = (
            self.tip.select_action(obs, current_atr=atr)
            if hasattr(self.tip, "select_action")
            else self.fast.select_action(obs)
        )
        if isinstance(tip_out, tuple):
            action = int(tip_out[0])
            model_used = str(tip_out[1]) if len(tip_out) > 1 else "unknown"
        else:
            action = int(tip_out)
            model_used = "fast"
        # BUG-010: Track predictions for concept drift detection
        if not hasattr(self, "_recent_predictions"):
            from collections import deque

            self._recent_predictions = deque(maxlen=2000)
        self._recent_predictions.append(float(action))
        if self.prom is not None and hasattr(self.prom, "set_sentiment"):
            self.prom.set_sentiment(bias)
        if hasattr(self.sentiment, "filter_signal"):
            action = int(self.sentiment.filter_signal(action, bias))

        spread_result = self.spread_vol_guard.check(features, bid=bid, ask=ask)
        if spread_result.blocked:
            self.trade_journal.record({"event": "blocked", "reason": spread_result.reason})
            return
        regime_result = self.regime_router.route(features, calendar_blocked=calendar_result.blocked)
        is_tip = hasattr(self, "tip") and hasattr(self.tip, "select_action")
        disagreement_result = self.disagreement_gate.check(
            action,
            obs,
            fast_model=self.fast,
            slow_model=self.slow,
            confidence=None,
            bypass_disagreement=is_tip,
        )
        if disagreement_result.blocked:
            self.trade_journal.record({"event": "blocked", "reason": disagreement_result.reason})
            return

        no_trade_result = self.no_trade_zone_gate.check(features)
        if no_trade_result.blocked:
            self.trade_journal.record({"event": "blocked", "reason": no_trade_result.reason})
            return

        safety = self.safety.allow_order(
            pair=self.pair,
            side="buy" if action == int(LiveAction.BUY) else "sell",
            lots=self.max_lots,
            bid=float(bid or mid),
            ask=float(ask or mid),
            equity=self.equity,
        )
        if not safety.get("ok"):
            self.trade_journal.record({"event": "blocked", "reason": safety.get("reason")})
            return

        ret = _last_float(features, "ret_5", 0.0)
        try:
            # R-1/R-2 fix: parametric_var now expects price-fraction returns
            # (NOT pip-scaled). The notional x price-fraction math gives dollar
            # VaR directly. Pass `ret` as-is; the auto-normalizer guards against
            # any future caller accidentally feeding pip values again.
            self.pvar.update_returns(self.pair, float(ret))
        except Exception:
            pass
        try:
            positions = dict(self.broker.get_positions() or {})
        except Exception:
            positions = {}
        positions[self.pair] = float(self._position)
        var_result = self.pvar.parametric_var(positions, self.equity)
        size_adj = (
            0.5 if float(var_result.get("var_pct", 0.0) or 0.0) > float(getattr(self.pvar, "max_var", 0.02)) else 1.0
        )

        hurst = _last_float(features, "hurst_60", 0.5)
        corr_stab = _last_float(features, "corr_break", 0.0)
        cols = list(features.columns)
        if "ret_5" in cols:
            if _POLARS and isinstance(features, pl.DataFrame):
                recent_returns = features.select("ret_5").tail(60).to_numpy().reshape(-1)
            else:
                recent_returns = np.asarray(features["ret_5"].tail(60), dtype=np.float64)
            recent_returns = np.nan_to_num(np.asarray(recent_returns, dtype=np.float64), nan=0.0)
        else:
            vol = _last_float(features, "vol_20", 0.001)
            recent_returns = np.full(60, float(ret if ret else vol * 0.1), dtype=np.float64)

        sizing = self.rck.size(
            self.equity,
            0.55,
            1.5,
            recent_returns,
            atr,
            corr_avg=float(var_result.get("correlation_avg", 0.0) or 0.0),
            hurst=float(hurst),
            corr_break=float(corr_stab),
        )
        if isinstance(regime_result, dict):
            regime_size_mult = float(regime_result.get("size_multiplier", 1.0) or 1.0)
        else:
            regime_size_mult = float(getattr(regime_result, "size_multiplier", 1.0) or 1.0)
        dae_mult = float(dae.get("size_multiplier", 1.0) or 1.0)
        lots = float(min(sizing.get("lots", 0.0) * regime_size_mult * size_adj * dae_mult, self.max_lots))
        if self.risk_engine is not None and getattr(self.risk_engine, "_soft_reduce", False):
            lots *= 0.5

        if lots > 0 and action in (int(LiveAction.BUY), int(LiveAction.SELL)):
            # P3: session exposure caps (DST SoT via SessionLimitsEnforcer)
            try:
                open_lots = sum(abs(float(v)) for v in positions.values())
                open_trades = sum(1 for v in positions.values() if abs(float(v)) > 1e-12)
            except Exception:
                open_lots = abs(float(self._position))
                open_trades = 1 if abs(self._position) > 1e-12 else 0
            sess_chk = self.session_limits.check(
                open_lots=open_lots,
                open_trades=open_trades,
                now=datetime.now(UTC),
            )
            if not sess_chk.get("allowed", True):
                self.trade_journal.record(
                    {
                        "event": "blocked",
                        "reason": "session_limits",
                        "details": sess_chk,
                    }
                )
                return

            if self.risk_engine is not None:
                _rd = self.risk_engine.check_order(
                    pair=self.pair,
                    lots=lots,
                    price=mid,
                )
                if not _rd.allowed:
                    self.trade_journal.record(
                        {
                            "event": "blocked",
                            "reason": f"risk_engine:{_rd.rule}",
                            "details": _rd.reason,
                        }
                    )
                    return

        if self._halt_new_orders:
            return

        if lots > 0 and action in (int(LiveAction.BUY), int(LiveAction.SELL)):
            buy = action == int(LiveAction.BUY)

            # Broker-side protective stops (P0 M1): attach SL/TP on a market order
            # so a crash/feed-gap cannot leave a naked position. Mirrors the
            # in-process ATR stop so both agree on the adverse-move distance.
            stop_dist = float(self.stop_loss_atr) * float(atr) if atr > 0 else 0.0
            tp_dist = float(self.take_profit_atr) * float(atr) if atr > 0 else 0.0
            if buy:
                sl = (mid - stop_dist) if stop_dist > 0 else None
                tp = (mid + tp_dist) if tp_dist > 0 else None
            else:
                sl = (mid + stop_dist) if stop_dist > 0 else None
                tp = (mid - tp_dist) if tp_dist > 0 else None

            # M2: only update engine state once the broker confirms the fill.
            # Some venues (BridgeBrokerAdapter) return {"ok": False} on reject;
            # others (OANDA direct) raise or return fill data without an ok key.
            def _place(side: str, qty: float) -> bool:
                r = self.broker.market_order(
                    self.pair,
                    side,
                    float(qty),
                    stop_loss=sl,
                    take_profit=tp,
                )
                if isinstance(r, dict) and r.get("ok") is False:
                    self.trade_journal.record(
                        {
                            "event": "order_rejected",
                            "side": side,
                            "lots": float(qty),
                            "venue": r.get("venue") or getattr(self.broker, "venue", "?"),
                        }
                    )
                    return False
                return True

            # BUG-004: close an existing opposite position before flipping.
            if buy and self._position < 0:
                if not _place("buy", abs(float(self._position))):
                    return
                self._risk_trade_closed(mid, "signal_flip")
                self._position = 0.0
                self._entry_price = 0.0
            elif not buy and self._position > 0:
                if not _place("sell", abs(float(self._position))):
                    return
                self._risk_trade_closed(mid, "signal_flip")
                self._position = 0.0
                self._entry_price = 0.0
            # Open the new leg.
            if not _place("buy" if buy else "sell", lots):
                return
            self._position = lots if buy else -lots
            self._entry_price = mid
            self._holding_bars = 0
            if self.risk_engine is not None:
                self.risk_engine.open_position(
                    self.pair, abs(self._position), self._entry_price, direction="long" if buy else "short"
                )
        elif action in (int(LiveAction.CLOSE), int(LiveAction.SCALE_OUT_100)):
            if abs(self._position) > 1e-12:
                _sc_ok = self.broker.close_position(self.pair)
                if not (isinstance(_sc_ok, dict) and _sc_ok.get("ok") is False):
                    self._risk_trade_closed(mid, "signal_close")
                    self._position = 0.0
                    self._holding_bars = 0
                    self._entry_price = 0.0
        elif action in (int(LiveAction.SCALE_OUT_25), int(LiveAction.SCALE_OUT_50)):
            if abs(self._position) > 1e-12:
                fraction = 0.25 if action == int(LiveAction.SCALE_OUT_25) else 0.50
                reduce_lots = round(abs(self._position) * fraction, 4)
                if reduce_lots > 0:
                    reduce_side = "sell" if self._position > 0 else "buy"
                    r = self.broker.market_order(self.pair, reduce_side, reduce_lots)
                    if not (isinstance(r, dict) and r.get("ok") is False):
                        if self._position > 0:
                            self._position -= reduce_lots
                        else:
                            self._position += reduce_lots
                        self._risk_trade_closed(mid, f"scale_out_{int(fraction * 100)}")
        elif action in (int(LiveAction.SCALE_IN_25), int(LiveAction.SCALE_IN_50), int(LiveAction.SCALE_IN_100)):
            if abs(self._position) > 1e-12:
                fraction = {int(LiveAction.SCALE_IN_25): 0.25, int(LiveAction.SCALE_IN_50): 0.50, int(LiveAction.SCALE_IN_100): 1.0}[action]
                add_lots = round(abs(lots) * fraction, 4)
                if add_lots > 0:
                    add_side = "buy" if self._position > 0 else "sell"
                    r = self.broker.market_order(self.pair, add_side, add_lots)
                    if not (isinstance(r, dict) and r.get("ok") is False):
                        if self._position > 0:
                            self._position += add_lots
                        else:
                            self._position -= add_lots
                        if self.risk_engine is not None:
                            self.risk_engine.open_position(
                                self.pair, abs(self._position), self._entry_price,
                                direction="long" if self._position > 0 else "short"
                            )
        elif action == int(LiveAction.HOLD):
            self._holding_bars += 1

        lat_total = (time.perf_counter() - t0) * 1000.0
        if self.prom is not None:
            self.prom.update_latency(lat_total)
            self.prom.set_position(self._position)
        self._bar_log.append(
            {
                "bar": int(bar_idx),
                "pair": self.pair,
                "action": int(action),
                "model": model_used,
                "lots": round(lots, 4),
                "equity": round(self.equity, 2),
                "sentiment": round(float(bias), 4),
                "latency_ms": round(lat_total, 2),
                "var_pct": float(var_result.get("var_pct", 0.0) or 0.0),
                "ts": datetime.now(UTC).isoformat(),
            }
        )
        if len(self._bar_log) % 60 == 0:
            self._save_log()

    def _maybe_hot_reload(self) -> None:
        """Poll reload_model.flag written by train_gpu after promotion."""
        if not consume_reload_flag(self._reload_flag):
            return
        self.logger.event(
            "INFO",
            "model_reload",
            "[Live] reload_model.flag detected - reloading weights",
            pair=self.pair,
        )
        try:
            fast, slow, meta = build_inference_agents(
                model_name=self._model_name,
                runtime=self._runtime,
                demo=False,
                seq_len=int(self._inference_meta.get("seq_len", 60)),
                n_features=self._inference_meta.get("n_features"),
                checkpoint_dir=self._checkpoint_dir,
            )
            self._agent_wrap_fast.set_model(fast, self._live_action_adapter(fast))
            self._agent_wrap_slow.set_model(slow, self._live_action_adapter(slow))
            self._inference_meta.update(meta)
            self.logger.event(
                "INFO",
                "model_reload_ok",
                f"[Live] Reloaded from {meta.get('source')} | pt={meta.get('pt_path')}",
                pair=self.pair,
                source=meta.get("source"),
            )
        except Exception as exc:
            self.logger.event(
                "ERROR",
                "model_reload_failed",
                f"[Live] Hot reload failed: {exc}",
                pair=self.pair,
            )

    def _feature_columns(self, features) -> list[str]:
        if self._expected_features is not None:
            missing = [c for c in self._expected_features if c not in features.columns]
            if missing:
                err_msg = (
                    f"Feature schema mismatch! Expected {len(self._expected_features)} features, "
                    f"missing {len(missing)}: {missing[:5]}."
                )
                self.logger.event("FATAL", "schema_mismatch", err_msg, pair=self.pair)
                raise RuntimeError(err_msg)
            return list(self._expected_features)
        return [
            c
            for c, dtype in zip(features.columns, features.dtypes, strict=False)
            if c != "timestamp_utc" and _is_numeric_dtype(dtype)
        ]

    def _check_drift(self, features) -> None:
        feature_cols = self._feature_columns(features)
        if _POLARS and isinstance(features, pl.DataFrame):
            X = features.select(feature_cols).to_numpy()
        else:
            X = np.asarray(features[feature_cols].to_numpy())

        # BUG-010: Use actual model predictions instead of random noise for labels.
        # This enables concept drift detection (target shift) in addition to covariate shift.
        if hasattr(self, "_recent_predictions") and len(self._recent_predictions) > 0:
            y = np.array(self._recent_predictions[-len(X) :])
            if len(y) < len(X):
                y = np.pad(y, (len(X) - len(y), 0), mode="edge")
        else:
            y = np.zeros(len(X))

        if not self._baseline_fitted:
            self.drift.fit_baseline(X, y)
            self._baseline_fitted = True
            return
        y_recent = y[-500:] if len(y) >= 500 else y
        result = self.drift.check(X[-500:], y_recent)
        if result.get("drift_detected"):
            self.logger.event(
                "WARN",
                "drift_detected",
                f"[Live] DRIFT DETECTED: {result.get('reasons')}",
                pair=self.pair,
                reasons=result.get("reasons") or [],
            )
            if self.prom is not None:
                self.prom.set_drift(True)
            self._trigger_retrain("drift", details=result.get("reasons") or [])
        elif self.prom is not None:
            self.prom.set_drift(False)

    def _trigger_retrain(self, reason: str, details=None) -> None:
        now = time.time()
        if now - self._last_retrain_ts < self._retrain_cooldown_s:
            return
        if self._retrain_lock.exists():
            return
        self._last_retrain_ts = now
        self._retrain_lock.parent.mkdir(parents=True, exist_ok=True)
        self._retrain_lock.write_text(
            json.dumps(
                {
                    "time": datetime.now(UTC).isoformat(),
                    "reason": reason,
                    "details": details,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        script = Path(__file__).resolve().parent.parent / "training" / "train_gpu.py"
        cmd = [sys.executable, str(script), "--model", "haelt", "--resume", "--live-retrain"]
        self.logger.event(
            "WARN",
            "retrain_trigger",
            f"[Live] Auto-retrain started ({reason}).",
            pair=self.pair,
            reason=reason,
            details=details,
        )
        try:
            subprocess.Popen(
                cmd,
                cwd=str(Path(__file__).resolve().parent.parent),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception as e:
            self.logger.event(
                "ERROR",
                "retrain_spawn_failed",
                f"[Live] Retrain spawn failed: {e}",
                pair=self.pair,
                reason=reason,
            )
            try:
                self._retrain_lock.unlink(missing_ok=True)
            except Exception:
                pass

    def _start_sentiment_loop(self):
        def _loop():
            while self._running:
                try:
                    headlines = get_latest_headlines(limit=12) or ["Market update"]
                    if hasattr(self.sentiment, "update_global_brain"):
                        self.sentiment.update_global_brain(headlines)
                except Exception:
                    pass
                time.sleep(60)

        threading.Thread(target=_loop, daemon=True).start()

    def _next_bar(self) -> datetime:
        return _align_next_bar(self.bar_freq)

    def _save_log(self):
        path = self.log_dir / f"live_{datetime.now(UTC):%Y%m%d}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            for entry in self._bar_log[-60:]:
                f.write(json.dumps(entry) + "\n")

    def stop(self):
        self.logger.event(
            "INFO",
            "shutdown",
            f"[Live] Stopping engine | bars logged: {len(self._bar_log)}",
            pair=self.pair,
            bars_logged=len(self._bar_log),
        )
        self._running = False
        try:
            self.broker.disconnect()
        except Exception:
            pass
        self._save_log()
        try:
            self.logger.close()
        except Exception:
            pass


class MultiPairLiveTradingEngine:
    """Synchronized multi-pair loop with shared broker session and risk budget."""

    def __init__(
        self,
        broker: BrokerInterface,
        fast_agent,
        slow_model,
        pairs: list[str],
        equity: float,
        max_lots: float,
        sentiment_mode: str = "auto",
        calendar_file: str | None = None,
        journal_path: str | None = None,
        max_spread_pips: float = 2.5,
        guard_min_confidence: float = 0.45,
        bar_freq: str = "1min",
        inference_meta: dict | None = None,
        stop_loss_atr: float = 1.5,
        take_profit_atr: float = 1.5,
        allow_paper_fallback: bool = False,
        risk_engine=None,
    ):
        self.broker = broker
        self.pairs = [p.upper() for p in pairs]
        self.allow_paper_fallback = bool(allow_paper_fallback)
        per_pair = float(max_lots) / max(1, len(self.pairs))

        shared_cross_asset = None
        try:
            end = pd.Timestamp.utcnow()
            start = end - pd.Timedelta(days=45)
            shared_cross_asset = load_cross_asset_panel(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                cache_dir=str(Path(PATHS.get("data_processed", "data/processed")) / "cross_asset"),
                source=os.getenv("CROSS_ASSET_SOURCE", "auto").strip() or "auto",
            )
            if shared_cross_asset is not None:
                print(f"[Live] Shared cross-asset loaded: {len(shared_cross_asset)} series across {len(self.pairs)} pairs")
        except Exception as exc:
            print(f"[Live] Shared cross-asset unavailable ({exc}); continuing without it")

        self.engines = [
            LiveTradingEngine(
                broker=broker,
                fast_agent=fast_agent,
                slow_model=slow_model,
                pair=p,
                equity=equity,
                max_lots=per_pair,
                sentiment_mode=sentiment_mode,
                prometheus_enabled=False,
                calendar_file=calendar_file,
                journal_path=None
                if journal_path is None
                else str(
                    Path(journal_path).with_name(
                        f"{Path(journal_path).stem}_{p.lower()}{Path(journal_path).suffix or '.jsonl'}"
                    )
                ),
                max_spread_pips=max_spread_pips,
                guard_min_confidence=guard_min_confidence,
                bar_freq=bar_freq,
                inference_meta=inference_meta,
                stop_loss_atr=stop_loss_atr,
                take_profit_atr=take_profit_atr,
                allow_paper_fallback=allow_paper_fallback,
                risk_engine=risk_engine,
                cross_asset=shared_cross_asset,
            )
            for p in self.pairs
        ]
        self.prom = None
        if bool(ALERTS.get("prometheus_enabled", True)):
            self.prom = ForexPrometheusExporter(
                port=int(ALERTS.get("prometheus_port", 8000)),
                initial_equity=float(equity),
            )
        self.bar_freq = str(bar_freq)
        self._running = False

    def start(self, max_bars: int | None = None):
        if not self.broker.connect():
            if not self.allow_paper_fallback and not isinstance(self.broker, PaperBroker):
                raise RuntimeError(
                    "[Live] Broker connection failed. Pass --allow-paper-fallback only for intentional paper testing."
                )
            print("[Live] Broker connection failed - using PaperBroker (explicit fallback)")
            self.broker = PaperBroker(initial_equity=self.engines[0].equity if self.engines else 10_000.0)
            if not self.broker.connect():
                raise RuntimeError("PaperBroker fallback failed to connect")
            for e in self.engines:
                e.broker = self.broker

        if hasattr(self.broker, "get_candles"):
            for e in self.engines:
                try:
                    hist_df = self.broker.get_candles(e.pair, count=120, granularity=self.bar_freq)
                    if hist_df is not None and not hist_df.empty:
                        e.buf.seed_bars(hist_df)
                        print(f"[Live] Preloaded {len(hist_df)} historical bars for {e.pair} buffer warmup")
                except Exception as exc:
                    print(f"[Live] Warning: Historical candle preload failed for {e.pair} ({exc})")

        self._running = True
        for e in self.engines:
            e._running = True
            e._start_sentiment_loop()
        if self.prom is not None:
            self.prom.start()
        print(f"[Live] MultiPair synchronized loop started for {self.pairs}")
        bar_count = 0
        next_bar_time = _align_next_bar(self.bar_freq)
        poll_interval = (
            0.1
            if (getattr(self.broker, "_zmq_sub", None) is not None or isinstance(self.broker, PaperBroker))
            else 0.5
        )
        while self._running:
            if max_bars and bar_count >= max_bars:
                break
            now = datetime.now(UTC)
            if now < next_bar_time:
                for e in self.engines:
                    bid, ask = self.broker.get_bid_ask(e.pair)
                    if bid and ask:
                        e.buf.push_tick(bid, ask)
                time.sleep(poll_interval)
                continue
            for e in self.engines:
                bars = e.buf.get_bars()
                if bars is not None and len(bars) >= 70:
                    e._on_new_bar(bars, bar_count)
            bar_count += 1
            next_bar_time = _align_next_bar(self.bar_freq)
        self.stop()

    def stop(self):
        self._running = False
        for e in self.engines:
            e._running = False
            e._save_log()
        if self.prom is not None:
            try:
                self.prom.stop()
            except Exception:
                pass
        self.broker.disconnect()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    try:
        import yaml
    except Exception:
        yaml = None

    def _pairs_from_run_yaml(path: Path) -> list[str]:
        if yaml is None or not path.exists():
            return []
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            data = doc.get("data", {}) if isinstance(doc, dict) else {}
            pairs = data.get("pairs", [])
            return [str(p).upper() for p in pairs if str(p).strip()]
        except Exception:
            return []

    p = argparse.ArgumentParser(description="Live Trading Engine")
    p.add_argument(
        "--strategy-mode",
        default="scalping",
        choices=sorted(STRATEGY_PROFILES.keys()),
        help="Trading horizon profile. scalping=1min; normal=1h slower trading.",
    )
    p.add_argument(
        "--bar-freq",
        default=None,
        help="Live aggregation frequency, e.g. 1min, 15min, 1h. Defaults to strategy profile.",
    )
    p.add_argument(
        "--broker",
        default="paper",
        choices=["paper", "lmax", "oanda", "mt5", "ibkr"],
        help="Venue: paper/lmax/oanda or BrokerBridge-backed mt5/ibkr",
    )
    p.add_argument("--mt5-login", type=int, default=None, help="MT5 account login (with --broker mt5)")
    p.add_argument("--mt5-password", default=None, help="MT5 password")
    p.add_argument("--mt5-server", default=None, help="MT5 server name")
    p.add_argument("--ibkr-host", default="127.0.0.1", help="IBKR TWS/Gateway host")
    p.add_argument("--ibkr-port", type=int, default=7497, help="IBKR port (7497 paper, 7496 live)")
    p.add_argument("--ibkr-client-id", type=int, default=1, help="IBKR client id")
    p.add_argument("--pair", default="EURUSD")
    p.add_argument(
        "--pairs",
        default="",
        help="Comma-separated pairs (overrides --pair). If empty, attempts config/run.yaml data.pairs",
    )
    p.add_argument("--pairs-config", default="config/run.yaml", help="YAML config path used to auto-load data.pairs")
    p.add_argument("--equity", type=float, default=10_000.0)
    p.add_argument("--max-lots", type=float, default=0.5)
    p.add_argument("--model", default="haelt")
    p.add_argument("--max-bars", type=int, default=None)
    p.add_argument(
        "--runtime",
        default="pytorch",
        choices=["pytorch", "onnx"],
        help="Inference backend: pytorch (CUDA) or onnx (AMD DirectML)",
    )
    p.add_argument(
        "--sentiment-mode",
        default="auto",
        choices=["auto", "ollama", "finbert", "off", "none", "neutral"],
        help="Sentiment backend priority (or 'off' to disable)",
    )
    p.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Sequence length used during training (default: strategy profile / run.yaml)",
    )
    p.add_argument(
        "--n-feat",
        type=int,
        default=None,
        help="Number of input features used during training (required for older ONNX exports)",
    )
    p.add_argument("--calendar-file", default=None, help="CSV/JSON economic calendar for live no-trade guard")
    p.add_argument("--journal-path", default=None, help="Optional JSONL path for structured trade journal")
    p.add_argument(
        "--max-spread-pips", type=float, default=2.5, help="Block new entries when live spread exceeds this value"
    )
    p.add_argument(
        "--guard-min-confidence",
        type=float,
        default=0.45,
        help="Minimum confidence for BUY/SELL when model confidence is available",
    )
    p.add_argument(
        "--demo", action="store_true", default=False, help="Use random DemoAgent instead of loading trained checkpoints"
    )
    p.add_argument(
        "--allow-paper-fallback",
        action="store_true",
        default=False,
        help="If live broker connect/pricing fails, fall back to PaperBroker "
        "(off by default - prevents silent paper trading on real runs)",
    )
    p.add_argument(
        "--risk-config",
        default=None,
        help="Optional JSON/YAML RiskEngine overrides (same schema as train_gpu --risk-config)",
    )
    args = p.parse_args()
    prof = strategy_profile(args.strategy_mode)
    if args.bar_freq is None:
        args.bar_freq = str(prof["bar_freq"])
    if args.seq_len is None:
        # Prefer training.seq_len from pairs-config YAML, else strategy profile (80 for scalping)
        args.seq_len = int(prof.get("seq_len", 80))
        try:
            import yaml

            cfg_path = Path(args.pairs_config)
            if cfg_path.is_file():
                with open(cfg_path, encoding="utf-8") as fh:
                    ycfg = yaml.safe_load(fh) or {}
                train_sl = (ycfg.get("training") or {}).get("seq_len")
                if train_sl is not None:
                    args.seq_len = int(train_sl)
                maturity = (ycfg.get("maturity") or {}).get("stage")
                if maturity and not getattr(args, "maturity_stage", None):
                    args.maturity_stage = str(maturity)
        except Exception:
            pass
    if args.max_spread_pips == 2.5 and args.strategy_mode != "scalping":
        args.max_spread_pips = float(prof["max_spread_pips"])
    if args.guard_min_confidence == 0.45 and args.strategy_mode != "scalping":
        args.guard_min_confidence = float(prof["guard_min_confidence"])
    stop_loss_atr = float(prof.get("stop_loss_atr", 1.5))
    take_profit_atr = float(prof.get("take_profit_atr", prof.get("profit_target_atr", 1.5)))
    ckpt_dir_override = prof.get("checkpoint_dir")

    ckpt_paths = resolve_checkpoint_paths(args.model, checkpoint_dir=ckpt_dir_override)
    print(f"[Live] Checkpoint dir     : {ckpt_paths.checkpoint_dir}")
    print(f"[Live] Target Model       : {args.model.upper()}")
    print(f"[Live] PyTorch checkpoint : {'OK' if ckpt_paths.pt_path else 'missing'} ({ckpt_paths.source})")
    if ckpt_paths.pt_path:
        print(f"[Live]   -> {ckpt_paths.pt_path}")
    print(f"[Live] ONNX checkpoint    : {'OK' if ckpt_paths.onnx_path else 'missing'}")
    if ckpt_paths.onnx_path:
        print(f"[Live]   -> {ckpt_paths.onnx_path}")

    _maturity = str(getattr(args, "maturity_stage", "") or "").lower()
    if not _maturity:
        try:
            from config.settings import MATURITY as _MAT

            _maturity = str(_MAT.get("stage", "paper")).lower()
        except Exception:
            _maturity = "paper"
    print(f"[Live] Maturity stage     : {_maturity}")

    # Non-paper live runs require a passed promotion gate artifact (fail-closed).
    if args.broker != "paper" and not args.demo:
        _promoted = False
        _prom_reasons: list[str] = []
        for _cand in (
            Path(ckpt_paths.checkpoint_dir) / args.model / "promotion_gate.json",
            Path(ckpt_paths.checkpoint_dir) / "promotion_gate.json",
            Path(ckpt_paths.checkpoint_dir) / "ensemble" / "promotion_gate.json",
            Path("checkpoints/ensemble/optimal_roadmap_certification.json"),
            Path("checkpoints/ensemble/promotion_gate.json"),
        ):
            if not _cand.exists():
                continue
            try:
                import json as _json

                _pg = _json.loads(_cand.read_text(encoding="utf-8"))
                if (
                    bool(_pg.get("promoted"))
                    or bool(_pg.get("quality_gate_passed"))
                    or _pg.get("status") == "CERTIFIED_READY_FOR_DEPLOYMENT"
                ):
                    _promoted = True
                    print(f"[Live] Promotion gate OK: {_cand}")
                    break
                _prom_reasons.append(f"{_cand.name}: promoted={_pg.get('promoted')}")
            except Exception as _pe:
                _prom_reasons.append(f"{_cand}: {_pe}")
        if not _promoted:
            raise SystemExit(
                "[Live] Refusing non-paper broker without promotion_gate.json "
                f"(promoted=true) under {ckpt_paths.checkpoint_dir}. "
                f"Checked: {_prom_reasons or 'no promotion_gate.json found'}. "
                "Use --broker paper for paper trading, or promote a model first."
            )
    elif _maturity == "production" and args.broker == "paper" and not args.demo:
        print(
            "[Live] WARN: maturity.stage=production with --broker paper - "
            "promote via promotion_gate.json before live capital."
        )

    fast_agent, slow_model, inference_meta = build_inference_agents(
        model_name=args.model,
        runtime=args.runtime,
        demo=args.demo,
        seq_len=args.seq_len,
        n_features=args.n_feat,
        checkpoint_dir=ckpt_paths.checkpoint_dir,
    )
    if inference_meta.get("demo") and not args.demo:
        raise SystemExit(
            "[Live] Refusing to run with DemoAgent without --demo. "
            "Train/promote a checkpoint, or pass --demo for paper testing."
        )

    cli_pairs = [p.strip().upper() for p in args.pairs.split(",") if p.strip()]
    yaml_pairs = _pairs_from_run_yaml(Path(args.pairs_config))
    pair_list = cli_pairs or yaml_pairs or [args.pair.upper()]
    print(f"[Live] Pairs: {pair_list}")

    broker_map = {"paper": PaperBroker, "lmax": LMAXBroker, "oanda": OANDABroker}
    if args.broker in ("mt5", "ibkr"):
        bridge_cfg: dict = {}
        if args.broker == "mt5":
            if args.mt5_login is not None:
                bridge_cfg["login"] = int(args.mt5_login)
            if args.mt5_password:
                bridge_cfg["password"] = args.mt5_password
            if args.mt5_server:
                bridge_cfg["server"] = args.mt5_server
            venue = "MT5"
        else:
            bridge_cfg = {
                "host": args.ibkr_host,
                "port": int(args.ibkr_port),
                "client_id": int(args.ibkr_client_id),
            }
            venue = "IBKR"
        broker = BridgeBrokerAdapter(venue=venue, config=bridge_cfg)
        print(f"[Live] BrokerBridge adapter: {venue}")
    else:
        broker = (
            broker_map[args.broker](initial_equity=args.equity) if args.broker == "paper" else broker_map[args.broker]()
        )
    # Paper broker is an intentional choice - allow its own "fallback" path trivially.
    allow_paper = bool(args.allow_paper_fallback) or args.broker == "paper"

    risk_engine = None
    if args.risk_config:
        try:
            import json as _json

            from risk.risk_engine import RiskConfig, RiskEngine

            raw = str(args.risk_config).strip()
            if raw.startswith("{"):
                cfg_dict = _json.loads(raw)
            else:
                path = Path(raw)
                text = path.read_text(encoding="utf-8")
                if path.suffix.lower() in (".yaml", ".yml"):
                    import yaml

                    cfg_dict = yaml.safe_load(text) or {}
                else:
                    cfg_dict = _json.loads(text)
            risk_engine = RiskEngine(equity=args.equity, cfg=RiskConfig.from_dict(cfg_dict))
            print("[Live] RiskEngine loaded from --risk-config")
        except Exception as e:
            print(f"[Live] WARN: --risk-config failed ({e}); using default RiskEngine")

    if len(pair_list) > 1:
        engine = MultiPairLiveTradingEngine(
            broker=broker,
            fast_agent=fast_agent,
            slow_model=slow_model,
            pairs=pair_list,
            equity=args.equity,
            max_lots=args.max_lots,
            sentiment_mode=args.sentiment_mode,
            calendar_file=args.calendar_file,
            journal_path=args.journal_path,
            max_spread_pips=args.max_spread_pips,
            guard_min_confidence=args.guard_min_confidence,
            bar_freq=args.bar_freq,
            inference_meta=inference_meta,
            stop_loss_atr=stop_loss_atr,
            take_profit_atr=take_profit_atr,
            allow_paper_fallback=allow_paper,
            risk_engine=risk_engine,
        )
        print(
            f"\n[Live] Starting {args.broker.upper()} multi-pair engine | {pair_list} | max {args.max_lots:.4f} lots total"
        )
    else:
        engine = LiveTradingEngine(
            broker=broker,
            fast_agent=fast_agent,
            slow_model=slow_model,
            pair=pair_list[0],
            equity=args.equity,
            max_lots=args.max_lots,
            sentiment_mode=args.sentiment_mode,
            calendar_file=args.calendar_file,
            journal_path=args.journal_path,
            max_spread_pips=args.max_spread_pips,
            guard_min_confidence=args.guard_min_confidence,
            bar_freq=args.bar_freq,
            inference_meta=inference_meta,
            stop_loss_atr=stop_loss_atr,
            take_profit_atr=take_profit_atr,
            allow_paper_fallback=allow_paper,
            risk_engine=risk_engine,
        )
        print(f"\n[Live] Starting {args.broker.upper()} engine | {pair_list[0]} | max {args.max_lots:.4f} lots")
    print(f"       Runtime: {args.runtime.upper()} | Strategy: {args.strategy_mode} | Bars: {args.bar_freq}")
    print("       Press Ctrl+C to stop and save logs\n")
    engine.start(max_bars=args.max_bars)
