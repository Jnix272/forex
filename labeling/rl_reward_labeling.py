"""
labeling/rl_reward_labeling.py
===============================
RL Reward Signal labeling: assign a reward to each bar based on the
forward P&L of entering at that bar, minus transaction costs.

Unlike Triple Barrier Method (which assigns fixed {-1,0,+1} labels),
RL reward labeling produces a continuous signal that directly represents
the risk-adjusted P&L the agent would have earned - making it the natural
training signal for PPO/DQN agents.

Label values:
  positive  -> profitable entry at this bar
  negative  -> losing entry
  near-zero -> no clear edge (hold)
"""

import numpy as np
import pandas as pd
from typing import Any, cast

from infrastructure.numerics import sanitize_frame

try:
    import polars as pl
except ImportError:  # pragma: no cover - pandas-only legacy environments
    pl = None


def _is_polars_frame(frame) -> bool:
    return pl is not None and isinstance(frame, pl.DataFrame)


def _label_regime_cfg() -> dict:
    try:
        from config.settings import LABEL_REGIME as _LR

        return _LR
    except Exception:
        return {}


def max_label_horizon_mult(label_regime: dict | None = None) -> float:
    """Largest multiplicative label horizon vs base LH (embargo / purge floor).

    Embargo must cover ``base_LH * max(regime horizon_mults) * max(session
    horizon_mults)`` - trending can reach 1.5x and asia/off can stretch further.
    Spread gates only shorten, so they are ignored here.
    """
    lr = label_regime if label_regime is not None else _label_regime_cfg()
    barrier = lr.get("barrier_scale") or {}
    regime_m = 1.0
    for cfg in barrier.values():
        if isinstance(cfg, dict) and "horizon_mult" in cfg:
            regime_m = max(regime_m, float(cfg["horizon_mult"]))
    session_m = 1.0
    for v in (lr.get("session_horizon_mult") or {}).values():
        session_m = max(session_m, float(v))
    return float(regime_m * session_m)


def resolve_session_key(
    session: str | None = None,
    *,
    asia_london: bool | float = False,
    london_ny: bool | float = False,
) -> str:
    """Prefer DST overlap flags over exclusive primary session for policy lookup."""
    if asia_london:
        return "asia_london"
    if london_ny:
        return "london_ny"
    if session is None:
        return "default"
    s = str(session).strip().lower()
    return s if s else "default"


def resolve_session_cost(
    session: str | None = None,
    *,
    asia_london: bool | float = False,
    london_ny: bool | float = False,
    label_regime: dict | None = None,
) -> float:
    """Session / overlap transaction-cost scale (applied before horizon)."""
    lr = label_regime if label_regime is not None else _label_regime_cfg()
    scales = lr.get("session_cost_scale") or {}
    key = resolve_session_key(session, asia_london=asia_london, london_ny=london_ny)
    return float(scales.get(key, 1.0))


def resolve_horizon(
    base_lh: int,
    regime: str | None = None,
    session: str | None = None,
    spread_z: float | None = None,
    *,
    asia_london: bool | float = False,
    london_ny: bool | float = False,
    regime_horizon_mult: float | None = None,
    label_regime: dict | None = None,
) -> int:
    """Resolve per-bar label horizon from base LH x regime x session x spread.

    Cost scaling is separate (``resolve_session_cost``); call that first when
    both are needed. ``regime_horizon_mult`` overrides table lookup when the
    caller already resolved barrier_scale for the bar.
    """
    lr = label_regime if label_regime is not None else _label_regime_cfg()
    base = max(1, int(base_lh))
    if regime_horizon_mult is not None:
        r_hm = float(regime_horizon_mult)
    else:
        barrier = lr.get("barrier_scale") or {}
        key = str(regime or "normal").lower()
        cfg = barrier.get(key) or barrier.get("normal") or {}
        r_hm = float(cfg.get("horizon_mult", 1.0)) if isinstance(cfg, dict) else 1.0

    sess_key = resolve_session_key(session, asia_london=asia_london, london_ny=london_ny)
    s_hm = float((lr.get("session_horizon_mult") or {}).get(sess_key, 1.0))

    sp_hm = 1.0
    if spread_z is not None and np.isfinite(spread_z):
        sp = lr.get("spread_horizon") or {}
        z = float(spread_z)
        z_ext = float(sp.get("z_extreme", 2.5))
        z_wide = float(sp.get("z_wide", 1.5))
        if z >= z_ext:
            sp_hm = float(sp.get("extreme_mult", 0.5))
        elif z >= z_wide:
            sp_hm = float(sp.get("wide_mult", 0.75))

    return max(3, int(base * r_hm * s_hm * sp_hm))


def _to_polars_with_timestamp(frame, timestamp_col: str = "timestamp_utc"):
    if pl is None:
        raise RuntimeError("Polars is required for Polars-native labeling inputs")
    assert pl is not None
    if isinstance(frame, pl.DataFrame):
        out = frame
    elif isinstance(frame, pd.DataFrame):
        pdf = frame.copy()
        if timestamp_col not in pdf.columns:
            index_name = pdf.index.name or timestamp_col
            pdf = pdf.reset_index().rename(columns={index_name: timestamp_col})
        out = pl.from_pandas(pdf)
    else:
        raise TypeError(f"Expected pandas or Polars DataFrame, got {type(frame).__name__}")
    for alt in ("timestamp", "time", "datetime"):
        if timestamp_col not in out.columns and alt in out.columns:
            out = out.rename({alt: timestamp_col})
            break
    if timestamp_col not in out.columns:
        raise ValueError(f"DataFrame missing {timestamp_col} column/index")
    return out.with_columns(pl.col(timestamp_col).cast(pl.Datetime("ns", "UTC"))).sort(timestamp_col)


def _polars_result_to_pandas(result, index_col: str = "timestamp_utc") -> pd.DataFrame:
    assert pl is not None
    out = result.to_pandas()
    out[index_col] = pd.to_datetime(out[index_col], utc=True)
    return out.set_index(index_col)


def compute_rl_reward_labels(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    atr_col: str = "atr_6",
    lookahead_bars: int = 30,  # match LABELING / strategy.lookahead_bars
    profit_atr_mult: float = 1.2,  # match LABELING.profit_target_atr
    stop_atr_mult: float = 0.8,  # match LABELING.stop_loss_atr
    tx_cost_pips: float = 1.5,
    pip_size: float = 0.0001,
    execution_delay_bars: int = 1,
) -> pd.DataFrame | Any:
    """
    Compute per-bar RL reward signals for long and short entries.

    For each bar t, simulates:
      - Long entry: buy at ask_close[t], exit at the first of:
          (a) price hits profit target (entry + ATR x profit_mult)
          (b) price hits stop loss   (entry - ATR x stop_mult)
          (c) lookahead_bars elapsed
      - Short entry: symmetric

    Returns a DataFrame with columns:
      reward_long  : net reward for a long entry at this bar
      reward_short : net reward for a short entry
      reward       : max(reward_long, reward_short)  ← training label
      label        : sign of reward -> {-1, 0, +1} for classifier use
      optimal_side : 'long' | 'short' | 'hold'
    """
    return_polars = _is_polars_frame(bars) or _is_polars_frame(features)
    if pl is not None:
        bars_pl = _to_polars_with_timestamp(bars)
        features_pl = _to_polars_with_timestamp(features)
        market_cols = ["timestamp_utc", "close"] + [c for c in ("ask_close", "bid_close") if c in bars_pl.columns]
        aligned = (
            features_pl.select(["timestamp_utc"] + [c for c in (atr_col, "spread_pips") if c in features_pl.columns])
            .join_asof(bars_pl.select(market_cols), on="timestamp_utc", strategy="backward")
            .sort("timestamp_utc")
        )
        index_values = aligned["timestamp_utc"]
        # DS-010: use float32 throughout - sufficient for pip-level forex accuracy
        close = aligned["close"].forward_fill().fill_null(0.0).to_numpy().astype(np.float32)
        if "ask_close" in aligned.columns and "bid_close" in aligned.columns:
            entry_long = aligned["ask_close"].forward_fill().fill_null(0.0).to_numpy().astype(np.float32)
            entry_short = aligned["bid_close"].forward_fill().fill_null(0.0).to_numpy().astype(np.float32)
            exit_long_path = entry_short.copy()  # bid
            exit_short_path = entry_long.copy()  # ask
        else:
            spread_vals = (
                np.asarray(aligned["spread_pips"].fill_null(0.5).to_numpy(), dtype=np.float32)
                if "spread_pips" in aligned.columns
                else np.full(len(close), 0.5, dtype=np.float32)
            )
            spread_half = np.asarray(spread_vals, dtype=np.float32) * pip_size / 2.0
            entry_long = close + spread_half
            entry_short = close - spread_half
            exit_long_path = close - spread_half
            exit_short_path = close + spread_half
        atr = (
            aligned[atr_col].fill_null(0.0005).to_numpy().astype(np.float32)
            if atr_col in aligned.columns
            else np.full(len(close), 0.0005, dtype=np.float32)
        )
    else:
        return_polars = False
        index_values = features.index
        # DS-010: use float32 throughout - no upcast to float64 needed
        close = bars["close"].reindex(features.index).ffill().values.astype(np.float32)
        if "ask_close" in bars.columns:
            entry_long = bars["ask_close"].reindex(features.index).ffill().values.astype(np.float32)
            entry_short = bars["bid_close"].reindex(features.index).ffill().values.astype(np.float32)
            exit_long_path = entry_short.copy()
            exit_short_path = entry_long.copy()
        else:
            spread_half = np.asarray(features["spread_pips"].values, dtype=np.float32) * pip_size / 2.0
            entry_long = close + spread_half
            entry_short = close - spread_half
            exit_long_path = close - spread_half
            exit_short_path = close + spread_half
        atr = (
            features[atr_col].values.astype(np.float32)
            if atr_col in features.columns
            else np.full(len(close), 0.0005, dtype=np.float32)
        )
    valid_market = (
        np.isfinite(close)
        & np.isfinite(entry_long)
        & np.isfinite(entry_short)
        & np.isfinite(atr)
        & (close > 0)
        & (entry_long > 0)
        & (entry_short > 0)
        & (atr > 0)
    )
    close = np.nan_to_num(close, nan=0.0, posinf=0.0, neginf=0.0)
    entry_long = np.nan_to_num(entry_long, nan=0.0, posinf=0.0, neginf=0.0)
    entry_short = np.nan_to_num(entry_short, nan=0.0, posinf=0.0, neginf=0.0)
    exit_long_path = np.nan_to_num(exit_long_path, nan=0.0, posinf=0.0, neginf=0.0)
    exit_short_path = np.nan_to_num(exit_short_path, nan=0.0, posinf=0.0, neginf=0.0)
    atr = np.nan_to_num(atr, nan=0.0005, posinf=0.0005, neginf=0.0005)
    tx_cost_pips * pip_size
    n = len(close)

    reward_long = np.zeros(n, dtype=np.float32)
    reward_short = np.zeros(n, dtype=np.float32)

    delay = max(0, int(execution_delay_bars))
    for i in range(n - lookahead_bars - delay):
        entry_i = i + delay
        if not valid_market[i] or not valid_market[entry_i]:
            continue
        el = entry_long[entry_i]
        es = entry_short[entry_i]
        tp_l = el + profit_atr_mult * atr[entry_i]
        sl_l = el - stop_atr_mult * atr[entry_i]
        tp_s = es - profit_atr_mult * atr[entry_i]
        sl_s = es + stop_atr_mult * atr[entry_i]

        # Simulate forward path
        horizon = close[entry_i + 1 : entry_i + 1 + lookahead_bars]
        horizon_l = exit_long_path[entry_i + 1 : entry_i + 1 + lookahead_bars]
        horizon_s = exit_short_path[entry_i + 1 : entry_i + 1 + lookahead_bars]

        if not np.isfinite(horizon).all() or not valid_market[entry_i + 1 : entry_i + 1 + len(horizon)].all():
            continue

        # Long
        pnl_l = None
        for p in horizon_l:
            if p >= tp_l:
                pnl_l = (tp_l - el) / pip_size
                break
            elif p <= sl_l:
                pnl_l = (sl_l - el) / pip_size
                break
        if pnl_l is None:
            pnl_l = (horizon_l[-1] - el) / pip_size
        reward_long[i] = pnl_l - tx_cost_pips

        # Short
        pnl_s = None
        for p in horizon_s:
            if p <= tp_s:
                pnl_s = (es - tp_s) / pip_size
                break
            elif p >= sl_s:
                pnl_s = (es - sl_s) / pip_size
                break
        if pnl_s is None:
            pnl_s = (es - horizon_s[-1]) / pip_size
        reward_short[i] = pnl_s - tx_cost_pips

    # Combined label - pick the best (most profitable) direction
    reward = np.maximum(reward_long, reward_short)
    label = np.select(
        [
            (reward_long > tx_cost_pips) & (reward_long >= reward_short),
            (reward_short > tx_cost_pips) & (reward_short > reward_long),
        ],
        [1, -1],
        default=0,
    )
    optimal = np.select([reward_long > reward_short, reward_short > reward_long], ["long", "short"], default="hold")

    assert pl is not None
    result_pl = pl.DataFrame(
        {
            "timestamp_utc": index_values,
            "reward_long": reward_long,
            "reward_short": reward_short,
            "reward": reward,
            "label": label,
            "optimal_side": optimal,
        }
    )
    valid_s = pl.Series("valid_market", valid_market)
    # Use max horizon for tail cut (consistent with regime function)
    try:
        from config.settings import LABELING as L
    except Exception:
        L = {}
    tail_cut = max(
        lookahead_bars,
        int(
            lookahead_bars
            * max_label_horizon_mult(
                {"barrier_scale": L.get("barrier_scale", {}), "session_horizon_mult": L.get("session_horizon_mult", {})}
            )
        ),
    ) + max(0, int(execution_delay_bars))
    result_pl = result_pl.with_columns(valid_s)
    result_pl = result_pl.with_columns(
        [
            pl.when(pl.col("valid_market")).then(pl.col("reward_long")).otherwise(0.0).alias("reward_long"),
            pl.when(pl.col("valid_market")).then(pl.col("reward_short")).otherwise(0.0).alias("reward_short"),
            pl.when(pl.col("valid_market")).then(pl.col("reward")).otherwise(0.0).alias("reward"),
            pl.when(pl.col("valid_market")).then(pl.col("label")).otherwise(0).alias("label"),
            pl.when(pl.col("valid_market"))
            .then(pl.col("optimal_side"))
            .otherwise(pl.lit("hold"))
            .alias("optimal_side"),
        ]
    ).drop("valid_market")
    if tail_cut > 0:
        result_pl = result_pl.head(max(0, len(result_pl) - tail_cut))
    counts = result_pl["label"].value_counts()
    count_col = "count" if "count" in counts.columns else counts.columns[-1]
    count_map = dict(zip(counts["label"].to_list(), counts[count_col].to_list(), strict=False))
    print(
        f"[RLLabeling] {len(result_pl):,} labels | "
        f"Long: {count_map.get(1, 0):,}  Hold: {count_map.get(0, 0):,}  Short: {count_map.get(-1, 0):,}"
    )
    if return_polars:
        return result_pl
    return _polars_result_to_pandas(result_pl)


def _compute_slippage_multiplier(
    atr: np.ndarray,
    spread: np.ndarray,
    vol_window: int = 120,
    spread_window: int = 120,
    vol_alpha: float = 0.5,
    liq_alpha: float = 0.3,
) -> np.ndarray:
    """
    E: Per-bar slippage cost multiplier.

    In practice, the broker's spread widens during high-volatility or
    thin-liquidity windows, causing real transaction costs to exceed the
    constant pip assumption. This function produces a per-bar multiplier:

        mult_t = 1  +  vol_alpha  x max(0, atr_t / median_atr - 1)
                    +  liq_alpha  x max(0, spread_t / median_spread - 1)

    Values: 1.0 = normal, 1.5 = moderately expensive, 2.0+ = rollover/news.
    The result is applied as:  effective_tx_cost = base_tx_cost x mult_t

    When ATR or spread data are unavailable (all-NaN / zeros), returns ones.
    """
    n = len(atr)
    mult = np.ones(n, dtype=np.float32)

    # Rolling median ATR
    med_atr = pd.Series(atr).rolling(vol_window, min_periods=10).median().ffill().to_numpy(dtype=np.float32)
    med_atr = np.where(np.isfinite(med_atr) & (med_atr > 0), med_atr, np.asarray(atr, dtype=np.float32))
    if med_atr.max() > 0:
        vol_ratio = np.maximum(0.0, atr / np.maximum(med_atr, 1e-10) - 1.0)
        mult += vol_alpha * vol_ratio

    # Rolling median spread
    med_spr = pd.Series(spread).rolling(spread_window, min_periods=10).median().ffill().to_numpy(dtype=np.float32)
    med_spr = np.where(np.isfinite(med_spr) & (med_spr > 0), med_spr, np.asarray(spread, dtype=np.float32))
    if med_spr.max() > 0:
        spr_ratio = np.maximum(0.0, spread / np.maximum(med_spr, 1e-10) - 1.0)
        mult += liq_alpha * spr_ratio

    return mult.astype(np.float32)


def compute_rl_reward_labels_regime(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    atr_col: str = "atr_6",
    lookahead_bars: int = 30,  # match LABELING / strategy.lookahead_bars
    profit_atr_mult: float | None = None,
    stop_atr_mult: float | None = None,
    pip_size: float = 0.0001,
    session_col: str | None = None,
    regime_col: str | None = None,
    no_trade_col: str | None = None,
    latency_col: str | None = None,
    latency_baseline_ms: float = 50.0,
    execution_delay_bars: int = 1,
) -> pd.DataFrame:
    """
    Regime-conditional RL reward labeling.

    Improvements over the base version:
      1. Barrier widths scale with vol regime (wider in high-vol, tighter in MR)
      2. Lookahead horizon scales with regime (shorter for MR, longer for trending)
      3. Transaction cost scales with session (wider in Asia / off-hours)
      4. Penalise bad wins: positive PnL but high MAE or late exit -> discounted
      5. No-trade zone: bars where no_trade_score is high get label=0 explicitly
      6. Multi-target output: direction + path_quality + confidence columns added
      7. (E) Slippage-aware costs: inflate tx cost when vol↑ or liquidity↓
      8. (F) Latency penalty: zero out edge when expected edge < latency-adjusted slippage

    Falls back to base parameters when regime columns are unavailable.
    """
    try:
        from config.settings import LABEL_REGIME as _LR
        from config.settings import NO_TRADE as _NT

        _LR_full = dict(_LR)
        barrier_scale = _LR["barrier_scale"]
        high_vol_pct = _LR["high_vol_pct"]
        low_vol_pct = _LR["low_vol_pct"]
        sess_cost_scale = _LR["session_cost_scale"]
        _NT["liquidity_vacuum_max"]
    except Exception:
        _LR_full = {}
        barrier_scale = {}
        high_vol_pct = 0.75
        low_vol_pct = 0.25
        sess_cost_scale = {}

    try:
        from config.settings import EXECUTION as _EX

        _vol_alpha = float(_EX.get("slippage_vol_alpha", 0.5))
        _liq_alpha = float(_EX.get("slippage_liq_alpha", 0.3))
        _vol_win = int(_EX.get("slippage_vol_window", 120))
        _spr_win = int(_EX.get("slippage_spread_window", 120))
        _lat_base = float(_EX.get("latency_baseline_ms", latency_baseline_ms))
    except Exception:
        _vol_alpha = 0.5
        _liq_alpha = 0.3
        _vol_win = 120
        _spr_win = 120
        _lat_base = float(latency_baseline_ms)

    try:
        from config.settings import LABELING as _LBL
    except Exception:
        _LBL = {}

    if profit_atr_mult is None:
        profit_atr_mult = float(_LBL.get("profit_target_atr", 1.2))
    if stop_atr_mult is None:
        stop_atr_mult = float(_LBL.get("stop_loss_atr", 0.8))

    close = np.asarray(bars["close"].reindex(features.index).ffill(), dtype=np.float64)
    if "ask_close" in bars.columns and "bid_close" in bars.columns:
        entry_long = np.asarray(bars["ask_close"].reindex(features.index).ffill(), dtype=np.float64)
        entry_short = np.asarray(bars["bid_close"].reindex(features.index).ffill(), dtype=np.float64)
        # Exit long at bid, exit short at ask (DS-001 / live realism).
        exit_long_path = entry_short.copy()
        exit_short_path = entry_long.copy()
    else:
        spread_half = np.asarray(
            features.get("spread_pips", pd.Series(0.5, index=features.index)).to_numpy(dtype=np.float64),
            dtype=np.float64,
        ) * pip_size / 2.0
        entry_long = close + spread_half
        entry_short = close - spread_half
        exit_long_path = entry_short.copy()
        exit_short_path = entry_long.copy()

    atr = np.asarray(features[atr_col], dtype=np.float64) if atr_col in features.columns else np.full(len(close), 0.0005)
    valid_market = (
        np.isfinite(close)
        & np.isfinite(entry_long)
        & np.isfinite(entry_short)
        & np.isfinite(atr)
        & (close > 0)
        & (entry_long > 0)
        & (entry_short > 0)
        & (atr > 0)
    )
    close = np.nan_to_num(close, nan=0.0, posinf=0.0, neginf=0.0)
    entry_long = np.nan_to_num(entry_long, nan=0.0, posinf=0.0, neginf=0.0)
    entry_short = np.nan_to_num(entry_short, nan=0.0, posinf=0.0, neginf=0.0)
    exit_long_path = np.nan_to_num(exit_long_path, nan=0.0, posinf=0.0, neginf=0.0)
    exit_short_path = np.nan_to_num(exit_short_path, nan=0.0, posinf=0.0, neginf=0.0)
    atr = np.nan_to_num(atr, nan=0.0005, posinf=0.0005, neginf=0.0005)
    n = len(close)

    # ── E: Slippage-aware cost multiplier (vol + liquidity inflation) ─────────
    _spread_vals = np.full(n, 0.5, dtype=np.float64)  # fallback: 0.5 pip spread
    if "spread_pips" in features.columns:
        _spread_vals = np.asarray(features["spread_pips"].fillna(0.5).to_numpy(), dtype=np.float64)
    elif "liquidity_vacuum" in features.columns:
        _spread_vals = np.asarray(features["liquidity_vacuum"].fillna(1.0).to_numpy(), dtype=np.float64)  # proxy

    slippage_mult = _compute_slippage_multiplier(
        atr=atr,
        spread=_spread_vals,
        vol_window=_vol_win,
        spread_window=_spr_win,
        vol_alpha=_vol_alpha,
        liq_alpha=_liq_alpha,
    )

    # ── F: Latency array (expected_latency_ms per bar, or constant baseline) ──
    latency_ms = np.full(n, _lat_base, dtype=np.float32)
    if latency_col and latency_col in features.columns:
        latency_ms = np.asarray(features[latency_col].fillna(_lat_base).to_numpy(), dtype=np.float32)
    # Latency cost premium: each extra ms above baseline = proportional edge reduction
    # lat_mult: fraction of 1-pip cost eaten by latency.  At baseline -> 0 penalty.
    latency_pip_cost = np.maximum(0.0, (latency_ms - _lat_base) / 1000.0)

    # ── Regime arrays ─────────────────────────────────────────────────────────
    # Priority: explicit regime_col from features > ATR-quantile fallback.
    # regime_col values: integer 0=low_vol, 1=normal, 2=high_vol (from HMM regime_class)
    # or string labels like "trending", "mean_rev", "ranging" (from RegimeGateClassifier).
    vol_series = pd.Series(atr, index=features.index)
    vol_q_hi = vol_series.rolling(500, min_periods=50).quantile(high_vol_pct).values
    vol_q_lo = vol_series.rolling(500, min_periods=50).quantile(low_vol_pct).values

    _use_regime_col = regime_col is not None and regime_col in features.columns
    if _use_regime_col:
        # Map integer regime_class (0=low_vol, 1=normal, 2=high_vol) to string labels
        _regime_int_map = {0.0: "low_vol", 1.0: "normal", 2.0: "high_vol"}
        regime_vals = np.asarray(features[regime_col].ffill().to_numpy(), dtype=object)
        regime_mapped = np.empty(len(regime_vals), dtype=object)
        for idx in range(len(regime_vals)):
            val = regime_vals[idx]
            if isinstance(val, str):
                regime_mapped[idx] = val.lower()
            elif isinstance(val, (int, np.integer)):
                regime_mapped[idx] = _regime_int_map.get(float(val), "normal")
            elif isinstance(val, (float, np.floating)):
                if np.isnan(val):
                    regime_mapped[idx] = "normal"
                else:
                    regime_mapped[idx] = _regime_int_map.get(val, "normal")
            else:
                regime_mapped[idx] = str(val).lower()
        regime_vals = regime_mapped
        for idx in range(len(regime_vals)):
            if regime_vals[idx] is None:
                regime_vals[idx] = "normal"
    else:
        regime_vals = None

    # Session / overlap cost multipliers (cost first; horizon resolved per bar).
    # DST-aware overlap flags (asia_london, london_ny) take precedence over
    # exclusive session_label when present (labeling-only aux columns).
    _asia_london = np.zeros(n, dtype=bool)
    _london_ny = np.zeros(n, dtype=bool)
    if "asia_london" in features.columns:
        _asia_london = np.asarray(features["asia_london"].fillna(0).values).astype(bool)
    if "london_ny" in features.columns and features["london_ny"].dtype != object:
        # Numeric overlap flag (DST from bars or FE); Utf8 session names ignored.
        _london_ny = np.asarray(features["london_ny"].fillna(0).values).astype(bool)
    # Prefer explicit overlap aux if FE kept crude london_ny in X and bars
    # attached a separate overlap column under the same name - both are 0/1 floats.
    sess_vals = None
    if session_col and session_col in features.columns:
        sess_vals = np.asarray(features[session_col].to_numpy(), dtype=object)
    sess_mult = np.ones(n)
    for i in range(n):
        s = str(sess_vals[i]) if sess_vals is not None else None
        sess_mult[i] = resolve_session_cost(
            s,
            asia_london=_asia_london[i],
            london_ny=_london_ny[i],
            label_regime=_LR_full or {"session_cost_scale": sess_cost_scale},
        )

    # Spread z for horizon gate (optional)
    _spread_z = None
    if "spread_zscore" in features.columns:
        _spread_z = features["spread_zscore"].fillna(0.0).values.astype(np.float64)

    # No-trade mask
    no_trade_mask = np.zeros(n, dtype=bool)
    if no_trade_col and no_trade_col in features.columns:
        no_trade_mask = features[no_trade_col].values > 0.67

    reward_long = np.zeros(n, dtype=np.float32)
    reward_short = np.zeros(n, dtype=np.float32)
    path_quality = np.zeros(n, dtype=np.float32)  # B: path quality
    confidence_target = np.zeros(n, dtype=np.float32)  # B: drawdown risk proxy

    # Max regime-scaled horizon for loop bounds (session/spread only shorten or
    # stretch within max_label_horizon_mult).
    _max_h = max(
        lookahead_bars,
        int(
            lookahead_bars
            * max_label_horizon_mult(
                {"barrier_scale": barrier_scale, "session_horizon_mult": _LR_full.get("session_horizon_mult", {})}
            )
        ),
    )

    delay = max(0, int(execution_delay_bars))
    for i in range(n - _max_h - delay):
        entry_i = i + delay
        if not valid_market[i] or not valid_market[entry_i]:
            continue
        # ── A: Regime-conditional barrier parameters ───────────────────────
        # Use explicit regime label when available; fall back to ATR quantile.
        v = atr[entry_i]
        if _use_regime_col:
            rv = str(regime_vals[entry_i]).lower()
            if "high_vol" in rv or "volatile" in rv:
                cfg = barrier_scale.get(
                    "high_vol",
                    {"tp_atr_mult": profit_atr_mult * 1.33, "sl_atr_mult": stop_atr_mult * 1.66, "horizon_mult": 0.7},
                )
            elif "mean_rev" in rv or "ranging" in rv:
                # Mean reversion: tighter barriers, shorter horizon - trades resolve quickly
                cfg = barrier_scale.get(
                    "mean_rev",
                    {"tp_atr_mult": profit_atr_mult * 0.66, "sl_atr_mult": stop_atr_mult * 0.88, "horizon_mult": 0.5},
                )
            elif "low_vol" in rv or "low" in rv:
                cfg = barrier_scale.get(
                    "low_vol",
                    {"tp_atr_mult": profit_atr_mult * 0.66, "sl_atr_mult": stop_atr_mult * 0.88, "horizon_mult": 1.3},
                )
            elif "trending" in rv:
                cfg = barrier_scale.get(
                    "trending",
                    {"tp_atr_mult": profit_atr_mult * 1.33, "sl_atr_mult": stop_atr_mult * 1.33, "horizon_mult": 1.5},
                )
            else:
                cfg = barrier_scale.get(
                    "normal", {"tp_atr_mult": profit_atr_mult, "sl_atr_mult": stop_atr_mult, "horizon_mult": 1.0}
                )
        else:
            # ATR-quantile fallback
            if v >= vol_q_hi[entry_i]:
                cfg = barrier_scale.get(
                    "high_vol",
                    {"tp_atr_mult": profit_atr_mult * 1.33, "sl_atr_mult": stop_atr_mult * 1.66, "horizon_mult": 0.7},
                )
            elif v <= vol_q_lo[entry_i]:
                cfg = barrier_scale.get(
                    "low_vol",
                    {"tp_atr_mult": profit_atr_mult * 0.66, "sl_atr_mult": stop_atr_mult * 0.88, "horizon_mult": 1.3},
                )
            else:
                cfg = barrier_scale.get(
                    "normal", {"tp_atr_mult": profit_atr_mult, "sl_atr_mult": stop_atr_mult, "horizon_mult": 1.0}
                )

        tp_mult = float(cfg["tp_atr_mult"])
        sl_mult = float(cfg["sl_atr_mult"])
        _sess = str(sess_vals[entry_i]) if sess_vals is not None else None
        _sz = float(_spread_z[entry_i]) if _spread_z is not None else None
        horizon = resolve_horizon(
            lookahead_bars,
            session=_sess,
            spread_z=_sz,
            asia_london=_asia_london[entry_i],
            london_ny=_london_ny[entry_i],
            regime_horizon_mult=float(cfg["horizon_mult"]),
            label_regime=_LR_full,
        )
        # E: inflate tx cost by slippage multiplier (vol + liquidity premium)
        tx_cost = 1.5 * pip_size * sess_mult[entry_i] * float(slippage_mult[entry_i])
        tx_pips = tx_cost / pip_size

        el = entry_long[entry_i]
        es = entry_short[entry_i]
        tp_l = el + tp_mult * atr[entry_i]
        sl_l = el - sl_mult * atr[entry_i]
        tp_s = es - tp_mult * atr[entry_i]
        sl_s = es + sl_mult * atr[entry_i]

        fwd_l = exit_long_path[entry_i + 1 : entry_i + 1 + horizon]
        fwd_s = exit_short_path[entry_i + 1 : entry_i + 1 + horizon]
        if len(fwd_l) == 0 or len(fwd_s) == 0:
            continue
        if (
            not np.isfinite(fwd_l).all()
            or not np.isfinite(fwd_s).all()
            or not valid_market[entry_i + 1 : entry_i + 1 + len(fwd_l)].all()
        ):
            continue

        # ── Long path simulation (exit at bid) ─────────────────────────────
        mae_l = 0.0
        mfe_l = 0.0
        pnl_l = None
        exit_bar_l = len(fwd_l)
        for j, p in enumerate(fwd_l):
            mae_l = max(mae_l, (el - p) / pip_size)  # max adverse excursion
            mfe_l = max(mfe_l, (p - el) / pip_size)  # max favourable excursion
            if p >= tp_l:
                pnl_l = (tp_l - el) / pip_size
                exit_bar_l = j + 1
                break
            elif p <= sl_l:
                pnl_l = (sl_l - el) / pip_size
                exit_bar_l = j + 1
                break
        if pnl_l is None:
            pnl_l = (fwd_l[-1] - el) / pip_size

        # ── C: Bad-win penalty ─────────────────────────────────────────────
        # Three types of bad wins - all get a 30% discount on the reward:
        #   1. Large MAE: price went badly against us before recovering
        #      (suggests the win was luck, not edge)
        #   2. Late exit: trade lingered until horizon without conviction
        #      (suggests weak trend, likely to reverse in live trading)
        #   3. Spike win: MFE >> TP (price briefly spiked past TP, then reversed)
        #      In live trading with limit orders this would not have been filled
        #      at the clean TP; real fill would be worse.
        bad_win_l = 0.0
        if pnl_l > 0:
            is_large_mae = mae_l > abs(pnl_l) * 1.5
            is_late_exit = exit_bar_l > horizon * 0.8
            is_spike_win = (
                mfe_l > tp_mult * atr[entry_i] / pip_size * 1.5  # MFE well past TP
                and exit_bar_l <= 2
            )  # but exited very fast
            if is_large_mae or is_late_exit or is_spike_win:
                bad_win_l = abs(pnl_l) * 0.3
        reward_long[i] = pnl_l - tx_pips - bad_win_l

        # ── Short path simulation (exit at ask) ────────────────────────────
        mae_s = 0.0
        mfe_s = 0.0
        pnl_s = None
        exit_bar_s = len(fwd_s)
        for j, p in enumerate(fwd_s):
            mae_s = max(mae_s, (p - es) / pip_size)
            mfe_s = max(mfe_s, (es - p) / pip_size)
            if p <= tp_s:
                pnl_s = (es - tp_s) / pip_size
                exit_bar_s = j + 1
                break
            elif p >= sl_s:
                pnl_s = (es - sl_s) / pip_size
                exit_bar_s = j + 1
                break
        if pnl_s is None:
            pnl_s = (es - fwd_s[-1]) / pip_size

        bad_win_s = 0.0
        if pnl_s > 0:
            is_large_mae = mae_s > abs(pnl_s) * 1.5
            is_late_exit = exit_bar_s > horizon * 0.8
            is_spike_win = mfe_s > tp_mult * atr[entry_i] / pip_size * 1.5 and exit_bar_s <= 2
            if is_large_mae or is_late_exit or is_spike_win:
                bad_win_s = abs(pnl_s) * 0.3
        reward_short[i] = pnl_s - tx_pips - bad_win_s

        # ── F: Latency penalty ─────────────────────────────────────────────
        lat_pen = float(latency_pip_cost[entry_i])
        if lat_pen > 0:
            reward_long[i] -= lat_pen
            reward_short[i] -= lat_pen

        # ── B: Multi-target outputs ────────────────────────────────────────
        # path_quality: how clean was the winning direction's path?
        #   1.0 = straight to TP with no adverse excursion
        #   0.0 = highly noisy / adverse path
        if reward_long[i] >= reward_short[i]:
            best_pnl, best_mae = abs(reward_long[i]), mae_l
        else:
            best_pnl, best_mae = abs(reward_short[i]), mae_s
        path_quality[i] = float(np.clip(1.0 - best_mae / (best_pnl + 1e-3), 0.0, 1.0))

        # confidence_target: normalized drawdown risk.
        # High value = this trade had large intra-trade drawdown -> risky.
        # Used as target for confidence/risk head (model should be less confident here).
        max_adverse = max(mae_l, mae_s)
        barrier_size = tp_mult * atr[i] / pip_size + 1e-6
        confidence_target[i] = float(np.clip(1.0 - max_adverse / barrier_size, 0.0, 1.0))

    # ── Combined label ─────────────────────────────────────────────────────
    reward = np.maximum(reward_long, reward_short)
    tx_pip = 1.5  # approx threshold in pips
    label = np.select(
        [
            (reward_long > tx_pip) & (reward_long >= reward_short),
            (reward_short > tx_pip) & (reward_short > reward_long),
        ],
        [1, -1],
        default=0,
    )

    # No-trade as first-class outcome
    no_trade_label = np.zeros(n, dtype=np.int8)
    no_trade_label[no_trade_mask] = 1

    optimal = np.select([reward_long > reward_short, reward_short > reward_long], ["long", "short"], default="hold")

    result = pd.DataFrame(
        {
            "reward_long": reward_long,
            "reward_short": reward_short,
            "reward": reward,
            "label": label,
            "path_quality": path_quality,  # B: path quality target
            "confidence_target": confidence_target,  # B: drawdown risk target
            "no_trade": no_trade_label,  # no-trade abstain flag
            "optimal_side": optimal,
        },
        index=features.index,
    )

    result.loc[~valid_market, ["reward_long", "reward_short", "reward"]] = 0.0
    result.loc[~valid_market, "label"] = 0
    result.loc[~valid_market, ["path_quality", "confidence_target"]] = 0.0
    result.loc[~valid_market, "no_trade"] = 1
    result.loc[~valid_market, "optimal_side"] = "hold"

    result.iloc[-(_max_h + delay) :] = np.nan
    result = result.dropna()

    counts = pd.Series(result["label"]).value_counts()
    nt_pct = no_trade_mask.mean() * 100
    pq_avg = float(result["path_quality"].mean())
    print(
        f"[RLLabelingRegime] {len(result):,} labels | "
        f"Long: {counts.get(1, 0):,}  Hold: {counts.get(0, 0):,}  Short: {counts.get(-1, 0):,} | "
        f"No-trade: {nt_pct:.1f}% | avg path_quality: {pq_avg:.3f} | "
        f"regime_col: {'yes' if _use_regime_col else 'ATR-fallback'}"
    )

    # TIER-1 INTEGRITY CHECK: "Wrong / Unstable Loss" prevention.
    # Mathematically guarantee no positive reward was ever assigned to a trade
    # if it couldn't beat the explicitly calculated spread and slippage cost.
    # (Since we subtract costs in the loop, if reward > 0, it intrinsically beat the costs,
    # but we add this explicit structural assert to prevent future code regressions).
    assert not (result["label"] == 1).any() or result.loc[result["label"] == 1, "reward_long"].min() > 0, (
        "Tier-1 Violation: Long label assigned with negative net reward."
    )
    assert not (result["label"] == -1).any() or result.loc[result["label"] == -1, "reward_short"].min() > 0, (
        "Tier-1 Violation: Short label assigned with negative net reward."
    )

    return result


def align_labels_with_features(
    labels_df: pd.DataFrame,
    features_df: pd.DataFrame,
    target_col: str = "label",
) -> tuple[pd.DataFrame | Any, pd.Series | Any, pd.DataFrame | Any]:
    """Inner-join and return (X, y, sidecar) ready for supervised or RL training.

    sidecar is a DataFrame aligned with X/y containing the true direction
    label plus auxiliary targets (path_quality, confidence_target, no_trade)
    when produced by the regime labeling function.  The direction label is
    kept even when y targets continuous reward/PnL so downstream caches can
    write an authoritative y_cls sidecar instead of inferring direction from
    sign(reward).
    """
    if _is_polars_frame(labels_df) or _is_polars_frame(features_df):
        assert pl is not None
        labels_pl = _to_polars_with_timestamp(labels_df)
        features_pl = _to_polars_with_timestamp(features_df)
        # Match join-key precision (pandas→Polars often yields μs; bar frames are ns).
        ts_dtype = features_pl.schema["timestamp_utc"]
        labels_pl = labels_pl.with_columns(pl.col("timestamp_utc").cast(ts_dtype))
        features_pl = features_pl.with_columns(pl.col("timestamp_utc").cast(ts_dtype))
        _sidecar_cols = [
            c
            for c in ("label", "path_quality", "confidence_target", "no_trade", "optimal_side")
            if c in labels_pl.columns
        ]
        if target_col not in labels_pl.columns:
            raise KeyError(f"target_col '{target_col}' not found in labels")
        join_cols = list(dict.fromkeys(["reward", "label", target_col, *_sidecar_cols]))
        combined = features_pl.join(
            labels_pl.select(["timestamp_utc"] + [c for c in join_cols if c in labels_pl.columns]),
            on="timestamp_utc",
            how="inner",
        ).drop_nulls()
        X_cols = [c for c in combined.columns if c not in {"timestamp_utc", *join_cols}]
        X = combined.select(X_cols)
        y = combined[target_col]
        # Keep timestamp_utc on the sidecar so callers can build time_idx / reindex
        # without a second Polars↔Pandas round-trip of the feature matrix.
        sc_keep = ["timestamp_utc"] + [c for c in _sidecar_cols if c in combined.columns]
        sidecar = combined.select(sc_keep)
        return X, y, sidecar

    _sidecar_cols = [
        c for c in ("label", "path_quality", "confidence_target", "no_trade", "optimal_side") if c in labels_df.columns
    ]
    if target_col not in labels_df.columns:
        raise KeyError(f"target_col '{target_col}' not found in labels")
    join_cols = list(dict.fromkeys(["reward", "label", target_col, *_sidecar_cols]))
    combined = features_df.join(labels_df[join_cols], how="inner")
    combined = combined.dropna()
    X = sanitize_frame(
        combined.drop(columns=join_cols),
        fill_value=0.0,
        context="align_labels_with_features.X",
    )
    y = combined[target_col]
    sidecar = combined[_sidecar_cols].copy() if _sidecar_cols else pd.DataFrame(index=combined.index)
    # Print statement removed to prevent massive IO bottlenecks with continuous RL rewards
    return X, y, sidecar


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "..")
    from data.data_ingestion import ForexDataPipeline, generate_synthetic_tick_data
    from features.feature_engineering import FeatureEngineer

    ticks = generate_synthetic_tick_data(n_rows=500_000)
    bars = ForexDataPipeline(bar_freq="1min", session_filter=False, apply_frac_diff=False).run(ticks)
    fe = FeatureEngineer()
    feats = fe.build(bars)

    labels = compute_rl_reward_labels(bars, feats)
    print(f"\nSample labels:\n{labels.head(5)}")
    X, y, sidecar = align_labels_with_features(labels, feats)
    print(f"Training shapes: X={X.shape} y={y.shape} sidecar={sidecar.shape}")
