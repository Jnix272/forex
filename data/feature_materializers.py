"""
Feature Materializers
=====================
Per-feature computation logic for materializing features from raw data.
Each materializer handles a specific feature type (price, volatility, microstructure, etc.).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from data.feature_definitions import (
    FeatureSource,
    FeatureSpec,
    MaterializationStrategy,
)
from data.feature_store import FeatureStore

# ════════════════════════════════════════════════════════════════════════════
# BASE MATERIALIZER
# ════════════════════════════════════════════════════════════════════════════

class BaseMaterializer:
    """Base class for feature materializers."""

    def __init__(self, store: FeatureStore):
        self.store = store

    def compute(
        self, spec: FeatureSpec, bars: pl.DataFrame, start: datetime, end: datetime
    ) -> pl.DataFrame | None:
        """Compute feature values from raw bars. Returns DataFrame with timestamp_utc + feature column."""
        raise NotImplementedError

    def validate(self, df: pl.DataFrame) -> None:
        if "timestamp_utc" not in df.columns:
            raise ValueError("Missing 'timestamp_utc' column")
        if len(df.columns) != 2:
            raise ValueError(f"Expected 2 columns (timestamp_utc + feature), got {len(df.columns)}")

    @property
    def requires_ohlcv(self) -> bool:
        return True

    @property
    def requires_ticks(self) -> bool:
        return False

    @property
    def requires_book(self) -> bool:
        return False


# ════════════════════════════════════════════════════════════════════════════
# PRICE MATERIALIZERS
# ════════════════════════════════════════════════════════════════════════════

class PriceMaterializer(BaseMaterializer):
    """Compute price-derived features (returns, spreads, etc.)."""

    def compute(
        self, spec: FeatureSpec, bars: pl.DataFrame, start: datetime, end: datetime
    ) -> pl.DataFrame | None:
        bars = bars.sort("timestamp_utc")

        if spec.name == "close":
            return bars.select("timestamp_utc", pl.col("close")).drop_nulls()

        if spec.name == "log_ret_1":
            return bars.select("timestamp_utc",
                (pl.col("close").log() - pl.col("close").shift(1).log()).alias("log_ret_1"),
            ).drop_nulls()

        if spec.name == "log_ret_5":
            return bars.select("timestamp_utc",
                (pl.col("close").log() - pl.col("close").shift(5).log()).alias("log_ret_5"),
            ).drop_nulls()

        if spec.name == "log_ret_20":
            return bars.select("timestamp_utc",
                (pl.col("close").log() - pl.col("close").shift(20).log()).alias("log_ret_20"),
            ).drop_nulls()

        if spec.name == "hlr":
            return bars.select("timestamp_utc",
                ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("hlr"),
            ).drop_nulls()

        if spec.name == "spread_bps":
            return bars.select("timestamp_utc",
                ((pl.col("close") - pl.col("open")) / pl.col("close") * 10_000).abs().alias("spread_bps"),
            ).drop_nulls()

        return None

    @property
    def requires_ohlcv(self) -> bool:
        return True


class VolatilityMaterializer(BaseMaterializer):
    """Compute volatility-derived features (ATR, rolling vol, Bollinger, etc.)."""

    def compute(
        self, spec: FeatureSpec, bars: pl.DataFrame, start: datetime, end: datetime
    ) -> pl.DataFrame | None:
        bars = bars.sort("timestamp_utc")

        window = spec.params.get("window", 20)

        if spec.name == "atr_6":
            window = 6
            tr = pl.max_horizontal(
                pl.col("high") - pl.col("low"),
                (pl.col("high") - pl.col("close").shift(1)).abs(),
                (pl.col("low") - pl.col("close").shift(1)).abs(),
            )
            return bars.select("timestamp_utc",
                tr.alias("tr_6")
            ).with_columns(
                pl.col("tr_6").rolling_mean(window_size=window).alias(spec.name)
            ).drop(["tr_6"]).drop_nulls()

        if spec.name == "atr_20":
            window = 20
            tr = pl.max_horizontal(
                pl.col("high") - pl.col("low"),
                (pl.col("high") - pl.col("close").shift(1)).abs(),
                (pl.col("low") - pl.col("close").shift(1)).abs(),
            )
            return bars.select("timestamp_utc",
                tr.alias("tr_20")
            ).with_columns(
                pl.col("tr_20").rolling_mean(window_size=window).alias(spec.name)
            ).drop(["tr_20"]).drop_nulls()

        if spec.name == "rolling_vol_20":
            log_rets = pl.col("close").log() - pl.col("close").shift(1).log()
            return bars.select("timestamp_utc",
                log_rets.rolling_std(window_size=20).alias(spec.name)
            ).drop_nulls()

        if spec.name == "bollinger_upper_20":
            close = pl.col("close")
            sma = close.rolling_mean(window_size=window)
            std = close.rolling_std(window_size=window)
            return bars.select("timestamp_utc",
                (sma + 2 * std).alias(spec.name)
            ).drop_nulls()

        if spec.name == "bollinger_lower_20":
            close = pl.col("close")
            sma = close.rolling_mean(window_size=window)
            std = close.rolling_std(window_size=window)
            return bars.select("timestamp_utc",
                (sma - 2 * std).alias(spec.name)
            ).drop_nulls()

        return None


class MicrostructureMaterializer(BaseMaterializer):
    """Compute market microstructure features (OFI, OBI, Hurst, fractal dimension, etc.)."""

    # ── These need tick/order book data; return pd-extended or estimated from OHLCV ──

    def compute(
        self, spec: FeatureSpec, bars: pl.DataFrame, start: datetime, end: datetime
    ) -> pl.DataFrame | None:
        bars = bars.sort("timestamp_utc")

        if spec.name == "ofi_20":
            # Order Flow Imbalance proxy from OHLCV: (close - open) / (high - low + 1e-12)
            return bars.select("timestamp_utc",
                ((pl.col("close") - pl.col("open")) / (pl.col("high") - pl.col("low") + 1e-12))
                .rolling_mean(window_size=20).alias(spec.name)
            ).drop_nulls()

        if spec.name == "obi_proxy":
            # Order Book Imbalance proxy using volume
            return bars.select("timestamp_utc",
                ((pl.col("volume") - pl.col("volume").shift(1)) / (pl.col("volume") + 1e-12))
                .cum_sum().alias(spec.name)
            ).drop_nulls()
            # Note: cum_sum maps to OBI accumulation in the forex scaling logic

        if spec.name == "hurst_120":
            return bars.select("timestamp_utc",
                pl.col("close").log().alias("log_close")
            ).with_columns(
                _hurst_exponent(pl.col("log_close"), 120).alias(spec.name)
            ).drop_nulls()

        if spec.name == "fractal_dim_30":
            return bars.select("timestamp_utc",
                _fractal_dimension(pl.col("log_ret_1"), 30).alias(spec.name)
            ).drop_nulls()

        if spec.name == "iv_proxy_20":
            # Implied Volatility proxy: intraday range normalized
            return bars.select("timestamp_utc",
                ((pl.col("high") - pl.col("low")) / pl.col("close"))
                .rolling_mean(window_size=20).alias(spec.name)
            ).drop_nulls()

        if spec.name == "skew_proxy_20":
            log_ret = pl.col("close").log() - pl.col("close").shift(1).log()
            return bars.select("timestamp_utc",
                log_ret.rolling_skew(window_size=20).alias(spec.name)
            ).drop_nulls()

        return None


class SessionMaterializer(BaseMaterializer):
    """Compute session-based features (time of day, day of week, etc.)."""

    def compute(
        self, spec: FeatureSpec, bars: pl.DataFrame, start: datetime, end: datetime
    ) -> pl.DataFrame | None:
        bars = bars.sort("timestamp_utc")

        if spec.name == "hour_sin":
            return bars.with_columns(
                pl.col("timestamp_utc").dt.hour().cast(pl.Float64).alias("hour_sin")
            ).select("timestamp_utc",
                (2 * np.pi * pl.col("hour_sin") / 24).sin().alias(spec.name)
            )

        if spec.name == "hour_cos":
            return bars.with_columns(
                pl.col("timestamp_utc").dt.hour().cast(pl.Float64).alias("hour_cos")
            ).select("timestamp_utc",
                (2 * np.pi * pl.col("hour_cos") / 24).cos().alias(spec.name)
            )

        if spec.name == "dow_sin":
            return bars.with_columns(
                pl.col("timestamp_utc").dt.weekday().cast(pl.Float64).alias("dow_sin")
            ).select("timestamp_utc",
                (2 * np.pi * (pl.col("dow_sin") - 1) / 7).sin().alias(spec.name)
            )

        if spec.name == "dow_cos":
            return bars.with_columns(
                pl.col("timestamp_utc").dt.weekday().cast(pl.Float64).alias("dow_cos")
            ).select("timestamp_utc",
                (2 * np.pi * (pl.col("dow_cos") - 1) / 7).cos().alias(spec.name)
            )

        if spec.name == "session_asia":
            hour = pl.col("timestamp_utc").dt.hour()
            return bars.select("timestamp_utc",
                hour.is_in([23, 0, 1, 2, 3, 4, 5, 6]).cast(pl.Int32).alias(spec.name)
            )

        if spec.name == "session_london":
            hour = pl.col("timestamp_utc").dt.hour()
            return bars.select("timestamp_utc",
                hour.is_in([7, 8, 9, 10, 11, 12, 13, 14]).cast(pl.Int32).alias(spec.name)
            )

        if spec.name == "session_ny":
            hour = pl.col("timestamp_utc").dt.hour()
            return bars.select("timestamp_utc",
                hour.is_in([13, 14, 15, 16, 17, 18, 19, 20]).cast(pl.Int32).alias(spec.name)
            )

        if spec.name == "is_monday":
            return bars.select("timestamp_utc",
                (pl.col("timestamp_utc").dt.weekday() == 1).cast(pl.Int32).alias(spec.name)
            )

        if spec.name == "is_friday":
            return bars.select("timestamp_utc",
                (pl.col("timestamp_utc").dt.weekday() == 5).cast(pl.Int32).alias(spec.name)
            )

        return None

    @property
    def requires_ohlcv(self) -> bool:
        return False


class MacroMaterializer(BaseMaterializer):
    """Compute macro-derived features from cross-asset / FRED panels (not FX OHLC)."""

    def _cache_dir(self) -> str:
        import os
        return os.environ.get(
            "CROSS_ASSET_CACHE",
            str(Path(getattr(self.store, "root", Path("data/feature_store"))) / "cross_asset_cache"),
        )

    def _load_panel(self, start: datetime, end: datetime) -> dict:
        from data.cross_asset import load_cross_asset_panel

        start_s = start.strftime("%Y-%m-%d") if hasattr(start, "strftime") else str(start)[:10]
        end_s = end.strftime("%Y-%m-%d") if hasattr(end, "strftime") else str(end)[:10]
        return load_cross_asset_panel(
            start_s, end_s, cache_dir=self._cache_dir(), source="auto",
        )

    @staticmethod
    def _series_to_frame(series, col: str) -> pl.DataFrame:
        import pandas as pd

        if series is None:
            return pl.DataFrame()
        if isinstance(series, pl.DataFrame):
            return series
        s = series if isinstance(series, pd.Series) else pd.Series(series)
        if s is None or len(s) == 0:
            return pl.DataFrame()
        idx = pd.DatetimeIndex(s.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        pdf = pd.DataFrame({"timestamp_utc": idx, col: s.to_numpy(dtype=float)})
        return pl.from_pandas(pdf).drop_nulls()

    def _align_to_bars(self, bars: pl.DataFrame, daily: pl.DataFrame, col: str) -> pl.DataFrame | None:
        if daily.is_empty() or col not in daily.columns:
            return None
        if bars is None or bars.is_empty() or "timestamp_utc" not in bars.columns:
            return daily.select("timestamp_utc", pl.col(col).alias(col)).drop_nulls()

        left = bars.select("timestamp_utc").unique().sort("timestamp_utc")
        right = daily.select("timestamp_utc", col).sort("timestamp_utc")
        # Match timezone awareness between frames
        left_tz = left["timestamp_utc"].dtype
        try:
            right = right.with_columns(pl.col("timestamp_utc").cast(left_tz))
        except Exception:
            pass
        joined = left.join_asof(right, on="timestamp_utc", strategy="backward")
        return joined.select("timestamp_utc", pl.col(col)).drop_nulls()

    def compute(
        self, spec: FeatureSpec, bars: pl.DataFrame, start: datetime, end: datetime
    ) -> pl.DataFrame | None:
        bars = bars.sort("timestamp_utc") if bars is not None and not bars.is_empty() else bars
        panel = self._load_panel(start, end)

        if spec.name == "yield_2y10y":
            daily = pl.DataFrame()
            if "YIELD_CURVE_SLOPE" in panel and panel["YIELD_CURVE_SLOPE"] is not None:
                daily = self._series_to_frame(panel["YIELD_CURVE_SLOPE"], "yield_2y10y")
            elif "US10Y" in panel and "US2Y" in panel:
                import pandas as pd
                us10 = panel["US10Y"]
                us2 = panel["US2Y"]
                idx = us10.index.union(us2.index)
                slope = us10.reindex(idx).ffill() - us2.reindex(idx).ffill()
                daily = self._series_to_frame(slope.dropna(), "yield_2y10y")
            if daily.is_empty():
                # FRED / synthetic via MacroYieldFeatureBuilder (never FX OHLC).
                from features.macro_features import MacroYieldFeatureBuilder
                built = MacroYieldFeatureBuilder().build(
                    bars if bars is not None else pl.DataFrame()
                )
                if (
                    built is not None
                    and not built.is_empty()
                    and "yield_curve_slope" in built.columns
                    and bars is not None
                    and not bars.is_empty()
                    and len(built) == bars.height
                ):
                    daily = bars.select("timestamp_utc").with_columns(
                        built["yield_curve_slope"].alias("yield_2y10y")
                    ).drop_nulls()
                elif built is not None and not built.is_empty() and "yield_curve_slope" in built.columns:
                    # Builder dropped timestamps; recover from load_yields panel.
                    import pandas as pd
                    start_ts = pd.Timestamp(start)
                    end_ts = pd.Timestamp(end)
                    yields = MacroYieldFeatureBuilder().load_yields(start_ts, end_ts)
                    if "US10Y" in yields and "US2Y" in yields:
                        idx = yields["US10Y"].index.union(yields["US2Y"].index)
                        slope = (
                            yields["US10Y"].reindex(idx).ffill()
                            - yields["US2Y"].reindex(idx).ffill()
                        ).dropna()
                        daily = self._series_to_frame(slope, "yield_2y10y")
            out = self._align_to_bars(bars, daily, "yield_2y10y")
            if out is None or out.is_empty():
                raise RuntimeError(
                    "MacroMaterializer: yield_2y10y unavailable from cross-asset/FRED panel"
                )
            return out

        if spec.name == "usd_index":
            if "DXY" not in panel or panel["DXY"] is None:
                raise RuntimeError(
                    "MacroMaterializer: usd_index requires DXY from cross-asset panel "
                    "(stooq/yahoo). No FX-close alias."
                )
            daily = self._series_to_frame(panel["DXY"], "usd_index")
            out = self._align_to_bars(bars, daily, "usd_index")
            if out is None or out.is_empty():
                raise RuntimeError("MacroMaterializer: usd_index panel empty after align")
            return out

        if spec.name == "vix_close":
            if "VIX" not in panel or panel["VIX"] is None:
                raise RuntimeError(
                    "MacroMaterializer: vix_close requires VIX from cross-asset panel. "
                    "No FX-close alias."
                )
            daily = self._series_to_frame(panel["VIX"], "vix_close")
            out = self._align_to_bars(bars, daily, "vix_close")
            if out is None or out.is_empty():
                raise RuntimeError("MacroMaterializer: vix_close panel empty after align")
            return out

        return None

    @property
    def requires_ohlcv(self) -> bool:
        # Macro series can materialize from the external panel alone; bars are
        # optional alignment targets when provided.
        return False


class DefaultMaterializer(BaseMaterializer):
    """Fallback materializer for features with no specialized implementation."""

    def compute(
        self, spec: FeatureSpec, bars: pl.DataFrame, start: datetime, end: datetime
    ) -> pl.DataFrame | None:
        return None


# ════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS (Polars expressions for complex computations)
# ════════════════════════════════════════════════════════════════════════════

def _hurst_exponent(log_close_expr: pl.Expr, max_lag: int = 120) -> pl.Expr:
    """
    Compute Hurst exponent via R/S analysis using Polars expressions
    over windows. This is a simplified rolling version.
    """
    mu = log_close_expr.rolling_mean(window_size=max_lag)
    diff = log_close_expr - mu
    z = diff.cum_sum()
    r = z.rolling_max(window_size=max_lag) - z.rolling_min(window_size=max_lag)
    s = diff.rolling_std(window_size=max_lag)
    rs = r / (s + 1e-12)
    # H ≈ ln(RS) / ln(n/2)
    return rs.log() / pl.lit(max_lag / 2.0).log()


def _fractal_dimension(return_expr: pl.Expr, window: int = 30) -> pl.Expr:
    """Estimate fractal dimension from returns using Higuchi's method approximation."""
    abs_diff = return_expr.abs().rolling_sum(window_size=window)
    total_range = return_expr.cum_sum().rolling_max(window_size=window) - \
                  return_expr.cum_sum().rolling_min(window_size=window)
    return (abs_diff.log() / (total_range + 1e-12).log()).alias("fractal_dim")


# ════════════════════════════════════════════════════════════════════════════
# MATERIALIZER DISPATCH
# ════════════════════════════════════════════════════════════════════════════

MATERIALIZER_MAP: dict[str, BaseMaterializer] = {}


def get_materializer(spec: FeatureSpec, store: FeatureStore) -> BaseMaterializer:
    """Get appropriate materializer for a feature spec."""
    name = spec.name
    if "session" in name or name in (
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "session_asia", "is_monday", "is_friday",
    ):
        return SessionMaterializer(store)
    elif spec.source == FeatureSource.MACRO or name in (
        "yield_2y10y", "usd_index", "vix_close",
    ):
        return MacroMaterializer(store)
    elif name.startswith(("atr_", "rolling_vol", "bollinger")):
        return VolatilityMaterializer(store)
    elif name in (
        "ofi_20", "obi_proxy", "hurst_120", "fractal_dim_30",
        "iv_proxy_20", "skew_proxy_20", "trade_arrival_rate_30",
    ):
        return MicrostructureMaterializer(store)
    elif name in ("close", "log_ret_1", "log_ret_5", "log_ret_20", "hlr", "spread_bps"):
        return PriceMaterializer(store)
    else:
        return DefaultMaterializer(store)


def materialize_feature(
    store: FeatureStore,
    feature_name: str,
    bars: pl.DataFrame,
    start: datetime,
    end: datetime,
    spec: FeatureSpec = None,
) -> pl.DataFrame | None:
    """Compute and store a single feature given raw bars."""
    if spec is None:
        spec = store.get_feature(feature_name)
        if spec is None:
            raise ValueError(f"Feature '{feature_name}' not registered")

    materializer = get_materializer(spec, store)
    result = materializer.compute(spec, bars, start, end)

    if result is not None and not result.is_empty():
        store._store_materialization(
            feature_name, result, start, end, MaterializationStrategy.EAGER_BATCH
        )

    return result


# ════════════════════════════════════════════════════════════════════════════
# BULK MATERIALIZATION
# ════════════════════════════════════════════════════════════════════════════

def materialize_feature_set(
    store: FeatureStore,
    feature_names: list[str],
    bars: pl.DataFrame,
    start: datetime,
    end: datetime,
) -> dict[str, pl.DataFrame]:
    """Materialize a set of features (with dependency resolution)."""
    names = store.registry.resolve_dependencies(feature_names)
    results = {}

    for name in names:
        if name in results:
            continue
        result = materialize_feature(store, name, bars, start, end)
        if result is not None:
            results[name] = result

    return {k: results[k] for k in feature_names if k in results}


# ════════════════════════════════════════════════════════════════════════════
# BATCH PIPELINE (full daily/weekly materialization)
# ════════════════════════════════════════════════════════════════════════════

def run_full_materialization(
    store: FeatureStore,
    bars: pl.DataFrame,
    start: datetime,
    end: datetime,
    feature_names: list[str] | None = None,
) -> dict[str, pl.DataFrame]:
    """
    Full featured pipeline: compute all (or specified) features for the given bar range
    and store them.
    """
    if feature_names is None:
        # All non-deprecated features
        from data.feature_definitions import list_features as get_builtin
        feature_names = [s.name for s in get_builtin() if not s.deprecated]

    return materialize_feature_set(store, feature_names, bars, start, end)


if __name__ == "__main__":
    print("Feature materializers loaded")
    print(f"Materializer map: {list(MATERIALIZER_MAP.keys())}")
