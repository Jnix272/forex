"""
Feature Data Contract
=====================
Contract for engineered feature data validation.
"""

from typing import ClassVar

import polars as pl

from config.feature_mask import FEATURE_MASK
from contracts.base import ContractMetadata, ContractVersion, DataContract, Stage


class FeatureContract(DataContract):
    contract_name = "feature"
    contract_version = ContractVersion.V1_1
    stage = Stage.FEATURE_ENGINEERING

    # Core required columns (always present)
    required_columns = {
        "timestamp_utc": pl.Datetime("ns", "UTC"),
    }

    # Market data columns (should be present from bars)
    market_columns: ClassVar[dict[str, pl.DataType]] = {
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
        "spread_avg": pl.Float64,
        "bid_close": pl.Float64,
        "ask_close": pl.Float64,
        "pair": pl.String,
    }

    # Feature columns from FEATURE_MASK (enabled features)
    @classmethod
    def _get_feature_columns(cls) -> dict[str, pl.DataType]:
        """Get feature columns from FEATURE_MASK"""
        try:
            from config.feature_mask import FEATURE_MASK, enabled_feature_names
            enabled = enabled_feature_names(FEATURE_MASK)
            # All features are float32 in the training matrix
            return dict.fromkeys(enabled, pl.Float32)
        except Exception:
            # Fallback if FEATURE_MASK not available
            return {}

    # Combined optional columns
    optional_columns = {
        **market_columns,
        # FinBERT embeddings (fb_*)
        **{f"fb_{i}": pl.Float32 for i in range(32)},
        # Regime features
        "regime_class": pl.Int32,
        "regime_label": pl.Float32,
        "vol_regime_state_0_prob": pl.Float32,
        "vol_regime_state_1_prob": pl.Float32,
        "vol_regime_state_2_prob": pl.Float32,
        # Session features
        "session_label": pl.String,
        "asia_london": pl.Float32,
        "london_ny": pl.Float32,
        # Time features
        "time_sin": pl.Float32,
        "time_cos": pl.Float32,
        "day_sin": pl.Float32,
        "day_cos": pl.Float32,
        # No-trade zones
        "no_trade_score": pl.Float32,
        # Quality flags
        "news_ok": pl.Float32,
        "pre_news": pl.Float32,
        "post_news": pl.Float32,
        "vol_ok": pl.Float32,
        "eco_surprise": pl.Float32,
        "eco_revision": pl.Float32,
        "buzz": pl.Float32,
        # Circuit breakers
        "drawdown": pl.Float32,
        "drawdown_breach": pl.Int32,
        "daily_loss_breach": pl.Int32,
        "var_95_breach": pl.Int32,
        "var_99_breach": pl.Int32,
        "max_position_lots": pl.Float32,
        "risk_per_lot": pl.Float32,
    }

    column_constraints = {
        "timestamp_utc": {"not_null": True},
        "close": {"not_null": True, "min": 0.0},
        "spread_avg": {"min": 0.0},
        "spread_pips": {"min": 0.0},
        "volume": {"min": 0.0},
        # Regime probabilities should sum to ~1
        "vol_regime_state_0_prob": {"min": 0.0, "max": 1.0},
        "vol_regime_state_1_prob": {"min": 0.0, "max": 1.0},
        "vol_regime_state_2_prob": {"min": 0.0, "max": 1.0},
        # Time features bounded
        "time_sin": {"min": -1.0, "max": 1.0},
        "time_cos": {"min": -1.0, "max": 1.0},
        "day_sin": {"min": -1.0, "max": 1.0},
        "day_cos": {"min": -1.0, "max": 1.0},
        # No-trade score
        "no_trade_score": {"min": 0.0, "max": 1.0},
        # Binary flags
        "news_ok": {"allowed": [0.0, 1.0]},
        "pre_news": {"allowed": [0.0, 1.0]},
        "post_news": {"allowed": [0.0, 1.0]},
        "vol_ok": {"allowed": [0.0, 1.0]},
        "drawdown_breach": {"allowed": [0, 1]},
        "daily_loss_breach": {"allowed": [0, 1]},
        "var_95_breach": {"allowed": [0, 1]},
        "var_99_breach": {"allowed": [0, 1]},
        # Position limits
        "max_position_lots": {"min": 0.0},
        "risk_per_lot": {"min": 0.0},
    }

    invariants = [
        "timestamp_utc IS NOT NULL",
        # Regime probabilities - at least one not null
        "vol_regime_state_0_prob IS NOT NULL OR vol_regime_state_1_prob IS NOT NULL OR vol_regime_state_2_prob IS NOT NULL",
    ]

    # Features that must not have lookahead bias
    no_lookahead_features: ClassVar[set[str]] = {
        "ret_5",
        "ret_20",
        "ret_60",
        "rsi_14",
        "macd",
        "macd_sig",
        "macd_hist",
        "bb_pct",
        "bb_width",
        "atr_6",
        "atr_20",
        "atr_60",
        "vol_6",
        "vol_20",
        "vol_60",
        "ofi",
        "obi_proxy",
        "vpin",
        "stoch_k",
        "stoch_d",
        "williams_r",
        "cci",
    }

    allow_unknown_columns = True  # Allow dynamic feature columns

    @classmethod
    def validate_frame(cls, df: pl.DataFrame, pair: str | None = None) -> tuple[pl.DataFrame, ContractMetadata]:
        errors = []
        warnings = []

        errors.extend(cls._check_required_columns(df))
        errors.extend(cls._check_constraints(df, pair))
        errors.extend(cls._check_invariants(df))

        # Check for NaN/inf in numeric columns
        numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
        for col in numeric_cols:
            null_count = df[col].null_count()
            if null_count > 0:
                warnings.append(f"Feature column '{col}' has {null_count} null values")

            # Check for inf
            inf_count = df.select(pl.col(col).is_infinite().sum()).item()
            if inf_count > 0:
                errors.append(f"Feature column '{col}' has {inf_count} infinite values")

        # Check FEATURE_MASK compliance if available
        try:
            enabled_features = {k for k, v in FEATURE_MASK.items() if v}
            built_features = set(df.columns)
            missing_enabled = enabled_features - built_features
            if missing_enabled:
                warnings.append(f"FEATURE_MASK enabled features missing from build: {sorted(missing_enabled)[:10]}")
        except Exception:
            pass  # FEATURE_MASK not available

        if errors:
            raise ValueError(f"FeatureContract validation failed for {pair}: {'; '.join(errors)}")

        metadata = ContractMetadata(
            contract_version=str(cls.contract_version),
            stage=cls.stage,
            pair=pair,
            timestamp_start=df["timestamp_utc"].min() if len(df) > 0 else None,
            timestamp_end=df["timestamp_utc"].max() if len(df) > 0 else None,
            n_rows=len(df),
            n_cols=len(df.columns),
            schema_hash=cls._compute_schema_hash(df),
            data_hash=cls._compute_data_hash(df),
            warnings=warnings,
        )

        return df, metadata


# Register the contract
from contracts.base import ContractRegistry  # noqa: E402

ContractRegistry.register(FeatureContract)
