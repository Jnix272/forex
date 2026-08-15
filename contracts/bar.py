"""
Bar Data Contract
=================
Contract for OHLCV bar data validation.
"""

from contracts.base import DataContract, Stage, ContractVersion, ContractMetadata
import polars as pl
from datetime import datetime


class BarContract(DataContract):
    contract_name = "bar"
    contract_version = ContractVersion.V1_1
    stage = Stage.RESAMPLING
    
    required_columns = {
        "timestamp_utc": pl.Datetime("ns", "UTC"),
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
    }
    
    optional_columns = {
        "spread_avg": pl.Float64,
        "bid_close": pl.Float64,
        "ask_close": pl.Float64,
        "spread_pips": pl.Float64,
        "pair": pl.String,
        "session_label": pl.String,
        "vwap": pl.Float64,
        "trade_count": pl.Int64,
    }
    
    column_constraints = {
        "open": {"not_null": True, "min": 0.0},
        "high": {"not_null": True, "min": 0.0},
        "low": {"not_null": True, "min": 0.0},
        "close": {"not_null": True, "min": 0.0},
        "volume": {"min": 0.0},
        "spread_avg": {"min": 0.0},
        "spread_pips": {"min": 0.0},
        "timestamp_utc": {"not_null": True},
        "vwap": {"min": 0.0},
        "trade_count": {"min": 0},
        "session_label": {"allowed": ["asia", "london", "ny", "asia_london", "london_ny", "off"]},
    }
    
    invariants = [
        "high >= low",
        "high >= open",
        "high >= close",
        "low <= open",
        "low <= close",
        "timestamp_utc IS NOT NULL",
        # Spread consistency
        "spread_avg IS NULL OR spread_avg >= 0",
    ]
    
    pair_overrides = {
        "USDJPY": {"open": {"min": 50.0}, "high": {"min": 50.0}, "low": {"min": 50.0}, "close": {"min": 50.0}},
        "GBPJPY": {"open": {"min": 100.0}, "high": {"min": 100.0}, "low": {"min": 100.0}, "close": {"min": 100.0}},
        "EURJPY": {"open": {"min": 100.0}, "high": {"min": 100.0}, "low": {"min": 100.0}, "close": {"min": 100.0}},
    }
    
    allow_unknown_columns = True
    
    @classmethod
    def validate_frame(cls, df: pl.DataFrame, pair: str | None = None) -> tuple[pl.DataFrame, ContractMetadata]:
        errors = []
        errors.extend(cls._check_required_columns(df))
        errors.extend(cls._check_unknown_columns(df))
        errors.extend(cls._check_constraints(df, pair))
        errors.extend(cls._check_invariants(df))
        
        if errors:
            raise ValueError(f"BarContract validation failed for {pair}: {'; '.join(errors)}")
        
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
        )
        
        return df, metadata


# Register the contract
from contracts.base import ContractRegistry
ContractRegistry.register(BarContract)