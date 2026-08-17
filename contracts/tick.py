"""
Tick Data Contract
==================
Contract for raw tick data validation.
"""

import polars as pl

from contracts.base import ContractMetadata, ContractVersion, DataContract, Stage


class TickContract(DataContract):
    contract_name = "tick"
    contract_version = ContractVersion.V1_1
    stage = Stage.INGESTION

    required_columns = {
        "timestamp_utc": pl.Datetime("ns", "UTC"),
        "bid": pl.Float64,
        "ask": pl.Float64,
    }

    optional_columns = {
        "bid_size": pl.Float64,
        "ask_size": pl.Float64,
        "trade_price": pl.Float64,
        "trade_size": pl.Float64,
        "trade_side": pl.String,
        "exchange": pl.String,
        "volume": pl.Float64,
        "spread": pl.Float64,
        "mid": pl.Float64,
        "pair": pl.String,
    }

    column_constraints = {
        "bid": {"not_null": True, "min": 0.0},
        "ask": {"not_null": True, "min": 0.0},
        "spread": {"min": 0.0},
        "volume": {"min": 0.0},
        "bid_size": {"min": 0.0},
        "ask_size": {"min": 0.0},
        "trade_size": {"min": 0.0},
        "timestamp_utc": {"not_null": True},
        "trade_side": {"allowed": ["buy", "sell", "unknown", "bid", "ask"]},
    }

    invariants = [
        "ask > bid",
        "timestamp_utc IS NOT NULL",
    ]

    pair_overrides = {
        "USDJPY": {"bid": {"min": 50.0}, "ask": {"min": 50.0}},
        "GBPJPY": {"bid": {"min": 100.0}, "ask": {"min": 100.0}},
        "EURJPY": {"bid": {"min": 100.0}, "ask": {"min": 100.0}},
        "XAUUSD": {"bid": {"min": 1000.0}, "ask": {"min": 1000.0}},
        "BTCUSD": {"bid": {"min": 10000.0}, "ask": {"min": 10000.0}},
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
            raise ValueError(f"TickContract validation failed for {pair}: {'; '.join(errors)}")

        # Ensure derived columns exist
        if "mid" not in df.columns:
            df = df.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"))
        if "spread" not in df.columns:
            df = df.with_columns((pl.col("ask") - pl.col("bid")).alias("spread"))
        if "pair" not in df.columns and pair:
            df = df.with_columns(pl.lit(pair).alias("pair"))

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
from contracts.base import ContractRegistry  # noqa: E402

ContractRegistry.register(TickContract)
