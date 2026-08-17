"""
Label Data Contract
===================
Contract for label data validation.
"""

from typing import ClassVar

import polars as pl

from contracts.base import ContractMetadata, ContractVersion, DataContract, Stage


class LabelContract(DataContract):
    contract_name = "label"
    contract_version = ContractVersion.V1_1
    stage = Stage.LABELING

    required_columns: ClassVar[dict[str, pl.DataType]] = {
        "timestamp_utc": pl.Datetime("ns", "UTC"),
    }

    # RL reward labels
    rl_columns: ClassVar[dict[str, pl.DataType]] = {
        "reward": pl.Float32,
        "label": pl.Int32,  # -1, 0, 1 for direction
    }

    # Triple-barrier labels
    tb_columns: ClassVar[dict[str, pl.DataType]] = {
        "tb_label": pl.Int32,  # -1, 0, 1
        "tb_return": pl.Float32,
        "tb_hit_time": pl.Int32,  # bars to hit barrier
        "tb_barrier_hit": pl.Int32,  # 0=none, 1=tp, 2=sl, 3=time
    }

    # Difficulty scores
    difficulty_columns: ClassVar[dict[str, pl.DataType]] = {
        "difficulty": pl.Int32,  # 0=easy, 1=medium, 2=hard
    }

    # Path quality
    path_quality_columns: ClassVar[dict[str, pl.DataType]] = {
        "path_quality": pl.Float32,
        "max_adverse_excursion": pl.Float32,
        "max_favorable_excursion": pl.Float32,
    }

    optional_columns: ClassVar[dict[str, pl.DataType]] = {
        **rl_columns,
        **tb_columns,
        **difficulty_columns,
        **path_quality_columns,
        "pair": pl.String,
    }

    column_constraints = {
        "timestamp_utc": {"not_null": True},
        "reward": {"not_null": True},
        "label": {"allowed": [-1, 0, 1]},
        "tb_label": {"allowed": [-1, 0, 1]},
        "tb_hit_time": {"min": 0},
        "tb_barrier_hit": {"allowed": [0, 1, 2, 3]},
        "difficulty": {"allowed": [0, 1, 2]},
        "path_quality": {"min": 0.0, "max": 1.0},
        "max_adverse_excursion": {"min": 0.0},
        "max_favorable_excursion": {"min": 0.0},
    }

    invariants = [
        "timestamp_utc IS NOT NULL",
        # Labels should not be all the same class (collapsed) - checked separately
    ]

    allow_unknown_columns = True

    @classmethod
    def validate_frame(cls, df: pl.DataFrame, pair: str | None = None) -> tuple[pl.DataFrame, ContractMetadata]:
        errors = []
        warnings = []

        errors.extend(cls._check_required_columns(df))
        errors.extend(cls._check_constraints(df, pair))
        errors.extend(cls._check_invariants(df))

        # Check label distribution
        if "label" in df.columns:
            label_counts = df["label"].value_counts()
            total = len(df)
            for row in label_counts.iter_rows():
                label, count = row
                pct = count / total * 100
                if pct < 1.0:
                    warnings.append(f"Label class {label} has very low frequency: {pct:.2f}%")
                if pct > 98.0:
                    errors.append(f"Label class {label} dominates: {pct:.2f}% (possible collapse)")

        # Check difficulty distribution
        if "difficulty" in df.columns:
            diff_counts = df["difficulty"].value_counts()
            for row in diff_counts.iter_rows():
                diff, count = row
                pct = count / total * 100
                if pct < 5.0:
                    warnings.append(f"Difficulty {diff} has low frequency: {pct:.2f}%")

        # Check for NaN/inf in reward
        if "reward" in df.columns:
            inf_count = df.select(pl.col("reward").is_infinite().sum()).item()
            null_count = df["reward"].null_count()
            if inf_count > 0:
                errors.append(f"Reward column has {inf_count} infinite values")
            if null_count > 0:
                errors.append(f"Reward column has {null_count} null values")

        if errors:
            raise ValueError(f"LabelContract validation failed for {pair}: {'; '.join(errors)}")

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

ContractRegistry.register(LabelContract)
