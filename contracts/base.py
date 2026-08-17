"""
Base Contract Classes
=====================
Core contract infrastructure for pipeline data validation.
"""

from __future__ import annotations

import hashlib
import json
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, ClassVar, TypeVar

import polars as pl
from pydantic import BaseModel, ConfigDict

T = TypeVar("T", bound="DataContract")


class ContractVersion(StrEnum):
    """Semantic versioning for contracts"""

    V1_0 = "1.0.0"
    V1_1 = "1.1.0"
    V2_0 = "2.0.0"
    CURRENT = V1_1


class Stage(StrEnum):
    """Pipeline stages for contract validation"""

    INGESTION = "ingestion"
    RESAMPLING = "resampling"
    FEATURE_ENGINEERING = "feature_engineering"
    LABELING = "labeling"
    DATASET_BUILD = "dataset_build"
    TRAINING = "training"


@dataclass(frozen=True)
class ContractMetadata:
    """Metadata attached to every validated DataFrame"""

    contract_version: str
    stage: Stage
    pair: str | None
    timestamp_start: datetime | None
    timestamp_end: datetime | None
    n_rows: int
    n_cols: int
    schema_hash: str
    data_hash: str
    validated_at: datetime = field(default_factory=datetime.now)
    validator_version: str = "1.0.0"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "stage": self.stage.value,
            "pair": self.pair,
            "timestamp_start": self.timestamp_start.isoformat() if self.timestamp_start else None,
            "timestamp_end": self.timestamp_end.isoformat() if self.timestamp_end else None,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "schema_hash": self.schema_hash,
            "data_hash": self.data_hash,
            "validated_at": self.validated_at.isoformat(),
            "validator_version": self.validator_version,
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContractMetadata:
        return cls(
            contract_version=data["contract_version"],
            stage=Stage(data["stage"]),
            pair=data.get("pair"),
            timestamp_start=datetime.fromisoformat(data["timestamp_start"]) if data.get("timestamp_start") else None,
            timestamp_end=datetime.fromisoformat(data["timestamp_end"]) if data.get("timestamp_end") else None,
            n_rows=data["n_rows"],
            n_cols=data["n_cols"],
            schema_hash=data["schema_hash"],
            data_hash=data["data_hash"],
            validated_at=datetime.fromisoformat(data["validated_at"]),
            validator_version=data.get("validator_version", "1.0.0"),
            warnings=data.get("warnings", []),
        )


class DataContract(ABC, BaseModel):
    """Base contract for all pipeline data stages"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
    )

    # Contract identity
    contract_name: ClassVar[str] = "base"
    contract_version: ClassVar[ContractVersion] = ContractVersion.CURRENT
    stage: ClassVar[Stage] = Stage.INGESTION

    # Required columns with Polars dtype specifications
    required_columns: ClassVar[dict[str, pl.DataType]] = {}
    optional_columns: ClassVar[dict[str, pl.DataType]] = {}

    # Column-level constraints
    column_constraints: ClassVar[dict[str, dict[str, Any]]] = {}

    # Cross-column invariants (Polars expressions as strings)
    invariants: ClassVar[list[str]] = []

    # Pair-specific overrides
    pair_overrides: ClassVar[dict[str, dict]] = {}

    # Allow unknown columns (for extensibility)
    allow_unknown_columns: ClassVar[bool] = True

    @classmethod
    @abstractmethod
    def validate_frame(
        cls: type[T], df: pl.DataFrame, pair: str | None = None
    ) -> tuple[pl.DataFrame, ContractMetadata]:
        """Validate and return cleaned DataFrame + metadata"""
        pass

    @classmethod
    def _compute_schema_hash(cls, df: pl.DataFrame) -> str:
        """Hash of column names + dtypes"""
        schema_str = json.dumps([(c, str(df.schema[c])) for c in sorted(df.columns)], sort_keys=True)
        return hashlib.sha256(schema_str.encode()).hexdigest()[:16]

    @classmethod
    def _compute_data_hash(cls, df: pl.DataFrame, sample_size: int = 1000) -> str:
        """Hash of data sample for lineage"""
        sample = df.head(min(sample_size, len(df)))
        data_str = sample.write_csv()
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]

    @classmethod
    def _check_required_columns(cls, df: pl.DataFrame) -> list[str]:
        """Return list of missing required columns"""
        errors = []
        for col, dtype in cls.required_columns.items():
            if col not in df.columns:
                errors.append(f"Missing required column: {col} ({dtype})")
            elif df.schema[col] != dtype:
                errors.append(f"Type mismatch for {col}: expected {dtype}, got {df.schema[col]}")
        return errors

    @classmethod
    def _check_unknown_columns(cls, df: pl.DataFrame) -> list[str]:
        """Check for columns not in contract"""
        if cls.allow_unknown_columns:
            return []
        known = set(cls.required_columns.keys()) | set(cls.optional_columns.keys())
        unknown = [c for c in df.columns if c not in known]
        if unknown:
            return [f"Unknown columns not in contract: {unknown}"]
        return []

    @classmethod
    def _check_constraints(cls, df: pl.DataFrame, pair: str | None) -> list[str]:
        """Check column-level constraints"""
        errors = []
        overrides = cls.pair_overrides.get(pair or "", {})

        for col, constraints in cls.column_constraints.items():
            if col not in df.columns:
                continue

            series = df[col]

            # Null check
            if constraints.get("not_null", False) and series.null_count() > 0:
                errors.append(f"{col}: null values not allowed (found {series.null_count()})")

            # Range checks
            if "min" in constraints:
                min_val = overrides.get(col, {}).get("min", constraints["min"])
                if series.min() is not None and series.min() < min_val:
                    errors.append(f"{col}: min value {series.min()} < {min_val}")

            if "max" in constraints:
                max_val = overrides.get(col, {}).get("max", constraints["max"])
                if series.max() is not None and series.max() > max_val:
                    errors.append(f"{col}: max value {series.max()} > {max_val}")

            # Allowed values
            if "allowed" in constraints:
                allowed = set(constraints["allowed"])
                invalid = set(series.unique().to_list()) - allowed
                if invalid:
                    errors.append(f"{col}: invalid values {invalid}")

            # Custom validator
            if "validator" in constraints:
                validator_fn = constraints["validator"]
                try:
                    result = validator_fn(series)
                    if result is not True:
                        errors.append(f"{col}: custom validator failed: {result}")
                except Exception as e:
                    errors.append(f"{col}: custom validator error: {e}")

        return errors

    @classmethod
    def _check_invariants(cls, df: pl.DataFrame) -> list[str]:
        """Evaluate cross-column invariants as Polars expressions"""
        errors = []
        for invariant in cls.invariants:
            try:
                # Use pl.sql_expr to parse and evaluate the expression
                expr = pl.sql_expr(invariant)
                result = df.select(expr)
                # Check if all values are True
                if result.height > 0:
                    col_name = result.columns[0]
                    if not result[col_name].all():
                        errors.append(f"Invariant violated: {invariant}")
            except Exception as e:
                errors.append(f"Invariant evaluation failed: {invariant} ({e})")
        return errors

    @classmethod
    def validate(
        cls: type[T], df: pl.DataFrame, pair: str | None = None, strict: bool = True
    ) -> tuple[pl.DataFrame, ContractMetadata]:
        """
        Main validation entry point.

        Args:
            df: DataFrame to validate
            pair: Currency pair for pair-specific overrides
            strict: If True, raise on validation errors; if False, return warnings

        Returns:
            Tuple of (validated DataFrame, metadata)
        """
        all_errors = []
        all_warnings = []

        all_errors.extend(cls._check_required_columns(df))
        all_errors.extend(cls._check_unknown_columns(df))
        all_errors.extend(cls._check_constraints(df, pair))
        all_errors.extend(cls._check_invariants(df))

        if all_errors:
            error_msg = f"{cls.contract_name} validation failed"
            if pair:
                error_msg += f" for {pair}"
            error_msg += f": {'; '.join(all_errors)}"
            if strict:
                raise ValueError(error_msg)
            else:
                all_warnings.extend(all_errors)

        metadata = ContractMetadata(
            contract_version=str(cls.contract_version),
            stage=cls.stage,
            pair=pair,
            timestamp_start=df["timestamp_utc"].min() if len(df) > 0 and "timestamp_utc" in df.columns else None,
            timestamp_end=df["timestamp_utc"].max() if len(df) > 0 and "timestamp_utc" in df.columns else None,
            n_rows=len(df),
            n_cols=len(df.columns),
            schema_hash=cls._compute_schema_hash(df),
            data_hash=cls._compute_data_hash(df),
            warnings=all_warnings,
        )

        return df, metadata


class ContractRegistry:
    """Registry for all data contracts with version management"""

    _contracts: dict[str, type[DataContract]] = {}
    _lock = threading.Lock()

    @classmethod
    def register(cls, contract_class: type[DataContract]) -> type[DataContract]:
        """Register a contract class"""
        with cls._lock:
            key = f"{contract_class.contract_name}:{contract_class.contract_version}"
            cls._contracts[key] = contract_class
        return contract_class

    @classmethod
    def get(cls, name: str, version: ContractVersion | str = ContractVersion.CURRENT) -> type[DataContract] | None:
        """Get a registered contract by name and version"""
        version_str = str(version) if isinstance(version, ContractVersion) else version
        key = f"{name}:{version_str}"
        return cls._contracts.get(key)

    @classmethod
    def get_latest(cls, name: str) -> type[DataContract] | None:
        """Get the latest version of a contract by name"""
        with cls._lock:
            matching = [c for k, c in cls._contracts.items() if k.startswith(f"{name}:")]
            if not matching:
                return None
            # Sort by version
            return max(matching, key=lambda c: c.contract_version.value)

    @classmethod
    def list_contracts(cls) -> list[dict[str, str]]:
        """List all registered contracts"""
        with cls._lock:
            return [
                {"name": c.contract_name, "version": str(c.contract_version), "stage": c.stage.value}
                for c in cls._contracts.values()
            ]

    @classmethod
    def validate_stage(
        cls, stage: Stage, df: pl.DataFrame, pair: str | None = None
    ) -> tuple[pl.DataFrame, ContractMetadata]:
        """Validate using the contract for a specific stage"""
        contract_map = {
            Stage.INGESTION: "tick",
            Stage.RESAMPLING: "bar",
            Stage.FEATURE_ENGINEERING: "feature",
            Stage.LABELING: "label",
            Stage.DATASET_BUILD: "dataset",
        }
        contract_name = contract_map.get(stage)
        if not contract_name:
            raise ValueError(f"No contract registered for stage: {stage}")

        contract_class = cls.get_latest(contract_name)
        if not contract_class:
            raise ValueError(f"No contract found for {contract_name}")

        return contract_class.validate(df, pair=pair)


# Auto-register contracts when imported
def _auto_register():
    """Auto-register all contract classes in this module"""
    # Contracts will be registered when their modules are imported
    pass
