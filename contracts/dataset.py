"""
Dataset Contract
================
Contract for final training dataset validation.
"""

from contracts.base import DataContract, Stage, ContractVersion, ContractMetadata
import polars as pl
from datetime import datetime
import numpy as np
from typing import ClassVar, Any


class DatasetContract(DataContract):
    contract_name = "dataset"
    contract_version = ContractVersion.V1_1
    stage = Stage.DATASET_BUILD
    
    # For Zarr/NPY datasets, we validate the metadata and sample
    required_columns: ClassVar[dict[str, pl.DataType]] = {
        # These are validated at the metadata level
    }
    
    optional_columns: ClassVar[dict[str, pl.DataType]] = {}
    
    column_constraints: ClassVar[dict[str, dict[str, Any]]] = {}
    
    invariants: ClassVar[list[str]] = []
    
    # Dataset-level constraints
    dataset_constraints: ClassVar[dict[str, Any]] = {
        "min_samples": 1000,
        "max_nan_pct": 0.01,  # 1% max NaN per feature
        "max_inf_pct": 0.0,
        "max_constant_features": 0,
        "min_feature_variance": 1e-10,
        "max_correlation": 0.999,  # Max pairwise correlation
        "sequence_length_consistency": True,
        "label_alignment": True,
    }
    
    allow_unknown_columns = True
    
    @classmethod
    def validate_frame(cls, df: pl.DataFrame, pair: str | None = None) -> tuple[pl.DataFrame, ContractMetadata]:
        """Validate a DataFrame representation of the dataset"""
        errors = []
        warnings = []
        
        errors.extend(cls._check_required_columns(df))
        errors.extend(cls._check_constraints(df, pair))
        errors.extend(cls._check_invariants(df))
        
        # Dataset-level checks
        if len(df) < cls.dataset_constraints["min_samples"]:
            errors.append(f"Dataset too small: {len(df)} < {cls.dataset_constraints['min_samples']}")
        
        # Check NaN percentage per column
        for col in df.columns:
            if df.schema[col].is_numeric():
                null_pct = df[col].null_count() / len(df)
                if null_pct > cls.dataset_constraints["max_nan_pct"]:
                    errors.append(f"Column '{col}' has {null_pct:.2%} NaN (max {cls.dataset_constraints['max_nan_pct']:.2%})")
                
                inf_count = df.select(pl.col(col).is_infinite().sum()).item()
                inf_pct = inf_count / len(df)
                if inf_pct > cls.dataset_constraints["max_inf_pct"]:
                    errors.append(f"Column '{col}' has {inf_pct:.2%} infinite values")
                
                # Check variance
                std = df[col].std()
                if std is not None and std < cls.dataset_constraints["min_feature_variance"]:
                    warnings.append(f"Column '{col}' has near-zero variance: {std}")
        
        # Check for constant columns
        constant_cols = [c for c in df.columns if df[c].n_unique() == 1]
        if len(constant_cols) > cls.dataset_constraints["max_constant_features"]:
            errors.append(f"Too many constant columns: {constant_cols}")
        
        # Check pairwise correlations (sample for speed)
        numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
        if len(numeric_cols) > 1:
            sample_df = df.select(numeric_cols).head(10000).to_pandas()
            corr_matrix = sample_df.corr().abs()
            # Find high correlations (excluding diagonal)
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > cls.dataset_constraints["max_correlation"]:
                        high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
            if high_corr:
                warnings.append(f"High correlation pairs found: {high_corr[:5]}")
        
        if errors:
            raise ValueError(f"DatasetContract validation failed: {'; '.join(errors)}")
        
        metadata = ContractMetadata(
            contract_version=str(cls.contract_version),
            stage=cls.stage,
            pair=pair,
            timestamp_start=None,
            timestamp_end=None,
            n_rows=len(df),
            n_cols=len(df.columns),
            schema_hash=cls._compute_schema_hash(df),
            data_hash=cls._compute_data_hash(df),
            warnings=warnings,
        )
        
        return df, metadata
    
    @classmethod
    def validate_zarr(cls, zarr_path: str, pair: str | None = None) -> tuple[dict, ContractMetadata]:
        """Validate a Zarr dataset on disk"""
        import zarr
        import numpy as np
        
        errors = []
        warnings = []
        
        try:
            root = zarr.open_group(zarr_path, mode='r')
        except Exception as e:
            raise ValueError(f"Failed to open Zarr group at {zarr_path}: {e}")
        
        # Check required arrays
        required_arrays = ["X", "y", "y_cls"]
        for arr_name in required_arrays:
            if arr_name not in root:
                errors.append(f"Missing required array: {arr_name}")
        
        if errors:
            raise ValueError(f"DatasetContract validation failed: {'; '.join(errors)}")
        
        X = root["X"]
        y = root["y"]
        y_cls = root["y_cls"]
        
        # Check shapes
        if X.shape[0] != y.shape[0]:
            errors.append(f"X and y shape mismatch: {X.shape[0]} vs {y.shape[0]}")
        if X.shape[0] != y_cls.shape[0]:
            errors.append(f"X and y_cls shape mismatch: {X.shape[0]} vs {y_cls.shape[0]}")
        
        n_samples = X.shape[0]
        n_features = X.shape[1] if X.ndim > 1 else 1
        seq_len = X.shape[2] if X.ndim > 2 else 1
        
        if n_samples < cls.dataset_constraints["min_samples"]:
            errors.append(f"Dataset too small: {n_samples} < {cls.dataset_constraints['min_samples']}")
        
        # Check for NaN/inf in sample
        sample_size = min(10000, n_samples)
        X_sample = X[:sample_size]
        y_sample = y[:sample_size]
        
        nan_count = np.isnan(X_sample).sum()
        if nan_count > 0:
            nan_pct = nan_count / X_sample.size
            if nan_pct > cls.dataset_constraints["max_nan_pct"]:
                errors.append(f"X has {nan_pct:.2%} NaN values")
            else:
                warnings.append(f"X has {nan_pct:.2%} NaN values")
        
        inf_count = np.isinf(X_sample).sum()
        if inf_count > 0:
            inf_pct = inf_count / X_sample.size
            if inf_pct > cls.dataset_constraints["max_inf_pct"]:
                errors.append(f"X has {inf_pct:.2%} infinite values")
        
        # Check y values
        if "label" in y.dtype.names or y.dtype == np.int32:
            unique_labels = np.unique(y_sample)
            if len(unique_labels) < 2:
                errors.append(f"Labels collapsed to single class: {unique_labels}")
        
        # Check sequence length consistency
        if cls.dataset_constraints["sequence_length_consistency"]:
            if X.ndim == 3:
                # All sequences should have same length
                pass  # Zarr enforces this
        
        if errors:
            raise ValueError(f"DatasetContract validation failed: {'; '.join(errors)}")
        
        # Create metadata
        metadata = ContractMetadata(
            contract_version=str(cls.contract_version),
            stage=cls.stage,
            pair=pair,
            timestamp_start=None,
            timestamp_end=None,
            n_rows=n_samples,
            n_cols=n_features,
            schema_hash=hashlib.sha256(str(X.shape).encode()).hexdigest()[:16],
            data_hash=hashlib.sha256(X_sample.tobytes()).hexdigest()[:16],
            warnings=warnings,
        )
        
        return {
            "n_samples": n_samples,
            "n_features": n_features,
            "seq_len": seq_len,
            "X_shape": X.shape,
            "y_shape": y.shape,
            "y_cls_shape": y_cls.shape,
        }, metadata


# Register the contract
from contracts.base import ContractRegistry
ContractRegistry.register(DatasetContract)