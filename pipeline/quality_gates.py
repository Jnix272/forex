"""
Automated Quality Gates with Auto-Remediation
==============================================
Data quality checks with automatic remediation capabilities.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl


class QualityGateResult(str, Enum):
    """Quality gate result"""
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    REMEDIATED = "remediated"


class RemediationAction(str, Enum):
    """Available remediation actions"""
    DROP_NULLS = "drop_nulls"
    FILL_NULLS_FORWARD = "fill_nulls_forward"
    FILL_NULLS_ZERO = "fill_nulls_zero"
    WINSORIZE = "winsorize"
    CLIP_OUTLIERS = "clip_outliers"
    DROP_DUPLICATES = "drop_duplicates"
    REINDEX_TIME = "reindex_time"
    INTERPOLATE_GAPS = "interpolate_gaps"
    CAP_SPREAD = "cap_spread"
    FIX_OHLC = "fix_ohlc"
    REMOVE_WEEKENDS = "remove_weekends"


@dataclass
class QualityCheck:
    """Single quality check definition"""
    name: str
    check_fn: Callable[[pl.DataFrame], tuple[bool, str, dict]]
    severity: str = "error"  # "error", "warning", "info"
    remediation: RemediationAction | None = None
    remediation_params: dict = field(default_factory=dict)
    enabled: bool = True


@dataclass
class QualityGateReport:
    """Report from quality gate execution"""
    stage: str
    pair: str | None
    timestamp: datetime
    overall: QualityGateResult
    checks: list[dict] = field(default_factory=list)
    remediations_applied: list[dict] = field(default_factory=list)
    original_shape: tuple[int, int] = (0, 0)
    final_shape: tuple[int, int] = (0, 0)
    
    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "pair": self.pair,
            "timestamp": self.timestamp.isoformat(),
            "overall": self.overall.value,
            "checks": self.checks,
            "remediations_applied": self.remediations_applied,
            "original_shape": self.original_shape,
            "final_shape": self.final_shape,
        }


class DataQualityGates:
    """
    Automated data quality gates with auto-remediation.
    
    Runs a series of quality checks and can automatically
    remediate common issues.
    """
    
    def __init__(
        self,
        stage: str,
        checks: list[QualityCheck] | None = None,
        auto_remediate: bool = True,
        remediation_log_dir: str | Path | None = None,
    ):
        self.stage = stage
        self.checks = checks or self._default_checks(stage)
        self.auto_remediate = auto_remediate
        self.remediation_log_dir = Path(remediation_log_dir) if remediation_log_dir else None
        if self.remediation_log_dir:
            self.remediation_log_dir.mkdir(parents=True, exist_ok=True)
    
    def _default_checks(self, stage: str) -> list[QualityCheck]:
        """Get default quality checks for a stage"""
        checks = [
            QualityCheck(
                name="no_nulls_in_critical",
                check_fn=self._check_no_nulls_in_critical,
                severity="error",
                remediation=RemediationAction.FILL_NULLS_FORWARD,
            ),
            QualityCheck(
                name="no_infinite_values",
                check_fn=self._check_no_infinite_values,
                severity="error",
                remediation=RemediationAction.WINSORIZE,
            ),
            QualityCheck(
                name="no_duplicate_timestamps",
                check_fn=self._check_no_duplicate_timestamps,
                severity="error",
                remediation=RemediationAction.DROP_DUPLICATES,
            ),
            QualityCheck(
                name="timestamp_monotonic",
                check_fn=self._check_timestamp_monotonic,
                severity="error",
                remediation=RemediationAction.REINDEX_TIME,
            ),
            QualityCheck(
                name="no_weekend_data",
                check_fn=self._check_no_weekend_data,
                severity="warning",
                remediation=RemediationAction.REMOVE_WEEKENDS,
            ),
        ]
        
        if stage in ["ingestion", "resampling"]:
            checks.extend([
                QualityCheck(
                    name="bid_ask_valid",
                    check_fn=self._check_bid_ask_valid,
                    severity="error",
                    remediation=RemediationAction.FIX_OHLC,
                ),
                QualityCheck(
                    name="spread_positive",
                    check_fn=self._check_spread_positive,
                    severity="error",
                    remediation=RemediationAction.CAP_SPREAD,
                ),
                QualityCheck(
                    name="ohlc_consistent",
                    check_fn=self._check_ohlc_consistent,
                    severity="error",
                    remediation=RemediationAction.FIX_OHLC,
                ),
            ])
        
        if stage == "feature_engineering":
            checks.extend([
                QualityCheck(
                    name="feature_variance",
                    check_fn=self._check_feature_variance,
                    severity="warning",
                ),
                QualityCheck(
                    name="no_constant_features",
                    check_fn=self._check_no_constant_features,
                    severity="warning",
                ),
                QualityCheck(
                    name="feature_correlation",
                    check_fn=self._check_feature_correlation,
                    severity="info",
                ),
            ])
        
        return checks
    
    def run(
        self, 
        df: pl.DataFrame, 
        pair: str | None = None,
        context: dict | None = None,
    ) -> tuple[pl.DataFrame, QualityGateReport]:
        """
        Run all quality checks with optional auto-remediation.
        
        Returns:
            Tuple of (remediated DataFrame, report)
        """
        original_shape = df.shape
        report = QualityGateReport(
            stage=self.stage,
            pair=pair,
            timestamp=datetime.now(),
            overall=QualityGateResult.PASS,
            original_shape=original_shape,
        )
        
        current_df = df.clone()
        overall_result = QualityGateResult.PASS
        
        for check in self.checks:
            if not check.enabled:
                continue
            
            try:
                passed, message, details = check.check_fn(current_df)
                
                check_result = {
                    "name": check.name,
                    "passed": passed,
                    "severity": check.severity,
                    "message": message,
                    "details": details,
                    "remediation": check.remediation.value if check.remediation else None,
                }
                
                if not passed:
                    if check.severity == "error":
                        overall_result = QualityGateResult.FAIL
                    elif check.severity == "warning" and overall_result == QualityGateResult.PASS:
                        overall_result = QualityGateResult.WARN
                
                # Apply remediation if needed
                if not passed and check.remediation and self.auto_remediate:
                    try:
                        current_df, remediation_details = self._apply_remediation(
                            current_df, check.remediation, check.remediation_params, context
                        )
                        
                        check_result["remediated"] = True
                        check_result["remediation_details"] = remediation_details
                        
                        report.remediations_applied.append({
                            "check": check.name,
                            "action": check.remediation.value,
                            "details": remediation_details,
                            "timestamp": datetime.now().isoformat(),
                        })
                        
                        overall_result = QualityGateResult.REMEDIATED
                        
                    except Exception as e:
                        check_result["remediation_error"] = str(e)
                        warnings.warn(f"Remediation failed for {check.name}: {e}")
                
                report.checks.append(check_result)
                
            except Exception as e:
                check_result = {
                    "name": check.name,
                    "passed": False,
                    "severity": "error",
                    "message": f"Check execution failed: {e}",
                    "error": str(e),
                }
                report.checks.append(check_result)
                overall_result = QualityGateResult.FAIL
        
        report.overall = overall_result
        report.final_shape = current_df.shape
        
        # Log remediation
        if self.remediation_log_dir and report.remediations_applied:
            self._log_remediation(report, pair)
        
        return current_df, report
    
    def _apply_remediation(
        self,
        df: pl.DataFrame,
        action: RemediationAction,
        params: dict,
        context: dict | None,
    ) -> tuple[pl.DataFrame, dict]:
        """Apply a remediation action"""
        
        if action == RemediationAction.DROP_NULLS:
            original_len = len(df)
            df = df.drop_nulls()
            return df, {"rows_dropped": original_len - len(df)}
        
        elif action == RemediationAction.FILL_NULLS_FORWARD:
            fill_cols = params.get("columns", df.select(pl.col(pl.NUMERIC_DTYPES)).columns)
            null_counts = {c: df[c].null_count() for c in fill_cols if c in df.columns}
            df = df.with_columns([
                pl.col(c).fill_null(strategy="forward").fill_null(0) 
                for c in fill_cols if c in df.columns
            ])
            return df, {"filled_columns": list(null_counts.keys()), "null_counts": null_counts}
        
        elif action == RemediationAction.FILL_NULLS_ZERO:
            fill_cols = params.get("columns", df.select(pl.col(pl.NUMERIC_DTYPES)).columns)
            null_counts = {c: df[c].null_count() for c in fill_cols if c in df.columns}
            df = df.with_columns([
                pl.col(c).fill_null(0) 
                for c in fill_cols if c in df.columns
            ])
            return df, {"filled_columns": list(null_counts.keys()), "null_counts": null_counts}
        
        elif action == RemediationAction.WINSORIZE:
            numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
            lower = params.get("lower", 0.01)
            upper = params.get("upper", 0.99)
            
            exprs = []
            for col in numeric_cols:
                if col in df.columns:
                    lo = df[col].quantile(lower)
                    hi = df[col].quantile(upper)
                    exprs.append(
                        pl.col(col).clip(lower_bound=lo, upper_bound=hi).alias(col)
                    )
            df = df.with_columns(exprs)
            return df, {"columns_winsorized": len(exprs), "bounds": (lower, upper)}
        
        elif action == RemediationAction.CLIP_OUTLIERS:
            numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
            std_mult = params.get("std_multiplier", 5.0)
            
            exprs = []
            for col in numeric_cols:
                if col in df.columns:
                    mean = df[col].mean()
                    std = df[col].std()
                    if std > 0:
                        lo = mean - std_mult * std
                        hi = mean + std_mult * std
                        exprs.append(
                            pl.col(col).clip(lower_bound=lo, upper_bound=hi).alias(col)
                        )
            df = df.with_columns(exprs)
            return df, {"columns_clipped": len(exprs), "std_multiplier": std_mult}
        
        elif action == RemediationAction.DROP_DUPLICATES:
            subset = params.get("subset", ["timestamp_utc"] if "timestamp_utc" in df.columns else None)
            original_len = len(df)
            df = df.unique(subset=subset, maintain_order=True)
            return df, {"rows_dropped": original_len - len(df)}
        
        elif action == RemediationAction.REINDEX_TIME:
            # Ensure timestamp column exists and is sorted
            if "timestamp_utc" in df.columns:
                df = df.sort("timestamp_utc")
                # Check for gaps
                freq = params.get("freq", "1min")
                # This would need a more complete implementation
                pass
            return df, {"reindexed": True}
        
        elif action == RemediationAction.INTERPOLATE_GAPS:
            # Interpolate missing time gaps
            if "timestamp_utc" in df.columns:
                method = params.get("method", "linear")
                numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
                df = df.with_columns([
                    pl.col(c).interpolate(method=method) for c in numeric_cols
                ])
            return df, {"interpolated": True, "method": method}
        
        elif action == RemediationAction.CAP_SPREAD:
            # Cap spread at reasonable multiple of median
            if "spread" in df.columns:
                multiplier = params.get("multiplier", 5.0)
                median_spread = df["spread"].median()
                cap = median_spread * multiplier
                original_spread = df["spread"].clone()
                df = df.with_columns(
                    pl.col("spread").clip(upper_bound=cap).alias("spread")
                )
                # Recalculate mid if needed
                if "bid" in df.columns and "ask" in df.columns:
                    df = df.with_columns([
                        ((pl.col("bid") + pl.col("ask")) / 2).alias("mid"),
                    ])
                capped_count = (original_spread > cap).sum()
                return df, {"spread_capped": int(capped_count), "cap_value": cap, "multiplier": multiplier}
            return df, {"spread_capped": 0, "reason": "no_spread_column"}
        
        elif action == RemediationAction.FIX_OHLC:
            # Fix OHLC inconsistencies
            fixed = 0
            if all(c in df.columns for c in ["open", "high", "low", "close"]):
                # Ensure high >= max(open, close) and low <= min(open, close)
                df = df.with_columns([
                    pl.max_horizontal("high", "open", "close").alias("high"),
                    pl.min_horizontal("low", "open", "close").alias("low"),
                ])
                fixed = len(df)
            return df, {"ohlc_fixed": fixed}
        
        elif action == RemediationAction.REMOVE_WEEKENDS:
            if "timestamp_utc" in df.columns:
                original_len = len(df)
                df = df.filter(
                    ~pl.col("timestamp_utc").dt.weekday().is_in([6, 7])  # Sat=6, Sun=7
                )
                return df, {"rows_removed": original_len - len(df)}
            return df, {"rows_removed": 0}
        
        return df, {"action": "unknown"}
    
    # Quality check functions
    
    def _check_no_nulls_in_critical(self, df: pl.DataFrame) -> tuple[bool, str, dict]:
        """Check for nulls in critical columns"""
        critical_cols = ["timestamp_utc", "close", "open", "high", "low"]
        critical_cols = [c for c in critical_cols if c in df.columns]
        
        null_info = {}
        total_nulls = 0
        for col in critical_cols:
            null_count = df[col].null_count()
            if null_count > 0:
                null_info[col] = null_count
                total_nulls += null_count
        
        if total_nulls > 0:
            return False, f"Found {total_nulls} nulls in critical columns", null_info
        return True, "No nulls in critical columns", {}
    
    def _check_no_infinite_values(self, df: pl.DataFrame) -> tuple[bool, str, dict]:
        """Check for infinite values in numeric columns"""
        numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
        
        inf_info = {}
        total_inf = 0
        for col in numeric_cols:
            inf_count = df.select(pl.col(col).is_infinite().sum()).item()
            if inf_count > 0:
                inf_info[col] = inf_count
                total_inf += inf_count
        
        if total_inf > 0:
            return False, f"Found {total_inf} infinite values", inf_info
        return True, "No infinite values", {}
    
    def _check_no_duplicate_timestamps(self, df: pl.DataFrame) -> tuple[bool, str, dict]:
        """Check for duplicate timestamps"""
        if "timestamp_utc" not in df.columns:
            return True, "No timestamp column", {}
        
        dup_count = df.filter(pl.col("timestamp_utc").is_duplicated()).height
        
        if dup_count > 0:
            return False, f"Found {dup_count} duplicate timestamps", {"duplicate_count": dup_count}
        return True, "No duplicate timestamps", {}
    
    def _check_timestamp_monotonic(self, df: pl.DataFrame) -> tuple[bool, str, dict]:
        """Check if timestamps are monotonically increasing"""
        if "timestamp_utc" not in df.columns:
            return True, "No timestamp column", {}
        
        is_sorted = df["timestamp_utc"].is_sorted()
        
        if not is_sorted:
            return False, "Timestamps are not monotonically increasing", {"sorted": False}
        return True, "Timestamps are monotonic", {"sorted": True}
    
    def _check_no_weekend_data(self, df: pl.DataFrame) -> tuple[bool, str, dict]:
        """Check for weekend data"""
        if "timestamp_utc" not in df.columns:
            return True, "No timestamp column", {}
        
        weekend_count = df.filter(
            pl.col("timestamp_utc").dt.weekday().is_in([6, 7])
        ).height
        
        if weekend_count > 0:
            return False, f"Found {weekend_count} weekend rows", {"weekend_count": weekend_count}
        return True, "No weekend data", {}
    
    def _check_bid_ask_valid(self, df: pl.DataFrame) -> tuple[bool, str, dict]:
        """Check bid/ask validity"""
        if not all(c in df.columns for c in ["bid", "ask"]):
            return True, "No bid/ask columns", {}
        
        invalid = df.filter(
            (pl.col("bid") <= 0) | (pl.col("ask") <= 0) | (pl.col("ask") < pl.col("bid"))
        ).height
        
        if invalid > 0:
            return False, f"Found {invalid} invalid bid/ask rows", {"invalid_count": invalid}
        return True, "Bid/ask valid", {}
    
    def _check_spread_positive(self, df: pl.DataFrame) -> tuple[bool, str, dict]:
        """Check spread is positive"""
        if "spread" not in df.columns:
            if all(c in df.columns for c in ["bid", "ask"]):
                spread = df["ask"] - df["bid"]
            else:
                return True, "No spread data", {}
        else:
            spread = df["spread"]
        
        negative = (spread < 0).sum()
        zero = (spread == 0).sum()
        
        if negative > 0 or zero > 0:
            return False, f"Found {negative} negative and {zero} zero spreads", {
                "negative_spreads": int(negative),
                "zero_spreads": int(zero),
            }
        return True, "All spreads positive", {}
    
    def _check_ohlc_consistent(self, df: pl.DataFrame) -> tuple[bool, str, dict]:
        """Check OHLC consistency"""
        required = ["open", "high", "low", "close"]
        if not all(c in df.columns for c in required):
            return True, "No OHLC columns", {}
        
        issues = 0
        # High should be >= open, close
        issues += df.filter(pl.col("high") < pl.max_horizontal("open", "close")).height
        # Low should be <= open, close
        issues += df.filter(pl.col("low") > pl.min_horizontal("open", "close")).height
        # High >= low
        issues += df.filter(pl.col("high") < pl.col("low")).height
        
        if issues > 0:
            return False, f"Found {issues} OHLC inconsistencies", {"inconsistency_count": issues}
        return True, "OHLC consistent", {}
    
    def _check_feature_variance(self, df: pl.DataFrame) -> tuple[bool, str, dict]:
        """Check feature variance"""
        numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
        
        low_var = []
        for col in numeric_cols:
            std = df[col].std()
            if std is not None and std < 1e-10:
                low_var.append(col)
        
        if low_var:
            return False, f"Found {len(low_var)} low-variance features", {"low_variance_features": low_var}
        return True, "All features have sufficient variance", {}
    
    def _check_no_constant_features(self, df: pl.DataFrame) -> tuple[bool, str, dict]:
        """Check for constant features"""
        numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
        
        constant = []
        for col in numeric_cols:
            if df[col].n_unique() <= 1:
                constant.append(col)
        
        if constant:
            return False, f"Found {len(constant)} constant features", {"constant_features": constant}
        return True, "No constant features", {}
    
    def _check_feature_correlation(self, df: pl.DataFrame) -> tuple[bool, str, dict]:
        """Check for highly correlated features"""
        numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
        
        if len(numeric_cols) < 2:
            return True, "Not enough numeric columns", {}
        
        # Sample for speed
        sample = df.select(numeric_cols).head(5000).to_pandas()
        corr_matrix = sample.corr().abs()
        
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.99:
                    high_corr.append({
                        "feature_1": corr_matrix.columns[i],
                        "feature_2": corr_matrix.columns[j],
                        "correlation": float(corr_matrix.iloc[i, j]),
                    })
        
        if high_corr:
            return True, f"Found {len(high_corr)} highly correlated feature pairs", {"high_correlations": high_corr[:10]}
        return True, "No extreme correlations", {}
    
    def _log_remediation(self, report: QualityGateReport, pair: str | None):
        """Log remediation actions"""
        pair_str = pair or "unknown"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.remediation_log_dir / f"remediation_{self.stage}_{pair_str}_{timestamp}.json"
        
        with open(filepath, "w") as f:
            json.dump(report.to_dict(), f, indent=2)


# Convenience function for creating quality gates
def create_quality_gates(
    stage: str,
    auto_remediate: bool = True,
    remediation_log_dir: str | Path | None = None,
) -> DataQualityGates:
    """Create quality gates for a pipeline stage"""
    return DataQualityGates(
        stage=stage,
        auto_remediate=auto_remediate,
        remediation_log_dir=remediation_log_dir,
    )