"""
Feature Quality Report
======================
Per-feature quality metrics for monitoring and debugging feature pipelines.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import polars as pl

# ════════════════════════════════════════════════════════════════════════════
# Data Classes
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class FeatureQuality:
    """Quality metrics for a single feature column."""

    name: str
    dtype: str
    null_count: int
    null_pct: float
    infinite_count: int
    constant: bool
    unique_count: int
    mean: float | None
    std: float | None
    min: float | None
    max: float | None
    skew: float | None
    kurtosis: float | None
    q25: float | None
    q50: float | None
    q75: float | None
    # Derived flags
    has_nulls: bool
    has_inf: bool
    is_constant: bool
    near_constant: bool  # std < 1e-8
    high_cardinality: bool  # unique > 0.9 * n


@dataclass
class FeatureQualityReport:
    """Aggregate quality report for a feature DataFrame."""

    n_rows: int
    n_cols: int
    features: list[FeatureQuality]
    summary: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "features": [asdict(f) for f in self.features],
            "summary": self.summary,
        }

    def get_problematic(self) -> list[FeatureQuality]:
        """Return features with quality issues."""
        return [f for f in self.features if f.has_nulls or f.has_inf or f.is_constant or f.near_constant]

    def get_clean(self) -> list[FeatureQuality]:
        """Return features passing all quality checks."""
        return [f for f in self.features if not (f.has_nulls or f.has_inf or f.is_constant or f.near_constant)]


# ════════════════════════════════════════════════════════════════════════════
# Core Computation
# ════════════════════════════════════════════════════════════════════════════


def compute_quality_report(
    df: pl.DataFrame,
    exclude_cols: list[str] | None = None,
    near_constant_threshold: float = 1e-8,
    high_cardinality_threshold: float = 0.9,
) -> FeatureQualityReport:
    """
    Compute comprehensive quality metrics for all numeric columns in a DataFrame.

    Args:
        df: Input DataFrame
        exclude_cols: Columns to skip (e.g., "timestamp_utc", target columns)
        near_constant_threshold: Std threshold below which feature is near-constant
        high_cardinality_threshold: Fraction of unique values for high cardinality flag

    Returns:
        FeatureQualityReport with per-feature metrics and summary
    """
    exclude = set(exclude_cols or [])
    numeric_cols = [c for c in df.columns if c not in exclude and df[c].dtype.is_numeric()]

    n_rows = len(df)
    features = []

    for col in numeric_cols:
        s = df[col]
        null_count = s.null_count()
        null_pct = null_count / n_rows if n_rows > 0 else 0.0

        # Convert to numpy for stats (drop nulls)
        vals = s.drop_nulls().to_numpy()
        finite_vals = vals[np.isfinite(vals)]

        infinite_count = int(np.sum(~np.isfinite(vals))) if len(vals) > 0 else 0

        if len(finite_vals) == 0:
            fq = FeatureQuality(
                name=col,
                dtype=str(s.dtype),
                null_count=null_count,
                null_pct=null_pct,
                infinite_count=infinite_count,
                constant=True,
                unique_count=0,
                mean=None,
                std=None,
                min=None,
                max=None,
                skew=None,
                kurtosis=None,
                q25=None,
                q50=None,
                q75=None,
                has_nulls=null_count > 0,
                has_inf=infinite_count > 0,
                is_constant=True,
                near_constant=True,
                high_cardinality=False,
            )
            features.append(fq)
            continue

        unique_count = int(np.unique(finite_vals).size)
        is_const = unique_count <= 1
        std_val = float(np.std(finite_vals))
        near_const = std_val < near_constant_threshold
        high_card = unique_count > (high_cardinality_threshold * n_rows)

        fq = FeatureQuality(
            name=col,
            dtype=str(s.dtype),
            null_count=null_count,
            null_pct=null_pct,
            infinite_count=infinite_count,
            constant=is_const,
            unique_count=unique_count,
            mean=float(np.mean(finite_vals)),
            std=std_val,
            min=float(np.min(finite_vals)),
            max=float(np.max(finite_vals)),
            skew=float(_skew(finite_vals)) if len(finite_vals) >= 3 else None,
            kurtosis=float(_kurtosis(finite_vals)) if len(finite_vals) >= 4 else None,
            q25=float(np.percentile(finite_vals, 25)),
            q50=float(np.percentile(finite_vals, 50)),
            q75=float(np.percentile(finite_vals, 75)),
            has_nulls=null_count > 0,
            has_inf=infinite_count > 0,
            is_constant=is_const,
            near_constant=near_const,
            high_cardinality=high_card,
        )
        features.append(fq)

    # Summary counts
    summary = {
        "total": len(features),
        "clean": sum(1 for f in features if not (f.has_nulls or f.has_inf or f.is_constant or f.near_constant)),
        "with_nulls": sum(1 for f in features if f.has_nulls),
        "with_inf": sum(1 for f in features if f.has_inf),
        "constant": sum(1 for f in features if f.is_constant),
        "near_constant": sum(1 for f in features if f.near_constant),
        "high_cardinality": sum(1 for f in features if f.high_cardinality),
    }

    return FeatureQualityReport(
        n_rows=n_rows,
        n_cols=len(features),
        features=features,
        summary=summary,
    )


def _skew(x: np.ndarray) -> float:
    """Sample skewness."""
    n = len(x)
    if n < 3:
        return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s == 0:
        return 0.0
    return (n / ((n - 1) * (n - 2))) * np.sum(((x - m) / s) ** 3)


def _kurtosis(x: np.ndarray) -> float:
    """Sample excess kurtosis."""
    n = len(x)
    if n < 4:
        return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s == 0:
        return 0.0
    return ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(((x - m) / s) ** 4) - (3 * (n - 1) ** 2) / (
        (n - 2) * (n - 3)
    )


# ════════════════════════════════════════════════════════════════════════════
# Convenience: Print Report
# ════════════════════════════════════════════════════════════════════════════


def print_quality_report(report: FeatureQualityReport, show_clean: bool = False) -> None:
    """Pretty-print a quality report to stdout."""
    print(f"\n{'=' * 80}")
    print(f"FEATURE QUALITY REPORT  |  rows={report.n_rows:,}  cols={report.n_cols}")
    print(f"{'=' * 80}")
    print(f"Summary: {report.summary}")

    problematic = report.get_problematic()
    if problematic:
        print(f"\n⚠️  PROBLEMATIC FEATURES ({len(problematic)}):")
        for f in problematic:
            issues = []
            if f.has_nulls:
                issues.append(f"nulls={f.null_pct:.1%}")
            if f.has_inf:
                issues.append(f"inf={f.infinite_count}")
            if f.is_constant:
                issues.append("CONSTANT")
            if f.near_constant:
                issues.append(f"near-const(std={f.std:.2e})")
            if f.high_cardinality:
                issues.append("high-card")
            print(f"  {f.name:30s} | {'; '.join(issues)}")

    if show_clean:
        clean = report.get_clean()
        print(f"\n✅ CLEAN FEATURES ({len(clean)}):")
        for f in clean:
            print(f"  {f.name:30s} | mean={f.mean:>10.4f} std={f.std:>8.4f} [{f.min:>8.4f}, {f.max:>8.4f}]")


# ════════════════════════════════════════════════════════════════════════════
# CLI / Standalone
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Quick self-test
    rng = np.random.default_rng(42)
    df = pl.DataFrame(
        {
            "good_feat": rng.normal(0, 1, 1000),
            "const_feat": np.ones(1000),
            "null_feat": [1.0 if i % 10 != 0 else None for i in range(1000)],
            "inf_feat": [1.0 if i != 500 else float("inf") for i in range(1000)],
            "high_card": rng.uniform(0, 1, 1000),
        }
    )

    report = compute_quality_report(df)
    print_quality_report(report)

    # Verify flags
    assert any(f.name == "const_feat" and f.is_constant for f in report.features)
    assert any(f.name == "null_feat" and f.has_nulls for f in report.features)
    assert any(f.name == "inf_feat" and f.has_inf for f in report.features)
    print("\n✅ Self-test passed")
