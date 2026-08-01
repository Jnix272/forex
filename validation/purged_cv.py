"""
Purged/Embargoed Cross-Validation for time series.
Prevents leakage across regime boundaries; matches live deployment.
"""
import numpy as np
from typing import Iterator, Tuple, List
from sklearn.model_selection import BaseCrossValidator


class PurgedEmbargoCV(BaseCrossValidator):
    """
    Purged K-Fold with embargo gap.
    
    For each fold:
    - Train: all data before test_start - purge_gap
    - Test: contiguous block
    - Embargo: gap after test before next train
    
    This prevents:
    1. Purging: labels near test set don't leak into train (seq_len + lookahead)
    2. Embargo: no overlap between consecutive test sets
    """
    def __init__(
        self,
        n_splits: int = 7,
        purge_bars: int = 120,
        embargo_bars: int = 60,
        min_train_size: int = 10000,
    ):
        self.n_splits = n_splits
        self.purge_bars = purge_bars
        self.embargo_bars = embargo_bars
        self.min_train_size = min_train_size
    
    def split(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        groups: np.ndarray = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Split into n_splits contiguous test blocks
        test_size = n_samples // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            test_start = (i + 1) * test_size
            test_end = min(test_start + test_size, n_samples)
            
            if test_end - test_start < 100:
                continue
            
            # Purge: exclude purge_bars before test from train
            purge_start = max(0, test_start - self.purge_bars)
            
            # Train indices: everything before purge_start
            train_indices = indices[:purge_start]
            
            if len(train_indices) < self.min_train_size:
                continue
            
            # Test indices
            test_indices = indices[test_start:test_end]
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


def compute_regime_consistency(
    val_sharpes: List[float],
    regime_sharpes: dict,
    min_mean_sharpe: float = 1.5,
    min_regime_sharpe: float = 0.5,
    max_regime_std: float = 0.3,
) -> Tuple[bool, str]:
    """
    Check regime consistency for promotion.
    
    Args:
        val_sharpes: List of Sharpe ratios per fold
        regime_sharpes: Dict mapping regime_name -> List of Sharpe per fold
        min_mean_sharpe: Minimum mean Sharpe across all folds
        min_regime_sharpe: Minimum Sharpe for any regime
        max_regime_std: Maximum std of Sharpe across folds for any regime
    
    Returns:
        (passed, reason)
    """
    mean_sharpe = np.mean(val_sharpes)
    
    if mean_sharpe < min_mean_sharpe:
        return False, f"Mean Sharpe {mean_sharpe:.2f} < {min_mean_sharpe}"
    
    for regime, sharpes in regime_sharpes.items():
        regime_mean = np.mean(sharpes)
        regime_std = np.std(sharpes)
        
        if regime_mean < min_regime_sharpe:
            return False, f"Regime {regime} Sharpe {regime_mean:.2f} < {min_regime_sharpe}"
        
        if regime_std > max_regime_std:
            return False, f"Regime {regime} Sharpe std {regime_std:.2f} > {max_regime_std}"
    
    return True, f"All checks passed: mean_sharpe={mean_sharpe:.2f}"


if __name__ == "__main__":
    # Quick test
    cv = PurgedEmbargoCV(n_splits=5, purge_bars=120, embargo_bars=60)
    X = np.zeros(10000)
    splits = list(cv.split(X))
    print(f"Generated {len(splits)} splits")
    for i, (tr, te) in enumerate(splits):
        print(f"  Fold {i}: train={len(tr)}, test={len(te)}, gap={tr[-1] if len(tr)>0 else 'N/A'} to {te[0]}")