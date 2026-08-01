"""
data/data_quality_report.py
===========================
Generates a deep audit of NaN rates, spread outliers, and class balances
to catch bad historical data.
"""

import json
from pathlib import Path
import logging

class DataQualityReporter:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)

    def generate_report(self, missing_bars: dict, zero_volume_periods: dict,
                        spread_outliers: dict, feature_nan_rates: dict,
                        label_class_balance: dict, reward_dist: dict,
                        per_regime_counts: dict):
        """Writes the data_quality_report.json."""
        report = {
            "missing_bars_by_pair": missing_bars,
            "zero_volume_periods": zero_volume_periods,
            "spread_outliers": spread_outliers,
            "feature_nan_rates": feature_nan_rates,
            "label_class_balance": label_class_balance,
            "reward_distribution": reward_dist,
            "per_regime_sample_counts": per_regime_counts
        }
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / "data_quality_report.json"
        
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            
        self.logger.info(f"Data quality report saved to {report_path}")
        
        # Check for catastrophic imbalances
        for k, v in label_class_balance.items():
            if v > 0.8:
                self.logger.warning(f"Extreme class imbalance detected! Class '{k}' is {v*100:.1f}% of data.")
