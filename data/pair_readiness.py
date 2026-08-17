"""
data/pair_readiness.py
======================
Upgrades the console checks into a structured JSON report that buckets
dropped bars into specific reasons, failing the build if a pair is unusable.
"""

import json
import logging
from pathlib import Path


class PairReadinessGate:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)
        self.reports = {}

    def add_pair(
        self,
        pair: str,
        source: str,
        start: str,
        end: str,
        freq: str,
        raw_ticks: int,
        duplicates: int,
        dropped_buckets: dict,
        valid_sequences: int,
        spreads: dict,
        atrs: dict,
        missing_columns: list,
    ):
        """Adds a pair to the readiness report, evaluating its pass/fail status."""

        status = "pass"
        fail_reason = ""

        if valid_sequences == 0:
            status = "fail"
            fail_reason = "Zero usable sequences after filtering."
        elif len(missing_columns) > 0:
            status = "fail"
            fail_reason = f"Missing required columns: {missing_columns}"
        elif dropped_buckets.get("nan_rate", 0) > 0.3:
            status = "warn"
            fail_reason = "High NaN rate (>30%)."

        self.reports[pair] = {
            "metadata": {"source": source, "start": start, "end": end, "freq": freq},
            "raw_ticks": raw_ticks,
            "duplicates": duplicates,
            "dropped_bars_by_reason": dropped_buckets,  # e.g. weekend, holiday, spread, news
            "valid_sequence_count": valid_sequences,
            "spread_stats": spreads,  # median, p95, max
            "atr_stats": atrs,
            "status": status,
            "fail_reason": fail_reason,
        }

    def execute_gate(self) -> bool:
        """Writes the report and fails training if any pair failed."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / "pair_readiness_report.json"

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(self.reports, f, indent=2)

        failures = [p for p, data in self.reports.items() if data["status"] == "fail"]

        if failures:
            self.logger.error(f"Pair Readiness Gate FAILED for: {failures}")
            for p in failures:
                self.logger.error(f"  {p}: {self.reports[p]['fail_reason']}")
            raise RuntimeError("One or more required pairs failed readiness checks. Halting training.")

        self.logger.info("All pairs passed readiness gate.")
        return True
