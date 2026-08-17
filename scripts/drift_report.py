"""
Generate an input-distribution drift report from cached training features.

Usage examples:
  python scripts/drift_report.py --cache-path data/processed/dataset_EURUSD_20000000_dukascopy_60_rl_reward.zarr
  python scripts/drift_report.py --cache-path data/processed/dataset_EURUSD_20000000_dukascopy_60_rl_reward --json-out logs/reports/drift_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config.settings import MONITORING
from monitoring.drift_gate import compute_drift_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Forex cache drift report")
    p.add_argument("--cache-path", type=str, required=True, help="Path to dataset cache (.zarr dir or NPY base path).")
    p.add_argument(
        "--baseline-samples", type=int, default=20_000, help="Rows from the start of cache used as baseline."
    )
    p.add_argument(
        "--live-samples", type=int, default=5_000, help="Rows from the end of cache used as recent/live slice."
    )
    p.add_argument("--psi-threshold", type=float, default=float(MONITORING.get("psi_threshold", 0.2)))
    p.add_argument("--ks-pvalue-threshold", type=float, default=float(MONITORING.get("ks_pvalue_threshold", 0.05)))
    p.add_argument("--top-k", type=int, default=20, help="Number of highest-PSI features to print.")
    p.add_argument("--json-out", type=str, default="", help="Optional output path for full JSON report.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = compute_drift_report(
        cache_path=args.cache_path,
        baseline_samples=args.baseline_samples,
        live_samples=args.live_samples,
        psi_threshold=args.psi_threshold,
        ks_pvalue_threshold=args.ks_pvalue_threshold,
        top_k_features=args.top_k,
    )

    status = "FAIL" if report["drift_detected"] else "PASS"
    print("=" * 72)
    print("Input Drift Report")
    print("=" * 72)
    print(f"Status: {status}")
    print(f"Cache: {report['cache_path']}")
    print(f"Rows: baseline={report['baseline_rows']:,}  live={report['live_rows']:,}  total={report['n_total']:,}")
    print(f"Features checked: {report['n_features_checked']:,}")
    print(f"psi_max={report['psi_max']:.6f}  (threshold={report['psi_threshold']:.6f})")
    print(f"ks_min_pvalue={report['ks_min_pvalue']:.6f}  (threshold={report['ks_pvalue_threshold']:.6f})")
    if report["reasons"]:
        print("Reasons:")
        for r in report["reasons"]:
            print(f"  - {r}")
    print("Top shifted features (by PSI):")
    for row in report["top_features"]:
        print(f"  - feature[{row['feature_idx']}] psi={row['psi']:.6f} ks_p={row['ks_pvalue']:.6f}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved JSON report: {out_path}")


if __name__ == "__main__":
    main()
