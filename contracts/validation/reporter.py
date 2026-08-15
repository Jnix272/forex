"""
Validation Reporter
===================
Generates comprehensive validation reports for pipeline runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from contracts.validation.gates import PipelineStageValidator, GateResult, ValidationResult
from contracts.validation.drift import SchemaDriftDetector, DriftReport


@dataclass
class ValidationReport:
    """Comprehensive validation report for a pipeline run"""
    run_id: str
    timestamp: datetime
    pair: str | None
    stages: dict[str, dict] = field(default_factory=dict)
    drift_reports: dict[str, DriftReport] = field(default_factory=dict)
    overall_status: str = "unknown"
    total_gates: int = 0
    total_passed: int = 0
    total_warnings: int = 0
    total_failed: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "pair": self.pair,
            "stages": self.stages,
            "drift_reports": {k: v.to_dict() for k, v in self.drift_reports.items()},
            "overall_status": self.overall_status,
            "total_gates": self.total_gates,
            "total_passed": self.total_passed,
            "total_warnings": self.total_warnings,
            "total_failed": self.total_failed,
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValidationReport":
        return cls(
            run_id=data["run_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            pair=data["pair"],
            stages=data["stages"],
            drift_reports={k: DriftReport.from_dict(v) for k, v in data.get("drift_reports", {}).items()},
            overall_status=data["overall_status"],
            total_gates=data["total_gates"],
            total_passed=data["total_passed"],
            total_warnings=data["total_warnings"],
            total_failed=data["total_failed"],
        )


class ValidationReporter:
    """Generates and manages validation reports"""
    
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(
        self,
        stage_validators: dict[str, PipelineStageValidator],
        drift_reports: dict[str, DriftReport] | None = None,
        run_id: str | None = None,
        pair: str | None = None,
    ) -> ValidationReport:
        """Generate comprehensive validation report"""
        
        run_id = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        stages = {}
        total_gates = 0
        total_passed = 0
        total_warnings = 0
        total_failed = 0
        
        for stage_name, validator in stage_validators.items():
            summary = validator.get_summary()
            stages[stage_name] = {
                "summary": summary,
                "gates": [r.to_dict() for r in validator.results],
            }
            
            total_gates += summary["total_gates"]
            total_passed += summary["passed"]
            total_warnings += summary["warnings"]
            total_failed += summary["failed"]
        
        # Overall status
        if total_failed > 0:
            overall_status = "fail"
        elif total_warnings > 0:
            overall_status = "warn"
        else:
            overall_status = "pass"
        
        report = ValidationReport(
            run_id=run_id,
            timestamp=datetime.now(),
            pair=pair,
            stages=stages,
            drift_reports=drift_reports or {},
            overall_status=overall_status,
            total_gates=total_gates,
            total_passed=total_passed,
            total_warnings=total_warnings,
            total_failed=total_failed,
        )
        
        return report
    
    def save_report(self, report: ValidationReport, filename: str | None = None) -> Path:
        """Save report to JSON file"""
        filename = filename or f"validation_report_{report.run_id}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, "w") as f:
            f.write(report.to_json())
        
        return filepath
    
    def save_html_report(self, report: ValidationReport, filename: str | None = None) -> Path:
        """Save report as HTML for easy viewing"""
        filename = filename or f"validation_report_{report.run_id}.html"
        filepath = self.output_dir / filename
        
        html = self._generate_html(report)
        
        with open(filepath, "w") as f:
            f.write(html)
        
        return filepath
    
    def _generate_html(self, report: ValidationReport) -> str:
        """Generate HTML report"""
        status_colors = {
            "pass": "#28a745",
            "warn": "#ffc107",
            "fail": "#dc3545",
            "unknown": "#6c757d",
        }
        
        color = status_colors.get(report.overall_status, "#6c757d")
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Validation Report - {report.run_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: {color}; color: white; padding: 20px; border-radius: 5px; }}
        .stage {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .stage-header {{ font-weight: bold; font-size: 1.2em; margin-bottom: 10px; }}
        .gate {{ padding: 10px; margin: 5px 0; border-radius: 3px; }}
        .gate-pass {{ background: #d4edda; border-left: 4px solid #28a745; }}
        .gate-warn {{ background: #fff3cd; border-left: 4px solid #ffc107; }}
        .gate-fail {{ background: #f8d7da; border-left: 4px solid #dc3545; }}
        .gate-skip {{ background: #e2e3e5; border-left: 4px solid #6c757d; }}
        .drift-report {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .drift-detected {{ background: #f8d7da; }}
        .drift-none {{ background: #d4edda; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
        .summary-box {{ padding: 15px; border-radius: 5px; color: white; text-align: center; }}
        .summary-pass {{ background: #28a745; }}
        .summary-warn {{ background: #ffc107; }}
        .summary-fail {{ background: #dc3545; }}
        .summary-total {{ background: #17a2b8; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Pipeline Validation Report</h1>
        <p>Run ID: {report.run_id}</p>
        <p>Timestamp: {report.timestamp.isoformat()}</p>
        <p>Pair: {report.pair or "All"}</p>
        <p>Overall Status: <span style="font-size: 1.5em; font-weight: bold;">{report.overall_status.upper()}</span></p>
    </div>
    
    <div class="summary">
        <div class="summary-box summary-total">
            <h3>{report.total_gates}</h3>
            <p>Total Gates</p>
        </div>
        <div class="summary-box summary-pass">
            <h3>{report.total_passed}</h3>
            <p>Passed</p>
        </div>
        <div class="summary-box summary-warn">
            <h3>{report.total_warnings}</h3>
            <p>Warnings</p>
        </div>
        <div class="summary-box summary-fail">
            <h3>{report.total_failed}</h3>
            <p>Failed</p>
        </div>
    </div>
"""
        
        # Stages
        for stage_name, stage_data in report.stages.items():
            html += f"""
    <div class="stage">
        <div class="stage-header">{stage_name} (Overall: {stage_data['summary']['overall']})</div>
        <table>
            <tr><th>Gate</th><th>Result</th><th>Message</th><th>Duration (ms)</th></tr>
"""
            for gate in stage_data["gates"]:
                result_class = f"gate-{gate['result']}"
                html += f"""
            <tr class="{result_class}">
                <td>{gate['gate_name']}</td>
                <td>{gate['result'].upper()}</td>
                <td>{gate['message']}</td>
                <td>{gate['duration_ms']:.1f}</td>
            </tr>
"""
            html += """
        </table>
    </div>
"""
        
        # Drift reports
        if report.drift_reports:
            html += """
    <h2>Schema Drift Detection</h2>
"""
            for stage_name, drift in report.drift_reports.items():
                drift_class = "drift-detected" if drift.drift_detected else "drift-none"
                html += f"""
    <div class="drift-report {drift_class}">
        <h3>{stage_name}</h3>
        <p>Drift Type: {drift.drift_type}</p>
        <p>Schema Hash: {drift.schema_hash} (ref: {drift.reference_schema_hash or 'N/A'})</p>
        <p>Data Hash: {drift.data_hash} (ref: {drift.reference_data_hash or 'N/A'})</p>
"""
                if drift.added_columns:
                    html += f"<p>Added Columns: {', '.join(drift.added_columns)}</p>"
                if drift.removed_columns:
                    html += f"<p>Removed Columns: {', '.join(drift.removed_columns)}</p>"
                if drift.type_changes:
                    html += "<p>Type Changes: " + ", ".join(f"{k}: {v[0]} → {v[1]}" for k, v in drift.type_changes.items()) + "</p>"
                if drift.high_psi_columns:
                    html += f"<p>High PSI Columns: {', '.join(drift.high_psi_columns)}</p>"
                
                html += """
    </div>
"""
        
        html += """
</body>
</html>
"""
        return html
    
    def load_report(self, filepath: str | Path) -> ValidationReport:
        """Load report from JSON file"""
        with open(filepath) as f:
            data = json.load(f)
        return ValidationReport.from_dict(data)
    
    def list_reports(self) -> list[Path]:
        """List all validation reports in output directory"""
        return sorted(self.output_dir.glob("validation_report_*.json"))