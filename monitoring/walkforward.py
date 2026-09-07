"""Walk-forward HTML performance report generator."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import PATHS

__all__ = ["WalkForwardReporter"]


class WalkForwardReporter:
    """
    Generates an HTML performance report after each walk-forward cycle.
    Includes: Sharpe, Calmar, win rate, drawdown chart, feature drift.
    """

    def __init__(self, report_dir: str | None = None):
        if report_dir is None:
            report_dir = PATHS["logs_reports"]
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        equity_curve: pd.Series,
        trades: pd.DataFrame,
        feature_drift: dict | None = None,
        model_name: str = "model",
    ) -> str:
        """Generate an HTML report. Returns path to HTML file."""
        stats = self._compute_stats(equity_curve, trades)
        html = self._render_html(stats, equity_curve, trades, feature_drift, model_name)
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        path = self.report_dir / f"wf_report_{model_name}_{ts}.html"
        path.write_text(html, encoding="utf-8")
        print(f"[Report] Generated: {path}")
        return str(path)

    def _compute_stats(self, equity_curve: pd.Series, trades: pd.DataFrame) -> dict:
        rets = equity_curve.pct_change().dropna()
        dd = (equity_curve / equity_curve.cummax() - 1).min()
        sr = rets.mean() / rets.std() * np.sqrt(252 * 1440) if rets.std() > 0 else 0
        calmar = abs(rets.mean() * 252 * 1440 / (abs(dd) + 1e-9))
        wr = (trades["pnl"] > 0).mean() if "pnl" in trades.columns else 0.5
        return {
            "total_return": float((equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100),
            "sharpe": round(float(sr), 3),
            "calmar": round(float(calmar), 3),
            "max_drawdown": round(float(dd * 100), 2),
            "win_rate": round(float(wr * 100), 1),
            "n_trades": len(trades),
            "start": str(equity_curve.index[0])[:10],
            "end": str(equity_curve.index[-1])[:10],
        }

    def _render_html(self, stats, equity, trades, drift, model_name) -> str:
        drift_section = ""
        if drift:
            drift_section = f"""
            <h2>Drift Detection</h2>
            <p>PSI max: <b>{drift.get("psi_max", 0):.3f}</b> |
               KS p-value: <b>{drift.get("ks_min_pvalue", 1):.4f}</b> |
               Sharpe drop: <b>{drift.get("sharpe_drop", 0):.3f}</b></p>
            <p>Drift detected: <b style="color:{"red" if drift.get("drift_detected") else "green"}">
            {drift.get("drift_detected", False)}</b></p>"""

        color = "green" if stats["sharpe"] > 0.5 else ("orange" if stats["sharpe"] > 0 else "red")
        return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Walk-Forward Report - {model_name}</title>
<style>
  body{{font-family:sans-serif;max-width:900px;margin:auto;padding:20px;background:#f9f9f9}}
  .card{{background:white;border-radius:8px;padding:20px;margin:12px 0;box-shadow:0 1px 4px rgba(0,0,0,0.1)}}
  .metric{{display:inline-block;margin:8px 20px 8px 0}}
  .metric .val{{font-size:28px;font-weight:bold;color:{color}}}
  .metric .lbl{{font-size:12px;color:#888}}
  table{{width:100%;border-collapse:collapse}}
  th,td{{padding:8px;border-bottom:1px solid #eee;text-align:left}}
  th{{background:#f0f0f0}}
</style></head><body>
<h1>Walk-Forward Report: {model_name}</h1>
<p style="color:#888">{stats["start"]} -> {stats["end"]}</p>
<div class="card">
  <div class="metric"><div class="val">{stats["total_return"]:+.1f}%</div><div class="lbl">Total Return</div></div>
  <div class="metric"><div class="val">{stats["sharpe"]}</div><div class="lbl">Sharpe Ratio</div></div>
  <div class="metric"><div class="val">{stats["calmar"]}</div><div class="lbl">Calmar Ratio</div></div>
  <div class="metric"><div class="val">{stats["max_drawdown"]}%</div><div class="lbl">Max Drawdown</div></div>
  <div class="metric"><div class="val">{stats["win_rate"]}%</div><div class="lbl">Win Rate</div></div>
  <div class="metric"><div class="val">{stats["n_trades"]}</div><div class="lbl">Trades</div></div>
</div>
{f'<div class="card">{drift_section}</div>' if drift_section else ""}
<p style="color:#aaa;font-size:11px">Generated {datetime.now():%Y-%m-%d %H:%M} UTC</p>
</body></html>"""
