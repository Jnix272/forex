"""Model results: fold estimates, refit, holdout gate, certificates, curves, baseline."""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gui_common import ROOT, list_run_dirs, read_json, short  # noqa: E402

st.set_page_config(page_title="Results", page_icon="📊", layout="wide")
st.title("Model results")

runs = list_run_dirs()
if not runs:
    st.info("No checkpoint runs under checkpoints/.")
    st.stop()
run = st.selectbox("Run", runs, format_func=lambda p: p.name)
model_dirs = sorted({p.parent for p in run.rglob("*_fold*_config.json")}) or [run]
mdir = st.selectbox("Model", model_dirs, format_func=lambda p: p.relative_to(run).as_posix() or p.name)

# ── Fold estimates ───────────────────────────────────────────────────────────
st.subheader("Walk-forward folds (performance estimates)")
rows = []
for cfg_path in sorted(mdir.glob("*_fold*_config.json")):
    c = read_json(cfg_path) or {}
    rows.append({
        "fold": c.get("fold_id"),
        "epoch": c.get("epoch"),
        "val_loss": c.get("best_val_loss"),
        "val_sharpe": c.get("best_val_sharpe_proxy"),
        "honest": "yes" if c.get("val_sharpe_is_honest") else "no (label proxy)",
        "sharpe CI low": c.get("honest_sharpe_ci_low"),
        "train rows": c.get("train_range"),
        "val rows": c.get("val_range"),
        "refit": c.get("refit"),
    })
if rows:
    st.dataframe(rows, use_container_width=True, hide_index=True)
    if not any(r["honest"] == "yes" for r in rows):
        st.warning("These folds pre-date the honest metric: their Sharpe is the label proxy, not a price-based Sharpe.")
else:
    st.info("No fold configs in this directory.")

sel = read_json(mdir / "fold_selection.json") or read_json(mdir.parent / "fold_selection.json")
if sel:
    if sel.get("selected") == "refit":
        st.success(f"Deployed checkpoint: **refit** on all pre-holdout data ({sel.get('refit_epochs')} epochs).")
    else:
        st.error(
            f"Deployed checkpoint is **fold {sel.get('selected_fold')}** (chosen by {sel.get('metric')}). "
            "Single-fold selection trains on part of the history; retrain to get a refit."
        )

# ── Curves ───────────────────────────────────────────────────────────────────
st.subheader("Training curves")
cv_files = sorted((run / "logs").glob("*_cv.json")) if (run / "logs").exists() else []
if cv_files:
    cvf = st.selectbox("CV history", cv_files, format_func=lambda p: p.name)
    hist = read_json(cvf) or []
    metric = st.radio("Metric", ["val_loss", "val_sharpe", "honest_sharpe_ci_low", "train_loss"], horizontal=True)
    fig = go.Figure()
    for entry in hist:
        ys = (entry.get("history") or {}).get(metric) or []
        if ys:
            fig.add_trace(go.Scatter(y=ys, mode="lines+markers", name=f"fold {entry.get('fold')}"))
    fig.update_layout(height=360, xaxis_title="epoch", yaxis_title=metric, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No *_cv.json history in this run's logs/.")

# ── Gate & certificate ───────────────────────────────────────────────────────
st.subheader("Holdout gate & certificate")
from validation.gate_policy import check_gate_artifact  # noqa: E402

cert = next((p for p in (mdir / "promotion_gate.json", mdir.parent / "promotion_gate.json") if p.exists()), None)
if cert:
    doc = read_json(cert) or {}
    ok, why = check_gate_artifact(doc)
    (st.success if ok else st.error)(f"`{cert.relative_to(ROOT)}` → {'VALID' if ok else 'NOT valid'}: {why}")
    gate = doc.get("gate") or doc
    gates = gate.get("gates") or {}
    if gates:
        st.dataframe([{"check": k, "pass": "✅" if v else "❌"} for k, v in gates.items()],
                     use_container_width=True, hide_index=True)
    det = gate.get("details") or {}
    keys = ["sharpe", "sharpe_ci_low", "sharpe_ci_high", "psr", "dsr", "n_trades", "profit_factor",
            "max_drawdown", "cost_pct", "n_research_trials"]
    st.json({k: det.get(k) for k in keys if k in det})
else:
    st.info("No promotion_gate.json for this model.")

rep = read_json(mdir / "pretrain_report.json")
if rep:
    st.subheader("Pretraining")
    st.write(f"status **{rep.get('status')}** · gate **{rep.get('quality_gate_result')}** · "
             f"loaded into training: **{rep.get('loaded_into_supervised_training')}**")

# ── Baseline ─────────────────────────────────────────────────────────────────
st.subheader("Honest baseline")
base_files = sorted((ROOT / "logs").glob("baseline_honest_*.json"))
if not base_files:
    st.info("No baseline yet. Run it from the Pipeline page.")
for bf in base_files:
    b = read_json(bf) or {}
    verdict = b.get("verdict", "?")
    (st.success if verdict == "SIGNAL" else st.warning)(f"{bf.name}: **{verdict}** on `{short(Path(str(b.get('cache', ''))).name, 80)}`")
    folds = b.get("folds") or []
    if folds:
        st.dataframe([{"fold": f.get("fold"), "sharpe": f.get("sharpe_net"), "CI low": f.get("sharpe_net_ci_low"),
                       "CI high": f.get("sharpe_net_ci_high"), "trades": f.get("n_trades"), **(f.get("per_pair") or {})}
                      for f in folds], use_container_width=True, hide_index=True)
