"""Forex model control panel (Streamlit).

    streamlit run gui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gui_common import (  # noqa: E402
    LIVE_LOGS,
    ROOT,
    list_caches,
    list_jobs,
    list_run_dirs,
    read_json,
    short,
)

st.set_page_config(page_title="Forex Control Panel", page_icon="📈", layout="wide")
st.title("Forex Control Panel")
st.caption(f"Project: `{ROOT}`")

jobs = list_jobs()
running = [j for j in jobs if j["running"]]
caches = list_caches()
runs = list_run_dirs()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Running jobs", len(running))
c2.metric("Dataset caches", len(caches))
c3.metric("Checkpoint runs", len(runs))
c4.metric("Live log files", len(list(LIVE_LOGS.glob("*.jsonl"))) if LIVE_LOGS.exists() else 0)

st.subheader("Dataset version")
try:
    from training.cache_integrity import DATASET_BUILD_VERSION

    current = [c for c in caches if f"_b{DATASET_BUILD_VERSION}" in c.name]
    if current:
        st.success(f"{len(current)} cache(s) built with the current code (`{DATASET_BUILD_VERSION}`).")
    else:
        st.warning(
            f"No cache built with the current code (`{DATASET_BUILD_VERSION}`). "
            "Rebuild it on the **Pipeline** page before training."
        )
except Exception as exc:
    st.info(f"Could not read DATASET_BUILD_VERSION: {exc}")

st.subheader("Certificates")
from validation.gate_policy import check_gate_artifact  # noqa: E402

rows = []
for cert in sorted(ROOT.glob("checkpoints/**/promotion_gate.json")) + sorted(
    ROOT.glob("checkpoints/**/optimal_roadmap_certification.json")
):
    doc = read_json(cert) or {}
    ok, why = check_gate_artifact(doc) if isinstance(doc, dict) else (False, "unreadable")
    rows.append({"file": short(str(cert.relative_to(ROOT)), 90), "valid": "✅" if ok else "❌", "reason": why})
if rows:
    st.dataframe(rows, use_container_width=True, hide_index=True)
else:
    st.info("No certificates found.")

st.subheader("Recent jobs")
if jobs:
    st.dataframe(
        [{"job": j["label"], "started": j["started"], "status": "running" if j["running"] else "finished"}
         for j in jobs[:10]],
        use_container_width=True, hide_index=True,
    )
else:
    st.info("No jobs yet. Start one on the **Pipeline** page.")
