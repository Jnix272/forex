"""Read-only view of the live / paper engine logs (no trading controls here)."""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gui_common import LIVE_LOGS, read_jsonl  # noqa: E402

st.set_page_config(page_title="Live Monitor", page_icon="📡", layout="wide")
st.title("Live monitor")
st.caption(
    "Read-only. Start the engine from a terminal "
    "(`python -m trading.live_engine --broker paper ...`); this page follows its logs."
)

if not LIVE_LOGS.exists():
    st.info(f"No live log directory at {LIVE_LOGS}.")
    st.stop()

_all = sorted(LIVE_LOGS.glob("live_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
# live_YYYYMMDD.jsonl = per-bar log; live_<run>_<pair>.jsonl = engine event log.
bar_logs = [p for p in _all if re.fullmatch(r"live_\d{8}\.jsonl", p.name)]
event_logs = [p for p in _all if p not in bar_logs]
journals = sorted(LIVE_LOGS.glob("trade_journal_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
auto = st.toggle("Auto-refresh every 5 s", value=False)


@st.fragment(run_every=5 if auto else None)
def _view():
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Bars")
        if not bar_logs:
            st.info("No bar logs yet.")
        else:
            f = st.selectbox("Bar log", bar_logs, format_func=lambda p: p.name)
            rows = read_jsonl(f, limit=5000)
            if rows:
                df = pd.DataFrame(rows)
                if "equity" in df:
                    fig = go.Figure(go.Scatter(y=df["equity"], mode="lines", name="equity"))
                    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="equity")
                    st.plotly_chart(fig, use_container_width=True)
                last = df.iloc[-1].to_dict()
                m1, m2, m3 = st.columns(3)
                m1.metric("Equity", f"{last.get('equity', float('nan')):,.2f}")
                m2.metric("Last action", str(last.get("action")))
                m3.metric("Latency ms", f"{last.get('latency_ms', float('nan')):.0f}")
                st.dataframe(df.tail(50).iloc[::-1], use_container_width=True, hide_index=True)
    with c2:
        st.subheader("Decisions & blocks")
        if not journals:
            st.info("No trade journals yet.")
        else:
            j = st.selectbox("Journal", journals, format_func=lambda p: p.name)
            recs = read_jsonl(j, limit=5000)
            events = Counter(r.get("event", "?") for r in recs)
            reasons = Counter(str(r.get("reason")) for r in recs if r.get("event") == "blocked")
            st.write({k: v for k, v in events.most_common()})
            if reasons:
                st.bar_chart(pd.Series(dict(reasons.most_common(12))), horizontal=True)
            orders = [r for r in recs if r.get("event") in ("order_filled", "order_rejected", "stop_loss", "take_profit")]
            if orders:
                st.dataframe(pd.DataFrame(orders).tail(50).iloc[::-1], use_container_width=True, hide_index=True)


_view()

st.subheader("Engine events")
if not event_logs:
    st.info("No engine event logs yet.")
else:
    ev = st.selectbox("Event log", event_logs[:200], format_func=lambda p: p.name)
    levels = st.multiselect("Severity", ["ERROR", "WARNING", "INFO", "DEBUG"], default=["ERROR", "WARNING"])
    recs = [r for r in read_jsonl(ev, limit=20000) if str(r.get("severity", "")).upper() in levels]
    if recs:
        df = pd.DataFrame(recs)
        cols = [c for c in ("ts", "severity", "event_type", "pair", "message") if c in df.columns]
        st.dataframe(df[cols].tail(200).iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.success("No events at the selected severity.")
