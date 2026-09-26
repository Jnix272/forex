"""Dataset caches: health checks (alignment, labels, spreads, feature health)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gui_common import list_caches, short  # noqa: E402
from data_checks import inspect_cache  # noqa: E402

st.set_page_config(page_title="Data Explorer", page_icon="🗂️", layout="wide")
st.title("Data explorer")

caches = list_caches()
extra = st.text_input("Or a cache path", "")
choices = ([Path(extra)] if extra else []) + caches
if not choices:
    st.info("No caches under data/processed/.")
    st.stop()
cache = st.selectbox("Cache", choices, format_func=lambda p: short(p.name, 110))

try:
    from training.cache_integrity import DATASET_BUILD_VERSION

    if f"_b{DATASET_BUILD_VERSION}" not in cache.name:
        st.warning(f"Built with older code (current version `{DATASET_BUILD_VERSION}`); rebuild before training.")
except Exception:
    pass

if st.button("Run checks", type="primary"):
    with st.spinner("Reading cache..."):
        st.session_state["checks"] = (str(cache), inspect_cache(cache))

res = st.session_state.get("checks")
if res and res[0] == str(cache):
    r = res[1]
    if r["problems"]:
        st.error(f"{len(r['problems'])} problem(s)")
        for p in r["problems"]:
            st.write("•", p)
    else:
        st.success("No problems found.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", r["arrays"]["X"][0])
    c2.metric("Features", r["features"]["n"])
    c3.metric("Constant features", len(r["features"]["constant"]))
    if r.get("time"):
        st.write(f"Time span **{r['time']['start']} → {r['time']['end']}** · duplicates {r['time']['duplicates']} · "
                 f"backwards steps {r['time']['backwards']}")
    if r["pairs"]:
        st.subheader("Per pair")
        df = pd.DataFrame(r["pairs"]).T
        st.dataframe(df.style.format(precision=3), use_container_width=True)
        st.bar_chart(df[["sell", "hold", "buy"]])
    with st.expander("Constant feature groups"):
        st.write(r["features"]["constant_bases"])
    with st.expander("Arrays & attributes"):
        st.json({"arrays": r["arrays"], "attrs": r["attrs"]})
