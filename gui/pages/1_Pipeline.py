"""Run pipeline steps as background jobs and follow their logs."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gui_common import CONFIGS, MODELS, list_caches, list_jobs, short, start_job, stop_job, tail  # noqa: E402

st.set_page_config(page_title="Pipeline", page_icon="⚙️", layout="wide")
st.title("Pipeline")
st.caption("Each step runs as a background process; closing the browser does not stop it.")

MODEL_NOTES = {
    "tft": "temporal fusion transformer",
    "haelt": "LSTM + transformer hybrid",
    "mamba": "state-space model",
    "gnn": "cross-pair graph network",
    "transformer": "plain transformer",
    "patchtst": "patch transformer",
    "expert": "mixture of experts",
    "glm": "linear baseline (no pretraining)",
}
default_cfg = CONFIGS.index("config/run.yaml") if "config/run.yaml" in CONFIGS else 0
tab_build, tab_base, tab_train, tab_opt = st.tabs(["1. Build dataset", "2. Baseline", "3. Train", "4. Optuna"])

with tab_build:
    st.write("Rebuild the dataset cache from tick data (hours for the full 2008–2025 range).")
    cfg = st.selectbox("Config", CONFIGS, index=default_cfg, key="b_cfg")
    c1, c2 = st.columns(2)
    start = c1.text_input("Start date (blank = config)", "", key="b_start")
    end = c2.text_input("End date (blank = config)", "", key="b_end")
    if st.button("Build dataset", type="primary"):
        args = ["-m", "training.train_gpu", "--config", cfg, "--build-only"]
        if start:
            args += ["--data-start", start]
        if end:
            args += ["--data-end", end]
        meta = start_job(f"build {start or 'cfg'}→{end or 'cfg'}", args)
        st.success(f"Started job {meta['id']} (pid {meta['pid']})")

with tab_base:
    st.write("Does a simple model beat 'no trade' after costs? Answer before training deep models.")
    caches = list_caches()
    cache = st.selectbox("Cache", ["(newest current-version)"] + [str(c) for c in caches],
                         format_func=lambda s: short(Path(s).name, 90) if s.startswith(("C", "D", "/")) else s)
    model = st.radio("Model", ["xgb", "logreg"], horizontal=True)
    folds = st.slider("Walk-forward folds", 3, 10, 5)
    if st.button("Run baseline", type="primary"):
        args = ["scripts/baseline_honest.py", "--model", model, "--folds", str(folds)]
        if not cache.startswith("("):
            args += ["--cache", cache]
        meta = start_job(f"baseline {model}", args)
        st.success(f"Started job {meta['id']}")

with tab_train:
    st.write("Walk-forward CV, final refit, holdout gate. Pretraining is off unless enabled in the config.")
    cfg_t = st.selectbox("Config", CONFIGS, index=default_cfg, key="t_cfg")
    try:
        from training.config_validate import SUPPORTED_SUPERVISED

        choices = sorted(SUPPORTED_SUPERVISED)
    except Exception:
        choices = [m for m in MODELS if m != "ensemble"]
    picked = st.multiselect(
        "Models to train (run one after another in a single job)",
        choices,
        default=[m for m in ("tft",) if m in choices],
        help=" · ".join(f"{k}: {v}" for k, v in MODEL_NOTES.items() if k in choices),
    )
    epochs = st.number_input("Epochs (0 = config)", 0, 200, 0)
    st.caption("The ensemble meta-learner and RL stages follow the config's ensemble / rl settings.")
    if st.button("Train", type="primary", disabled=not picked):
        args = ["-m", "training.train_gpu", "--config", cfg_t]
        if len(picked) == 1:
            args += ["--model", picked[0], "--no-all-models"]
        else:
            args += ["--all-models", "--models", ",".join(picked)]
        if epochs:
            args += ["--epochs", str(int(epochs))]
        meta = start_job(f"train {'+'.join(picked)}", args)
        st.success(f"Started job {meta['id']}")

with tab_opt:
    st.write("Hyperparameter search. Run only after the baseline shows signal.")
    m_o = st.selectbox("Model", ["tft", "haelt", "transformer"], key="o_m")
    mode = st.radio("Mode", ["cheap", "deep"], horizontal=True)
    trials = st.number_input("Trials", 1, 500, 20)
    if st.button("Start study", type="primary"):
        meta = start_job(f"optuna {m_o} {mode}",
                         ["scripts/optuna_tune.py", "--model", m_o, "--mode", mode, "--trials", str(int(trials))])
        st.success(f"Started job {meta['id']}")

st.divider()
st.subheader("Jobs")
jobs = list_jobs()
if not jobs:
    st.info("No jobs yet.")
else:
    labels = [f"{'🟢' if j['running'] else '⚪'} {j['label']} — {j['started']}" for j in jobs]
    idx = st.selectbox("Job", range(len(jobs)), format_func=lambda i: labels[i])
    job = jobs[idx]
    c1, c2, c3 = st.columns([1, 1, 4])
    if job["running"] and c1.button("Stop job"):
        stop_job(job)
        st.warning("Stop requested.")
    auto = c2.toggle("Auto-refresh", value=job["running"])
    c3.code(" ".join(job["cmd"][1:]), language="bash")

    @st.fragment(run_every=3 if auto else None)
    def _log_view():
        st.code(tail(job["log"], 300) or "(no output yet)", language="text")

    _log_view()
