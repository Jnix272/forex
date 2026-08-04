# Continue

**Updated:** 2026-08-04  
**Status:** Improvement backlog (items 1–7 + A1–D2 wiring) complete. Config consistency gates live. `train_gpu.py` split complete. Training-cache / loop perf remediations landed (Polars chunk path, FP16 Zarr, fused AdamW, Linux lz4).

---

## Living docs

| Doc | Purpose |
|-----|---------|
| [`README.md`](README.md) | Docs index |
| [`TRAINING_PIPELINE_AUDIT.md`](TRAINING_PIPELINE_AUDIT.md) | Stage-by-stage training audit (current) |
| [`CONFIG_CONSISTENCY.md`](CONFIG_CONSISTENCY.md) | Settings ↔ YAML ↔ curriculum ↔ dataset schema gates |
| [`IMPROVEMENT_PLAN.md`](IMPROVEMENT_PLAN.md) | Longer roadmap |
| [`NEWS_DATA_GUIDE.md`](NEWS_DATA_GUIDE.md) | News acquisition |
| [`SESSION_REPORT.md`](SESSION_REPORT.md) | Session log |
| [`archive/`](archive/) | Historical audit reports (read-only) |

---

## Just landed

- **Training cache / loop perf (2026-08-04 evening):**
  - `_build_chunk`: stay Polars through FE → mask → align; pandas only for labeling APIs; HTF context is Polars (`group_by_dynamic` + `join_asof`).
  - Zarr **X** stored as **FP16** (`ZARR_FEATURE_DTYPE`); labels/market sidecars FP32. Rebuild caches to benefit.
  - `build_adamw()`: fused `torch.optim.AdamW` (apex fallback) in supervised loop + DivFT.
  - Linux Zarr default **lz4@1** (`default_zarr_compression`); `run_ubuntu.yaml` + ubuntu hardware profiles match; `auto` elsewhere → platform pick.
- **Module slice 26–36** (curriculum / GPU util): audited in [`TRAINING_PIPELINE_AUDIT.md`](TRAINING_PIPELINE_AUDIT.md).
  - `SharpeProxyLoss` + multitask Sharpe: softsign instead of `tanh`.
  - YAML `distillation.student_model` → `--model` when KD enabled.
- Config multi-part gates — see CONFIG_CONSISTENCY.
- Docs cleaned: audits → `archive/`; this file is the short tracker.

```bash
uv run pytest tests/test_zarr_prefetch.py tests/test_fused_adamw.py tests/test_model_full_data_flow.py -q
uv run python -m training.train_gpu --validate-config --config config/run.yaml
```

---

## train_gpu split (done)

`training/train_gpu.py` **~2.2k lines** (was ~15k, −85%). Logic lives in focused modules
(`dataset_builder`, `supervised_loop`, `gpu_cli`, `rl_runner`, …) with back-compat re-exports.
Further cuts would only shrink `main()`.

Smoke: `uv run pytest tests/test_training_smoke.py tests/test_cv.py -q`

---

## Suggested next

1. **Rebuild training Zarr** on Linux (FP16 X + lz4@1) before long runs — old FP32/zstd caches still readable.
2. **Paper + promote a model**, then optional live (non-paper requires `promotion_gate.json`).
3. **Optional:** wire `BrokerBridge` into `live_engine` (still paper/LMAX/OANDA today).

**Audit backlog cleared:** Top 5 + DS-002/PIT + tabular purged CV + pretrain guardrails + stage timings + GPU util + BrokerBridge + live promotion gate + curriculum/GPU util 26–36 + cache/loop perf above. See [`TRAINING_PIPELINE_AUDIT.md`](TRAINING_PIPELINE_AUDIT.md).

---

## Archive note

Completed CONTINUE deliverables (Risk, Metrics, MC wiring, drift, audit, alerting, module wiring)
are in CHANGELOG + SESSION_REPORT. Old `*_AUDIT_REPORT.md` / `FIXES_APPLIED.md` → [`archive/`](archive/).
