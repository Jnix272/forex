# Continue

**Updated:** 2026-08-09  
**Status SoT:** [`IMPROVEMENTS.md`](IMPROVEMENTS.md) — Done / Open / Other in one place.

> ✅ **2026-08-09 (end of session):** the `supervised_loop.py` epoch loop is
> **repaired** and the curriculum/adversarial consolidation is **complete**.
> Legacy adaptive-curriculum references stripped (Task A); `graph_pgd`
> auto-select wired for `model_name == "gnn"` (Task E); per-sample curriculum
> weights applied to the loss via `_apply_curriculum_weights` (Task F);
> `OneCycleLR` switched to `total_steps` mode (Task G); `HardExampleMiner`
> leftovers cleaned (Task B). P0/P1 audit triaged in [`FIXES.md`](FIXES.md)
> (5 code fixes + 10 already-fixed + 7 false-positives + 12 design-deferred).
> Training smoke + 232 tests pass. Remaining open: §9.2 design-gaps (backtest
> realism, portfolio limits) and §9.3 P2 tech-debt.
>
> ✅ **2026-08-09 (late):** **Per-model training profiles** implemented.
> Central `ModelTrainingProfile` registry in `config/model_training_profile.py`
> auto-applies optimal config per architecture (adversarial, curriculum,
> miner, pretrain, SWA, RL). 6 models × 12 training dimensions. Full CLI
> override support. All 6 model types compile and validated.
>
> ✅ **2026-08-09 (late):** **Cross-cutting integrations** implemented.
> - Adversarial + Curriculum coordination: eps scales with difficulty level
> - Online Miner → Curriculum: forgetting/easy ratios freeze/accelerate pace
> - PGD hardening: L2 grad norm, warmup steps, per-dim eps multipliers
> - Pretrain → Adversarial: feature vulnerability from hard examples
> - ONNX scaler verification: `verify_onnx_scaler` CLI subcommand
> All modules compile clean.
>
> ✅ **2026-08-09 (late):** **YAML + CLI integration** complete.
> New config sections in `run.yaml` / `run_ubuntu.yaml`: `training.adversarial`,
> `training.training_framework`, `training.pretrain_framework`, `training.rl_framework`,
> `curriculum.miner_feedback`, `curriculum.self_paced`, `curriculum.loss_weighting`,
> `pretrain.framework`, `rl.framework`. Mapped via `_YAML_MAP` in `gpu_cli.py`.
>
> ✅ **2026-08-09 (late):** **Unified Monitoring System** implemented.
> New `monitoring/` package with unified logging, checking, alerting, and dashboard:
> - `monitoring/events.py` — Unified event schema (LOG, CHECK, ALERT, METRIC, CHECKPOINT, HEARTBEAT, PROGRESS)
> - `monitoring/event_bus.py` — Async priority queue with deduplication, SQLite persistence, backpressure
> - `monitoring/unified_logger.py` — Single entry point replacing train_logger, sidecar, logging_utils
> - `monitoring/checks/` — 24 built-in checks (NaN, grad norm, loss plateau, representation collapse, checkpoint load, data drift PSI/KS, GPU/CPU/disk resources)
> - `monitoring/alerts/engine.py` — 10 built-in alert rules with rate limiting, multi-channel dispatch
> - `monitoring/dashboard/app.py` — FastAPI + WebSocket live dashboard with Chart.js metrics visualization
> - `monitoring/__init__.py` — Single import for all components
> All modules compile clean, integration tests pass.
>
> ✅ **2026-08-09 (late):** **Data Pipeline Audit & Fixes** complete.
> Deep audit of 14 files across `data/`, `training/`, `labeling/`, `config/`.
> 21 issues found (2 Critical, 4 High, 10 Medium, 5 Low). 9 fixed:
> - Multi-pair Zarr resizeability (C1)
> - DataQualityReporter wired into `build_dataset_chunked` (C2)
> - Per-bar tx cost used for label threshold instead of hardcoded 1.5 pips (H2)
> - `sanitize_array` clip range disabled for features (H3)
> - Scaler shape validation at DataLoader load time (H4)
> - Feature mask allowlist expanded (M7)
> - Triple barrier sequential fallback bid/ask fix (L1)
> - Dead expression removed (L4)
> All modules compile clean.

---

## Read first

| Doc | Use |
|-----|-----|
| [`IMPROVEMENTS.md`](IMPROVEMENTS.md) | **Canonical** fixed vs open backlog |
| [`CONFIG_CONSISTENCY.md`](CONFIG_CONSISTENCY.md) | Settings ↔ YAML ↔ curriculum ↔ schema gates |
| [`SESSION_AUDIT.md`](SESSION_AUDIT.md) | Session/DST technical detail (status mirrored in IMPROVEMENTS) |
| [`TRAINING_PIPELINE_AUDIT.md`](TRAINING_PIPELINE_AUDIT.md) | Stage-by-stage training audit |
| [`IMPROVEMENT_PLAN.md`](IMPROVEMENT_PLAN.md) | Longer data/HW roadmap |
| [`SESSION_REPORT.md`](SESSION_REPORT.md) | Append-only session log |
| [`../CHANGELOG.md`](../CHANGELOG.md) | User-facing change history |
| [`archive/`](archive/) | Historical audits (read-only) |

---

## Suggested next

1. **§9.2 P1 design-gaps** (see [`FIXES.md`](FIXES.md)): backtest execution realism (#16-18: partial-TP tracking, spread-aware stops, order-type-aware impact); conformal `apply_no_trade_zones` `main_logits` param (#19); macro forward-fill leakage test (#20); configurable numerics clip range (#22); news pipeline init-time FinBERT/Ollama load (#23); per-fold feature-quality monitor (#25); portfolio-level session limits (#27); emergency kill-switch design (#28); gradient checkpointing flag (#30); TrainingController crash checkpointing (#31); scaler checksum validation (#32); hard-example-miner temporal-separation test (#35).
2. **§9.3 P2 tech-debt** (Week 3): Numba dedup, GPU backtester SL/TP, slippage calibration, LMAX FIX routing, FinBERT thread-safety, audio timeout, logging hygiene, `main.py` CLI, session-mapping/spread calibration, subprocess timeouts, missing scripts, heartbeat/JSONL rotation, Docker healthchecks/resource-limits, scaler-fusion convention sync, lot/notional convention sync, circuit-breaker coordination.
3. **§8 remaining factory wiring** (opt-in): `create_curriculum_callback`, `create_pretrain_adapter`, `create_rl_adapter` factories exist + are unit-tested but are not called in the production pipeline (P3-2/P3-3/P3-4). `create_adversarial_attack` IS wired (P3-1, Task E).
4. Rebuild training Zarr on Linux (FP16 X + lz4@1) before long runs — `lr*` / mask digests invalidate on LABEL_REGIME or FEATURE_MASK edits
5. Paper + promote a model; non-paper live needs `promotion_gate.json`
6. Set `FRED_API_KEY` (or fix Stooq) to exercise ~14 yield/cross-asset skips
7. Exercise `--broker mt5` / `ibkr` against a paper terminal

```bash
# Quick validation (loop is now green — no NameError)
uv run pytest tests/ -q
uv run python -m training.train_gpu --validate-config --config config/run.yaml
# Session SoT / limits / slip names:
.venv/bin/python3 -m pytest \
  tests/test_session_sot_p1_p3_p4.py \
  tests/test_risk_execution.py::TestSessionLimitsEnforcer -q
# Phase 3 tests (all green post-repair):
.venv/bin/python3 -m pytest \
  tests/test_adversarial_generator.py \
  tests/test_curriculum_callbacks.py \
  tests/test_pretrain_adapter.py \
  tests/test_rl_adapter.py -q
```

---

## Just landed (pointer)

Full tables live in [`IMPROVEMENTS.md`](IMPROVEMENTS.md). Recent: P0–P2 audit remediations, dataset/label column fixes, dynamic LH + DST session P2, performance Numba/GPU sync, `train_gpu` split, config gates. Detail also in `CHANGELOG.md` / `SESSION_REPORT.md`.

**Phase 3 Architectural Replacements (2026-08-08):** AdversarialGenerator → PGD/FGSM/FreeLB, Curriculum → Composer/Lightning callbacks, Pretraining → lightly-ssl/Solo-learn adapters, RL → CleanRL/SB3 adapters, ONNX scaler fusion. New files: `training/adversarial_generator.py`, `training/curriculum_callbacks.py`, `training/pretrain_adapter.py`, `training/rl_adapter.py` + 4 test files. Modified: `training/supervised_loop.py`, `training/gpu_cli.py`, `inference/onnx_inference.py`. See `SESSION_REPORT.md` for details.

**2026-08-09 curriculum/adversarial consolidation (Improvements #1–4):** ✅ **complete.** `CurriculumManager` + `OnlineHardExampleMiner` adopted in-loop; `GraphAdversarialAttack` added and **wired** (auto-select for `gnn`); per-model `pretrain_method` done; legacy adaptive-curriculum loop body **removed**; `OneCycleLR` `total_steps` mode; per-sample curriculum weights **applied to the loss** via `_apply_curriculum_weights`. See [`FIXES.md`](FIXES.md) for the full audit + verdicts.
