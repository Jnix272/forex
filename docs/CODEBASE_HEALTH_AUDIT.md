# Codebase Health Audit — Bugs, Dead Code & Disconnected Wiring
**Date:** 2026-08-01  
**Project:** `forex-scaling-model` v6.5.0  
**Scope:** Full codebase — logic bugs, error handling, dead code, disconnected infrastructure, config mismatches  
**Total NEW Issues Found:** 18 (+ 14 confirmed overlaps with prior reports)

---

## Relationship to Prior Reports

This audit was conducted independently after the following existing reports:
- `SYSTEMS_AUDIT_REPORT.md` — 8 issues (Sharpe inflation, auto-tune, calibration, curriculum)
- `TRAINING_MODEL_AUDIT_REPORT.md` — 13 issues (SharpeProxyLoss, MCDropout, CatBoost, ensemble)
- `SHARPE_FEATURES_LOGGING_REPORT.md` — 13 issues (Sharpe math, FinBERT zeros, logging)
- `INFRASTRUCTURE_AUDIT_REPORT.md` — 14 issues (leakage, sidecar, audit trail, readiness)
- `DATASET_IMPROVEMENT_REPORT.md` — 10 issues (label leakage, EMA bias, timezone, sentiment)
- `PIPELINE_IMPROVEMENT_REPORT.md` — 11 issues (broker stub, pip value, NLP, DST)
- `DEAD_CODE_REPORT.md` — 315 findings (vulture + ruff)

**Master index in `INFRASTRUCTURE_AUDIT_REPORT.md` appendix: 43 total issues.**

Issues below marked with **(NEW)** are not covered in any prior report. Issues marked **(CONFIRMS X)** validate a prior finding from a different angle.

---

## Executive Summary

This audit covers three dimensions of codebase health:

1. **Logic bugs & error handling** — 14 issues found in live trading, risk management, backtesting, and RL training code. Several are production-critical (incorrect position tracking, permanent safety halts, stale equity sizing). **7 are NEW findings.**

2. **Configuration & wiring gaps** — 9 issues where config references features/params that don't exist, infrastructure that was scaffolded but never integrated, and pipeline stages that aren't orchestrated. **3 are NEW findings.**

3. **Dead code & broken imports** — 19 dead root-level scripts, 1 dead package (`sizing/`), 50+ unused classes, 5 files with UTF-8 BOM issues, 2 broken symbol imports. **8 are NEW findings beyond `DEAD_CODE_REPORT.md`.**

---

## Part 1: Critical Bugs (Live Trading / Real Money Risk)

---

### BUG-001 — OANDA Net Position Calculation Fragile **(NEW)**

| Attribute | Detail |
|-----------|--------|
| **File** | `trading/live_engine.py` |
| **Line** | ~671–676 |
| **Severity** | Critical |

```python
short_u = float(p.get("short", {}).get("units", 0))
net = long_u - abs(short_u)
```

OANDA reports `short.units` as a negative number. Applying `abs()` then subtracting happens to work but is fragile and undocumented. If both long and short hedge positions exist simultaneously, tracking can desync.

**Fix:** Use `net = long_u + short_u` (since `short_u` is already negative from OANDA).

---

### BUG-002 — LiveSafetyGate Never Resets — Permanent Halt After One Bad Day **(NEW)**

| Attribute | Detail |
|-----------|--------|
| **File** | `trading/live_engine.py` |
| **Line** | ~449–495 |
| **Severity** | Critical |

Once `self.halted = True`, there is no mechanism to reset at the start of a new trading day. The `starting_equity` never updates. After one day hits the loss limit, the engine is permanently halted until manually restarted. No alert is sent.

**Fix:** Add a `new_day()` method (similar to `DrawdownAwareExitManager`) that resets `halted` and updates `starting_equity` at midnight.

---

### BUG-003 — Stale Equity on Broker Failure → Wrong Position Sizing **(NEW)**

| Attribute | Detail |
|-----------|--------|
| **File** | `trading/live_engine.py` |
| **Line** | ~973–979 |
| **Severity** | Critical |

```python
try:
    acct = self.broker.get_account() or {}
    if "equity" in acct:
        self.equity = float(acct["equity"])
except Exception:
    pass
```

If the broker API fails repeatedly, `self.equity` retains a stale value. All Kelly sizing downstream uses this. No warning is logged.

**Fix:** Log a warning, increment a failure counter, and halt trading if N consecutive equity fetches fail.

---

### BUG-004 — Position Reversal Doesn't Close Existing Position **(NEW)**

| Attribute | Detail |
|-----------|--------|
| **File** | `trading/live_engine.py` |
| **Line** | ~1109–1120 |
| **Severity** | High |

When reversing long→short, a single sell order is sent without first closing the existing long. `self._position` jumps directly to `-lots` ignoring the old position size. OANDA nets internally, but other brokers won't.

**Fix:** Close existing position first, then open the new direction. Or send `old_size + new_size` as the order quantity.

---

### BUG-005 — `DrawdownAwareExitManager.new_day()` Never Called **(NEW)**

| Attribute | Detail |
|-----------|--------|
| **File** | `risk/execution.py` + `trading/live_engine.py` |
| **Severity** | High |

The live engine calls `self.dae.update(equity, pnl)` every bar but never calls `new_day()` at midnight. The "daily loss limit" becomes a cumulative-from-start limit, defeating its purpose.

**Fix:** Detect day boundary in the bar loop and call `self.dae.new_day()`.

---

## Part 2: High-Impact Logic Errors

---

### BUG-006 — Kelly Division by Zero **(NEW)**

| Attribute | Detail |
|-----------|--------|
| **File** | `sizing/kelly_criterion.py` |
| **Line** | ~15–18 |
| **Severity** | High |

```python
def kelly_binary(win_prob: float, win_loss_ratio: float) -> float:
    q = 1 - win_prob
    return win_prob - q / win_loss_ratio
```

If `win_loss_ratio == 0` (no winning trades in lookback), this crashes with `ZeroDivisionError`. Exposed via the `/kelly_sizing` API endpoint.

**Fix:** Guard with `if win_loss_ratio <= 0: return 0.0`.

---

### BUG-007 — GPU Backtester Look-Ahead Bias **(NEW)**

| Attribute | Detail |
|-----------|--------|
| **File** | `backtesting/gpu_backtester.py` |
| **Line** | ~38–39 |
| **Severity** | High |

Comment says "shift signals by 1" but code does `d_positions = d_signals[:-1]` which only truncates — no actual shift. Signal from bar `i` is applied to the return from bar `i` to `i+1`, using the same close price the signal was computed from.

**Fix:** `d_positions = d_signals[:-1]` should become a proper lag: multiply `d_returns[1:]` by `d_signals[:-1]`.

---

### BUG-008 — RL Reward Normalizer Doesn't Subtract Mean **(NEW)**

| Attribute | Detail |
|-----------|--------|
| **File** | `models/rl_agents.py` |
| **Line** | ~627–634 |
| **Severity** | High |

```python
return float(np.clip(reward / std, -5.0, 5.0))
```

Divides raw reward by std without subtracting the mean. If mean reward drifts positive, the output isn't centered, breaking the intended ~N(0,1) normalization.

**Fix:** `return float(np.clip((reward - self.mean) / std, -5.0, 5.0))`

---

### BUG-009 — Backtest Commission Double-Count on Partial Closes **(CONFIRMS SYS-001 area — new specific mechanism)**

| Attribute | Detail |
|-----------|--------|
| **File** | `backtesting/backtest.py` |
| **Line** | ~273–276 |
| **Severity** | Medium |

On each partial close, `pnl_usd = gross_pnl_usd - commission` subtracts the TOTAL accumulated commission (including prior legs) from only the current partial's gross P&L. Makes backtests appear worse than reality.

**Fix:** Track commission per-leg separately, or only subtract the incremental commission.

---

### BUG-010 — Drift Detection Uses Random Labels **(NEW)**

| Attribute | Detail |
|-----------|--------|
| **File** | `trading/live_engine.py` |
| **Line** | ~1199–1202 |
| **Severity** | Medium |

```python
self.drift.fit_baseline(X, np.random.normal(0.001, 0.01, len(X)))
result = self.drift.check(X[-500:], np.random.normal(0, 0.01, 500))
```

Random noise is passed as labels. Concept drift (target shift) is undetectable — only covariate shift works.

**Fix:** Pass actual model predictions or trade outcomes as labels.

---

### BUG-011 — VaR Zero-Pads Short Return Histories **(NEW)**

| Attribute | Detail |
|-----------|--------|
| **File** | `risk/execution.py` |
| **Line** | ~232–233 |
| **Severity** | Medium |

For new pairs with few returns, the fallback `[0.0]*ml` creates fake zero-return histories. Correlation matrix becomes unreliable, VaR underestimates risk.

**Fix:** Exclude pairs with insufficient history from portfolio VaR, or require a minimum observation count.

---

### BUG-012 — Replay Buffer O(N) Sampling Bottleneck **(NEW)**

| Attribute | Detail |
|-----------|--------|
| **File** | `models/rl_agents.py` |
| **Line** | ~499–513 |
| **Severity** | Medium (Performance) |

With `buf_size=1_000_000`, the Python loop iterates ALL elements to build probability weights on every `sample()` call (every training step).

**Fix:** Maintain a running weight array updated on `push()`, or use segment trees for O(log N) prioritized sampling.

---

## Part 3: Configuration & Wiring Disconnections

---

### WIRE-001 — `label_quality` Features Never Computed **(NEW — CRITICAL)**

| Attribute | Detail |
|-----------|--------|
| **Config** | `config/run.yaml` lines 104–112, `config/feature_mask.py` lines 96–100 |
| **Severity** | Critical |

Five features are referenced in config and feature masks but have **no computation logic anywhere**:
- `rolling_hit_rate_240`
- `rolling_label_sharpe_240`
- `recent_loss_cluster_30`
- `label_confidence_prior_240`
- `rolling_false_breakout_rate_240`

These will always be NaN/zero at runtime — the model trains on empty columns.

**Fix:** Either implement the computations in `features/feature_engineering_pl.py` or remove from config/mask.

---

### WIRE-002 — `dim_feedforward` Config Silently Ignored **(NEW)**

| Attribute | Detail |
|-----------|--------|
| **Config** | `config/models.py` line 56 |
| **Training** | `training/train_gpu.py` line 7285 expects `args.dim_ff` |
| **Severity** | Moderate |

The `_normalize_architecture_profile()` function maps generic fields but never translates `dim_feedforward` → `dim_ff`. The configured value is dead — model uses fallback `d_model * 2`.

**Fix:** Add `dim_feedforward` → `dim_ff` mapping in the normalization function.

---

### WIRE-003 — Kafka/TimescaleDB Infrastructure Never Integrated **(CONFIRMS INF scaffolding findings)**

| Attribute | Detail |
|-----------|--------|
| **File** | `infrastructure/timescale_kafka.py` |
| **Severity** | Moderate |

`TimescaleDBStore` and `KafkaTickConsumer` are fully implemented but never imported or called from `trading/live_engine.py` or anywhere in the trading pipeline. The live engine uses its own broker interfaces for data.

**Status:** Dead scaffolding. Remove or wire in as a tick source option.

---

### WIRE-004 — `data_cache` Path is Windows-Only **(NEW)**

| Attribute | Detail |
|-----------|--------|
| **File** | `config/run.yaml` line 374 |
| **Severity** | Moderate |

```yaml
data_cache: D:/forex_scaling_model/data/processed
```

Invalid on the current Linux system. Should be a relative path or environment variable.

**Fix:** Use `data_cache: ./data/processed` or `${DATA_CACHE_DIR}`.

---

### WIRE-005 — Duplicate Discord Notifier

| Attribute | Detail |
|-----------|--------|
| **Files** | `infrastructure/discord_notifier.py` vs `monitoring/discord_alerts.py` |
| **Severity** | Low |

Two independent Discord notification implementations. Only `monitoring/discord_alerts.py` (`DiscordAlerter`) is actually used by the live engine and training. The infrastructure version is dead.

**Fix:** Remove `infrastructure/discord_notifier.py`.

---

### WIRE-006 — Drift Detection Not Auto-Scheduled

| Attribute | Detail |
|-----------|--------|
| **Config** | `config/run.yaml` → `drift_detection.enabled: true` |
| **Reality** | Only triggered via `--drift-gate` flag or retraining orchestrator |
| **Severity** | Low |

The config implies automatic drift detection, but it only runs when explicitly requested.

---

### WIRE-007 — Pipeline Script Only Covers Download + Train

| Attribute | Detail |
|-----------|--------|
| **File** | `scripts/run_pipeline.py` |
| **Severity** | Low |

Only dispatches `download`, `train`, and `all`. Feature engineering, ONNX export, backtesting, and RL training are not orchestrated — must be run manually.

---

### WIRE-008 — `finbert_proj_dim` Config Key Doesn't Exist

| Attribute | Detail |
|-----------|--------|
| **File** | `config/feature_mask.py` line 119 (comment reference) |
| **Severity** | Low |

Comment references `SENTIMENT["finbert_proj_dim"]` from `settings.py` but no such key exists. The 8-dim hardcode works but is undocumented.

---

### WIRE-009 — `test_dashboard.py` Referenced but Missing **(NEW)**

| Attribute | Detail |
|-----------|--------|
| **File** | `run_e2e_tests.py` |
| **Severity** | Low |

The E2E test runner references `test_dashboard.py` which does not exist in the `tests/` directory. The test suite will skip or error on this.

---

## Part 4: Dead Code & Disconnected Modules

*(See also `DEAD_CODE_REPORT.md` for the prior 315-finding static analysis.)*

### 4A. Dead Root-Level Scripts (19 files — never imported or called)

These are one-off utility scripts sitting at the repo root that no pipeline, test, or module references:

| File | Type |
|------|------|
| `fix_imports.py` | One-off code fixer |
| `fix_remaining.py` | One-off code fixer |
| `add_features.py` | One-off modifier |
| `add_micro_features.py` | One-off modifier |
| `add_validation.py` | One-off modifier |
| `edit_config.py` | One-off modifier |
| `edit_feature_groups.py` | One-off modifier |
| `update_path.py` | One-off modifier |
| `insert_features.py` | One-off modifier |
| `recovered_patch.py` | One-off patch |
| `ast_check.py` | One-off checker |
| `find_section.py` | One-off tool |
| `find_section2.py` | One-off tool |
| `find_insert.py` | One-off tool |
| `find_insert2.py` | One-off tool |
| `test_week4.py` | Standalone test (not in pytest suite) |
| `test_week4_validation.py` | Standalone test (not in pytest suite) |
| `count_ticks.py` | Duplicated as `scripts/count_ticks.py` |
| `best_date.py` | Duplicated as `scripts/best_date.py` |

**Action:** Move to a `_scratch/` archive directory or delete.

---

### 4B. Dead Package: `sizing/`

The entire `sizing/` package (`kelly_criterion.py` + `__init__.py`) is **never imported by anything in the codebase**. The API (`api/main.py`) imports `PositionSizer` and `vol_target_scalar` from `sizing.kelly_criterion` — but per the wiring audit, these symbols exist and work. However, no other module in the trading pipeline, training, or backtesting imports from `sizing/`.

**Note:** The `api/main.py` route DOES use it for the HTTP endpoint, so it's not fully dead — but the live trading engine (`trading/live_engine.py`) has its own inline Kelly implementation and does NOT use this package.

---

### 4C. Broken Symbol Imports (Guarded but Dead)

| File | Line | Import | Issue |
|------|------|--------|-------|
| `training/train_gpu.py` | ~11188 | `from monitoring.demotion_monitor import PROD_CHECKPOINT` | Symbol doesn't exist in target module |
| `training/train_gpu.py` | ~14220 | `from monitoring.demotion_monitor import PROD_CHECKPOINT, PREV_CHECKPOINT` | Both symbols undefined |

Both are wrapped in `try/except` so they won't crash, but the feature they support is non-functional.

---

### 4D. UTF-8 BOM Issues (Potential SyntaxError)

These files have a UTF-8 BOM that can cause `ast.parse()` and some Python interpreters to fail:

- `models/xgboost_model.py` (also has mixed CRLF/LF line endings)
- `tests/test_model_diagnostics.py`
- `training/train_catboost.py`
- `training/train_xgboost.py`
- `validation/model_diagnostics.py`

**Fix:** Strip BOM with `sed -i '1s/^\xEF\xBB\xBF//' <file>` and normalize line endings.

---

### 4E. Major Unused Classes (~50 classes never referenced outside their file)

| Module | Dead Classes | Notes |
|--------|-------------|-------|
| `backtesting/gpu_backtester.py` | `GPUBacktester` | Entire module unused |
| `data/feature_materializers.py` | `BaseMaterializer`, `DefaultMaterializer`, `MacroMaterializer` | Whole framework unused |
| `data/feature_schema.py` | `FeatureSchemaEnforcer` | Schema enforcement unused |
| `execution/broker_bridge.py` | `BrokerBridge` | Execution layer unused |
| `execution/order_manager.py` | `OrderManager` | Execution layer unused |
| `features/audio_sentiment.py` | `AudioSentimentPipeline` | Entire audio module unused |
| `infrastructure/timescale_kafka.py` | `KafkaTickConsumer`, `TimescaleDBStore` | Kafka/Timescale unused |
| `infrastructure/deployment.py` | `SHAPMonitor`, `ShadowModeManager` | Deployment helpers unused |
| `pretrain/guardrails.py` | `PretrainGuardrails` | Pretrain support unused |
| `pretrain/handoff_logic.py` | `PretrainHandoffGate` | Pretrain support unused |
| `pretrain/hard_example_mining.py` | `PretrainHardExampleMiner` | Pretrain support unused |
| `risk/portfolio_allocator.py` | `PortfolioAllocator` | Portfolio allocation unused |
| `trading/live_engine.py` | `BrokerInterface`, `LMAXBroker`, `LiveTickBuffer`, `MultiPairLiveTradingEngine` | Defined but never instantiated externally |
| `training/training_controller.py` | `TrainingController` | Training controller unused |

---

### 4F. Duplicate Export in `features/__init__.py`

`compute_quality_report` appears twice in `__all__` — mapped to both `features.feature_engineering_pl` and `features.quality`. The version in `feature_engineering_pl` is silently shadowed.

---

## Priority Fix Order

| Priority | IDs | Rationale |
|----------|-----|-----------|
| **P0 — Fix Now** | BUG-001, BUG-002, BUG-003, BUG-005 | Active live trading risk: wrong sizing, permanent halts, position desync |
| **P1 — Fix Before Next Training Run** | WIRE-001, BUG-007, BUG-008 | Training on empty features, inflated backtest metrics, broken RL normalization |
| **P2 — Fix Soon** | BUG-004, BUG-006, BUG-009, BUG-010, BUG-011, WIRE-002 | Correctness issues in risk/backtest/config |
| **P3 — Cleanup** | BUG-012, WIRE-003–009 | Performance, dead scaffolding, config hygiene |
| **P4 — Hygiene** | Dead scripts (4A), BOM issues (4D), unused classes (4E) | Remove dead code, fix encoding |

---

## Appendix: Files Most Needing Attention

| File | Issue Count | Notes |
|------|-------------|-------|
| `trading/live_engine.py` | 7 | Core live trading — most critical fixes needed here |
| `risk/execution.py` | 2 | VaR + daily loss limit |
| `models/rl_agents.py` | 2 | Reward normalization + replay buffer |
| `backtesting/backtest.py` | 1 | Commission double-count |
| `backtesting/gpu_backtester.py` | 1 | Look-ahead bias |
| `sizing/kelly_criterion.py` | 1 | Division by zero |
| `config/run.yaml` | 2 | Dead features + Windows path |
| `config/models.py` | 1 | Ignored param |
| `infrastructure/timescale_kafka.py` | 1 | Dead scaffolding |
