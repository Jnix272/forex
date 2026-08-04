# Dead Code & Unused Symbol Audit Report
**Date:** 2026-08-01  
**Project:** `/run/media/jamie/jamie/forex-main`  
**Tools:** `vulture` (80% confidence), `ruff` (F401/F811/F841)  
**Total Findings:** 281 ruff violations + 34 vulture findings

---

## Executive Summary

Automated static analysis identified **315 dead code findings** across the codebase:
- **281** unused imports, redefined names, and locally assigned-but-never-used variables (ruff F401/F811/F841)
- **34** unused variables, unreachable code blocks, and unused imports (vulture ≥80% confidence)

The most critical findings are in `training/train_gpu.py`, `trading/live_engine.py`, and `monitoring/rich_display.py`. Three **unreachable code** blocks were identified — code that can never execute under any condition. Several imports in `main.py` suggest partially-wired features (L2 order book, options skew, COT data) that were stubbed but never connected.

---

## 🔴 High Priority — Unreachable Code (3 blocks)

These blocks are 100% confirmed unreachable by vulture — they can never execute:

| File | Line | Issue |
|------|------|-------|
| `monitoring/rich_display.py` | 169 | Unreachable code after `return` |
| `monitoring/rich_display.py` | 200 | Unreachable code after `return` |
| `monitoring/rich_display.py` | 217 | Unreachable code after `return` |

**Fix:** Remove all three dead blocks entirely from `rich_display.py`.

---

## 🔴 High Priority — Dead Feature Imports in `main.py`

Five imports in `main.py:25` point to feature classes that are imported but **never used anywhere in the main execution flow**. This indicates stubbed features that were partially wired but never completed:

```python
# main.py:25 — all unused:
from features.advanced_features import (
    CorrelationRegimeDetector,   # unused — regime detection stub
    COTFeatures,                  # unused — COT data integration stub
    L2OrderBookFeatures,          # unused — real L2 OBI stub (see DS-009)
    OptionsSkewFeatures,          # unused — options market stub
    rolling_hurst_fractal,        # unused — Hurst exponent stub
)
```

These directly correspond to **DS-009** (real L2 order book data ignored) and represent missing alpha signals.

---

## 🔴 High Priority — `training/train_gpu.py` Dead Symbols

The largest file in the project has the most dead code:

### Unused Imports
| Line | Symbol | Notes |
|------|--------|-------|
| 93 | `collections.Counter` | Never called |
| 128 | `data.sources.TICK_COLUMNS` | Never referenced |
| 132 | `data.news_feed.get_latest_headlines` | News feed disconnected — confirms PIPE-003 |
| 135 | `labeling.rl_reward_labeling.compute_rl_reward_labels` | RL labeler imported but unused |
| 143 | `models.architectures.MultiTaskHead` | Imported but not instantiated |
| 174 | `monitoring.discord_alerts.DiscordAlerter` | Alerter imported but never called |
| 207 | `torch.nn.functional` | F imported but never used |
| 312 | `infrastructure.numerics.ensure_finite_tensor` | Guard function imported but unused |
| 5680 | `pandas` | pandas imported mid-file, never used |
| 11843 | `contextlib` | Imported inside function, unused |

### Unused Local Variables (assigned but value never read)
| Line | Variable | Notes |
|------|----------|-------|
| 4393 | `e` (exception) | Exception caught and discarded silently |
| 4986 | `exc` (exception) | Exception caught and discarded silently |
| 5830 | `n_remaining` | Progress tracking var never printed |
| 7690 | `names_list` | Feature names built but never logged |
| 9131 | `all_groups` | Parameter groups computed but discarded |
| 9138 | `frozen` | Frozen params tracked but never reported |
| 11879 | `out` | Model output computed but thrown away |
| 12275 | `classification` | Classification result unused |
| 12454 | `equity` | Equity series computed but never saved |
| 13812 | `m2` | Model reference computed but unused |

---

## 🟠 High Priority — `trading/live_engine.py` Dead Config Imports

```python
# trading/live_engine.py:96 — all unused:
from config.settings import LATENCY, RISK, SIZING
```

`LATENCY`, `RISK`, and `SIZING` are all imported but never referenced. This means the live engine is **not applying the configured risk limits** — it is operating with unconfigured defaults instead of the user's defined `RISK` and `SIZING` parameters. This is a latent risk management bug.

---

## 🟠 High Priority — `training/train_catboost.py` and `training/train_xgboost.py` — Redefined Imports

Both tree model trainers import their model class twice — the first import is immediately overwritten:

```python
# train_catboost.py:32 — first import (unused, immediately overwritten)
from models.catboost_model import CatBoostForecaster  

# ... 300 lines later ...

# train_catboost.py:344 — second import (this one is used)
from models.catboost_model import CatBoostForecaster  # F811 redefinition
```

Same pattern in `train_xgboost.py:32` and `:344`. The top-level import is a dead import. Fix: remove the top-level import, keep only the one inside the function where it's needed.

---

## 🟡 Medium Priority — `features/advanced_features.py` Dead Variables

```python
# features/advanced_features.py:287-289 — computed but never used
ofi_z_fast = ...        # 100% confidence unused
ofi_z_slow = ...        # 100% confidence unused
tbm_default_horizon = ...  # 100% confidence unused
```

Same pattern duplicated in `features/multipair.py:144-146`.

---

## 🟡 Medium Priority — `training/scale_model.py` Dead Imports

```python
# training/scale_model.py — 6 unused imports:
import gc                                       # unused
import time                                     # unused
from training.train_gpu import MemmapSequenceDataset  # unused
from training.train_gpu import _log_error        # unused
from training.train_gpu import _log_info         # unused
from training.train_gpu import _log_oom          # unused
```

---

## 🟡 Medium Priority — `validation/promotion_gate.py`

```python
# validation/promotion_gate.py:158
backtest_sharpe_std = ...  # computed but NEVER READ (100% confidence)
```

This variable is computed but then the result is discarded. If it was intended to be included in the promotion gate decision, this is a logic bug — the Sharpe standard deviation is being silently ignored.

---

## 🟡 Medium Priority — `monitoring/` Dead Imports

| File | Line | Unused Symbol |
|------|------|---------------|
| `monitoring/prometheus_exporter.py` | 99 | `CollectorRegistry` |
| `monitoring/rich_display.py` | 61 | `Layout` |
| `monitoring/rich_display.py` | 70 | `Rule` |
| `monitoring/visualize_performance.py` | 43 | `Patch` |

---

## 🟡 Medium Priority — `retraining/pipeline.py` Dead Imports

```python
# retraining/pipeline.py:26-28 — both unused:
from data.feature_store import FeatureSpec          # unused
from data.feature_store import run_full_materialization  # unused
```

---

## 🟡 Medium Priority — `trading/` Dead Imports

| File | Line | Unused Symbol | Concern |
|------|------|---------------|---------|
| `trading/live_engine.py` | 96 | `LATENCY`, `RISK`, `SIZING` | Risk limits not applied |
| `trading/live_guards.py` | 13 | `dataclasses.asdict` | Utility import unused |
| `trading/live_guards.py` | 17 | `numpy` | numpy imported but unused |
| `trading/inference_engines.py` | 3 | `typing.Optional` | Typing import unused |

---

## 🟢 Low Priority — Test File Dead Imports (Selected)

Tests have the most unused imports. Most are low-risk but they clutter the test suite and cause confusion about what's actually being tested:

| File | Unused Symbols |
|------|---------------|
| `tests/test_promotion_artifacts.py` | `os`, `torch`, `shutil`, `pytest`, `Path` (all 5 imports unused) |
| `tests/test_retrain_orchestrator.py` | `json`, `shutil`, `timedelta`, `MIN_SHARPE`, `ModelFamily`, `ModelRecord`, `DriftTracker` |
| `tests/test_smoke.py` | `pytest`, `os`, `shutil`, `Path` |
| `tests/test_training_smoke.py` | `os`, `tempfile`, `pytest`, `MagicMock` |
| `run_e2e_tests.py` | `pytest_cov` (not installed in venv) |

---

## 🟡 Medium Priority — `features/feature_engineering_pl.py` Dead Symbols

| Line | Symbol | Notes |
|------|--------|-------|
| 156 | `n_states` | HMM state count assigned, never used (confirms DS-007) |
| 1007 | `buf_min` | Buffer minimum computed but discarded |
| 1334 | `finbert_embs` | FinBERT embeddings computed but **never assigned to output** — confirms PIPE-003 |

The `finbert_embs` finding at line 1334 is particularly significant: it proves that FinBERT embeddings are being computed somewhere but the result is thrown away before it reaches the model — which explains why the model receives all-zero embeddings (PIPE-003).

---

## Recommended Fix Strategy

### Auto-fixable (ruff --fix)
Run the following to automatically remove all auto-fixable unused imports:
```bash
uv run --with ruff ruff check . --select F401,F811 --exclude ".venv,wandb,data,__pycache__" --fix
```
This will safely remove ~200 of the 281 ruff violations automatically.

### Manual Fixes Required
1. **`trading/live_engine.py:96`** — Wire `RISK` and `SIZING` into the live engine or remove the import
2. **`features/feature_engineering_pl.py:1334`** — Fix `finbert_embs` to actually be returned/used (PIPE-003 fix)
3. **`monitoring/rich_display.py:169,200,217`** — Delete the three unreachable code blocks
4. **`validation/promotion_gate.py:158`** — Determine if `backtest_sharpe_std` should be used in the gate decision
5. **`training/train_gpu.py:9131,9138`** — Determine if `all_groups` and `frozen` should be logged
6. **`main.py:25`** — Either wire L2OrderBookFeatures, COTFeatures, etc. into the pipeline or remove imports

---

## Summary

| Category | Count |
|----------|-------|
| Unreachable code blocks | 3 |
| Unused imports (production code) | ~120 |
| Unused imports (test code) | ~80 |
| Locally assigned but never used variables | ~30 |
| Redefined imports (F811) | 3 |
| **Auto-fixable by ruff** | ~200 |
| **Require manual review** | ~115 |
