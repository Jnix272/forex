# Systems Deep Audit Report
**Date:** 2026-08-01  
**Project:** `/run/media/jamie/jamie/forex-main`  
**Analyst:** Antigravity Deep Systems Audit  
**Areas Audited:** 8 — Sharpe Inflation, Heuristic Auto-Tune, Training Control & Memory, Training Control Report, Multitask Calibration, Adaptive Curriculum, Economic Prior & News Features, Config Preflight Validation  
**Total Issues Found:** 8

---

## Executive Summary

The audit reveals a mix of critical correctness bugs and significant data leakage patterns baked into the training and evaluation loop. The most severe finding is that the **Sharpe ratio is structurally inflated** by computing returns only at trade close events rather than mark-to-market — this affects every backtest and evaluation gate in the system. Additionally, the validation set is being reused for three separate purposes simultaneously (early stopping, hyperparameter tuning, and temperature calibration), creating a systematic leak that makes all three results overly optimistic.

---

## 🚨 Critical Issues (1)

---

### SYS-001 — Sharpe Ratio is Structurally Inflated

| Attribute | Detail |
|-----------|--------|
| **File** | `backtesting/backtest.py` |
| **Line** | 441 |
| **Area** | Backtesting / Evaluation |
| **Severity** | 🚨 Critical |

**Description:**  
Returns are calculated as:
```python
returns = self.results_df["equity"].pct_change().dropna()
sharpe = (returns.mean() / returns.std(ddof=1)) * ann_factor
```
The `equity` column only updates **when a trade is closed**. During open trade holding periods, equity is flat — zero variance rows. When these flat rows are included in `pct_change()`, the standard deviation of returns is artificially deflated toward zero, which causes the Sharpe denominator to be much smaller than it should be and the Sharpe ratio to be massively inflated.

**Additionally:** The risk-free rate is omitted from the Sharpe calculation (excess return should be `mean - rf_rate`).

**Example of Impact:**  
A strategy that holds a position for 3 days will show 3 days of 0% daily return before the close bar. Those zeros shrink the std significantly. A realistic Sharpe of 0.8 could easily appear as 2.5+ in this formulation.

**Recommended Fix:**
```python
# In backtesting/backtest.py line 441 — replace:
returns = self.results_df["equity"].pct_change().dropna()

# With mark-to-market equity (updates every bar, not just on trade close):
returns = self.results_df["total_value"].pct_change().dropna()

# Also apply risk-free rate:
ANNUAL_RF = 0.05  # or load from config
bar_rf = ANNUAL_RF / ann_factor
excess_returns = returns - bar_rf
sharpe = (excess_returns.mean() / excess_returns.std(ddof=1)) * ann_factor
```
Ensure `total_value` is computed every bar as `cash + open_position_unrealised_pnl`.

---

## 🔴 High Priority Issues (3)

---

### SYS-002 — Heuristic Auto-Tune Leaks Validation Data into Hyperparameter Decisions

| Attribute | Detail |
|-----------|--------|
| **File** | `training/train_gpu.py` (L12585–12720) |
| | `scripts/optuna_tune.py` |
| **Area** | Hyperparameter Tuning |
| **Severity** | 🔴 High |

**Description:**  
`_auto_tune_next_run` uses `history["val_sharpe"]` and `history["val_loss"]` to drive hyperparameter decisions (adjusting `dropout`, `seq_len`, learning rate, etc.). However, this is the **exact same validation dataset** used for early stopping. The same held-out fold is simultaneously:
1. Stopping training early (early stopping gate)
2. Deciding the next hyperparameter configuration (auto-tune gate)

This means hyperparameters are fitted to the validation set — the same data leakage pattern as training on the test set.

**Additionally:** The tuned parameters overwrite `config/run.yaml` in place with no versioning — there is no record of what was changed between runs.

**Recommended Fix:**
1. Create a **three-way split**: `train` → `val` (early stopping only) → `tune_eval` (auto-tune decisions only, never seen during training).
2. Generate versioned config files on each auto-tune cycle: `run_[timestamp]_trial_[N].yaml` instead of overwriting `run.yaml`.
3. In Optuna, set `sampler` direction based on the `tune_eval` fold metrics only.

---

### SYS-003 — Temperature Calibration Uses Same Data as Early Stopping

| Attribute | Detail |
|-----------|--------|
| **File** | `models/architectures.py` (L1047–1130) |
| **Area** | Multitask Calibration |
| **Severity** | 🔴 High |

**Description:**  
`TemperatureScaler` calibrates the model's confidence logits using `val_loader` — the same validation set used for early stopping. This means the temperature parameter `T` is optimised to look good on already-seen data, making the model appear better calibrated than it actually is on true out-of-sample data.

**Additionally:** `TemperatureScaler.forward` unpacks the multi-task output tuple, scales only `logits[0]`, and returns a **single tensor** instead of the expected `(scaled_logits, y_ret, y_conf)` tuple. Any downstream code expecting the full tuple will break silently.

**Recommended Fix:**
```python
# Fix 1 — Use a dedicated calibration set (never used in training or early stopping)
cal_loader = DataLoader(cal_dataset, ...)  # separate held-out fold
scaler.fit(cal_loader)


# Fix 2 — Fix forward() to preserve the full output tuple
def forward(self, logits):
    direction_logits, ret_pred, conf_pred = logits  # unpack
    scaled_direction = direction_logits / self.temperature
    return (scaled_direction, ret_pred, conf_pred)  # repack full tuple
```

---

### SYS-004 — Config Preflight Validation Ignores All Risk Parameters

| Attribute | Detail |
|-----------|--------|
| **File** | `training/config_validate.py` (L228–260) |
| | `config/settings.py` (L726–755) |
| **Area** | Config Preflight Validation |
| **Severity** | 🔴 High |

**Description:**  
`config_validate.py` only validates basic training arguments (`epochs`, `batch_size`, etc.). It entirely skips validation of:
- `LIVE_RISK["kelly_fraction"]` — should be > 0 and < 1.0
- `LIVE_RISK["max_drawdown_pct"]` — should be > 0 and < 100
- `SIZING["risk_pct"]` — should be > 0 and < 10
- `TRAINING["learning_rate"]` — should be > 0
- `TRAINING["grad_clip"]` — should be > 0

Missing or out-of-range values accessed via `.get()` fall back to silent `None` defaults that can cause downstream crashes or catastrophic position sizing.

**Recommended Fix:**  
Implement a Pydantic-based validation layer that runs on application startup:
```python
from pydantic import BaseModel, validator, Field


class LiveRiskConfig(BaseModel):
    kelly_fraction: float = Field(..., gt=0.0, lt=1.0)
    max_drawdown_pct: float = Field(..., gt=0.0, lt=100.0)
    max_position_pct: float = Field(..., gt=0.0, lt=100.0)

    @validator("kelly_fraction")
    def kelly_not_too_aggressive(cls, v):
        if v > 0.25:
            raise ValueError("kelly_fraction > 0.25 is dangerously aggressive")
        return v
```
Run `LiveRiskConfig(**settings.LIVE_RISK)` at startup before any training or live trading begins.

---

## 🟡 Medium Priority Issues (3)

---

### SYS-005 — Adaptive Curriculum Uses Validation Sharpe as Progression Gate

| Attribute | Detail |
|-----------|--------|
| **File** | `training/train_gpu.py` (L3732–3810) |
| | `training/curriculum_controller.py` (L131–180) |
| **Area** | Adaptive Curriculum |
| **Severity** | 🟡 Medium |

**Description:**  
Curriculum progression (how fast difficulty increases) is gated by `val_sharpe` stabilising. This means the curriculum schedule — which shapes what training examples the model sees — is implicitly optimised against the validation set. The curriculum becomes a third consumer of the same validation fold, alongside early stopping and auto-tuning.

**Additionally:** In `_compute_difficulty_scores`, the spread signal uses:
```python
pd.Series(spr).rolling(120, min_periods=10).median().bfill()
```
The `.bfill()` propagates the rolling median **backwards**, leaking future spread volatility into the first 120 bars of each training chunk.

**Recommended Fix:**
1. Remove `.bfill()` — replace with `.fillna(method='ffill').fillna(0)` to avoid look-ahead.
2. Gate curriculum progression on **training loss plateaus** (no validation data needed) or a fully separate OOS gating set.

---

### SYS-006 — Economic Surprise Feature Has 1-Bar Look-Ahead Bias

| Attribute | Detail |
|-----------|--------|
| **File** | `data/economic_calendar.py` (L275–305) |
| **Area** | Economic Prior & News Category Features |
| **Severity** | 🟡 Medium |

**Description:**  
The economic surprise vector (`eco_surprise_norm`) is assigned to the bar at the exact timestamp of the economic release (e.g., 08:30:00 for NFP). This implies the model has instantaneous, zero-latency access to the actual figure — as if the code parses the release in the same microsecond it is published.

In reality, there is always at least 1 bar of latency before the surprise can be acted upon (data feed parsing, order routing, etc.).

**Impact:**  
The model learns to trade on the surprise value at the exact release bar, which is impossible in live trading. This inflates performance metrics on news-driven events.

**Recommended Fix:**
```python
# Shift surprise assignment forward by 1 bar
eco_surprise_norm = eco_surprise_norm.shift(1).fillna(0.0)
```
This ensures the model only sees the surprise value starting from the **next bar** after the release.

---

### SYS-007 — Curriculum .bfill() Leaks Future Spread Into Difficulty Scores

*(This is the second issue in SYS-005, extracted for clarity.)*

| Attribute | Detail |
|-----------|--------|
| **File** | `training/train_gpu.py` (L3760 approx) |
| **Area** | Adaptive Curriculum |
| **Severity** | 🟡 Medium |

```python
# Current (leaks future):
pd.Series(spr).rolling(120, min_periods=10).median().bfill()

# Fix (no look-ahead):
pd.Series(spr).rolling(120, min_periods=10).median().ffill().fillna(0.0)
```

---

## 🟢 Low Priority Issues (1)

---

### SYS-008 — Training Epoch NaN Batch Handling Across Accumulation Boundaries

| Attribute | Detail |
|-----------|--------|
| **File** | `training/train_gpu.py` (L7940–8080) |
| **Area** | Training Control & Memory |
| **Severity** | 🟢 Low |

**Description:**  
AMP scaler and gradient clipping are implemented correctly (`unscale_` before `clip_grad_norm_`). Memory management is sound. However, when a NaN batch is encountered mid-accumulation window (e.g., step 2 of a 4-step accumulation), the current code may skip the entire accumulation window but leave partial gradients in the computation graph from the valid prior steps. This is benign in practice but could cause subtle instability.

**Recommended Fix:**  
On NaN detection, explicitly call `opt.zero_grad(set_to_none=True)` before skipping to the next accumulation window, ensuring no partial gradients carry over:
```python
if torch.isnan(loss):
    opt.zero_grad(set_to_none=True)  # flush partial accum
    scaler.update()
    continue
```

---

## Summary Table

| # | Severity | Area | Issue | File |
|---|----------|------|-------|------|
| SYS-001 | 🚨 Critical | Backtesting | Sharpe inflated by closed-trade-only equity | `backtesting/backtest.py:441` |
| SYS-002 | 🔴 High | Auto-Tune | Val set reused for hyperparameter decisions | `training/train_gpu.py:12585` |
| SYS-003 | 🔴 High | Calibration | Temperature scaling uses same val as early stop | `models/architectures.py:1047` |
| SYS-004 | 🔴 High | Config | Preflight skips all risk parameter validation | `training/config_validate.py:228` |
| SYS-005 | 🟡 Medium | Curriculum | Val Sharpe gates curriculum progression | `training/curriculum_controller.py:131` |
| SYS-006 | 🟡 Medium | News Features | Economic surprise has 1-bar look-ahead | `data/economic_calendar.py:275` |
| SYS-007 | 🟡 Medium | Curriculum | .bfill() leaks future spread into difficulty | `training/train_gpu.py:3760` |
| SYS-008 | 🟢 Low | Training | NaN batch mid-accumulation leaves partial grads | `training/train_gpu.py:7940` |

---

## Recommended Fix Order

| Priority | Fix | Effort |
|----------|-----|--------|
| 1 | SYS-001: Switch equity to mark-to-market for Sharpe | Low |
| 2 | SYS-006: Shift economic surprise by 1 bar | Low |
| 3 | SYS-007: Replace .bfill() with .ffill() in curriculum | Low |
| 4 | SYS-003: Fix TemperatureScaler tuple unpacking | Low |
| 5 | SYS-004: Add Pydantic risk config validation | Medium |
| 6 | SYS-002: Separate val/tune/calibration sets | High |
| 7 | SYS-005: Gate curriculum on train loss not val Sharpe | Medium |
| 8 | SYS-008: Zero grad on NaN mid-accumulation | Low |
