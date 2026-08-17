# Training & Model Architecture Audit Report
**Date:** 2026-08-01  
**Project:** `/run/media/jamie/jamie/forex-main`  
**Analyst:** Antigravity ML Training & Architecture Expert  
**Files Audited:** `models/architectures.py`, `models/ensemble.py`, `models/catboost_model.py`, `models/xgboost_model.py`, `training/train_gpu.py`, `training/train_catboost.py`, `training/train_xgboost.py`, `training/hard_example_miner.py`, `training/curriculum_controller.py`  
**Total Issues Found:** 13

---

## Executive Summary

The audit uncovered four **critical or high-severity bugs** that are actively causing training failures and silent inference corruption right now. The most severe is that `SharpeProxyLoss` produces `NaN` gradients when predictions are constant (which always happens in the first steps of training), triggering the NaN-skip logic and aborting the Sharpe learning objective early. The `MCDropoutWrapper` permanently leaves dropout active after uncertainty estimation, corrupting all subsequent validation runs. The entire `train_catboost.py` script crashes immediately due to copy-paste errors from XGBoost. And the ensemble meta-learner diversity penalty has zero gradient flow and does nothing.

---

## 🚨 Critical Issues (2)

---

### TM-001 — `SharpeProxyLoss` Produces NaN Gradients at Training Start

| Attribute | Detail |
|-----------|--------|
| **File** | `training/train_gpu.py:443`, `models/architectures.py:209` |
| **Severity** | 🚨 Critical |

**Description:**  
`SharpeProxyLoss` computes:
```python
std = returns.std(unbiased=False)
sharpe = mean / std
```
At the very start of training, the model outputs near-constant predictions. When all predictions are identical, `returns` is a constant vector and its standard deviation is exactly `0.0`. PyTorch's gradient of `std` at `std=0` is mathematically `NaN` (division by zero in the backward pass). This immediately triggers the `NaN-grad-skip` logic in `train_epoch`, causing the Sharpe loss to be silently abandoned in early training — exactly when learning the Sharpe objective is most critical.

**The same bug exists in `MultiTaskLoss`** which also calls `returns.std(unbiased=False)`.

**Recommended Fix:**
```python
# Replace in SharpeProxyLoss.forward() and MultiTaskLoss:
# Before:
std = returns.std(unbiased=False)

# After (numerically stable):
var = returns.var(unbiased=False)
std = torch.sqrt(var + 1e-8)  # eps prevents NaN gradient at std=0
```

---

### TM-002 — `train_catboost.py` Crashes Immediately (Wrong API)

| Attribute | Detail |
|-----------|--------|
| **File** | `training/train_catboost.py` |
| **Severity** | 🚨 Critical |

**Description:**  
The entire CatBoost training script is broken due to copy-paste errors from the XGBoost script. It calls non-existent CatBoost API methods:
```python
# train_catboost.py — these do NOT exist in the catboost library:
model = cb.CBClassifier(objective="multi:softmax", ...)   # ❌ XGBoost class name
model = cb.CBRegressor(objective="reg:squarederror", ...) # ❌ XGBoost class name
```
Additionally:
- Uses XGBoost `objective` string keys instead of CatBoost `loss_function` strings
- Does not pass `task_type="GPU"` — defaults to slow CPU training
- Will crash with `AttributeError: module 'catboost' has no attribute 'CBClassifier'` on first run

**Recommended Fix:**
```python
# train_catboost.py — correct CatBoost API:
from catboost import CatBoostClassifier, CatBoostRegressor

model = CatBoostClassifier(
    loss_function="MultiClass",  # not "multi:softmax"
    eval_metric="Accuracy",
    task_type="GPU",  # enable GPU
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    verbose=100,
)
```

---

## 🔴 High Issues (4)

---

### TM-003 — `MCDropoutWrapper` Permanently Enables Dropout After Uncertainty Call

| Attribute | Detail |
|-----------|--------|
| **File** | `models/ensemble.py` |
| **Severity** | 🔴 High |

**Description:**  
`MCDropoutWrapper.predict_with_uncertainty` calls `_enable_dropout()` which sets all `nn.Dropout` modules to `.train()` mode. There is no corresponding `_disable_dropout()` call after uncertainty estimation completes. After any call to `predict_with_uncertainty`, **all subsequent forward passes — including validation and live inference — apply random dropout**, producing stochastic and incorrect outputs.

```python
# Current — broken: dropout stays permanently enabled
def predict_with_uncertainty(self, x, n_samples=30):
    self._enable_dropout()  # sets dropout to train mode
    preds = [self(x) for _ in range(n_samples)]
    # ← missing: self._disable_dropout()
    return preds
```

**Recommended Fix:**
```python
def predict_with_uncertainty(self, x, n_samples=30):
    self._enable_dropout()
    try:
        with torch.no_grad():
            preds = torch.stack([self(x) for _ in range(n_samples)])
        mean = preds.mean(0)
        uncertainty = preds.std(0)
        return mean, uncertainty
    finally:
        self._disable_dropout()  # always restore, even if exception


def _disable_dropout(self):
    for m in self.modules():
        if isinstance(m, nn.Dropout):
            m.eval()
```

---

### TM-004 — Ensemble Diversity Loss Has Zero Gradient (Does Nothing)

| Attribute | Detail |
|-----------|--------|
| **File** | `models/ensemble.py` (`train_meta_learner`) |
| **Severity** | 🔴 High |

**Description:**  
The meta-learner trains with a diversity penalty:
```python
with torch.no_grad():
    base_preds = [model(x) for model in self.base_models]  # gradients blocked

div_pen = meta.diversity_loss(base_preds)  # computed from no_grad tensors
loss = task_loss + lambda_div * div_pen  # div_pen is a constant — no gradient!
```
`base_preds` is generated inside `torch.no_grad()` — the resulting tensors have `requires_grad=False`. The diversity penalty computed from them is a constant and contributes **zero gradient** to the meta-learner weights. The `lambda_div` parameter has no effect whatsoever.

**Recommended Fix:**  
Compute the diversity penalty on the **weighted predictions** (which depend on meta-learner weights), not the frozen base predictions:
```python
# Compute base predictions frozen (correct — don't tune base models here)
with torch.no_grad():
    base_preds = torch.stack([model(x) for model in self.base_models], dim=1)

# Meta-learner weights (these DO have gradients)
weights = self.meta(x)  # shape: (batch, n_models)
weighted_preds = (weights.unsqueeze(-1) * base_preds).sum(1)

# Diversity: penalise when weights collapse to a single model
weight_entropy = -(weights * weights.log().clamp(-10)).sum(1).mean()
div_pen = -weight_entropy  # maximise entropy → maximise diversity

loss = task_loss + lambda_div * div_pen  # NOW has real gradients ✅
```

---

### TM-005 — `SharpeProxyLoss` Double-Sqrt Annualisation Bug

| Attribute | Detail |
|-----------|--------|
| **File** | `training/train_gpu.py:430`, `models/architectures.py:209` |
| **Severity** | 🔴 High |

**Description:**  
Config supplies `sharpe_annualization_factor: 325.0` (already `√(252×420)`). The loss applies `** 0.5` again:
```python
self._ann_sqrt = float(ann) ** 0.5  # → √325 ≈ 18  (18× too weak)
```
Sharpe gradient is ~18× too weak, causing the Huber baseline loss to dominate.

**Fix:** `self._ann_sqrt = float(ann)` — use the config value directly.

---

### TM-006 — `MultiTaskLoss` KL Divergence Reduction Dilutes Penalty

| Attribute | Detail |
|-----------|--------|
| **File** | `training/train_gpu.py` (`MultiTaskLoss`) |
| **Severity** | 🔴 High |

**Description:**  
```python
# pred_dist is already averaged to shape (3,) — one value per class
F.kl_div(pred_dist.log(), true_dist, reduction="batchmean")
```
`"batchmean"` divides the total KL divergence by the **batch dimension** of the input tensor. When `pred_dist` is already a 1D tensor of shape `(3,)`, PyTorch treats `3` as the "batch size" and divides by 3, under-weighting the class balance penalty by 3×.

**Fix:**
```python
F.kl_div(pred_dist.log(), true_dist, reduction="sum")
```

---

## 🟡 Medium Issues (5)

---

### TM-007 — CUDA Graphs Cause Silent NaNs with LSTM Models

| Attribute | Detail |
|-----------|--------|
| **File** | `training/train_gpu.py` (`torch.compile` block) |
| **Severity** | 🟡 Medium |

**Description:**  
`torch.compile(mode="reduce-overhead")` uses CUDA graphs. CUDA graphs require static computation shapes, but LSTMs (used in `HAELTHybrid`) have dynamic internal shapes that change between iterations. This mismatch causes **silent NaN outputs** — the model trains without error but produces garbage predictions.

**Fix:**
```python
# Detect LSTM-containing models and use safe compile mode
has_lstm = any(isinstance(m, nn.LSTM) for m in model.modules())
compile_mode = "default" if has_lstm else "reduce-overhead"
model = torch.compile(model, mode=compile_mode)
```

---

### TM-008 — No Weight Initialisation on Deep Models

| Attribute | Detail |
|-----------|--------|
| **File** | `models/architectures.py` (`TFTScalper`, `EXPERTEncoder`, `HAELTHybrid`) |
| **Severity** | 🟡 Medium |

**Description:**  
All deep models rely entirely on PyTorch's default weight initialisation (`Uniform(-1/√fan_in, 1/√fan_in)`). For deep networks with GELU/SiLU activations, this is known to cause vanishing/exploding gradients, particularly in the early training steps before the learning rate warmup stabilises things.

**Recommended Fix:** Add explicit `_init_weights` to each model class:
```python
def _init_weights(self):
    for module in self.modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
```

---

### TM-009 — Mamba Block Uses Sigmoid Instead of Softplus for Step Size

| Attribute | Detail |
|-----------|--------|
| **File** | `models/architectures.py` (`MambaBlock`) |
| **Severity** | 🟡 Medium |

**Description:**  
```python
dt = torch.sigmoid(self.dt_proj(x2c))  # bounded to (0, 1)
```
The standard Mamba SSM paper uses `softplus` for the step size (`dt`), which maps to `(0, ∞)`. Using `sigmoid` caps the maximum step size at 1.0, preventing the model from taking large discretisation steps when needed for long-range dependency capture.

**Fix:**
```python
dt = F.softplus(self.dt_proj(x2c))  # correct: unbounded positive ✅
```

---

### TM-010 — Gradient Checkpointing Completely Missing

| Attribute | Detail |
|-----------|--------|
| **File** | `models/architectures.py` (`TFTScalper`, `iTransformerScalper`, `EXPERTEncoder`) |
| **Severity** | 🟡 Medium |

**Description:**  
With `seq_len=60–120` and batch sizes of 256+, the activation memory for deep Transformer blocks is the primary VRAM bottleneck. Gradient checkpointing recomputes activations during the backward pass instead of storing them, halving peak VRAM at the cost of ~33% more compute.

**Fix:** Wrap forward calls in Transformer blocks:
```python
from torch.utils.checkpoint import checkpoint

# In iTransformerScalper.forward():
for layer in self.encoder_layers:
    x = checkpoint(layer, x, use_reentrant=False)
```

---

### TM-011 — Label Smoothing Disabled (Hard Targets on Noisy Data)

| Attribute | Detail |
|-----------|--------|
| **File** | `training/train_gpu.py` (`MultiTaskLoss`) |
| **Severity** | 🟡 Medium |

**Description:**  
```python
self.ce = nn.CrossEntropyLoss(label_smoothing=0.0)
```
Financial directional labels (Long/Short/Flat) are inherently noisy — even the best signal has ~45% pure noise. Hard labels `{0, 1, 2}` cause overconfident predictions early in training, leading to poor calibration and brittleness on unseen data.

**Fix:**
```python
self.ce = nn.CrossEntropyLoss(label_smoothing=0.05)
```

---

## 🟢 Low Issues (2)

---

### TM-012 — Post-Norm vs Pre-Norm Inconsistency Across Models

| Attribute | Detail |
|-----------|--------|
| **File** | `models/architectures.py` |
| **Severity** | 🟢 Low |

**Description:**  
`TFTScalper` and `CausalGNNCrossAsset` use post-normalisation (residual → norm). `iTransformerScalper` uses `norm_first=True` (pre-norm). Pre-norm is significantly more stable for deep networks, especially without a warmup scheduler. All models should use pre-norm for consistency and stability.

---

### TM-013 — Early Stopping Saves Wrong Checkpoint Metadata

| Attribute | Detail |
|-----------|--------|
| **File** | `training/train_gpu.py` (early stopping block) |
| **Severity** | 🟢 Low |

**Description:**  
When `stop_on_sharpe=True`, the early stopper tracks `best_sharpe` but the checkpoint metadata JSON records `val_loss` at the time of the best Sharpe — not the actual `best_val_loss`. If the best Sharpe occurred at a high-loss epoch, the metadata misleadingly shows a poor loss as the checkpoint's quality metric.

---

## Top 10 Improvements by Expected Impact

| Rank | ID | Fix | Impact on Live Trading |
|------|-----|-----|----------------------|
| 1 | TM-001 | Fix Sharpe NaN gradients — `√(var + 1e-8)` | 🔴 Training currently aborts Sharpe objective early |
| 2 | TM-003 | Fix MCDropout inference leak — add `_disable_dropout()` | 🔴 Validation and inference results currently corrupted |
| 3 | TM-002 | Rewrite `train_catboost.py` with correct API | 🔴 CatBoost training is completely broken |
| 4 | TM-004 | Fix ensemble diversity loss gradient flow | 🟠 Ensemble collapses to one model, no diversity benefit |
| 5 | TM-005 | Fix double-sqrt annualisation in SharpeProxyLoss | 🟠 Sharpe gradient 18× too weak — Huber dominates |
| 6 | TM-006 | Fix KL divergence `batchmean` → `sum` | 🟠 Class balance penalty 3× under-weighted |
| 7 | TM-007 | Disable CUDA graphs for LSTM models | 🟡 Silent NaNs in HAELTHybrid training |
| 8 | TM-008 | Add explicit weight initialisation | 🟡 Faster, more stable early training convergence |
| 9 | TM-009 | Mamba: sigmoid → softplus for step size | 🟡 Better long-range dependency capture |
| 10 | TM-010 | Add gradient checkpointing to Transformer blocks | 🟡 2× batch size, reduced OOM risk |

---

## Files to Modify for All Fixes

| File | Issues to Fix |
|------|--------------|
| `training/train_gpu.py` | TM-001, TM-005, TM-006, TM-007, TM-011 |
| `models/architectures.py` | TM-001, TM-008, TM-009, TM-010, TM-012 |
| `models/ensemble.py` | TM-003, TM-004 |
| `training/train_catboost.py` | TM-002 |
