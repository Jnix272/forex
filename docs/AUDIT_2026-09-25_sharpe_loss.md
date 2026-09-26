# Audit: Model Loss & Sharpe Metrics (2026-09-25)

**Scope:**
- the supervised training objective: `models/architectures.MultiTaskLoss` / `MultiPairMultiTaskLoss`, built by `training/loop_losses._build_train_loss`
- how validation turns model outputs into directions, accuracy and Sharpe (`training/loop_epochs.validate_epoch`, `_non_overlapping_sharpe`, `training/honest_eval`)
- how those numbers drive checkpoint selection, early stopping, Optuna and the gate

**Method:** static review at `main` @ `674f925`, plus the saved fold histories from the model audit (A1–A3).

**Bottom line:**
- The loss the models actually minimise is not the one the config and Optuna describe.
- Most of the options passed to it are silently dropped, the cost-aware direction labels are never used, and one head is trained towards two conflicting targets.
- On the metric side, validation Sharpe measures an always-in-the-market policy against a synthetic label rather than the policy that trades live.

---

## Loss

### S1. Most loss options are silently ignored (P0)
- For multitask models `_build_train_loss` builds `MultiTaskLoss(class_weights=..., class_balance_weight=..., entropy_weight=..., direction_weight_floor=..., focal_gamma=..., class_prior=..., w_sharpe=..., sharpe_ann=..., label_smoothing=...)`.
- `MultiTaskLoss.__init__` accepts `**kwargs` and uses **none** of them. Only `w_dir`, `w_ret`, `w_conf`, `huber_delta`, `w_quantile` and `quantiles` do anything.
- **Consequences:**
  - Class weighting and class-prior correction never happened, so HOLD imbalance is untreated.
  - Focal loss and label smoothing never happened.
  - `loss: sharpe_huber` or `sharpe_weight` never reach the multitask path.
  - Optuna has been tuning `mt_focal_gamma`, `mt_class_balance_weight` and `mt_direction_weight_floor`: three dead dimensions.
- `MultiPairMultiTaskLoss` wraps the same class and inherits the problem.
- **Fix:** implement the options in the loss, or remove them from the config and Optuna. Make `**kwargs` raise on unknown keys.

### S2. The direction labels (`y_cls`) are never used (P0)
- `forward(logits, ret_hat, conf, y_cls, y_cont, ...)` never reads `y_cls`. The direction target is `(y_cont > 0)`, with 0.5 only when `y_cont == 0` exactly.
- The cost-aware CPAR labels (BUY/SELL only when that side beats the spread, otherwise HOLD) have no effect on training. The model learns the sign of the mid-price reward, which is the old, cost-blind target.
- **Fix:** train the direction head as 3-class cross-entropy on `y_cls`, or at minimum mask HOLD rows out of the BCE, with the class weights from S1.

### S3. One logit, two conflicting targets (P0)
- The direction logit gets both `Huber(logit, y_cont)` and `BCE(logit, 1[y_cont > 0])`.
- Huber pulls the logit towards the continuous reward, which ranges up to ±17 ATR units. BCE wants a large-magnitude logit with the right sign and a small one when uncertain.
- The gradients fight: large |y_cont| rows push towards big logits whatever the confidence, and near-zero rows pull the logit to 0.
- The regression target already has its own head (`ret_hat`, trained with the same Huber), so the Huber term on the direction logit is redundant and harmful.
- **Fix:** keep BCE or CE only on the direction logit.

### S4. `bet_size` / per-sample weights are never applied (P0)
- `MultiTaskLoss.forward` accepts `bet_size` and never uses it, and `MultiPairMultiTaskLoss` passes it through to the same function.
- So the period-balancing weights added on 2026-09-25 (folded into `bet_size`) and the curriculum sample weights do nothing for multitask models.
- **Fix:** reduce per-sample (`reduction="none"`), multiply by the weights, then take a weighted mean.

### S5. Heavy-tailed targets dominate the loss (P1)
- `y_cont` is the CPAR reward divided by a 30-minute ATR, with tails of about ±17 and hard clips at ±50 (dataset audit B4, E6).
- `huber_delta = 1` makes Huber linear in the tails, which helps, but the loss mean is still set by a small share of extreme rows.
- `ret_hat`, the two quantile heads and the Huber on the direction logit (S3) all regress on this tail.
- **Fix:** winsorize or rank-transform the regression target per fold, and report the loss on a clipped target.

### S6. The confidence head learns a constant (P1)
- `tgt_conf = (|y_cont| > 1e-4)` is 1 for essentially every row: CPAR rewards are almost never exactly 0.
- The BCE therefore teaches `conf → +∞`, and any later use of `conf` for sizing or gating is a constant.
- `y_conf` from the dataset's `pq` (path quality) is used only when passed in.
- **Fix:** define confidence as "the direction was right and cleared costs", or drop the head.

### S7. Validation loss isn't a meaningful selection metric (P1)
- The checkpoint is chosen on validation loss (or its SACS variant), which is the S1–S6 objective.
- Because of the tails (S5) and conflicting terms (S3), a lower validation loss does not mean better trades.
- The fold histories show exactly this: identical losses across architectures (model audit A2), with Sharpe uncorrelated to loss.
- **Fix:** select on the honest held-out net Sharpe lower confidence bound, and keep loss only as a sanity check.

---

## Sharpe

### S8. Validation measures an always-in-the-market policy (P0)
- In `validate_epoch` the multitask direction is `d = pred_cls − 1`, where `pred_cls = sign(logit) + 1` for scalar logits.
- A position is taken on every validation row, with no HOLD and no confidence threshold.
- Live trading opens only when `proba.max() ≥ hold_threshold` (0.45), the gate's backtest has its own threshold, and the honest metric trades every horizon.
- Three different policies are scored, and the reported Sharpe describes none of the live ones.
- **Fix:** one `decide(outputs) -> {-1, 0, +1}` function, with the live threshold and HOLD handling, used by validation, the gate, the baseline and live.

### S9. `dir_sharpe` and `cost_sharpe` score a synthetic label, not prices (P1)
- The label-based Sharpes multiply the direction by the CPAR reward, which is ATR-normalised, clipped and had the spread cancel until the fix.
- Cost is a flat `tx_cost_bps × 4 / 1e4` in reward units. That isn't in price units, and it's much smaller than the real spread in reward terms.
- These are what produced the ±30 "Sharpes" (model audit A3). They still appear in logs, in `history["val_sharpe"]` and in the training summary `best_val_sharpe`.
- The honest net Sharpe (`[Val][honest]`) is the only price-based one, but `val_sharpe` still drives:
  - the "Sharpe collapse" controller signals
  - the early-stop composite (`- 0.1 × v_sh`)
  - `history["val_sharpe"]`, read by the Optuna `val_sharpe` metric and the fold scorers
- **Fix:** make the honest net Sharpe the only reported and consumed Sharpe. Keep the label Sharpe under a clearly diagnostic name, or drop it.

### S10. Directional accuracy counts HOLD rows as automatic misses (P1)
- `dir_acc` compares a 2-way prediction (BUY or SELL, see S8) against 3-class `y_cls`.
- Every HOLD row (about 12% overall and 28% in the last period, dataset audit B4) counts as wrong, so accuracy is capped at the non-HOLD share.
- The reported 0.42–0.49 therefore isn't comparable to a 50% coin flip.
- **Fix:** report accuracy on non-HOLD rows, plus balanced 3-class accuracy once S2 and S8 are done.

### S11. Honest net Sharpe: remaining gaps (P1)
- **Costs:** the cost is one spread per round trip at mid prices. Commission and slippage (0.5–0.7 pips in the backtests) are excluded, which flatters scalping at 30-bar horizons.
- **Annualisation:** it uses `√(bars_per_year / h)`, assuming a possible trade every h bars, and scales by trade count rather than calendar time. With HOLDs (after S8), `periods_per_year` should be the *realised* trades per year: n_trades ÷ holdout years.
- **Pooling:** per-pair trades at the same timestamp are treated as independent, but EURUSD and GBPUSD returns are about 0.7–0.8 correlated. The pooled CI is too narrow. Use a block bootstrap over timestamps (resampling all pairs of a bar together).
- **Minimum sample:** the Sharpe is 0.0 below `min_trades=30`. With 7 folds of about 600 non-overlapping trades, the per-fold CI is roughly ±2 Sharpe, so any fold-level selection is mostly noise unless the CI lower bound is used.

### S12. Sharpe definitions in the code base (P2)

| Where | Returns | Annualisation | Status |
|---|---|---|---|
| `validate_epoch` dir/cost Sharpe | sign × CPAR label, flat bps cost | `sharpe_annualization_factor` (config; 325 until 2026-09-25) | diagnostic only (S9) |
| `honest_eval.net_pnl_metrics` | price returns − spread, non-overlapping | √(bpy/h) | primary; gaps S11 |
| `backtest.ForexScalingBacktest` | per-bar equity returns | √bars_per_year | OK |
| `rl_agents` env | per-step PnL net of costs (fixed today) | √bars_per_year | OK after fix |
| `PromotionGate` PSR/DSR | from any of the above | periods_per_year (fixed today) | depends on the input |
| Optuna `val_sharpe` metric | max over epochs of the label Sharpe | as above | should use honest (S9) |

---

## Recommended order

1. **S1–S4 (the loss):**
   - direction as CE on `y_cls` with class weights
   - no Huber on the direction logit
   - apply per-sample weights
   - make unknown loss options raise
   - remove the dead Optuna dimensions
2. **S8:** a single decision function shared by validation, the gate, the baseline and live.
3. **S9 + S7:** honest net Sharpe (CI lower bound) as the only selection and early-stop signal. Rename or drop the label Sharpes.
4. **S5, S6, S10, S11:** robust targets, a meaningful confidence target, HOLD-aware accuracy, full costs, realised-trade annualisation, and a timestamp-block bootstrap.

Items 1–3 change what every model learns and how it's selected, so do them before the post-rebuild retrain.

---

## Fix log

| # | Fix |
|---|---|
| S1 | `MultiTaskLoss` implements `class_weights`, `focal_gamma`, `label_smoothing`, `class_balance_weight` (penalty on mean P(buy) − 0.5) and `w_sharpe`. Unknown options raise `TypeError`. `direction_weight_floor`, `entropy_weight` and `class_prior` were removed from the caller, and `mt_direction_weight_floor` from Optuna. |
| S2 | Direction is a BCE on `y_cls` (BUY vs SELL) over tradable rows; HOLD rows are masked. With no `y_cls`, it falls back to the sign of the reward. |
| S3 | No Huber on the direction logit, in either full training or direction warm-up. |
| S4 | Every term is a per-sample weighted mean using `bet_size` (period balance and curriculum weights now apply). |
| S5 | The regression target is winsorised to ±`training.regression_target_clip` (default 5). |
| S6 | The confidence head predicts tradability (`y_cls != HOLD`). |
| S7 | With the honest metric available (≥ 30 trades), the checkpoint is selected on −(95% CI lower bound of the net Sharpe), and early stopping uses the same score. |
| S8 | `training/decision.py`: `action_proba` / `decide` form [SELL, HOLD, BUY] = [(1−p_buy)·p_trade, 1−p_trade, p_buy·p_trade] with the 0.45 threshold. Validation, the gate's holdout backtest and live (PyTorch and the ONNX wrapper, which now outputs log action-probabilities) all use it. |
| S9 | `validate_epoch` returns the honest net Sharpe as `val_sharpe` (history, collapse controller, early stop, Optuna). The label Sharpe is kept as `last_label_sharpe` for diagnostics. |
| S10 | With HOLD decisions possible, 3-class accuracy compares like with like. |
| S11 | The honest metric adds commission and slippage (`EXECUTION.honest_extra_cost_pips`, default 1 pip, in each pair's own pips), annualises by realised trades per calendar year, and pools pairs into one equal-weight portfolio return per timestamp. The gate uses the same `periods_per_year`. |

**Tests:** `tests/test_loss_sharpe_fixes_2026_09_25.py` plus updated gate and honest-eval tests. In the wider loss, metric, inference and gate run, the 5 remaining failures all fail on the prior commit too.
