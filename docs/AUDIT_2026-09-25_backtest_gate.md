# Audit: Backtest & Promotion Gate (2026-09-25)

**Scope:**
- `validation/promotion_gate.py` (`PromotionGate`, PSR/DSR)
- `validation/gate_policy.py` (the certification that live trading trusts)
- `scripts/auto_optimal_roadmap.py` (Stage 4 certification)
- `training/post_train.py` and `training/train_gpu.py` (gate invocation)
- `scripts/backtest_model.py::run_execution_backtest` (the gate's backtest)
- `backtesting/backtest.py` (`ForexScalingBacktest`)

**Method:** static review at `main` @ `d45d3e9`. Nothing was run.

**Bottom line:** no promotion decision to date is trustworthy.
- The gate's backtest feeds the model inputs built differently from training.
- Its statistical tests pass or fail almost regardless of the data.
- The artifact that live trading accepts comes from a separate, much weaker path that never calls `PromotionGate`.

---

## P0: decisions can't be trusted

### G1. The gate backtest doesn't use the model's training inputs
`run_execution_backtest` (the "execution-aware" forward gate) rebuilds features itself:
- **Different feature path:** `FeatureEngineer` plus `AdvancedFeatureBuilder` on freshly loaded ticks. There's no `apply_feature_mask`, no multi-pair assembly and no macro, COT or news joins, unlike the dataset builder.
- **No scaler:** raw features go straight into a model trained on robust-scaled, clipped inputs.
- **Positional fitting:** `_fit_feature_width` truncates or zero-pads each pair's columns to `n_features / n_pairs`, and `X` is then truncated or padded again to `n_features`. Columns are matched by position, not by name.
- **Look-ahead in sentiment:** `finbert_sentiment` is set to today's `get_latest_headlines()` score for *every historical bar*.
- **Possibly synthetic, and tiny:** data comes from `load_or_generate(..., n_rows=100_000)`, a function that can fall back to generated data. It's also capped at 100k ticks, a few days of EURUSD.
- ~~**Wrong JPY scale**~~ *(correction: this script's `PIP_SIZES` is keyed by pair, so USDJPY did get 0.01. Not a bug; now uses the shared `get_pip_size`.)*

**Fix:** score the gate on the cached holdout rows (`X`, `close_pairs`, `spread_pairs`), with the checkpoint's own scaler, the feature-names sidecar and the per-pair pip size. That's what `training/honest_eval.py` already does for validation.

### G2. The Probabilistic Sharpe Ratio (PSR) gate is fed an annualised Sharpe
- `probabilistic_sharpe_ratio(sharpe, 0, n_obs, ...)` is written for a per-observation Sharpe.
- Every caller passes the annualised figure, and `n_obs` is the trade count.
- At Sharpe 1.5 with 500 observations, z ≈ 1.5 / √((1 + 1.0·2.25)/499) ≈ 18.6, so PSR ≈ 1.0. The "95% confidence" gate passes for any positive Sharpe. In the other direction, a small negative annualised Sharpe gives PSR ≈ 0.
- The deflated Sharpe (DSR) compares against `E[max SR]` on the per-observation scale, so it's mis-scaled too.
- **Fix:** convert with `sr_obs = sharpe / sqrt(periods_per_year)`, using the same periods-per-year as the Sharpe itself.

### G3. The per-model gate can never pass
- `post_train` calls `evaluate(..., regime_pnl={})`, and `evaluate` sets `regime_ok = False` when `regime_pnl` is empty.
- So every per-model `promotion_gate.json` is a rejection.
- Live now requires a gate artifact, so the only way anything reaches live is the RL roadmap certificate (G5).

### G4. The multiple-testing correction is off or under-counted
- `GateConfig.strict_psr` defaults to `False`, so the DSR is skipped.
- Where it's on (`post_train`), `n_backtest_trials = number of CV folds` (7). That is not the number of configurations tried.
- Across the 7 folds × 4 architectures, 22+ Optuna trials, threshold sweeps and seeds, the honest trial count is in the hundreds. With that correction, the expected maximum Sharpe from noise alone is large.

### G5. The certificate live trading trusts is weak and self-attested
`scripts/auto_optimal_roadmap.py` Stage 4 writes the certificate that `validation.gate_policy.check_gate_artifact` accepts. Problems:
- **Inputs:**
  - It reads the RL training report `rl_report.json`, not an independent backtest. It never calls `PromotionGate`, so there's no PSR, profit factor, cost-ratio, regime or DSR check.
  - If `eval_return_pct` is missing it falls back to **`train_return_pct`**, an in-sample number.
  - The fold check reads `best_val_sharpe_proxy` from fold configs. That's the noisy ±30 validation Sharpe (model audit A3), and values with |v| ≥ 100 are silently dropped.
  - The Sharpe comes from the RL environment's own accounting, which excluded costs until this week (A5).
- **Trust:**
  - The live check trusts JSON fields: `status`, `quality_gate_passed`, `gate_version` and `rejection_reasons`.
  - Nothing ties the artifact to the checkpoint's contents: no hash of the weights or of the data window. The mtime check added this week only catches newer files.
  - Editing the JSON, or copying an old one next to a new model with a later mtime, passes.
- **Fix:** have one gate, `PromotionGate` on the honest holdout backtest, write the certificate. Embed the SHA-256 of every certified checkpoint and of the holdout index range, and have live verify those hashes.

### G6. The CV "gate simulation" invents its evidence
- `_gate_sim_for_hist` in `train_gpu.py` sets `profit_factor = 1 + 0.12·Sharpe`, `max_drawdown = 0.12 − 0.015·Sharpe`, `n_trades = 150` and `n_obs = 800`, and uses the **max** validation Sharpe over epochs.
- These are made-up numbers derived from one noisy statistic, then fed to the gate as if measured.
- Anything that reads `gate_sim`, such as fold selection logs, gets false evidence.
- **Fix:** remove it, or compute real fold metrics with `honest_eval`.

---

## P1: backtest mechanics

### B1. Pip size and pip value default to EURUSD
- `ForexScalingBacktest` defaults to `pip_size=0.0001` and `pip_value_per_lot=1.0`.
- Callers that don't override them mis-scale USDJPY by about 100×, both in the slippage (`slippage_pips × pip_size`) and in USD P&L. USD-base pairs also need conversion by price.

### B2. Stop and take-profit exits ignore the spread
- `_check_stops` triggers on mid-price `low`/`high` and fills at `stop ± slippage`.
- A long stop should fill at the *bid*, so it's missing half the spread, the same for shorts at the ask, and TP fills likewise.
- Entries correctly use `ask_close`/`bid_close`, so the backtest is spread-asymmetric: entries pay the spread, exits don't.

### B3. Profit factor is ∞ with no losing trades
- `profit_factor = inf` when no trade lost, so the `pf_ok` gate passes on tiny samples. It should be undefined or rejected below a minimum trade count.

### B4. The cost gate passes when costs aren't tracked
- `evaluate` sets `cost_ok = True` when `transaction_costs == 0`.
- `ForexScalingBacktest` only counts commission as "cost" (`total_commission_usd`); spread and slippage are embedded in prices and never counted. `cost_pct` therefore understates costs, heavily for a scalping strategy.

### B5. One holdout, reused
- The promotion holdout is the last `promote_forward_frac` of the cache.
- Every retrain, Optuna study and threshold sweep is judged on that same tail, which overfits the holdout over time.
- There's no rolling or never-touched final test set, and no record of how many times it's been looked at.

### B6. The Sharpe stability check depends on trial count
- `sharpe_stability_ok` only runs when `n_backtest_trials > 1`, conflating folds with trials.
- When it runs, it compares fold std with the *holdout* Sharpe, two different samples.

---

## P2

- **G7.** `GateConfig` thresholds (Sharpe ≥ 1.5, PF ≥ 1.5, DD ≤ 20%, ≥ 500 trades) differ from `gate_policy` (Sharpe > 0.5, ≥ 1,000 trades, DD ≤ 15%). There are two sources of truth with no shared constants.
- **G8.** `evaluate` accepts `n_obs` with a default of 1000. Any caller that forgets it gets an arbitrary PSR sample size.
- **G9.** The confidence threshold is swept (`sweep_confidence_threshold`) on the same data the gate scores, so there's selection bias on the gate data unless the sweep uses a separate slice.
- **G10.** Regime P&L concentration is required but no backtest produces it, so G3 fails every time. Either compute it (the per-trade regime label exists in the features) or make the check optional with an explicit flag.

---

## Recommended order

1. **G1 + B1 + B2:** one honest holdout backtest built on the cached features, scaler, feature sidecar and per-pair prices and spreads. Extend `honest_eval.pooled_pair_metrics` with an equity curve, drawdown and profit factor, rather than a third feature pipeline.
2. **G2 + G4:** per-observation PSR and DSR, with the trial count taken from the Optuna DB, the number of architectures and the seeds.
3. **G5 + G7:** one gate (`PromotionGate` with one config) writes one certificate. It should contain SHA-256s of the checkpoints, the scaler, the feature sidecar and the holdout range, and live should verify them.
4. **G3/G10, G6, B3–B6:** real regime P&L, remove the synthetic gate simulation, a finite profit factor, all costs counted, and a final test set that is used once.

Until 1–3 are done, no certificate means anything. The current ones are already refused by gate v2 and by preflight's `_features.json` requirement.

---

## Fix log

| # | Fix |
|---|---|
| G1 | New `training/honest_eval.holdout_gate_metrics` scores the gate on the cached holdout rows, with the checkpoint's own scaler (clipped like training), per-pair close/spread for per-pair heads, and non-overlapping trades net of spread. `post_train._evaluate_forward_gate` uses it; `run_execution_backtest` is no longer on the gate path. A missing scaler sidecar is a REJECT. |
| G2 | PSR and DSR take `periods_per_year` and work on the per-observation scale. Without it, PSR and DSR fail closed. The returns-based DSR in `evaluation/metrics.py` now uses sd(SR) × E[max of N normals]. |
| G3 / G10 | The regime check is optional (`GateConfig.require_regime_pnl`, default False) instead of failing every gate. |
| G4 | `strict_psr` (DSR) is on by default. The trial count is `count_research_trials()`: fold checkpoints plus Optuna trials. |
| G5 | Gate v3: certificates carry SHA-256 hashes of every certified artifact (weights, scaler, feature sidecar) and `check_gate_artifact` recomputes them. `train_gpu` writes certificates via `certificate_from_gate`. The roadmap certificate requires a valid ensemble PromotionGate certificate, never uses `train_return_pct`, and reads honest fold Sharpes (median > 0, ≥ 70% of folds positive). |
| G6 | The CV gate simulation (invented PF/DD) is removed. |
| G7 | `gate_policy` thresholds are derived from `GateConfig`. |
| G8 | `n_obs` no longer defaults to 1000 (falls back to `n_trades`). |
| G9 | The gate backtest has no confidence scores, so the threshold sweep no longer tunes on gate data. |
| B1 | `ForexScalingBacktest(pair=...)` derives pip size and USD pip value (USD-base pairs converted by median price). `backtest_model` passes the pair. |
| B2 | Stop and TP exits pay half the spread (Python and Numba paths). |
| B3 | Profit factor is capped at 100, and 0 with no trades. |
| B4 | The backtest reports `spread_slippage_cost_usd` and `total_cost_usd`; the gate's cost input is the measured spread cost. |
| B5 | `logs/holdout_usage.json` counts holdout scorings and warns after 20. |
| B6 | Sharpe stability uses `fold_sharpes` (≥ 3 folds, mean > 0, CV < 1). |

**Tests:** `tests/test_gate_fixes_2026_09_25.py`, plus updated gate, cost-gate and backtest tests. The two remaining failures (`test_bug02_gpu_backtester_commission_and_ruin`, `test_auto_retrain_on_drift_no_data`) already failed before these changes.

**Consequence:** every existing certificate is now invalid (gate v3 needs artifact hashes). Certification requires models retrained with scaler and feature sidecars, gated on the rebuilt cache.
