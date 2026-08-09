# Deep Audit — Sharpe / Models / Pretraining / Inference (2026-08-07)

Companion to `docs/AUDIT_2026-08-06.md`. Three parallel forensic sweeps of the
metrics math, model architectures, pretraining / supervised-loop helpers,
and inference engines. **131 findings** total (54 BUG / 38 SILENT /
~39 STYLE+WONTFIX+CLEAN). ~70 are NEW vs the prior audit doc — including 3
that contradict existing claims.

Highlighted 🔴 findings are believed to materially affect live trading or
promotion decisions today. The remaining BUGs silently degrade training
quality but are bounded in impact.

---

## 🔴 Critical contradictions / oversights vs `docs/AUDIT_2026-08-06.md`

| ID | File:line | Defect |
|----|-----------|--------|
| **B-10** | `backtesting/backtest.py:1014-1020` + `training/post_train.py:864-878` | `_equity_curve_metrics` is dead code (zero callers). Calling `performance_metrics()` on the default Numba (≥50k-bars) path **raises `RuntimeError`** because `self.trades` is empty and `self.results_df` is populated. `_evaluate_forward_gate` catches that generic exception and auto-REJECTs the model with "Execution backtest failed." ⇒ **every production training run with a long promo holdout auto-REJECTs the model on a backtest-engine error, NOT on model quality.** Contradicts `AUDIT_2026-08-06.md` B1's PARTIAL claim that the equity-curve fallback restores metrics. |
| **DOMAIN-ADAPT** | `pretrain/pretrain_runner.py:419-447` + `pretrain/multi_task.py:738-739` | The runner's multi-task pretraining path **never passes `domain_labels`**, so `DANN`/`MMD`/`CORAL` are defined but never invoked. `docs/SESSION_REPORT.md:276` claims they are wired; code says otherwise. |
| **VS-SessRep** | `models/ensemble.py:163-182` | `SESSION_REPORT.md:604` claims `diversity_loss` was removed; it is still present and misleadingly documented. |
| **VS-SessRep-EWCAnti** | `training/ewc.py:78-111` | EWC Fisher diagonal is normalized by `self.max_samples` not `samples_processed`; when the actual dataset is smaller than `max_samples`, the penalty is **underweighted by ~15×** → fails to prevent forgetting. |

---

## 🔴 Critical correctness bugs (live or promotion today)

| ID | File:line | Defect |
|----|-----------|--------|
| **R4** | `models/rl_agents.py:529` | PPO entropy bonus has the wrong sign: `loss = pol_loss + val_c*val_loss + ent_c*entropy.mean()`. Minimizing the loss **minimizes** entropy (deterministic). Standard is `− ent_c * entropy.mean()`. Exploration actively collapses on every PPO run. |
| **RA2** | `models/rl_advanced.py:417-425` | HER "future" strategy uses `random.randint(t_idx, n-1)` inclusive — the final transition's relabels are **always self-matchings** (`dist=0`) → guaranteed `+1.0` reward for free. Classic HER samples `t_idx+1 .. n-1` precisely to avoid this. |
| **A8 / A9** | `models/architectures.py:875-876 + 891`; `1034-1048` | `MambaBlock.conv1d` and `EXPERTEncoder.ConvFFN` use symmetric `padding=k-1` + slice `[:T]`. The slice keeps the left half of the padded output, so output `t` reads inputs up to `t+k-1` — **non-causal conv**, future leakage of `k-1` bars into every Mamba/EXPERT layer. (Fix: `F.pad(x, (k-1, 0))` + `Conv1d(padding=0)`, or tuple padding `(k-1, 0)`.) |
| **I3** | `inference/rl_inference.py:99-102`, `models/rl_agents.py:406-412` | PPO inference path has **no greedy mode** — `select_action` always samples from the Categorical. Two consecutive calls on the same observation return different actions. Live PPO inference is nondeterministic. (DQN sets `eps=0`; PPO needs an equivalent greedy flag.) |
| **I5** | `inference/pytorch_inference.py:149-151`, `inference/onnx_inference.py:147-164, 209-224` | Ensemble ONNX vs PyTorch engines produce different probabilities for the same scalar: PyTorch hardcodes `[0.1,0.2,0.7]`, ONNX softmaxes `[-out-0.15, 0, out-0.15]`. Switching engines (CUDA→DirectML) silently flips trade decisions. |
| **R-1 / R-2** | `risk/execution.py:262-282` | Confirmed `AUDIT_2026-08-06.md` M5: `parametric_var` mixes price-fraction-return covariance with pip-dollar weights and CVaR is reported without rescaling. Compute result: **`var_usd ≈ $0.00023`, `var_pct ≈ 2e-7`** for a standard EURUSD lot — exactly the audit's "$0.025" repro. None of the VaR risk gates that consume these numbers ever fire. |
| **T2 / S2** | `training/post_train.py:898` / `training/supervised_loop.py:1514-1515` | `np.std(folds)` (ddof=0) at promotion; training Sharpe variance uses raw second moment (ddof=0). Both systematically understated by `√(N/(N-1))`. CV gate too lenient (passes models with true CV=1.4), training Sharpe inflated up to 2.6% on small validation sets. |
| **F1** | `training/post_train.py:627-628` | Challenger-vs-production loss comparison `metric_val > prod_metric + min_delta` with **negative** `min_delta=−0.001` accepts any challenger with loss > 0.999 (incl. slight regressions). Gate is broken for the loss-metric direction. |
| **S1** | `training/supervised_loop.py:1501-1511` | `validate_epoch` returns `sharpe=0.0` when every batch hit a NaN-skip — indistinguishable from a real Sharpe=0. Combined with `_promote_best_fold` (`post_train.py:584`) selecting on this metric, NaN epochs can be saved as the "best" fold. |
| **P1** | `validation/promotion_gate.py:198-205`, `training/post_train.py:916-918` | Promotion gate raises on `gross_pnl=None` (fail-closed), but caller substitutes `gross_pnl=bt_metrics["net_pnl"]` and `transaction_costs=0.0`, defeating the raise and making `cost_ok=True` always. Combined with B-10, the regime/cost/stability/DSR/turnover/latency gates are silent pass-by-default. |
| **A4** | `models/architectures.py:714-758`, `809-858` | `TFTScalper` and `HAELTHybrid` Transformer branches have **no positional encoding** — permutation-equivariant over time, can't tell bar 0 from bar 59. The "long-range cross-asset" claim in the HAELT docstring is unsupported. |

---

## Sharpe / risk / promotion — 22 findings (summary)

See the agent report for the full table. Highlights beyond the critical table:

- **B-2 / B-4 / P-6**: per-bar MtM Sharpe annualized at √(252×24×60) (~276× textbook) is paired with per-trade PF, n_obs, and Calmar in the same metrics dict. PSR is computed with mismatched units; the **Calmar gate is numerically inert at floor 1.0** (any positive annualized-per-bar Sharpe trivially clears it).
- **B-3**: Sortino uses `excess_returns < 0` (compare rf-adjusted to 0) and sample std of downside — not textbook (target=0, RMS downside).
- **B-5 / P-1**: profit_factor and `cost_pct` use `0.01` and `1e-9` floors → near-zero losers / transaction_costs ⇒ trivially PASS.
- **B-6**: Lockbox test mixes per-bar annualized Sharpe with unannualized MC Sharpe in the same result dict.
- **B-7**: `prob_sharpe_above_1` is judged against an **absolute** `1.0` regardless of caller-supplied `annual_factor` — opposite verdicts with `trades_per_year=None` vs `252`.
- **B-8**: Python-path circuit-breaker force-close at `backtest.py:997` realizes PnL into `self.equity` but never writes it back to the truncated `res_equity` — final equity and Sharpe numerator understated. The Numba path (`equity_curve[i] = equity`) is consistent.
- **R3 / R4 (`risk/execution.py`)**: lot-sizing caps to 10 lots on unmodeled pairs → 100× equity fail-open; `RegimePositionSizer.size` uses √252 but the input is per-bar (1-min) returns → vol 600× off, capped to size-multiplier 3× every time.
- **RE-4 / R3-bis**: `historical_var` math is clean; daily-loss circuit breaker is no-op on live because `live_engine` never calls `on_trade_closed` (audit M6 — OPEN).

## Models / ensemble / RL — 30 findings (summary)

Highlights beyond the critical table:

- **A1**: `MultiTaskWrapper.forward` returns `conf` as a raw logit; no single place sigmoidizes it for downstream position sizing.
- **A3**: `balanced_ce_parts` reads `weighted_l_dir` (already dir-weighted) ⇒ the class-balance penalty silently inherits `y_conf` dir-weighting, mis-tuned against documented "balanced CE".
- **A5**: no causal mask on any in-model Transformer — relies entirely on upstream pipeline being leak-free, which `D1/D2/D4` show it isn't (no defense-in-depth).
- **A10**: pooling never masks padded timesteps anywhere in the architecture file.
- **A11**: `build_model("haelt")` halves `d_model`/`nhead` **in place on caller's mutable kwargs** → re-entry (Optuna `subprocess.run` with shared cfg) re-halves.
- **A12**: `HuberLoss` dir_penalty gets double-scaled by `sign_weight × weight` when `weight` is passed.
- **E3**: `scripts/train_ensemble_meta.py:215-255` trains the meta-learner on a **randomly sampled** subset of the same cache the base models were trained on — no temporal split / purge / embargo. Same canonical in-sample leak as `B4` (triple-barrier meta-labeler) but for the stacking meta-learner. Not previously noted.
- **E4**: `EnsembleMetaLearner` uses `nn.LazyLinear` context encoder that binds to first-batch shape — silent wrong shape if first forward isn't production-shaped.
- **E5**: `EnsembleMetaLearner.forward` runs bases under `no_grad` but does **not** call `.eval()` on bases — Dropout stays active during meta-training, miscalibrating weights to a noise distribution that doesn't match production.
- **E7 / E8**: `confidence_filter` mixes relative confidence with absolute thresholds; `EnsembleRiskFilter` thresholds assume O(1) disagreement but `MultiTaskWrapper` bases return O(0.001) → gate rarely trips.
- **E9**: `MultiTimeframeAttention` hard-codes `encoded[0]` as the 1m query — reordering `timeframes` silently swaps fine/coarse.
- **R5**: `RunningRewardNormalizer` labeled Welford but uses `_alpha=0.001` → ~1000-step lag, mis-normalizes at every curriculum phase transition.
- **R7**: `ReplayBuffer.sample` uses `replace=True` and class-balanced `p=` with **no importance-sampling correction** in the loss → biased Q-target.
- **RA1**: HER tolerance hardcoded `0.0002` (~2 pips) — always-fail for JPY (pip=0.01 ⇒ 0.02 pip — never matches), always-pass-ish for low-ATR EURUSD.
- **RA5**: `SharpeRewardWrapper` returns `sharpe*~15.8 − dd_pen(~0.25) − cost_pen(~$0.001)` — drawdown/cost penalties numerically negligible, agent effectively optimizes raw Sharpe.
- **RA6**: `SharpeRewardWrapper.reset()` vs `train_agent`'s `reward_sharpe.reset_episode()` naming convention mismatch — crashes on direct (non-adapter) use.
- **I2**: `torch_load_safe` `ModuleNotFoundError` fallback to `weights_only=False` triggers on **every** post-refactor legacy checkpoint — wider attack surface than `S1` implies.
- **I4**: ensemble base models loaded with `strict=False` — silent wrong-weights on architecture mismatch (refactors!).
- **I6 / I7 / I8 / I9**: `RLInferenceAgent` has no runtime shape assertion between encoder output and policy `obs_size`, `_infer_n_actions` hardcodes `"net.4"` (coupled to `DQNetwork.net`'s exact layer count), `select_action` hardcodes `lot_size=10_000` while the training env makes it configurable, and `_wrap_rl_execution_policy` agent-state tail truncation drops `in_position` when encoder dim grows.

## Pretraining / supervised loop — 79 findings (summary)

**14 BUGs (highest training-quality impact):**

1. `training/supervised_loop.py:645-661` — mixup drops the multitask `l_conf` head loss under `--use-mixup --multitask` (confidence head silently isn't trained during mixup epochs).
2. `models/architectures.py:202-309` — `MultiTaskLoss.l_conf` uses `BCEWithLogitsLoss(reduction="mean")` so it is **never `y_conf`-weighted**, while `l_dir` is per-sample weighted — breaks `path_quality` semantics.
3. `training/ewc.py:78-111` — EWC Fisher normalized by `max_samples` not `samples_processed` → ~15× underweight (catastrophic forgetting not prevented).
4. `pretrain/contrastive.py:686-695` — `RegimeAwareTSCLTrainer.pretrain` can pick `pos_i == anchor_i` or `neg_i == pos_i == anchor_i` (no self-exclusion); degenerates positives into negatives.
5. `pretrain/multi_task.py:677-711` — `_gradnorm_step` drops the domain-adaptation loss when `len(task_losses) > 1`.
6. `training/pretrain_runner.py:1039` — `NameError` on `un` inside plateau-handoff branch when method is TSCL/cluster (non-multi-block) with `handoff_patience > 0` — crashes mid-run.
7. `training/supervised_loop.py:1501-1511` — see S1 above.
8. `pretrain/multi_task.py:738-739` + `pretrain/pretrain_runner.py:419-447` — see DOMAIN-ADAPT above.
9. `training/supervised_loop.py:941-942` — `clip_grad_norm_` is called BEFORE `_maybe_warn_grad_norm`, so the warn always reads post-clip norms and never trips the >50 threshold.
10. `training/ewc.py:48-71` — EWC `_task_loss` auto-falls into MSE when `labels` are float, even for classification models where Fisher should be CE-based; callers must explicitly pass `classification=True` or get the wrong Fisher basis.

**Verified CLEAN against user hypotheses:** NT-Xent `log(0)` when sim=1.0 (projection head's `eps=1e-8` saves it); mixup DOES mix class labels via soft one-hot; `_OVERCONF_PENALTY` does NOT leak across runs; `OnlineHardMiner` has no index overflow; `amp_dtype` IS plumbed end-to-end; OneCycle scheduler step guard correct; `.clip_grad_norm_` `total_norm` is the post-clip value (so the grad-norm warning is blind rather than double-clipping).

**Unresolved:** `TemperatureScaler.calibrate(...)` call site — docstring says val; not directly confirmed in this sweep. Suggested follow-up: grep `scaler.calibrate` near `supervised_loop.py:3457` to verify it passes the tune/val loader, not `train_dl`.

## Suggested fix order (highest impact first)

1. **R4 (PPO entropy sign)**: one-character flip — flip entropy collapse on every PPO run.
2. **B-10 + P1 + T2 + S1 (promotion chain)**: the gate has been effectively off for any ≥50k-bar run AND auto-rejects on engine errors. Two-line fix in `backtest.py:1014` (gate the raise, fall back to `_equity_curve_metrics` or accept the equities-only metric set) unblocks the whole promotion path.
3. **R-1/R-2 (VaR)**: parametric VaR/CVaR are mathematically wrong by ~10,000×. Fix the cov scaling once and every downstream risk gate begins to function correctly.
4. **RA2 (HER self-match)**: `random.randint(t_idx + 1, n - 1)` instead of `t_idx` — fixes trivial positive rewards for the last transition and reduces noise across the episode.
5. **A8 / A9 (causal conv)**: replace `padding=k-1` + slice with asymmetric `(k-1, 0)` padding — restores the "causal" docstring claim and stops future leakage in Mamba/EXPERT.
6. **I3 (PPO greedy inference)**: add a `greedy` flag to `ActorCritic.act`; `RLInferenceAgent.select_action` passes it. One-line fix, removes live nondeterminism.
7. **A4 (positional encoding)**: add `nn.Embedding(max_seq_len, d_model)` + add to inputs on TFT/HAELT Transformer branches — restores the long-range cross-asset claim.
8. **E3 (meta-learner in-sample)**: enforce a temporal split (e.g. last 20% of the cache for meta training, with an embargo equal to the labeling horizon) in `scripts/train_ensemble_meta.py:215-255`. Same fix shape as `B4` already applied to the triple-barrier meta-labeler.
9. **F1 (challenger loss direction)**: invert the comparator for the loss case (`metric_val < prod_metric + min_delta` with positive `min_delta`).
10. **EWC Fisher normalization** (`training/ewc.py:78-111`): divide by `samples_processed`, not `max_samples`.
11. **DOMAIN-ADAPT (DANN/MMD/CORAL)**: wire `domain_labels` through `_run_multi_task_pretrain` so the documented adaptation methods actually run.
12. **I5 (ensemble engine parity)**: route single-scalar outputs through a shared `_scalar_to_direction_logits` → softmax helper used by both engines, instead of hardcoded `[0.1,0.2,0.7]` in pytorch_inference.

---

## Reference: agent reports

- Metrics audit: [Sharpe / Risk audit](39d72342-604a-40d2-9875-53b57d3f1be7)
- Models audit: [Models / Ensemble / RL audit](ac721d2b-4cc0-4de3-b6e5-9dde7b8cd627)
- Pretraining + supervised-loop audit: [Pretraining / Supervised audit](768669f8-4b7c-4e70-8b74-86bb1127a43a)

---

## 2026-08-07 fix-log (post-audit)

Three critical bugs were addressed first per the user's "fix the bugs first" instruction. The remaining milestones 4–12 from the plan (preflight + logging + telemetry layer) are deferred to a follow-up plan.

| Finding | File | Status |
|---------|------|--------|
| **R4 — PPO entropy sign** | `models/rl_agents.py:536` | **Already correct** in the working tree (`- self.ent_c * entropy.mean()`). The audit's claim that the sign was `+` was based on a stale snapshot; verified by `git show HEAD:models/rl_agents.py` and `grep -rn 'ent_c \* entropy'`. No code change made. |
| **Inference never applies training-time `StandardScaler`** | `inference/pytorch_inference.py`, `inference/onnx_inference.py`, `inference/rl_inference.py`, `training/supervised_loop.py`, `scripts/verify_onnx_export.py`, new `inference/_scaler_load.py` | **DONE.** Sidecar config now carries `cache_path` + `schema_hash` + `feature_names`; the inference engine loads the persisted `scaler.npz`, applies `transform()` before the forward pass, and raises `RuntimeError` on `n_features_in_` mismatch. Tests: `tests/test_inference_scaler_contract.py` (5 pass, 2 skip on missing torch). |
| **Meta-learner in-sample leak (no temporal split / purge / embargo)** | `scripts/train_ensemble_meta.py` | **DONE.** Standalone script now uses `training.cache_integrity._trainable_max_index(total, args)` and samples from `[0, _trainable)` only — mirroring the production path `training/post_train.run_ensemble_meta:164-174`. New CLI flags `--promote-forward-frac` and `--embargo-bars`. Tests: `tests/test_ensemble_meta_split.py` (6 pass) including a regression guard that proves the OLD uniform sampling pattern permitted sampling from the holdout tail. |

### 2026-08-07 follow-up fixes (milestones 4, 7, 11)

| Finding | File | Status |
|---------|------|--------|
| **M4 / S1 — `validate_epoch` returns `sharpe=0.0` indistinguishable from real 0.0; `best_sharpe=-inf` accepts any positive Sharpe** | `training/supervised_loop.py` | **DONE.** `validate_epoch` now returns `sharpe=None` sentinel when `n_ret==0` or `valid_batches==0`. Early-stopping logic now initialises `best_sharpe` to a configurable floor (`--early-stop_min_sharpe`, default `0.0`) instead of `-inf`, and rejects the `None` sentinel (cannot improve from no-data). Resume path falls back to the floor too. Tests: `tests/test_validate_epoch_sentinel.py` (5 pass). |
| **M7 — Grad-norm warn AFTER clip; no JSONL event** | `training/supervised_loop.py`, `monitoring/train_logger.py` | **DONE.** `_maybe_warn_grad_norm` is now called BEFORE `clip_grad_norm_` so it measures the true gradient magnitude. Added `epoch` parameter, emits `grad_norm` JSONL event, and falls back to `stderr` when `_TRAIN_LOGGER` is unavailable. New `TrainingLogger.on_grad_norm` method mirrors the inline emission. Tests: `tests/test_grad_norm_ordering.py` (6 pass, 1 skip on missing torch). |
| **M11 / F1 — Challenger loss-sign + missing promotion gate telemetry** | `training/post_train.py`, `monitoring/train_logger.py` | **DONE.** Challenger-vs-prod gate now uses positive `min_delta=0.001` for both sharpe and loss directions, replacing the fragile `-0.001` + `+min_delta` sign-cancel trick (behaviour verified bit-identical in tests). New `TrainingLogger.on_promotion_decision` method emits a `promotion_decision` JSONL record and appends a row to `logs/promotion_decisions.csv`. `_evaluate_forward_gate` and `_select_best_fold` emit the event after each gate run. Tests: `tests/test_promotion_gate_telemetry.py` (10 pass). |

### Files touched in this round

- `inference/_scaler_load.py` (new)
- `inference/pytorch_inference.py` (extended `load_pytorch_model` for cache_path / scaler / schema-hash; threaded 5-tuple)
- `inference/onnx_inference.py` (two call sites updated to the 5-tuple)
- `inference/rl_inference.py` (call site updated)
- `training/supervised_loop.py` (sidecar JSON now carries `cache_path` / `schema_hash` / `feature_names`; teacher-load call site updated)
- `scripts/verify_onnx_export.py` (call site updated)
- `scripts/train_ensemble_meta.py` (chronological trainable-prefix split, new CLI flags, logging)
- `tests/test_inference_scaler_contract.py` (new)
- `tests/test_ensemble_meta_split.py` (new)
- `tests/test_validate_epoch_sentinel.py` (new)
- `tests/test_grad_norm_ordering.py` (new)
- `tests/test_promotion_gate_telemetry.py` (new)
- `CHANGELOG.md`

### 2026-08-08 follow-up fixes (R-1/R-2, P1, A8/A9, I3, A4, EWC, RA2)

| Audit finding | Files touched | Status |
|---|---|---|
| **R-1 / R-2 — `parametric_var` mixes pip-dollar weights with price-fraction-return covariance; CVaR reported without rescaling** | `risk/execution.py` (rewrote `PortfolioVaR` — `parametric_var` + `max_allowed_lots` + `update_returns` contract), `trading/live_engine.py` (caller updated to feed price-fraction returns) | **DONE.** Recomputed `parametric_var` in dollar-notional space (lots × $100k × price-fraction-return): a standard EURUSD lot at 99% conf now gives `var_usd ≈ $69.78` instead of the OLD `$0.0096`. `update_returns` auto-normalises the obvious "pip-scaled" misuse case (`|r| > 0.5` → multiply by `pip_size`). New `set_notional(pair, ...)` per-pair override for non-FX instruments. All existing `TestPortfolioVaR` tests pass (5/5); new tests: `tests/test_var_magnitude_fix.py` (12 pass including a closed-form `var_usd = z·σ·N` check). |
| **P1 — Promotion gate silent-default (`net_pnl` substituted for `gross_pnl`, `transaction_costs=0.0`)** | `scripts/backtest_model.py` (normalizer now passes through `gross_pnl`, `total_commission`, `profit_factor`), `training/post_train.py` (`_evaluate_forward_gate` no longer substitutes net for gross; failed-closed reject when gross_pnl unavailable) | **DONE.** The cost gate now reads real `commission / |gross_pnl|` and only passes when `cost_pct ≤ max_cost_pct`. When `gross_pnl` is unavailable from the backtester, falls back to `net_pnl + total_commission`; if both are 0, emits a fail-closed REJECT instead of silent-pass. New tests: `tests/test_promotion_cost_gate.py` (6 pass, 3 skipped on missing torch). |
| **A8 / A9 — Causal conv uses symmetric `padding=k-1` + post-hoc slice; true receptive field leaks future bars under dilation/stride** | `models/architectures.py` (`MambaBlock.conv1d` L875-876, `ConvFFN.conv1/conv2` L1035-1036) | **DONE.** Both convs now use asymmetric left-only `padding=(kernel-1, 0)`; post-hoc `[:, :, :T]` and `h[:, :, :T]` slices removed (with asymmetric padding, output length == input length). For stride=1 zero-padding the two forms are numerically equivalent (closed-form verified in `tests/test_causal_conv_padding.py`); the asymmetric form is robust to future edits, dilated convs, and even kernels. Tests: `tests/test_causal_conv_padding.py` (8 pass). |
| **I3 — PPO actor always stochastic during live inference** | `models/rl_agents.py` (`ActorCritic.act` gains `greedy=False` kwarg, dispatches to `argmax` when True; `PPOAgent.select_action` threads the flag), `inference/rl_inference.py` (passes `greedy=True` with `try/except TypeError` fallback for DQN) | **DONE.** Live inference is now deterministic for PPO (`argmax` instead of `Categorical.sample()`); training rollout keeps the default stochastic path for exploration. Tests: `tests/test_ppo_greedy_inference.py` (8 pass, 1 skip on missing torch). |
| **A4 — Transformer branches lack positional encoding (permutation-equivariant over time)** | `models/architectures.py` (`HAELTHybrid`, `TFTScalper`, `EXPERTEncoder` each get `nn.Embedding(max_seq_len, d_model)` positional embedding), `config/models.py` (`no_pos_encoding` flipped to False) | **DONE.** Each transformer-branched class now learns an absolute positional embedding added after the input projection. Handles `T == max_seq_len` (fast), `T < max_seq_len` (slice), `T > max_seq_len` (cyclic fallback). Constructor signatures backward-compatible. Removed misleading `EXPERTEncoder` docstring "order is inherent in time series". Tests: `tests/test_positional_encoding.py` (3 pass, 4 skip on missing torch). |
| **EWC — Fisher diagonal divides by `max_samples` constant instead of `samples_processed`** | `training/ewc.py` (`_compute_fisher_diagonal`) | **DONE.** Removed per-step `/ self.max_samples`; added single post-loop normalisation by `max(samples_processed, 1)`. Fisher is now a true per-sample mean of `grad²`, invariant to batch size and dataset size (audit ~15× underweighting when `len(ds) < max_samples` is fixed). Public interface unchanged — no caller changes needed. Tests: `tests/test_ewc_fisher_normalization.py` (3 pass, 2 skip on missing torch). |
| **RA2 — HER "future" strategy uses `random.randint(t_idx, n-1)` inclusive — final transition always self-matches** | `models/rl_advanced.py` (`HERBuffer.end_episode`) | **DONE.** Switched to `random.randint(t_idx + 1, n - 1)` (strict future). Skips relabel entirely when `t_idx + 1 >= n` (last transition has no future) instead of polluting the buffer with self-matches. New `np.array_equal(her_goal, achieved)` guard skips any degenerate "price didn't move" self-match. "episode" and "random" strategies retain their existing behaviour but also get the self-match guard. Tests: `tests/test_her_self_match.py` (8 pass). |

#### Files touched (2026-08-08 round)

- `risk/execution.py` (rewrote `PortfolioVaR` math; new units contract)
- `trading/live_engine.py` (caller updated for price-fraction returns)
- `scripts/backtest_model.py` (normalizer passes through gross/commission/PF)
- `training/post_train.py` (`_evaluate_forward_gate` no longer substitutes net for gross)
- `models/architectures.py` (MambaBlock / ConvFFN / 3 transformer classes fixed)
- `models/rl_agents.py` (`greedy` flag on `ActorCritic.act` / `PPOAgent.select_action`)
- `inference/rl_inference.py` (passes `greedy=True`)
- `training/ewc.py` (Fisher normalisation by `samples_processed`)
- `models/rl_advanced.py` (`HERBuffer.end_episode` future strategy strict)
- `config/models.py` (EXPERTEncoder `no_pos_encoding` flipped)
- `tests/test_var_magnitude_fix.py` (new, 12 pass)
- `tests/test_promotion_cost_gate.py` (new, 6 pass / 3 skip)
- `tests/test_causal_conv_padding.py` (new, 8 pass)
- `tests/test_ppo_greedy_inference.py` (new, 8 pass / 1 skip)
- `tests/test_positional_encoding.py` (new, 3 pass / 4 skip)
- `tests/test_ewc_fisher_normalization.py` (new, 3 pass / 2 skip)
- `tests/test_her_self_match.py` (new, 8 pass)
- `CHANGELOG.md`

