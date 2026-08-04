# Continue — Remaining Improvement Items

**Updated:** 2026-08-04
**Progress:** 20 / 20 items complete
**Remaining:** —
> ✅ All items complete: A1, A2, B1, B2, C1, C2, C3, C4, D1, D2 (wiring), Risk Controls, Metrics, Data drift, Model drift, Audit, Alerting. See `docs/SESSION_REPORT.md` (session 2026-08-04 18:01 UTC).

> **Wiring note (2026-08-04):** The previously-completed improvement modules have now been wired into production entry points as opt-in, behavior-preserving CLI flags (feature quality gate, no-trade zones, meta-labeling, off-policy rewards, CV strategies, HPO scheduler, multi-task pretrainer, curriculum manager, execution engine) and the legacy Monte Carlo classes now delegate to `evaluation/monte_carlo.py` via a result-schema-preserving facade. See `CHANGELOG.md` and `docs/SESSION_REPORT.md` (session 2026-08-04 16:46 UTC).

> One item at a time. After each item: update `CHANGELOG.md` (`## [Unreleased] → ### Added`), append a `## Session — <UTC timestamp>` block to `docs/SESSION_REPORT.md`, and tick the todo list.
>
> **Test command:** `uv run pytest <path> -q` — always run the new module's tests plus `tests/test_feature_pipeline.py` + `tests/test_labeling_pipeline.py` for regressions.

---

## 1. Risk Controls — real-time risk engine, FX Greeks, portfolio-level

**Priority:** high
**Status:** ✅ complete (2026-08-04)

**Deliverables:**
- `risk/risk_engine.py` — real-time risk engine:
  - Pre-trade checks: max position size, max notional, max daily loss (stop-trading threshold), max order frequency, single-instrument concentration caps.
  - Post-trade monitoring: drawdown limits, VaR (historical + parametric), Expected Shortfall / CVaR, exposure by currency and by pair, gap-risk flags.
  - Circuit breakers / kill-switch: auto-flatten or auto-standby when limits breached.
  - Audit log of every risk decision (timestamp, rule, value, action).
- `risk/fx_greeks.py` — FX Greeks:
  - Delta, gamma, theta, vega, rho for FX option-like exposures; position-adjusted portfolio Greeks.
  - FX delta expressed in base currency; handle JPY/USD quoting conventions.
- `risk/portfolio_monitor.py` — portfolio-level:
  - Aggregate exposure across pairs, net currency exposure, correlation-aware exposure, liquidity tiering.
- Wire into `config/settings.py` (new `RISK` block) and `training/train_gpu.py` for live/dry-run enforcement.
- `tests/test_risk_engine.py` (~15 tests), `tests/test_fx_greeks.py` (~10 tests), `tests/test_portfolio_monitor.py` (~8 tests).

**Verify:** `uv run pytest tests/test_risk_engine.py tests/test_fx_greeks.py tests/test_portfolio_monitor.py -q`

---

## 2. Metrics — PSR, Deflated Sharpe, Calmar, Omega, Tail ratio

**Priority:** high
**Status:** ✅ complete (2026-08-04)

**Deliverables:**
- `evaluation/metrics.py`:
  - Probabilistic Sharpe Ratio (PSR) with skew/kurtosis adjustment.
  - Deflated Sharpe Ratio (DSR) — Bailey-López de Prado correction for multiple trials.
  - Calmar ratio (CAGR / max drawdown), Omega ratio (gain/loss threshold), Tail ratio (95/5 quantile), Sortino, downside deviation.
  - Minimum backtest length required for significance.
- Integration helpers: compute on a returns series or on a portfolio/backtest result object.
- `tests/test_metrics.py` (~15 tests), incl. known-value sanity checks against textbook numbers.

**Verify:** `uv run pytest tests/test_metrics.py -q`

---

## 3. Monte Carlo — block bootstrap, stationary bootstrap, path-dependent sims

**Priority:** medium
**Status:** ✅ complete (2026-08-04)

**Implemented:** `evaluation/monte_carlo.py`
- `block_bootstrap_indices` / `block_bootstrap` — fixed-length block bootstrap (contiguous blocks, sampled with replacement, concatenated+truncated).
- `stationary_bootstrap_indices` / `stationary_bootstrap` — Politis–Romano, geometric random block lengths, circular wrap.
- `pl_block_bootstrap` — Polars-native bootstrap (one column per path).
- `PathMonteCarlo` — replay a `signal_fn` over resampled return paths → distribution of final equity / total return / max DD / Sharpe.
- `TradeSequenceMonteCarlo` — keep the trade sequence (entry/exit bars + signed size) fixed, resample the market path.
- `summarize_simulation`, `monte_carlo_backtest`, `Trade`, `SimResult`.
- `evaluation/` package added to `pyproject.toml`; `tests/test_monte_carlo.py` (22 tests, green). Regressions green across feature/labeling/execution.
- **Wired into production (2026-08-04):** legacy `backtesting/improvements.MonteCarloBacktest` and `monitoring/pipeline.MonteCarloBacktest` now delegate resampling to this suite (`block_bootstrap_indices`, block_length=1 i.i.d.) via a facade that preserves their historical `.run`/`.run_from_backtest` result-key schemas, so `scripts/backtest_model.py` and `main.py` keep working. Facade tests added in `tests/test_monte_carlo.py`.

---

## 4. Data drift — feature-attribution (SHAP), concept drift, adversarial validation

**Priority:** medium
**Status:** ✅ complete (2026-08-04)

**Deliverables:**
- `drift/data_drift.py`:
  - SHAP-based feature-attribution drift: compare train-time SHAP distributions vs live; flag features whose importance shifts.
  - Concept drift detectors: ADWIN, Page-Hinkley, DDM/EDDM on model error; streaming drift score.
  - Feature distribution drift: KS test / Wasserstein distance + PSI (reuse `features/feature_quality_monitor.py`).
  - Adversarial validation: train a classifier to distinguish train vs live samples; AUC > threshold ⇒ drift alert.
- Emit structured drift events consumable by the alerting module.
- `tests/test_data_drift.py` (~14 tests).

**Verify:** `uv run pytest tests/test_data_drift.py -q`

---

## 5. Model drift — champion-challenger, canary rollout, automated rollback

**Priority:** medium
**Status:** ✅ complete (2026-08-04)

**Deliverables:**
- `drift/model_drift.py`:
  - Champion-challenger comparison harness (shadow-mode eval of challenger vs champion on live stream).
  - Canary rollout: route `x%` of live signals to challenger; escalate on success.
  - Automated rollback: monitor live performance vs champion baseline; breach thresholds → rollback + alert.
  - Performance degradation triggers (drawdown, PSR drop, error spike).
- Integration with promotion/readiness gates already in the repo.
- `tests/test_model_drift.py` (~12 tests).

**Verify:** `uv run pytest tests/test_model_drift.py -q`

---

## 6. Audit — full lineage + reproducibility manifests

**Priority:** medium
**Status:** ✅ complete (2026-08-04)

**Deliverables:**
- `audit/lineage.py`:
  - Data lineage: dataset/version → preprocessing steps → feature set → label version → training run.
  - Model registry hooks: record params, data hash, code commit, seed, env (GPU, lib versions), dataset hash.
  - Reproducibility manifests: JSON/YAML snapshot of every artifact + inputs + environment for a training run; store alongside checkpoints.
  - Audit trail of all promotion/rollback decisions.
- `audit/manifest.py` — generate/verify manifests (hash inputs, assert reproducibility).
- CLI or helper script to regenerate a run's manifest from a checkpoint dir.
- `tests/test_audit.py` (~12 tests).

**Verify:** `uv run pytest tests/test_audit.py -q`

---

## 7. Alerting — multi-channel + escalation + runbooks

**Priority:** low
**Status:** ✅ complete (2026-08-04)

**Deliverables:**
- `monitoring/alerting.py`:
  - Multi-channel sinks: Slack webhook, Discord webhook, email (SMTP), stdout/console, log file.
  - Severity levels (info/warning/critical) with per-channel routing + deduplication/rate limiting.
  - Escalation: if critical alert unacked for N minutes → escalate to next channel/severity.
  - Runbooks: per-alert-type remediation steps stored alongside alert definitions.
  - Test-mode channel (in-memory recorder) so tests don't send real messages.
- Wire drift events (item 4) + risk violations (item 1) into the alerting entry point.
- `tests/test_alerting.py` (~12 tests).

**Verify:** `uv run pytest tests/test_alerting.py -q`

---

## Completion

All 7 items complete (plus the D1/D2 wiring). `CHANGELOG.md`, `docs/SESSION_REPORT.md`
(session 2026-08-04 18:01 UTC), and this file have been updated.

Re-run the full suite:

```bash
uv run pytest tests/ -q
```

> Note: `tests/test_system.py` and `tests/test_streaming_pipeline.py` have
> pre-existing collection errors (unrelated to this work) and the full run still
> times out (>120s). Use the targeted suites below to verify green state:
> ```bash
> uv run pytest tests/test_risk_engine.py tests/test_fx_greeks.py \
>   tests/test_portfolio_monitor.py tests/test_metrics.py tests/test_data_drift.py \
>   tests/test_model_drift.py tests/test_audit.py tests/test_alerting.py \
>   tests/test_feature_pipeline.py tests/test_labeling_pipeline.py tests/test_monte_carlo.py -q
> ```
