"""
validation/promotion_gate.py
=============================
Formal model promotion gate - checks multiple criteria before a model
can move from Staging -> Production.

Core gates (all must pass to promote):
  1.  Out-of-sample Sharpe         ≥ min_sharpe (default 1.5)
  2.  Profit factor                ≥ min_profit_factor (default 1.5)
  3.  Max drawdown                 ≤ max_drawdown_pct (default 20%)
  4.  Calmar ratio (Sharpe / DD)   ≥ min_calmar (default 1.0)
  5.  Number of trades             ≥ min_trades (default 500)
  6.  Regime concentration         no single regime > max_regime_conc of P&L
  7.  Transaction cost / gross PnL < max_cost_pct (default 30%)
  8.  Probabilistic Sharpe Ratio   ≥ min_psr (default 0.95)
  9.  Deflated Sharpe Ratio        ≥ min_psr (only when strict_psr + trials>1)

Capital-efficiency gates (optional - pass when input data unavailable):
  10. Sharpe / turnover_rate       ≥ min_sharpe_per_turnover
  11. Sharpe / max_drawdown        ≥ min_sharpe_per_drawdown
  12. Sharpe / avg_latency_ms      ≥ min_sharpe_per_latency

Sharpe stability gate (optional - passes when multi-trial data unavailable):
  13. CV of Sharpe across folds   < 100%

Usage:
    from validation.promotion_gate import PromotionGate
    gate   = PromotionGate()
    result = gate.evaluate(
        sharpe=1.8,
        profit_factor=1.6,
        max_drawdown=0.12,
        n_trades=800,
        regime_pnl={"trending": 0.6, "neutral": 0.3, "mean_rev": 0.1},
        gross_pnl=5000.0,
        transaction_costs=800.0,
        n_backtest_trials=1,      # set >1 for Deflated Sharpe
        backtest_sharpe_std=0.0,  # std across walk-forward folds
    )
    print(result["promoted"], result["reasons"])
"""

import math
import warnings
from dataclasses import dataclass
from typing import Optional

warnings.filterwarnings("ignore")


# ── configuration ────────────────────────────────────────────────────────────


@dataclass
class GateConfig:
    """All promotion thresholds in one place."""

    min_sharpe: float = 1.5
    min_sharpe_emergency: float = 1.3  # for emergency retrain DAG
    min_profit_factor: float = 1.5
    max_drawdown_pct: float = 0.20
    min_trades: int = 500
    max_regime_conc: float = 0.75  # no regime > 75% of P&L
    max_cost_pct: float = 0.30  # transaction cost / gross P&L
    min_calmar: float = 1.0  # Sharpe / max_drawdown floor
    # B-M4: PSR confidence threshold. 0.5 (the old hardcoded value) only asks
    # "is the true Sharpe probably > 0", which is trivially weak. 0.95 requires
    # 95% confidence the true Sharpe beats the benchmark before promoting.
    min_psr: float = 0.95
    min_dsr: float = 0.95
    strict_psr: bool = False  # enable Deflated Sharpe (requires trials>1)

    # K: Capital efficiency gates - predict live survivability better than raw Sharpe.
    # All three are optional gates (skipped when input data is unavailable).
    min_sharpe_per_turnover: float = 0.10  # Sharpe / daily_turnover_rate
    min_sharpe_per_drawdown: float = 8.0  # Sharpe / max_drawdown  ≈ Calmar x annualisation
    min_sharpe_per_latency: float = 0.05  # Sharpe / avg_latency_ms  (ms⁻¹ units)


# ── Probabilistic Sharpe Ratio ────────────────────────────────────────────────


def probabilistic_sharpe_ratio(
    sr_hat: float,
    sr_benchmark: float,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 5.0,
) -> float:
    """
    P(SR* > SR_benchmark) using Bailey & de Prado (2012).
    Corrects for non-normality of returns.

    Default kurtosis is 5.0 (FX-typical leptokurtosis), not 3.0 (Gaussian).
    Understating kurtosis makes PSR look better than it is for fat-tailed FX.

    Returns a probability in [0, 1].
    Typical threshold: PSR > 0.95 (95% confident true Sharpe beats benchmark).
    """
    if n_obs < 5:
        return 0.5  # not enough data
    try:
        from scipy.stats import norm
    except ImportError:
        # Manual normal CDF approximation
        def _ncdf(x):
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))

        norm = type("N", (), {"cdf": staticmethod(_ncdf)})()

    variance = (1 / (n_obs - 1)) * (1 - skewness * sr_hat + ((kurtosis - 1) / 4) * sr_hat**2)
    if variance <= 0:
        return 0.5
    z = (sr_hat - sr_benchmark) / math.sqrt(max(variance, 1e-12))
    return float(norm.cdf(z))


def deflated_sharpe_ratio(
    sr_hat: float,
    sr_trials: int,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 5.0,
) -> float:
    """
    Deflated Sharpe Ratio from Bailey & de Prado (2014).
    Accounts for multiple testing / hyperparameter search.
    """
    if sr_trials <= 1:
        return sr_hat
    # Expected maximum SR under multiple trials (approximation)
    gamma_euler = 0.5772156649
    e_max_sr = (1 - gamma_euler) * math.sqrt(2 * math.log(sr_trials)) + gamma_euler / math.sqrt(2 * math.log(sr_trials))
    psr = probabilistic_sharpe_ratio(sr_hat, e_max_sr, n_obs, skewness, kurtosis)
    return float(psr)


# ── main gate ────────────────────────────────────────────────────────────────


class PromotionGate:
    """
    Evaluates whether a model should be promoted to production.

    All 7 gates must pass (or their override is set) for `promoted=True`.
    """

    def __init__(self, config: GateConfig | None = None):
        self.cfg = config or GateConfig()

    def evaluate(
        self,
        sharpe: float,
        profit_factor: float,
        max_drawdown: float,
        n_trades: int,
        regime_pnl: dict[str, float] | None = None,
        gross_pnl: float | None = None,
        transaction_costs: float = 0.0,
        n_obs: int = 1000,
        n_backtest_trials: int = 1,
        backtest_sharpe_std: float = 0.0,
        skewness: float = 0.0,
        kurtosis: float = 5.0,
        emergency_retrain: bool = False,
        # K: Capital efficiency inputs
        turnover_rate: float | None = None,  # trades per day (or fraction of bars traded)
        avg_latency_ms: float | None = None,  # average model inference + broker RTT in ms
        val_logits: Optional["np.ndarray"] = None,  # (Optional) logits for conformal
        val_labels: Optional["np.ndarray"] = None,  # (Optional) labels for conformal
    ) -> dict:
        """
        Evaluate all promotion gates.

        Parameters
        ----------
        sharpe              : Out-of-sample annualised Sharpe ratio.
        profit_factor       : Gross profit / gross loss.
        max_drawdown        : Max drawdown as a fraction (0.15 = 15%).
        n_trades            : Number of trades in the backtest.
        regime_pnl          : Dict mapping regime name -> fraction of total P&L.
                              e.g. {"trending": 0.7, "neutral": 0.3}
        gross_pnl           : Gross profit from winning trades (before costs), in
                              any currency unit. Used as the denominator for the
                              transaction-cost ratio gate.
        transaction_costs   : Total transaction costs incurred.
        n_obs               : Number of return observations (for PSR).
        n_backtest_trials   : Number of hyperparameter trials run (for DSR).
        backtest_sharpe_std : Std of Sharpe across walk-forward folds.
        skewness            : Skewness of returns (for PSR correction).
        kurtosis            : Kurtosis of returns.
        emergency_retrain   : If True, use lower min_sharpe_emergency threshold.
        turnover_rate       : K - average trades per day. High turnover erodes edge
                              through slippage at scale. None = gate skipped.
        avg_latency_ms      : K - average end-to-end latency in ms (inference +
                              broker round-trip). Latency-sensitive edges disappear
                              in production. None = gate skipped.
        """
        # B-M4: gross_pnl is required. A silent default of 1.0 makes the cost
        # ratio (transaction_costs / gross_pnl) explode to absurd values if the
        # caller forgets to pass it, which would spuriously fail the cost gate.
        if gross_pnl is None:
            raise ValueError(
                "PromotionGate.evaluate: gross_pnl is required (total gross P&L "
                "before costs). Pass the backtest's gross profit, not a placeholder."
            )

        if not all(math.isfinite(v) for v in [sharpe, profit_factor, max_drawdown, n_trades]):
            return {
                "promoted": False,
                "gates": {},
                "details": {},
                "reasons": ["non-finite input value"],
                "summary": "REJECT ❌  Non-finite inputs",
                "emergency_mode": emergency_retrain,
            }

        min_sr = self.cfg.min_sharpe_emergency if emergency_retrain else self.cfg.min_sharpe
        gates: dict[str, bool] = {}
        details: dict[str, float] = {}

        # 1. Sharpe
        gates["sharpe_ok"] = sharpe >= min_sr
        details["sharpe"] = sharpe
        details["min_sharpe"] = min_sr

        # 2. Profit factor
        gates["pf_ok"] = profit_factor >= self.cfg.min_profit_factor
        details["profit_factor"] = profit_factor
        details["min_profit_factor"] = self.cfg.min_profit_factor

        # 3. Max drawdown
        gates["dd_ok"] = max_drawdown <= self.cfg.max_drawdown_pct
        details["max_drawdown"] = max_drawdown
        details["max_dd_limit"] = self.cfg.max_drawdown_pct

        # 4. Calmar Ratio
        calmar = sharpe / max(max_drawdown, 0.01)
        gates["calmar_ok"] = calmar >= self.cfg.min_calmar
        details["calmar"] = calmar
        details["min_calmar"] = self.cfg.min_calmar

        # 5. Sample size
        gates["sample_ok"] = n_trades >= self.cfg.min_trades
        details["n_trades"] = float(n_trades)
        details["min_trades"] = float(self.cfg.min_trades)

        # 6. Regime concentration
        if regime_pnl:
            max_conc = max(abs(v) for v in regime_pnl.values())
            dominant = max(regime_pnl, key=lambda k: abs(regime_pnl[k]))
            gates["regime_ok"] = max_conc <= self.cfg.max_regime_conc
        else:
            max_conc, dominant = 0.0, "unknown"
            gates["regime_ok"] = False
        details["max_regime_conc"] = max_conc
        details["dominant_regime"] = dominant  # type: ignore[assignment]

        # 7. Transaction cost ratio
        if transaction_costs > 0.0:
            cost_pct = transaction_costs / max(abs(gross_pnl), 1e-9)
            gates["cost_ok"] = cost_pct <= self.cfg.max_cost_pct
        else:
            # No transaction costs tracked - cost gate passes (cost_pct = 0)
            cost_pct = 0.0
            gates["cost_ok"] = True
        details["cost_pct"] = cost_pct
        details["cost_limit"] = self.cfg.max_cost_pct

        # 8. Probabilistic Sharpe
        psr = probabilistic_sharpe_ratio(sharpe, 0.0, n_obs, skewness, kurtosis)
        gates["psr_ok"] = psr > self.cfg.min_psr  # B-M4: 95% confident by default
        details["psr"] = psr
        details["min_psr"] = self.cfg.min_psr

        if self.cfg.strict_psr:
            if n_backtest_trials > 1:
                dsr = deflated_sharpe_ratio(sharpe, n_backtest_trials, n_obs, skewness, kurtosis)
                gates["dsr_ok"] = dsr > self.cfg.min_dsr
                details["dsr"] = dsr
                details["min_dsr"] = self.cfg.min_dsr
            else:
                gates["dsr_ok"] = True  # No multi-trial data - gate passes
                details["dsr"] = -1.0
                details["dsr_skipped"] = False
                print("[PromotionGate] INFO: dsr_ok skipped (strict_psr on but n_backtest_trials<=1)")
        else:
            # DSR gate is optional when strict_psr is off
            gates["dsr_ok"] = True
            details["dsr"] = -1.0
            details["dsr_skipped"] = True
            # print("[PromotionGate] INFO: dsr_ok skipped (strict_psr off)")

        # Sharpe stability gate: reject models whose walk-forward Sharpe is
        # highly variable (low confidence that OOS performance will persist).
        # backtest_sharpe_std > 0 indicates multiple fold results are available.
        if backtest_sharpe_std > 0 and n_backtest_trials > 1:
            sharpe_cv = backtest_sharpe_std / max(abs(sharpe), 1e-6)
            gates["sharpe_stability_ok"] = sharpe_cv < 1.0  # CV < 100%
            details["backtest_sharpe_std"] = backtest_sharpe_std
            details["sharpe_cv"] = sharpe_cv
        else:
            # No multi-trial data - gate passes (optional gate)
            gates["sharpe_stability_ok"] = True
            details["backtest_sharpe_std"] = 0.0
            details["sharpe_stability_skipped"] = True
            # Optional: print debug if needed
            # print("[PromotionGate] INFO: sharpe_stability_ok skipped (no multi-trial data)")

        # ── K: Capital efficiency metrics ──────────────────────────────────
        # These three ratios predict live survivability better than raw Sharpe.
        # A model with high raw Sharpe but poor efficiency ratios will degrade
        # in production due to slippage (turnover), execution risk (latency),
        # or unacceptable loss events (drawdown).

        # K1: Sharpe / turnover - measures edge per unit of market impact.
        # A high-Sharpe scalper trading 50x/day has less edge per trade than
        # a swing trader at 2x/day. At scale, slippage erodes the scalper first.
        if turnover_rate is not None and turnover_rate > 0:
            sharpe_per_turnover = sharpe / turnover_rate
            gates["efficiency_turnover_ok"] = sharpe_per_turnover >= self.cfg.min_sharpe_per_turnover
            details["turnover_rate"] = turnover_rate
            details["sharpe_per_turnover"] = sharpe_per_turnover
        else:
            # No turnover data - gate passes (optional gate)
            gates["efficiency_turnover_ok"] = True
            details["sharpe_per_turnover"] = -1.0
            details["efficiency_turnover_skipped"] = True

        # K2: Sharpe / max_drawdown - Calmar-flavoured ratio.
        # Penalises models that achieve Sharpe through concentration or
        # large drawdown tolerance. Directly maps to position sizing headroom:
        # a model with Sharpe=2, DD=30% needs 3x larger account buffer than
        # one with Sharpe=2, DD=10% to survive an unlucky sequence.
        if max_drawdown > 0:
            sharpe_per_drawdown = sharpe / max_drawdown
        else:
            sharpe_per_drawdown = sharpe * 100.0  # near-zero drawdown: excellent
        gates["efficiency_drawdown_ok"] = sharpe_per_drawdown >= self.cfg.min_sharpe_per_drawdown
        details["sharpe_per_drawdown"] = sharpe_per_drawdown

        # K3: Sharpe / latency - latency sensitivity of the edge.
        # Edges that require sub-10 ms execution are fragile: network jitter,
        # broker queue delays, or a cold model cache will randomly destroy them.
        # A model that still works at 100 ms is robust to infrastructure variance.
        if avg_latency_ms is not None and avg_latency_ms > 0:
            sharpe_per_latency = sharpe / avg_latency_ms
            gates["efficiency_latency_ok"] = sharpe_per_latency >= self.cfg.min_sharpe_per_latency
            details["avg_latency_ms"] = avg_latency_ms
            details["sharpe_per_latency"] = sharpe_per_latency
        else:
            # No latency data - gate passes (optional gate)
            gates["efficiency_latency_ok"] = True
            details["sharpe_per_latency"] = -1.0
            details["efficiency_latency_skipped"] = True

        # ── Final decision ─────────────────────────────────────────────────
        promoted = all(gates.values())
        reasons = [f"{k}: {'✓' if v else '✗'}" for k, v in gates.items()]

        if val_logits is not None and val_labels is not None:
            cp_info = self.compute_conformal_coverage(val_logits, val_labels)
            details["cp_coverage"] = cp_info["coverage"]
            details["cp_threshold"] = cp_info["threshold"]
            details["cp_avg_set_size"] = cp_info["avg_set_size"]
            # Diagnostic log
            print(
                f"[Diagnostic] Conformal Prediction -> Coverage: {cp_info['coverage']:.3f}, Threshold: {cp_info['threshold']:.3f}, Avg Set Size: {cp_info['avg_set_size']:.3f}"
            )

        return {
            "promoted": promoted,
            "gates": gates,
            "details": details,
            "reasons": reasons,
            "emergency_mode": emergency_retrain,
            "summary": (
                "PROMOTE ✅" if promoted else "REJECT ❌  Failed: " + ", ".join(k for k, v in gates.items() if not v)
            ),
        }

    def evaluate_from_history(
        self,
        trade_pnls: "list[float]",
        equity_curve: "list[float]",
        regime_labels: "list[str] | None" = None,
        tx_costs: "list[float] | None" = None,
        annualization: float = 252.0,
        n_bars: int | None = None,  # total bars in backtest (for turnover calc)
        avg_latency_ms: float | None = None,
        **kwargs,
    ) -> dict:
        """
        Convenience wrapper - compute metrics from raw trade data.

        Parameters
        ----------
        trade_pnls    : Per-trade net P&L values.
        equity_curve  : Equity curve as list of equity values.
        regime_labels : Optional list of regime label per trade.
        tx_costs      : Optional list of transaction cost per trade.
        n_bars        : Total bars in backtest. Used to compute turnover_rate
                        (trades / bars x annualization). Pass this to enable
                        the K1 Sharpe/turnover efficiency gate.
        avg_latency_ms: End-to-end latency for K3 gate. None = skipped.
        """
        import numpy as np

        pnls = np.array(trade_pnls, dtype=float)
        eq = np.array(equity_curve, dtype=float)

        # Sharpe
        if len(pnls) > 1 and pnls.std(ddof=1) > 1e-12:
            sharpe = float(pnls.mean() / pnls.std(ddof=1) * np.sqrt(annualization))
        else:
            sharpe = 0.0

        # Profit factor
        gross_profit = float(pnls[pnls > 0].sum())
        gross_loss = float(abs(pnls[pnls < 0].sum()))
        profit_factor = gross_profit / max(gross_loss, 1e-9)

        # Max drawdown
        if len(eq) > 0:
            peak = np.maximum.accumulate(eq)
            dd = (eq - peak) / np.maximum(peak, 1e-9)
            max_dd = float(abs(dd.min()))
        else:
            max_dd = 0.0

        # Regime concentration
        regime_pnl: dict[str, float] | None = None
        if regime_labels and len(regime_labels) == len(pnls):
            from collections import defaultdict

            rp: dict[str, float] = defaultdict(float)
            for label, pnl in zip(regime_labels, pnls, strict=False):
                rp[label] += pnl
            total = sum(abs(v) for v in rp.values()) + 1e-9
            regime_pnl = {k: v / total for k, v in rp.items()}

        # Costs - gross_total is gross profit from winning trades, used as the
        # denominator for the transaction-cost ratio (costs / gross_wins).
        tx_total = float(sum(tx_costs)) if tx_costs else 0.0
        gross_total = float(gross_profit)

        # K1: turnover rate = trades per day
        # n_bars assumed to be 1-min bars; 1440 bars per day
        turnover_rate: float | None = None
        if n_bars and n_bars > 0:
            bars_per_day = 1440.0  # 1-minute bars
            trading_days = n_bars / bars_per_day
            turnover_rate = len(pnls) / max(trading_days, 1.0)

        return self.evaluate(
            sharpe=sharpe,
            profit_factor=profit_factor,
            max_drawdown=max_dd,
            n_trades=len(pnls),
            regime_pnl=regime_pnl,
            gross_pnl=gross_total,
            transaction_costs=tx_total,
            n_obs=len(pnls),
            turnover_rate=turnover_rate,
            avg_latency_ms=avg_latency_ms,
            **kwargs,
        )

    def sweep_confidence_threshold(
        self,
        trade_pnls: "list[float]",
        confidence_scores: "list[float]",
        thresholds: "list[float] | None" = None,
        annualization: float = 252.0,
        min_trades: int = 30,
    ) -> dict:
        """Sweep confidence thresholds to find the one maximising Sharpe.

        For each threshold, keep only trades whose confidence >= threshold,
        recompute Sharpe, and record the result. Returns a dict with per-
        threshold stats and the optimal threshold / Sharpe.
        """
        import numpy as np

        pnls = np.asarray(trade_pnls, dtype=float)
        confs = np.asarray(confidence_scores, dtype=float)
        if len(pnls) != len(confs):
            return {
                "error": "pnls and confidence_scores length mismatch",
                "optimal_threshold": None,
                "optimal_sharpe": None,
                "sweeps": [],
            }
        if len(pnls) == 0:
            return {"optimal_threshold": None, "optimal_sharpe": None, "sweeps": []}

        if thresholds is None:
            thresholds = [round(t, 2) for t in np.arange(0.35, 0.65, 0.01).tolist()]

        sweeps = []
        best_sr = -1e9
        best_thr = None

        for thr in thresholds:
            mask = confs >= thr
            sub = pnls[mask]
            n = int(mask.sum())
            if n < min_trades:
                sweeps.append({"threshold": thr, "n_trades": n, "sharpe": None, "skipped": True})
                continue
            std = float(sub.std(ddof=1)) if n > 1 else 0.0
            sr = float(sub.mean() / std * math.sqrt(annualization)) if std > 1e-12 else 0.0
            sweeps.append({"threshold": thr, "n_trades": n, "sharpe": round(sr, 6), "skipped": False})
            if sr > best_sr:
                best_sr = sr
                best_thr = thr

        return {
            "optimal_threshold": best_thr,
            "optimal_sharpe": round(best_sr, 6) if best_thr is not None else None,
            "sweeps": sweeps,
        }

    def compute_conformal_coverage(
        self, val_logits: "np.ndarray", val_labels: "np.ndarray", alpha: float = 0.10
    ) -> dict:
        """
        Computes nonconformity scores: `score[i] = 1 - softmax(logits[i])[true_label[i]]`
        Calibration: finds the (1-alpha) quantile of scores on the calibration set
        Returns dict with keys: `coverage`, `threshold`, `avg_set_size`
        """
        import numpy as np

        # softmax
        exp_l = np.exp(val_logits - np.max(val_logits, axis=-1, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)

        n = len(val_labels)
        scores = 1.0 - probs[np.arange(n), val_labels]

        # Empirical (1-alpha) quantile
        threshold = float(np.quantile(scores, 1.0 - alpha))

        # Prediction sets: all classes j where 1 - probs[j] <= threshold
        sets = (1.0 - probs) <= threshold

        coverage = float(np.mean(sets[np.arange(n), val_labels]))
        avg_set_size = float(np.mean(np.sum(sets, axis=-1)))

        return {"coverage": coverage, "threshold": threshold, "avg_set_size": avg_set_size}


# ── threshold tuning I/O ─────────────────────────────────────────────────────


def write_threshold_tuning_json(
    sweep_result: dict,
    path: str,
    model_name: str = "unknown",
    extra_meta: dict | None = None,
) -> bool:
    """Persist the output of PromotionGate.sweep_confidence_threshold to JSON.

    Returns True on success, False on error.
    """
    import json as _json
    from pathlib import Path as _Path

    out = _Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model_name,
        "optimal_threshold": sweep_result.get("optimal_threshold"),
        "optimal_sharpe": sweep_result.get("optimal_sharpe"),
        "n_sweep_points": len(sweep_result.get("sweeps", [])),
        "sweeps": sweep_result.get("sweeps", []),
    }
    if extra_meta:
        payload["meta"] = extra_meta

    try:
        with open(out, "w", encoding="utf-8") as f:
            _json.dump(payload, f, indent=2, default=str)
        return True
    except Exception:
        return False


def rank_models(
    results: "list[dict]",
    model_names: "list[str] | None" = None,
    primary: str = "sharpe_per_drawdown",
) -> "list[dict]":
    """
    K: Rank a list of PromotionGate.evaluate() results by capital efficiency.

    Sorts descending by `primary` metric (default: Sharpe/drawdown).
    Each result gets a `rank` field and an `efficiency_score` composite.

    efficiency_score = mean of the three normalised ratios:
        norm(Sharpe/turnover) + norm(Sharpe/drawdown) + norm(Sharpe/latency)
    Skipped metrics (value == -1.0) are excluded from the mean.

    Args:
        results    : List of dicts returned by PromotionGate.evaluate().
        model_names: Optional list of names aligned with results.
        primary    : Key in details to sort by. Options:
                     "sharpe_per_drawdown" | "sharpe_per_turnover" | "sharpe_per_latency"
                     | "sharpe" | "psr"

    Returns:
        Sorted list with added fields: rank, model_name, efficiency_score.
    """
    import numpy as np

    ranked = []
    for i, r in enumerate(results):
        d = r.get("details", {})
        entry = dict(r)
        entry["model_name"] = model_names[i] if model_names and i < len(model_names) else f"model_{i}"
        entry["_sort_key"] = float(d.get(primary, d.get("sharpe", 0.0)) or 0.0)
        ranked.append(entry)

    ranked.sort(key=lambda x: x["_sort_key"], reverse=True)

    # Normalised composite efficiency score
    metric_keys = ["sharpe_per_turnover", "sharpe_per_drawdown", "sharpe_per_latency"]
    for key in metric_keys:
        vals = [r["details"].get(key, -1.0) for r in ranked]
        valid = [v for v in vals if v >= 0]
        if not valid:
            continue
        vmin, vmax = min(valid), max(valid)
        span = max(vmax - vmin, 1e-9)
        for r in ranked:
            v = r["details"].get(key, -1.0)
            r.setdefault("_norm", {})[key] = (v - vmin) / span if v >= 0 else 0.5

    for i, r in enumerate(ranked):
        norms = list(r.pop("_norm", {}).values())
        r["efficiency_score"] = float(np.mean(norms)) if norms else 0.0
        r["rank"] = i + 1
        r.pop("_sort_key", None)

    return ranked


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np

    gate = PromotionGate()

    print("=== Gate 1: Excellent model (should PROMOTE) ===")
    r = gate.evaluate(
        sharpe=1.8,
        profit_factor=1.7,
        max_drawdown=0.12,
        n_trades=700,
        regime_pnl={"trending": 0.5, "neutral": 0.3, "mean_rev": 0.2},
        gross_pnl=10_000.0,
        transaction_costs=1_500.0,
        n_obs=700,
        turnover_rate=3.5,  # 3.5 trades/day
        avg_latency_ms=55.0,  # 55 ms end-to-end
    )
    print(f"  {r['summary']}")
    for reason in r["reasons"]:
        print(f"    {reason}")
    d = r["details"]
    print(f"  K1 Sharpe/turnover  = {d['sharpe_per_turnover']:.3f}  (min {gate.cfg.min_sharpe_per_turnover})")
    print(f"  K2 Sharpe/drawdown  = {d['sharpe_per_drawdown']:.2f}  (min {gate.cfg.min_sharpe_per_drawdown})")
    print(f"  K3 Sharpe/latency   = {d['sharpe_per_latency']:.4f} (min {gate.cfg.min_sharpe_per_latency})")

    print("\n=== Gate 2: High-Sharpe scalper - fails turnover gate (should REJECT) ===")
    r2 = gate.evaluate(
        sharpe=1.9,
        profit_factor=1.4,
        max_drawdown=0.15,
        n_trades=5000,
        gross_pnl=9_000.0,
        transaction_costs=4_000.0,
        n_obs=5000,
        turnover_rate=48.0,  # 48 trades/day - scalper; slippage will kill edge at scale
        avg_latency_ms=8.0,  # needs sub-10ms - fragile to infra variance
    )
    print(f"  {r2['summary']}")
    for reason in r2["reasons"]:
        print(f"    {reason}")

    print("\n=== Gate 3: Low-Sharpe high-drawdown (should REJECT) ===")
    r3 = gate.evaluate(
        sharpe=1.6,
        profit_factor=1.3,
        max_drawdown=0.25,
        n_trades=300,
        regime_pnl={"trending": 0.92, "neutral": 0.08},
        gross_pnl=8_000.0,
        transaction_costs=5_000.0,
        n_obs=300,
    )
    print(f"  {r3['summary']}")

    print("\n=== Gate 4: evaluate_from_history + rank_models ===")
    rng = np.random.default_rng(42)
    pnls = rng.normal(0.002, 0.01, 600)
    equity = np.cumprod(1 + pnls / 100) * 10_000
    r4 = gate.evaluate_from_history(
        trade_pnls=pnls.tolist(),
        equity_curve=equity.tolist(),
        n_bars=864_000,  # 600 days x 1440 bars
        avg_latency_ms=60.0,
    )
    print(f"  {r4['summary']}")
    print(f"  turnover_rate computed = {r4['details'].get('turnover_rate', 'n/a'):.2f} trades/day")

    print("\n=== K: rank_models (3 hypothetical models) ===")
    candidates = [
        gate.evaluate(
            sharpe=2.1,
            profit_factor=1.8,
            max_drawdown=0.10,
            n_trades=800,
            gross_pnl=12000,
            transaction_costs=1200,
            n_obs=800,
            turnover_rate=2.0,
            avg_latency_ms=50.0,
        ),
        gate.evaluate(
            sharpe=2.4,
            profit_factor=1.6,
            max_drawdown=0.28,
            n_trades=4000,
            gross_pnl=11000,
            transaction_costs=4000,
            n_obs=4000,
            turnover_rate=40.0,
            avg_latency_ms=6.0,
        ),
        gate.evaluate(
            sharpe=1.7,
            profit_factor=1.9,
            max_drawdown=0.08,
            n_trades=400,
            gross_pnl=9000,
            transaction_costs=900,
            n_obs=400,
            turnover_rate=1.5,
            avg_latency_ms=80.0,
        ),
    ]
    ranked = rank_models(candidates, model_names=["haelt", "mamba_scalper", "tft_swing"])
    print(f"  {'Rank':<6} {'Model':<16} {'Sharpe':>7} {'S/DD':>7} {'S/Turn':>8} {'S/Lat':>8} {'Eff':>7} {'Status'}")
    print(f"  {'-' * 65}")
    for m in ranked:
        d = m["details"]
        st = "✅ PROMOTE" if m["promoted"] else "❌ REJECT"
        print(
            f"  #{m['rank']:<5} {m['model_name']:<16} "
            f"{d['sharpe']:>7.2f} {d['sharpe_per_drawdown']:>7.1f} "
            f"{d.get('sharpe_per_turnover', -1):>8.3f} "
            f"{d.get('sharpe_per_latency', -1):>8.4f} "
            f"{m['efficiency_score']:>7.3f}  {st}"
        )
    print("OK ✓")
