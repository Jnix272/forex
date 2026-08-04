"""
Off-Policy Rewards & QR-DQN Distributional Labels (Improvement #5)
==================================================================
Counterfactual (inverse-propensity / doubly-robust) reward estimation for
off-policy evaluation of RL agents, and distributional QR-DQN reward labels.

Everything is deterministic and self-contained (numpy/scipy only):

  softmax_probs / behavior vs target policies
  ipw_value_estimate        : IPS estimate of a target policy's value, with
                              optional weight clipping and bootstrap CI
  counterfactual_reward_by_action : per-action IPS counterfactual rewards
  doubly_robust_reward_by_action  : per-action doubly-robust estimates
  quantile_reward_labels    : QR-DQN distributional labels — per-bar quantiles
                              of the forward exit-reward distribution (long/short)
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

# ════════════════════════════════════════════════════════════════════════════
# 1. Propensity helpers
# ════════════════════════════════════════════════════════════════════════════

def softmax_probs(logits: np.ndarray) -> np.ndarray:
    """Softmax over logits (last axis) -> action probability distribution."""
    l = np.asarray(logits, dtype=float)
    l = l - np.max(l, axis=-1, keepdims=True)
    e = np.exp(l)
    return e / np.sum(e, axis=-1, keepdims=True)


# ════════════════════════════════════════════════════════════════════════════
# 2. Inverse Propensity Score (IPS) estimators
# ════════════════════════════════════════════════════════════════════════════

def _valid_index(actions, rewards, behavior_probs, target_probs):
    n = len(actions)
    idx = np.arange(n)
    return idx[np.isfinite(rewards)]


def _extract_act_prob(probs: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Return the probability each realized action was assigned.

    Accepts either a 1-D array (per-sample probability of the realized action)
    or a 2-D policy matrix (n_samples x n_actions).
    """
    p = np.asarray(probs, dtype=float)
    if p.ndim == 1:
        return p
    a = np.asarray(actions).astype(int)
    n = min(len(a), len(p))
    return np.take_along_axis(p[:n], a[:n, None], axis=1).ravel()


def ipw_value_estimate(
    actions: np.ndarray,
    rewards: np.ndarray,
    behavior_probs: np.ndarray,
    target_probs: np.ndarray,
    clip: float | None = None,
    n_bootstrap: int = 0,
    seed: int = 0,
) -> dict[str, float]:
    """IPS estimate of the value of a target policy.

    ``actions``          : int array of realized actions (behavior policy)
    ``rewards``          : realized rewards for those actions
    ``behavior_probs``   : probability the behavior policy assigned to each
                           realized action (same length as ``actions``)
    ``target_probs``     : probability the target policy assigns to each
                           realized action (same length as ``actions``)
    ``clip``             : optional winsorization cap on IPS weights
    ``n_bootstrap``      : >0 to also report a bootstrap 95% CI

    Returns ``{"ips", "se", "ci95_lo", "ci95_hi", "weights"}`` (weights
    included when ``clip`` is None; else clipped weights).
    """
    actions = np.asarray(actions).astype(int)
    rewards = np.asarray(rewards, dtype=float)
    bp = _extract_act_prob(behavior_probs, actions)
    tp = _extract_act_prob(target_probs, actions)
    n = min(len(actions), len(rewards), len(bp), len(tp))
    if n == 0:
        return {"ips": 0.0, "se": 0.0, "ci95_lo": 0.0, "ci95_hi": 0.0}

    w = tp[:n] / np.maximum(bp[:n], 1e-12)
    if clip is not None:
        w = np.minimum(w, float(clip))
    vals = w * rewards[:n]

    ips = float(np.mean(vals))
    se = float(np.std(vals) / np.sqrt(n))
    lo, hi = ips - 1.96 * se, ips + 1.96 * se

    if n_bootstrap > 0:
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(int(n_bootstrap)):
            boots.append(float(np.mean(rng.choice(vals, size=n, replace=True))))
        qs = np.quantile(boots, [0.025, 0.975])
        lo, hi = float(qs[0]), float(qs[1])

    return {"ips": ips, "se": se, "ci95_lo": lo, "ci95_hi": hi,
            "weights": w}


def counterfactual_reward_by_action(
    actions: np.ndarray,
    rewards: np.ndarray,
    behavior_probs: np.ndarray,
    target_probs: np.ndarray,
    n_actions: int,
    clip: float | None = None,
) -> np.ndarray:
    """Per-action IPS counterfactual reward estimates.

    For each action ``a``: mean of target-policy reweighted rewards among
    samples that actually took ``a``. Returns an (n_actions,) array; actions
    with no samples under the behavior policy get 0.0.
    """
    actions = np.asarray(actions).astype(int)
    rewards = np.asarray(rewards, dtype=float)
    bp = _extract_act_prob(behavior_probs, actions)
    tp = _extract_act_prob(target_probs, actions)
    n = min(len(actions), len(rewards), len(bp), len(tp))
    out = np.zeros(n_actions, dtype=float)
    if n == 0:
        return out
    w = tp[:n] / np.maximum(bp[:n], 1e-12)
    if clip is not None:
        w = np.minimum(w, float(clip))
    for a in range(n_actions):
        m = actions[:n] == a
        if m.sum() == 0:
            continue
        out[a] = float(np.mean(w[m] * rewards[:n][m]))
    return out


# ════════════════════════════════════════════════════════════════════════════
# 3. Doubly-robust estimator
# ════════════════════════════════════════════════════════════════════════════

def doubly_robust_reward_by_action(
    actions: np.ndarray,
    rewards: np.ndarray,
    behavior_probs: np.ndarray,
    target_probs: np.ndarray,
    n_actions: int,
    q: np.ndarray | None = None,
    clip: float | None = None,
) -> np.ndarray:
    """Per-action doubly-robust counterfactual reward estimates.

    ``q`` : optional (n,) outcome-model prediction of each sample's reward
    under its realized action. When None, a per-action sample-mean outcome
    model is used (constant within action).

    DR_a = q_mean_a + mean over samples taking ``a`` of
           w_i * (r_i - q_i),  w_i = target_pi(a|x)/behavior_mu(a|x).
    """
    actions = np.asarray(actions).astype(int)
    rewards = np.asarray(rewards, dtype=float)
    bp = _extract_act_prob(behavior_probs, actions)
    tp = _extract_act_prob(target_probs, actions)
    n = min(len(actions), len(rewards), len(bp), len(tp))
    out = np.zeros(n_actions, dtype=float)
    if n == 0:
        return out

    if q is None:
        q_model = np.zeros(n, dtype=float)
        for a in range(n_actions):
            m = actions[:n] == a
            if m.sum() > 0:
                q_model[m] = np.mean(rewards[:n][m])
    else:
        q_model = np.asarray(q, dtype=float)[:n]

    w = tp[:n] / np.maximum(bp[:n], 1e-12)
    if clip is not None:
        w = np.minimum(w, float(clip))
    residual = rewards[:n] - q_model

    for a in range(n_actions):
        m = actions[:n] == a
        if m.sum() == 0:
            out[a] = 0.0
            continue
        base = float(np.mean(q_model[m]))
        adj = float(np.mean(w[m] * residual[m]))
        out[a] = base + adj
    return out


# ════════════════════════════════════════════════════════════════════════════
# 4. QR-DQN distributional reward labels
# ════════════════════════════════════════════════════════════════════════════

def quantile_reward_labels(
    close: np.ndarray,
    atr: np.ndarray,
    lookahead: int,
    pip_size: float = 0.0001,
    tx_cost_pips: float = 1.5,
    profit_atr_mult: float = 1.5,
    stop_atr_mult: float = 1.0,
    levels: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
    spread_half: np.ndarray | None = None,
    execution_delay_bars: int = 0,
    valid: np.ndarray | None = None,
) -> pl.DataFrame:
    """Per-bar quantile (QR-DQN) reward labels.

    For each bar, the distribution of outcomes is built from the forward
    path over ``lookahead`` bars: for every possible time-exit k in
    1..lookahead, the net reward of a long exit and a short exit is computed
    (against entry ± ATR×mult barriers), and the better side is taken. The
    quantiles of that {reward}_k distribution become the distributional labels.

    Returns a Polars DataFrame with one row per bar:
      reward_q05..reward_q95 : reward quantiles (in pips, net of cost)
      reward_q_med           : median (alias of reward_q50)
      label_quantile         : {-1,0,+1} sign of the median, 0 when |median| < cost
      conviction_q           : P(best-exit reward > 0) from the distribution
    """
    close = np.asarray(close, dtype=float)
    atr = np.asarray(atr, dtype=float)
    n = len(close)
    H = max(1, int(lookahead))
    if spread_half is not None:
        spread_half = np.asarray(spread_half, dtype=float)
        entry_long = close + spread_half
        entry_short = close - spread_half
        exit_long_path = close - spread_half
        exit_short_path = close + spread_half
    else:
        entry_long = close.copy()
        entry_short = close.copy()
        exit_long_path = close.copy()
        exit_short_path = close.copy()

    valid_m = np.ones(n, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    delay = max(0, int(execution_delay_bars))
    lev = list(levels)

    out_q = np.zeros((n, len(lev)), dtype=float)
    out_conv = np.zeros(n, dtype=float)
    for i in range(n - H - delay):
        ei = i + delay
        if not valid_m[i] or not valid_m[ei]:
            continue
        el, es = entry_long[ei], entry_short[ei]
        tp_l, sl_l = el + profit_atr_mult * atr[ei], el - stop_atr_mult * atr[ei]
        tp_s, sl_s = es - profit_atr_mult * atr[ei], es + stop_atr_mult * atr[ei]
        dist_l = []
        dist_s = []
        for k in range(1, H + 1):
            p = ei + k
            if p >= n:
                break
            rl = exit_long_path[p]
            rs = exit_short_path[p]
            if rl >= tp_l:
                pnl_l = tp_l - el
            elif rl <= sl_l:
                pnl_l = sl_l - el
            else:
                pnl_l = rl - el
            if rs <= tp_s:
                pnl_s = es - tp_s
            elif rs >= sl_s:
                pnl_s = es - sl_s
            else:
                pnl_s = es - rs
            dist_l.append(pnl_l / pip_size - tx_cost_pips)
            dist_s.append(pnl_s / pip_size - tx_cost_pips)
        if not dist_l:
            continue
        best = np.maximum(np.asarray(dist_l), np.asarray(dist_s))
        out_q[i] = np.quantile(best, lev)
        out_conv[i] = float(np.mean(best > 0.0))

    cols = {f"reward_q{int(round(q * 100)):02d}": out_q[:, k] for k, q in enumerate(lev)}
    med_idx = lev.index(0.5) if 0.5 in lev else min(range(len(lev)), key=lambda k: abs(lev[k] - 0.5))
    med = out_q[:, med_idx]
    cost = float(tx_cost_pips)
    label = np.where(med > cost, 1, np.where(med < -cost, -1, 0)).astype(int)
    return pl.DataFrame({
        **cols,
        "reward_q_med": med,
        "label_quantile": label,
        "conviction_q": out_conv,
    })


# ════════════════════════════════════════════════════════════════════════════
# 5. Orchestrator
# ════════════════════════════════════════════════════════════════════════════

def compute_off_policy_rewards(
    actions: np.ndarray,
    rewards: np.ndarray,
    behavior_logits: np.ndarray | None = None,
    behavior_probs: np.ndarray | None = None,
    target_logits: np.ndarray | None = None,
    target_probs: np.ndarray | None = None,
    n_actions: int = 3,
    clip: float | None = 10.0,
    n_bootstrap: int = 200,
    seed: int = 0,
) -> pl.DataFrame:
    """Compute IPS + doubly-robust counterfactual rewards for an RL episode.

    Either (logits) or (probs) may be provided for the behavior and target
    policies. Returns a Polars DataFrame with:
      action, ips_reward, dr_reward, behavior_n, weight_mean, weight_max,
      ipw_value, dr_value, ci95_lo, ci95_hi.
    """
    if behavior_probs is None:
        if behavior_logits is None:
            raise ValueError("behavior_probs or behavior_logits required")
        behavior_probs = softmax_probs(np.asarray(behavior_logits, dtype=float))
    if target_probs is None:
        if target_logits is None:
            raise ValueError("target_probs or target_logits required")
        target_probs = softmax_probs(np.asarray(target_logits, dtype=float))

    actions = np.asarray(actions).astype(int)
    rewards = np.asarray(rewards, dtype=float)
    n = min(len(actions), len(rewards), len(behavior_probs), len(target_probs))
    actions, rewards = actions[:n], rewards[:n]

    ips_by_action = counterfactual_reward_by_action(
        actions, rewards, behavior_probs, target_probs, n_actions, clip)
    dr_by_action = doubly_robust_reward_by_action(
        actions, rewards, behavior_probs, target_probs, n_actions, clip=clip)

    est = ipw_value_estimate(actions, rewards, behavior_probs, target_probs, clip, n_bootstrap, seed)
    w = est["weights"]

    rows = []
    for a in range(n_actions):
        m = actions == a
        rows.append({
            "action": int(a),
            "ips_reward": float(ips_by_action[a]),
            "dr_reward": float(dr_by_action[a]),
            "behavior_n": int(m.sum()),
            "weight_mean": float(np.mean(w[m])) if m.sum() else 0.0,
            "weight_max": float(np.max(w[m])) if m.sum() else 0.0,
        })
    out = pl.DataFrame(rows)
    out = out.with_columns([
        pl.lit(float(est["ips"])).alias("ipw_value"),
        pl.lit(float(np.mean(dr_by_action))).alias("dr_value"),
        pl.lit(float(est["ci95_lo"])).alias("ci95_lo"),
        pl.lit(float(est["ci95_hi"])).alias("ci95_hi"),
    ])
    return out
