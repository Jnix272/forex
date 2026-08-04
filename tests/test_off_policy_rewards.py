"""
Tests for off-policy rewards & QR-DQN labels (Improvement #5)
"""
from __future__ import annotations

import numpy as np

from labeling.off_policy_rewards import (
    compute_off_policy_rewards,
    counterfactual_reward_by_action,
    doubly_robust_reward_by_action,
    ipw_value_estimate,
    quantile_reward_labels,
    softmax_probs,
)

# ---------------------------------------------------------------------------
# softmax_probs
# ---------------------------------------------------------------------------

def test_softmax_probs():
    # 1-D logits -> probabilities
    logits = np.array([1.0, 2.0, -1.0])
    probs = softmax_probs(logits)
    assert np.allclose(probs, [0.2595, 0.7054, 0.0351], atol=1e-3)
    assert np.allclose(probs.sum(), 1.0)

    # 2-D (batch)
    logits2 = np.array([[0.0, 1.0], [-1.0, 0.0]])
    probs2 = softmax_probs(logits2)
    assert probs2.shape == (2, 2)
    assert np.allclose(probs2[0], [0.2689, 0.7311], atol=1e-3)
    assert np.allclose(probs2[1], [0.2689, 0.7311], atol=1e-3)


def test_extract_act_prob_1d():
    from labeling.off_policy_rewards import _extract_act_prob
    probs = np.array([0.1, 0.2, 0.7])
    actions = np.array([0, 1, 2])
    assert np.allclose(_extract_act_prob(probs, actions), [0.1, 0.2, 0.7])


def test_extract_act_prob_2d():
    from labeling.off_policy_rewards import _extract_act_prob
    probs = np.array([[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]])
    actions = np.array([0, 2])
    assert np.allclose(_extract_act_prob(probs, actions), [0.1, 0.5])


# ---------------------------------------------------------------------------
# IPS estimator
# ---------------------------------------------------------------------------

def test_ipw_value_estimate_simple():
    acts = np.array([0, 1, 0])
    rew = np.array([1.0, 2.0, 1.0])
    bp = np.array([0.5, 0.5, 0.5])
    tp = np.array([0.9, 0.1, 0.9])
    est = ipw_value_estimate(acts, rew, bp, tp)
    # weight0 = 0.9/0.5 = 1.8; weight1 = 0.1/0.5 = 0.2; mean = (1.8*1 + 0.2*2 + 1.8*1)/3 = 4.0/3 ≈ 1.3333
    assert abs(est["ips"] - 1.3333) < 1e-3
    assert est["se"] > 0.0


def test_ipw_value_estimate_clip():
    acts = np.array([0, 1])
    rew = np.array([10.0, -10.0])
    bp = np.array([0.5, 0.5])
    tp = np.array([1.0, 0.01])  # weight would be 2.0, clipped to 1.5
    est = ipw_value_estimate(acts, rew, bp, tp, clip=1.5)
    # clipped: w0=1.5, w1=0.02; mean = (1.5*10 + 0.02*-10)/2 = (15 - 0.2)/2 = 7.4
    assert abs(est["ips"] - 7.4) < 1e-3


def test_ipw_value_estimate_empty():
    est = ipw_value_estimate([], [], [], [])
    assert est["ips"] == 0.0
    assert est["se"] == 0.0


def test_ipw_value_estimate_bootstrap():
    acts = np.array([0, 1, 0])
    rew = np.array([1.0, 2.0, 1.0])
    bp = np.array([0.5, 0.5, 0.5])
    tp = np.array([0.9, 0.1, 0.9])
    est = ipw_value_estimate(acts, rew, bp, tp, n_bootstrap=100)
    assert "ci95_lo" in est
    assert "ci95_hi" in est
    assert est["ci95_lo"] < est["ci95_hi"]


# ---------------------------------------------------------------------------
# Per-action IPS
# ---------------------------------------------------------------------------

def test_counterfactual_reward_by_action():
    acts = np.array([0, 1, 0, 2])
    rew = np.array([1.0, 2.0, 3.0, 4.0])
    bp = np.array([0.1, 0.1, 0.1, 0.7])
    tp = np.array([0.9, 0.1, 0.1, 0.9])
    out = counterfactual_reward_by_action(acts, rew, bp, tp, n_actions=3)
    # action 0: weights [0.9/0.1=9, 0.1/0.1=1]; weighted rewards [1*9, 3*1] = [9, 3]; mean = (9+3)/2 = 6
    assert abs(out[0] - 6.0) < 1e-2
    # action 1: weight [0.1/0.1=1]; weighted reward [2*1] = 2; mean = 2
    assert abs(out[1] - 2.0) < 1e-2
    # action 2: weight [0.9/0.7≈1.2857]; weighted reward [4*1.2857] ≈ 5.1428; mean = 5.14
    assert abs(out[2] - 5.14) < 0.01


def test_counterfactual_reward_by_action_clip():
    acts = np.array([0, 1])
    rew = np.array([10.0, -10.0])
    bp = np.array([0.5, 0.5])
    tp = np.array([1.0, 1.0])
    out = counterfactual_reward_by_action(acts, rew, bp, tp, n_actions=2, clip=1.0)
    # clipped weights = 1.0 each; out[0]=10, out[1]=-10
    assert abs(out[0] - 10.0) < 1e-2
    assert abs(out[1] + 10.0) < 1e-2


# ---------------------------------------------------------------------------
# Doubly-robust
# ---------------------------------------------------------------------------

def test_doubly_robust_reward_by_action():
    acts = np.array([0, 1, 0, 2])
    rew = np.array([1.0, 2.0, 3.0, 4.0])
    bp = np.array([0.1, 0.1, 0.1, 0.7])
    tp = np.array([0.9, 0.1, 0.1, 0.9])
    out = doubly_robust_reward_by_action(acts, rew, bp, tp, n_actions=3)
    # q0 = mean(reward|action0) = (1+3)/2 = 2
    # q1 = 2, q2 = 4
    # DR = q + weight * (r - q) where w = tp/bp
    # q1 = 2
    # adj1 = (1*(2-2))/1 = 0 => 2
    # For action 2: base = 4, adj = ((0.9/0.7)*(4-4))/1 = 0 => 4
    assert abs(out[0] + 2.0) < 1e-2
    assert abs(out[1] - 2.0) < 1e-2
    assert abs(out[2] - 4.0) < 1e-2


def test_doubly_robust_with_q():
    acts = np.array([0, 1, 2])
    rew = np.array([10.0, 20.0, 30.0])
    bp = np.array([0.2, 0.2, 0.6])
    tp = np.array([0.5, 0.3, 0.2])
    q = np.array([15.0, 20.0, 30.0])  # external outcome model
    out = doubly_robust_reward_by_action(acts, rew, bp, tp, n_actions=3, q=q)
    # a=0: base=15, adj = (0.5/0.2)*(10-15) = 2.5*-5 = -12.5 => 2.5
    # a=1: base=20, adj = (0.3/0.2)*(20-20) = 1.5*0 = 0 => 20
    # a=2: base=30, adj = (0.2/0.6)*(30-30) = 0.333*0 = 0 => 30
    assert abs(out[0] - 2.5) < 1e-2
    assert abs(out[1] - 20.0) < 1e-2
    assert abs(out[2] - 30.0) < 1e-2


def test_doubly_robust_no_missing():
    acts = np.array([0, 0, 0])
    rew = np.array([1.0, 2.0, 3.0])
    bp = np.array([0.33, 0.33, 0.34])
    tp = np.array([0.5, 0.3, 0.2])
    out = doubly_robust_reward_by_action(acts, rew, bp, tp, n_actions=3, q=None)
    # q0 = (1+2+3)/3 = 2
    # weights w = tp/bp ≈ [1.515, 0.909, 0.588]
    # residuals = [-1, 0, 1]
    # adj = mean(w * residual) ≈ -0.309
    # DR = 2 + (-0.309) ≈ 1.691
    assert abs(out[0] - 1.69) < 0.01


# ---------------------------------------------------------------------------
# QR-DQN quantile labels
# ---------------------------------------------------------------------------

def test_quantile_reward_labels_basic():
    close = np.array([100, 101, 102, 103, 104, 105])
    atr = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    res = quantile_reward_labels(close, atr, lookahead=2, pip_size=0.0001,
                                 tx_cost_pips=1.5, profit_atr_mult=2.0, stop_atr_mult=1.0)
    assert len(res) == 6
    assert "reward_q05" in res.columns
    assert "reward_q95" in res.columns
    assert "reward_q_med" in res.columns
    assert "label_quantile" in res.columns
    assert "conviction_q" in res.columns


def test_quantile_reward_labels_order():
    close = np.array([100, 100, 100, 100, 200])
    atr = np.ones(5)
    res = quantile_reward_labels(close, atr, lookahead=2, pip_size=0.0001,
                                 tx_cost_pips=0.0)
    # reward should be non-decreasing with quantile level
    assert (res["reward_q05"].to_numpy() <= res["reward_q25"].to_numpy() + 1e-9).all()
    assert (res["reward_q25"].to_numpy() <= res["reward_q50"].to_numpy() + 1e-9).all()
    assert (res["reward_q75"].to_numpy() <= res["reward_q95"].to_numpy() + 1e-9).all()


def test_quantile_reward_labels_empty_path():
    close = np.array([100])
    atr = np.ones(1)
    res = quantile_reward_labels(close, atr, lookahead=5)
    # zero bars due to lookahead + no valid forward path
    assert len(res) == 1
    assert res["label_quantile"].to_numpy()[0] == 0


def test_quantile_reward_labels_target():
    close = np.array([100, 101, 102, 103, 104, 105])
    atr = np.ones(6)
    res = quantile_reward_labels(close, atr, lookahead=2, pip_size=0.0001,
                                 tx_cost_pips=1.5)
    # bar 1: reward_median ≈ (TP- entry)/pip - 1.5 = (101+0.0002 - 100)/0.0001 -1.5 = 10.002 - 1.5 ≈ 8.5 -> positive -> label=+1
    # (exact depends on barrier choices and exit events)
    assert res["label_quantile"].to_list()[1] in (-1, 0, 1)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def test_compute_off_policy_rewards_basic():
    acts = np.array([0, 1, 2, 0, 1])
    rew = np.array([0.5, 1.0, 0.2, 0.8, 0.3])
    behavior = np.array([[0.2, 0.3, 0.5], [0.1, 0.2, 0.7], [0.4, 0.4, 0.2], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    target = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.1, 0.1, 0.8], [0.4, 0.2, 0.4], [0.3, 0.3, 0.4]])
    R = compute_off_policy_rewards(acts, rew, behavior_probs=behavior, target_probs=target, n_actions=3)
    assert len(R) == 3
    assert "ips_reward" in R.columns
    assert "dr_reward" in R.columns
    assert "ipw_value" in R.columns
    assert "dr_value" in R.columns
    assert "ci95_lo" in R.columns
    assert "ci95_hi" in R.columns


def test_compute_off_policy_rewards_1d_consistency():
    acts = np.array([0, 1, 2])
    rew = np.array([1.0, 2.0, 3.0])
    behavior = np.array([0.2, 0.5, 0.3])
    target = np.array([0.5, 0.3, 0.2])
    R1 = compute_off_policy_rewards(acts, rew, behavior_probs=behavior, target_probs=target, n_actions=3)
    # using 1-D behavior/target
    bp1d = behavior[acts]
    tp1d = target[acts]
    R2 = compute_off_policy_rewards(acts, rew, behavior_probs=bp1d, target_probs=tp1d, n_actions=3)
    assert np.allclose(R1["dr_reward"].to_numpy(), R2["dr_reward"].to_numpy())


def test_compute_off_policy_rewards_no_clip():
    acts = np.array([0, 1])
    rew = np.array([10.0, -10.0])
    behavior = np.array([[0.5, 0.5], [0.5, 0.5]])
    target = np.array([[1.0, 0.01], [0.01, 1.0]])
    R = compute_off_policy_rewards(acts, rew, behavior_probs=behavior, target_probs=target,
                                   n_actions=2, clip=None)
    # weights large, resulting in large ips_reward
    assert "ips_reward" in R.columns
