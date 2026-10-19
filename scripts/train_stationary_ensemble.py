"""Production Model Trainer: Leak-Free Out-of-Sample Split for the 4H Stationary Ensemble.

Trains and exports the full Cross-Sectional + Cost-Vol Ridge + Differentiable
Turnover-Penalized Net-Sharpe Policy + LightGBM Booster + Hysteresis Buffer
parameters (`z_entry`, `z_exit`) strictly on pre-holdout rows (`row < 39,065`,
`bar_idx < HOLDOUT_START - EMBARGO`).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_walkforward_4h import (
    ALL_PAIRS,
    CLEAN_BASE_FEATURES,
    SLIP,
    build_engineered_feature_matrix,
    fit_diff_net_sharpe_policy,
    predict_diff_sharpe,
    sharpe,
    simulate_hysteresis_daily,
)

print("=" * 95)
print("TRAINING LEAK-FREE PRODUCTION 4-HOUR CROSS-SECTIONAL + HYSTERESIS ENSEMBLE (PRE-HOLDOUT SPLIT)")
print("=" * 95, flush=True)

t0 = time.time()
data = np.load(ROOT / ".tmp" / "all_584_audit_data.npz")
X_all = data["X"]
idx = data["idx"]
close = data["close"]
spread = data["spread"]
feature_names = json.load(open(ROOT / "checkpoints" / "haelt" / "haelt_fold0_best_features.json"))
m = len(X_all)

ACTIVE_PAIRS = ["EURUSD", "USDJPY", "USDCAD"]
pair_indices = {p: i for i, p in enumerate(ALL_PAIRS)}

STATIONARY_BASE_FEATURES = list(CLEAN_BASE_FEATURES)
base_names_146 = [fn.split("::")[-1] for fn in feature_names[:146]]
curated_indices = [j for j, fn in enumerate(feature_names) if fn.split("::")[-1] in STATIONARY_BASE_FEATURES]
curated_names = [feature_names[j] for j in curated_indices]
X = X_all[:, curated_indices]

# Build full engineered cross-sectional feature matrices per pair (89 features/pair)
per_pair_X, atr_60_pips, eng_feat_names = build_engineered_feature_matrix(X_all, idx, close, feature_names)

H_BARS = 48
EMBARGO_BARS = 96
BPD = 288
# Strictly exclude the last 20% of samples (~2025 holdout span) from training
TRAIN_CUTOFF_ROW = int(m * 0.80)
TRAIN_MAX_BAR_IDX = int(idx[TRAIN_CUTOFF_ROW]) - EMBARGO_BARS

print(
    f"Loaded {m:,} samples | Training strictly on rows [0, {TRAIN_CUTOFF_ROW:,}) "
    f"(bar_idx < {TRAIN_MAX_BAR_IDX:,}) | Reserved {m - TRAIN_CUTOFF_ROW:,} OOS rows"
)

out_dir = ROOT / "checkpoints" / "stationary_ensemble"
out_dir.mkdir(parents=True, exist_ok=True)
models = {}
thresholds = {}
hysteresis_params = {}
oos_metrics = {}
cs_policy_bundle: dict[str, np.ndarray] = {}

lo_day_oos = int(idx[TRAIN_CUTOFF_ROW]) // BPD
n_days_oos = (int(idx[-1]) + H_BARS) // BPD - lo_day_oos + 1

for pr in ALL_PAIRS:
    pi = pair_indices[pr]
    pip = 0.01 if "JPY" in pr else 0.0001
    cost_bps = ((spread[:, pi] + SLIP * pip) / np.maximum(close[:, pi], 1e-6)) * 1e4
    cost_pips = (spread[:, pi] + SLIP * pip) / pip
    cp = close[:, pi]
    Xp = per_pair_X[pi]
    atrp = atr_60_pips[:, pi]

    print(f"\nTraining leak-free production model & CS-Hysteresis policy for {pr}...", flush=True)

    ok = idx + H_BARS < close.shape[0]
    fwd = np.full(m, np.nan, dtype=np.float32)
    fwd[ok] = ((close[idx[ok] + H_BARS, pi] / close[idx[ok], pi]) - 1.0) * 1e4
    fwd_net = np.sign(fwd) * np.maximum(0.0, np.abs(fwd) - 0.5 * cost_bps[idx])

    fwd_pips = np.full(m, np.nan, dtype=np.float64)
    fwd_pips[ok] = (cp[idx[ok] + H_BARS] - cp[idx[ok]]) / pip

    tr_mask = np.isfinite(fwd_net) & (idx < TRAIN_MAX_BAR_IDX)
    oos_mask = np.isfinite(fwd) & (idx >= int(idx[TRAIN_CUTOFF_ROW]))
    tr_indices = np.where(tr_mask)[0]
    oos_indices = np.where(oos_mask)[0]

    # 1. Fit LightGBM Booster on curated 180 stationary base features
    mdl = lgb.LGBMRegressor(
        max_depth=4,
        learning_rate=0.03,
        n_estimators=130,
        min_child_samples=150,
        reg_lambda=15.0,
        subsample=0.8,
        colsample_bytree=0.7,
        n_jobs=4,
        verbose=-1,
        random_state=42,
    )
    mdl.fit(X[tr_mask], np.clip(fwd_net[tr_mask], -120, 120))

    preds_tr_lgb = mdl.predict(X[tr_mask])
    thresh = float(np.percentile(np.abs(preds_tr_lgb), 75.0))
    thresholds[pr] = thresh

    model_path = out_dir / f"{pr.lower()}_4h_lgb.txt"
    mdl.booster_.save_model(str(model_path))
    models[pr] = mdl

    # 2. Fit Cross-Sectional + Cost-Vol Ridge + Differentiable Net-Sharpe Policy strictly on tr_indices
    y_cost_vol = np.zeros(m, dtype=np.float64)
    fin = np.isfinite(fwd_pips)
    y_cost_vol[fin] = np.clip(
        np.sign(fwd_pips[fin])
        * np.maximum(0.0, np.abs(fwd_pips[fin]) - 0.5 * cost_pips[idx[fin]])
        / np.maximum(atrp[fin], 2.0),
        -4.0,
        4.0,
    )
    sample_w = np.clip(atrp / np.maximum(cost_pips[idx], 0.5), 0.5, 5.0)
    roll_indices = tr_indices[-6500:] if len(tr_indices) > 6500 else tr_indices

    # Model A: Expanding Cost-Vol Ridge
    mu_exp = np.median(Xp[tr_indices], axis=0)
    sd_exp = np.std(Xp[tr_indices], axis=0) + 1e-6
    Z_tr_exp = np.clip((Xp[tr_indices] - mu_exp) / sd_exp, -4.0, 4.0)
    ridge_exp = Ridge(alpha=5000.0)
    ridge_exp.fit(Z_tr_exp, y_cost_vol[tr_indices], sample_weight=sample_w[tr_indices])
    pred_exp_tr = ridge_exp.predict(Z_tr_exp)
    exp_mean, exp_std = float(np.mean(pred_exp_tr)), float(np.std(pred_exp_tr) + 1e-9)

    # Model B: Rolling-Window Top-35 Ridge
    mu_rol = np.median(Xp[roll_indices], axis=0)
    sd_rol = np.std(Xp[roll_indices], axis=0) + 1e-6
    Z_tr_rol = np.clip((Xp[roll_indices] - mu_rol) / sd_rol, -4.0, 4.0)
    corrs_rol = np.abs(
        (Z_tr_rol.T @ (y_cost_vol[roll_indices] - y_cost_vol[roll_indices].mean()))
        / (len(roll_indices) * (np.std(Z_tr_rol, axis=0) * np.std(y_cost_vol[roll_indices]) + 1e-9))
    )
    top35 = np.argsort(np.nan_to_num(corrs_rol, nan=0.0))[::-1][:35]
    ridge_rol = Ridge(alpha=3000.0)
    ridge_rol.fit(Z_tr_rol[:, top35], y_cost_vol[roll_indices], sample_weight=sample_w[roll_indices])
    pred_rol_tr = ridge_rol.predict(Z_tr_rol[:, top35])
    rol_mean, rol_std = float(np.mean(pred_rol_tr)), float(np.std(pred_rol_tr) + 1e-9)

    # Model C: Differentiable Turnover-Penalized Net-Sharpe Policy
    sel_ds, mu_ds, sc_ds, theta_ds = fit_diff_net_sharpe_policy(
        Xp[tr_indices],
        fwd_pips[tr_indices],
        cost_pips[idx[tr_indices]],
        atrp[tr_indices],
        top_k=35,
    )
    pred_ds_tr = predict_diff_sharpe(Xp[tr_indices], sel_ds, mu_ds, sc_ds, theta_ds)
    ds_mean, ds_std = float(np.mean(pred_ds_tr)), float(np.std(pred_ds_tr) + 1e-9)

    # Model D: Zero-Overfit Cross-Sectional Composite
    cs_col = Xp[:, -1]
    cs_mean, cs_std = float(np.mean(cs_col[tr_indices])), float(np.std(cs_col[tr_indices]) + 1e-9)
    cot_comm_idx = base_names_146.index("cot_net_comm") if "cot_net_comm" in base_names_146 else -1
    cot_std_val = (
        float(np.std(X_all[:15000, pi * 146 + cot_comm_idx]) + 1e-6) if cot_comm_idx >= 0 else 1.0
    )

    if pr in ("EURUSD", "GBPUSD"):
        w_blend = np.array([0.40, 0.25, 0.20, 0.15], dtype=np.float64)
        z_tr_blend = (
            0.40 * ((pred_exp_tr - exp_mean) / exp_std)
            + 0.40 * ((pred_ds_tr - ds_mean) / ds_std)
            + 0.20 * ((cs_col[tr_indices] - cs_mean) / cs_std)
        )
    else:
        w_blend = np.array([0.30, 0.35, 0.35, 0.00], dtype=np.float64)
        z_tr_blend = (
            0.50 * ((pred_exp_tr - exp_mean) / exp_std)
            + 0.50 * ((pred_ds_tr - ds_mean) / ds_std)
        )
    blend_std = float(np.std(z_tr_blend) + 1e-9)

    z_in = 1.40 if pr != "USDCAD" else 1.50
    z_out = -0.25 if pr != "USDCAD" else -0.30
    hysteresis_params[pr] = {"z_entry": z_in, "z_exit": z_out}

    # Evaluate on untouched OOS slice
    Z_oos_exp = np.clip((Xp[oos_indices] - mu_exp) / sd_exp, -4.0, 4.0)
    pred_exp_oos = (ridge_exp.predict(Z_oos_exp) - exp_mean) / exp_std

    Z_oos_rol = np.clip((Xp[oos_indices] - mu_rol) / sd_rol, -4.0, 4.0)
    pred_rol_oos = (ridge_rol.predict(Z_oos_rol[:, top35]) - rol_mean) / rol_std

    pred_ds_oos = (predict_diff_sharpe(Xp[oos_indices], sel_ds, mu_ds, sc_ds, theta_ds) - ds_mean) / ds_std
    pred_cs_oos = (cs_col[oos_indices] - cs_mean) / cs_std

    z_oos = (
        w_blend[0] * pred_exp_oos
        + w_blend[1] * pred_rol_oos
        + w_blend[2] * pred_ds_oos
        + w_blend[3] * pred_cs_oos
    ) / blend_std

    d_hy, n_hy, p_hy = simulate_hysteresis_daily(
        z_oos,
        idx[oos_indices],
        cp,
        cost_pips,
        atrp[oos_indices],
        pip,
        lo_day_oos,
        n_days_oos,
        z_entry=z_in,
        z_exit=z_out,
        vol_size=True,
    )
    oos_ic = float(np.corrcoef(z_oos, fwd_pips[oos_indices])[0, 1])
    oos_sr = sharpe(d_hy)
    oos_metrics[pr] = {
        "oos_corr": round(oos_ic, 4),
        "oos_hysteresis_sharpe": round(oos_sr, 3),
        "oos_net_pips": round(float(p_hy), 1),
        "oos_trades": int(n_hy),
        "train_samples": int(len(tr_indices)),
        "oos_samples": int(len(oos_indices)),
    }

    # Save pair policy arrays into bundle
    pfx = pr.lower()
    cs_policy_bundle[f"{pfx}_mu_exp"] = mu_exp.astype(np.float64)
    cs_policy_bundle[f"{pfx}_sd_exp"] = sd_exp.astype(np.float64)
    cs_policy_bundle[f"{pfx}_coef_exp"] = ridge_exp.coef_.astype(np.float64)
    cs_policy_bundle[f"{pfx}_intercept_exp"] = np.array([float(ridge_exp.intercept_)], dtype=np.float64)
    cs_policy_bundle[f"{pfx}_exp_stats"] = np.array([exp_mean, exp_std], dtype=np.float64)

    cs_policy_bundle[f"{pfx}_mu_rol"] = mu_rol.astype(np.float64)
    cs_policy_bundle[f"{pfx}_sd_rol"] = sd_rol.astype(np.float64)
    cs_policy_bundle[f"{pfx}_top35"] = top35.astype(np.int64)
    cs_policy_bundle[f"{pfx}_coef_rol"] = ridge_rol.coef_.astype(np.float64)
    cs_policy_bundle[f"{pfx}_intercept_rol"] = np.array([float(ridge_rol.intercept_)], dtype=np.float64)
    cs_policy_bundle[f"{pfx}_rol_stats"] = np.array([rol_mean, rol_std], dtype=np.float64)

    cs_policy_bundle[f"{pfx}_sel_ds"] = sel_ds.astype(np.int64)
    cs_policy_bundle[f"{pfx}_mu_ds"] = mu_ds.astype(np.float64)
    cs_policy_bundle[f"{pfx}_sc_ds"] = sc_ds.astype(np.float64)
    cs_policy_bundle[f"{pfx}_theta_ds"] = theta_ds.astype(np.float64)
    cs_policy_bundle[f"{pfx}_ds_stats"] = np.array([ds_mean, ds_std], dtype=np.float64)

    cs_policy_bundle[f"{pfx}_cs_stats"] = np.array([cs_mean, cs_std], dtype=np.float64)
    cs_policy_bundle[f"{pfx}_w_blend"] = w_blend
    cs_policy_bundle[f"{pfx}_blend_std"] = np.array([blend_std], dtype=np.float64)
    cs_policy_bundle[f"{pfx}_cot_std"] = np.array([cot_std_val], dtype=np.float64)

    print(
        f"  Saved {pr} booster & CS-Hysteresis policy | OOS IC: {oos_ic:+.4f} | "
        f"OOS Hysteresis Sharpe: {oos_sr:+.2f} ({p_hy:+.1f} net pips, {n_hy} trades)"
    )

policy_npz_path = out_dir / "cs_policy_weights.npz"
np.savez_compressed(policy_npz_path, **cs_policy_bundle)

metadata = {
    "created_at": time.time(),
    "version": "3.0.0-cs-hysteresis-leak-free",
    "strategy_name": "stationary_4h_cross_sectional_hysteresis",
    "horizon_bars": H_BARS,
    "train_cutoff_row": TRAIN_CUTOFF_ROW,
    "train_max_bar_idx": TRAIN_MAX_BAR_IDX,
    "active_pairs": ACTIVE_PAIRS,
    "all_pairs": ALL_PAIRS,
    "thresholds_bps": thresholds,
    "hysteresis_params": hysteresis_params,
    "conviction_percentile": 75.0,
    "take_profit_atr": 3.50,
    "stop_loss_atr": 0.80,
    "curated_feature_indices": curated_indices,
    "curated_feature_names": curated_names,
    "base_features": STATIONARY_BASE_FEATURES,
    "base_names_146": base_names_146,
    "engineered_feature_names": eng_feat_names,
    "policy_weights_file": "cs_policy_weights.npz",
    "oos_diagnostics": oos_metrics,
}

meta_path = out_dir / "metadata.json"
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nSaved policy weights -> {policy_npz_path}")
print(f"Saved metadata       -> {meta_path}")
print(f"Leak-free CS-Hysteresis production ensemble trained and ready in {time.time() - t0:.1f}s!")
print("=" * 95)
