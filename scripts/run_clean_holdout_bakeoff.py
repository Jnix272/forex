"""
scripts/run_clean_holdout_bakeoff.py
====================================
Honest Out-of-Sample Holdout Bake-off (March - December 2025, 19,127 bars)
Evaluates all 4 tiers across EURUSD, USDCAD, and USDJPY:
  1. Dedicated HAELT Specialist (`haelt_best.pt` promoted from best CV fold, native H=30 + Hysteresis)
  2. Dedicated Mamba Specialist (`mamba_best.pt` or best validated CV fold, native H=30 + Hysteresis)
  3. Leak-Free 4H Stationary Ensemble (trained strictly on pre-holdout rows + Hysteresis)
  4. Multi-Architecture Specialist Zoo Ensemble (HAELT + Mamba + 4H Stationary + Hysteresis)
Real-world execution costs:
  - Live Dukascopy variable spread per bar
  - Slippage: 0.7 pips per round-trip trade
"""

from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import zarr

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.pytorch_inference import PyTorchInferenceEngine
from trading.inference_engines import (
    MultiArchitectureSpecialistEnsembleInferenceEngine,
    StationaryEnsembleInferenceEngine,
)
from training.decision import action_proba
from scripts.run_walkforward_4h import build_engineered_feature_matrix, predict_diff_sharpe

CS_POLICY_PATH = ROOT / "checkpoints" / "stationary_ensemble" / "cs_policy_weights.npz"
CS_PAIR_SLOT = {"EURUSD": 0, "USDJPY": 1, "USDCAD": 3}


def _cs_policy_z(pair: str, per_pair_X: list) -> np.ndarray | None:
    """Apply the exported CS + Cost-Vol Ridge + Diff-Sharpe policy exactly as trained (no holdout tuning)."""
    if not CS_POLICY_PATH.exists():
        return None
    pol = np.load(CS_POLICY_PATH)
    pfx = pair.lower()
    if f"{pfx}_mu_exp" not in pol:
        return None
    Xp = per_pair_X[CS_PAIR_SLOT[pair]]
    z_exp = np.clip((Xp - pol[f"{pfx}_mu_exp"]) / pol[f"{pfx}_sd_exp"], -4.0, 4.0)
    p_exp = (z_exp @ pol[f"{pfx}_coef_exp"] + pol[f"{pfx}_intercept_exp"][0] - pol[f"{pfx}_exp_stats"][0]) / pol[f"{pfx}_exp_stats"][1]
    z_rol = np.clip((Xp - pol[f"{pfx}_mu_rol"]) / pol[f"{pfx}_sd_rol"], -4.0, 4.0)
    p_rol = (z_rol[:, pol[f"{pfx}_top35"]] @ pol[f"{pfx}_coef_rol"] + pol[f"{pfx}_intercept_rol"][0] - pol[f"{pfx}_rol_stats"][0]) / pol[f"{pfx}_rol_stats"][1]
    p_ds = (predict_diff_sharpe(Xp, pol[f"{pfx}_sel_ds"], pol[f"{pfx}_mu_ds"], pol[f"{pfx}_sc_ds"], pol[f"{pfx}_theta_ds"]) - pol[f"{pfx}_ds_stats"][0]) / pol[f"{pfx}_ds_stats"][1]
    p_cs = (Xp[:, -1] - pol[f"{pfx}_cs_stats"][0]) / pol[f"{pfx}_cs_stats"][1]
    w = pol[f"{pfx}_w_blend"]
    return (w[0] * p_exp + w[1] * p_rol + w[2] * p_ds + w[3] * p_cs) / pol[f"{pfx}_blend_std"][0]

PAIRS = ["EURUSD", "USDCAD", "USDJPY"]
HOLDOUT_START = 172144

SPECIALIST_CKPTS = {
    "EURUSD": "checkpoints/haelt_eurusd_v1/haelt/haelt_best.pt",
    "USDCAD": "checkpoints/haelt_usdcad_v1/haelt/haelt_best.pt",
    "USDJPY": "checkpoints/haelt_usdjpy_v1/haelt/haelt_best.pt",
}


def compute_metrics(pnl: np.ndarray, months: float = 9.5) -> tuple[int, float, float, float, float]:
    n_tr = len(pnl)
    if n_tr == 0:
        return 0, 0.0, 0.0, 0.0, 0.0
    wr = float((pnl > 0).mean() * 100.0)
    sh = float((pnl.mean() / (pnl.std() + 1e-8)) * np.sqrt(max(1.0, n_tr * (12.0 / months)))) if n_tr >= 4 else 0.0
    wins = float(pnl[pnl > 0].sum())
    losses = float(abs(pnl[pnl < 0].sum()))
    pf = float(wins / losses) if losses > 1e-8 else (9.99 if wins > 0 else 0.0)
    net = float(pnl.sum())
    return n_tr, wr, net, sh, pf


def simulate_hysteresis_holdout(
    z_signal: np.ndarray,
    close: np.ndarray,
    cost_pips: np.ndarray,
    valid_step: np.ndarray,
    pip: float,
    step_bars: int = 30,
    z_entry: float = 1.35,
    z_exit: float = -0.20,
) -> np.ndarray:
    """
    Hysteresis Buffer State Machine on holdout bars:
    - Rebalances every `step_bars` when `valid_step[i]` is True (no overnight/weekend gap).
    - Holds existing position without paying exit/re-entry spread as long as `z * sign(pos) >= z_exit`.
    - If an overnight/weekend gap is encountered (`not valid_step[i]`), closes open position before gap.
    """
    n = len(close)
    pos = 0
    entry_price = 0.0
    entry_cost = 0.0
    trade_pnls = []

    i = 0
    while i < n - step_bars:
        if not valid_step[i]:
            if pos != 0:
                gross = (close[i] - entry_price) / pip if pos == 1 else (entry_price - close[i]) / pip
                trade_pnls.append(gross - 0.5 * (entry_cost + cost_pips[i]))
                pos = 0
            i += 1
            continue

        z = float(z_signal[i])
        if pos == 0:
            if z >= z_entry:
                pos = 1
                entry_price = float(close[i])
                entry_cost = float(cost_pips[i])
                i += step_bars
            elif z <= -z_entry:
                pos = -1
                entry_price = float(close[i])
                entry_cost = float(cost_pips[i])
                i += step_bars
            else:
                i += 1
        elif pos == 1:
            if z < z_exit:
                gross = (float(close[i]) - entry_price) / pip
                trade_pnls.append(gross - 0.5 * (entry_cost + float(cost_pips[i])))
                pos = 0
                if z <= -z_entry:
                    pos = -1
                    entry_price = float(close[i])
                    entry_cost = float(cost_pips[i])
                    i += step_bars
                else:
                    i += 1
            else:
                i += step_bars
        else:
            if -z < z_exit:
                gross = (entry_price - float(close[i])) / pip
                trade_pnls.append(gross - 0.5 * (entry_cost + float(cost_pips[i])))
                pos = 0
                if z >= z_entry:
                    pos = 1
                    entry_price = float(close[i])
                    entry_cost = float(cost_pips[i])
                    i += step_bars
                else:
                    i += 1
            else:
                i += step_bars

    if pos != 0:
        gross = (float(close[-1]) - entry_price) / pip if pos == 1 else (entry_price - float(close[-1])) / pip
        trade_pnls.append(gross - 0.5 * (entry_cost + float(cost_pips[-1])))

    return np.asarray(trade_pnls, dtype=np.float64)


def _infer_specialist_z(eng: PyTorchInferenceEngine, m3_root: object, pair_slot: int, n_holdout: int) -> np.ndarray:
    all_probs = []
    chunk_size = 2000
    core_model = getattr(eng.model, "inner", eng.model)
    feat_slice = slice(pair_slot * 146, (pair_slot + 1) * 146)
    for s in range(0, n_holdout, chunk_size):
        e = min(s + chunk_size, n_holdout)
        w_chunk = np.asarray(m3_root["X"][HOLDOUT_START + s : HOLDOUT_START + e, :, feat_slice])
        scaled = eng._transform_window(w_chunk)
        x = torch.as_tensor(scaled, dtype=torch.float32, device=eng.device)
        with torch.no_grad():
            out = core_model(x)
            pr = action_proba(out).cpu().numpy()
            all_probs.append(pr)
    probs = np.concatenate(all_probs, axis=0)
    p_sell, _, p_buy = probs[:, 0], probs[:, 1], probs[:, 2]
    p_trade = np.clip(p_sell + p_buy, 1e-6, 1.0)
    signed_edge = (p_buy - p_sell) * p_trade
    return (signed_edge - np.mean(signed_edge[:2000])) / (np.std(signed_edge[:2000]) + 1e-8)


def run_bakeoff():
    print("=" * 115, flush=True)
    print("   LEAK-FREE 2025 OUT-OF-SAMPLE HOLDOUT BAKE-OFF (MULTI-ARCHITECTURE SPECIALIST ZOO + HYSTERESIS)", flush=True)
    print("=" * 115, flush=True)

    m_X_path = ROOT / "data" / "processed" / "holdout_2025_4h_features.npy"
    if not m_X_path.exists():
        multi_pat = "data/processed/dataset_scalping_5m_EURUSD-USDCAD-USDJPY_0_dukascopy_120_triple_barrier_reward*.zarr"
        m_files = glob.glob(str(ROOT / multi_pat))
        m_root = zarr.open(store=m_files[0], mode="r")
        parts = [np.asarray(m_root["X"][s : min(s + 2000, 191271), 119, :]) for s in range(HOLDOUT_START, 191271, 2000)]
        m_X = np.concatenate(parts, axis=0)
        np.save(str(m_X_path), m_X)
    else:
        m_X = np.load(str(m_X_path))

    n_holdout = len(m_X)
    # 3-pair Zarr order is EURUSD, USDJPY, USDCAD; 4-pair layout is EURUSD, USDJPY, GBPUSD, USDCAD.
    # GBPUSD slot is proxied with EURUSD (not present in the holdout Zarr).
    X_584 = np.empty((n_holdout, 584), dtype=np.float32)
    X_584[:, 0:292] = m_X[:, 0:292]
    X_584[:, 292:438] = m_X[:, 0:146]
    X_584[:, 438:584] = m_X[:, 292:438]
    m3 = zarr.open(store=glob.glob(str(ROOT / "data/processed/dataset_scalping_5m_EURUSD-USDCAD-USDJPY_0_dukascopy_120_triple_barrier_*.zarr"))[0], mode="r")
    cp3 = np.asarray(m3["close_pairs"][HOLDOUT_START : HOLDOUT_START + n_holdout], dtype=np.float64)
    close4 = np.column_stack([cp3[:, 0], cp3[:, 1], cp3[:, 0], cp3[:, 2]])
    feat_names = json.load(open(ROOT / "checkpoints" / "haelt" / "haelt_fold0_best_features.json"))
    per_pair_X, _, _ = build_engineered_feature_matrix(X_584, np.arange(n_holdout), close4, feat_names)
    print(
        f"[Holdout] 2025 Out-of-Sample Slice: bars {HOLDOUT_START:,} to {HOLDOUT_START + n_holdout:,} "
        f"({n_holdout:,} bars, ~9.5 months)\n",
        flush=True,
    )

    results = []
    portfolio_zoo_pips = 0.0
    portfolio_zoo_trades = 0
    portfolio_zoo_pnls = []

    PAIR_TO_SLOT_3P = {"EURUSD": 0, "USDJPY": 1, "USDCAD": 2}

    for pair in PAIRS:
        print(f"{'=' * 48} {pair} {'=' * 48}", flush=True)
        ci = PAIR_TO_SLOT_3P[pair]

        close = cp3[:, ci]
        spread = np.asarray(m3["spread_pairs"][HOLDOUT_START : HOLDOUT_START + n_holdout, ci], dtype=np.float64)
        t_ns = (
            np.asarray(m3["t_ns"][HOLDOUT_START : HOLDOUT_START + n_holdout], dtype=np.int64)
            if "t_ns" in m3
            else None
        )

        pip = 0.01 if "JPY" in pair else 0.0001
        slip_pips = 0.7
        cost_pips = (spread + slip_pips * pip) / pip

        # Build gap-free mask for H=30 (2.5h) and H=48 (4h) so trades never jump across 18:00-07:00 UTC or weekends
        bar_ns = 300_000_000_000  # 5 minutes in ns
        valid_30 = np.ones(n_holdout, dtype=bool)
        valid_48 = np.ones(n_holdout, dtype=bool)
        if t_ns is not None:
            valid_30[:-30] = (t_ns[30:] - t_ns[:-30]) <= int(30 * bar_ns * 1.15)
            valid_30[-30:] = False
            valid_48[:-48] = (t_ns[48:] - t_ns[:-48]) <= int(48 * bar_ns * 1.15)
            valid_48[-48:] = False

        # ---------------------------------------------------------
        # Model 1: Dedicated Single-Pair Specialist (HAELT, Native H=30 + Hysteresis)
        # ---------------------------------------------------------
        ckpt = str(ROOT / SPECIALIST_CKPTS[pair])
        fold_sel_path = ROOT / Path(SPECIALIST_CKPTS[pair]).parent / "fold_selection.json"
        sel_fold_info = ""
        if fold_sel_path.exists():
            fs_data = json.loads(fold_sel_path.read_text())
            sel_fold_info = f" [selected_fold={fs_data.get('selected_fold')}, metric={fs_data.get('metric_value')}]"
        print(f"[{pair}] Running HAELT Specialist Inference ({Path(ckpt).name}{sel_fold_info})...", flush=True)
        eng_haelt = PyTorchInferenceEngine(ckpt, "haelt", seq_len=120)
        t0 = time.time()
        z_haelt = _infer_specialist_z(eng_haelt, m3, ci, n_holdout)
        print(f"[{pair}] HAELT inference complete in {time.time() - t0:.2f}s", flush=True)

        pnl_spec = simulate_hysteresis_holdout(
            z_haelt, close, cost_pips, valid_30, pip, step_bars=30, z_entry=1.45, z_exit=-0.15
        )
        n_sp, wr_sp, net_sp, sh_sp, pf_sp = compute_metrics(pnl_spec)
        results.append({
            "pair": pair,
            "model": "HAELT Specialist",
            "arch": "HAELT (Validated Fold + Hyst)",
            "trades": n_sp,
            "win_rate": wr_sp,
            "net_pips": net_sp,
            "sharpe": sh_sp,
            "pf": pf_sp,
            "horizon": "2.5h (30b)",
        })
        print(
            f"[{pair} - HAELT    ] Trades: {n_sp:<4} | Win: {wr_sp:4.1f}% | "
            f"Net Pips: {net_sp:+7.1f} | Sharpe: {sh_sp:+5.2f} | PF: {pf_sp:4.2f}",
            flush=True,
        )

        # ---------------------------------------------------------
        # Model 2: Dedicated Mamba Specialist (if available for pair)
        # ---------------------------------------------------------
        z_mamba = None
        mamba_ckpt = MultiArchitectureSpecialistEnsembleInferenceEngine._discover_best_specialist_ckpt("mamba", pair)
        if mamba_ckpt is not None:
            print(f"[{pair}] Running Mamba Specialist Inference ({mamba_ckpt.name})...", flush=True)
            eng_mamba = PyTorchInferenceEngine(str(mamba_ckpt), "mamba", seq_len=120)
            t_m = time.time()
            z_mamba = _infer_specialist_z(eng_mamba, m3, ci, n_holdout)
            print(f"[{pair}] Mamba inference complete in {time.time() - t_m:.2f}s", flush=True)

            pnl_mamba = simulate_hysteresis_holdout(
                z_mamba, close, cost_pips, valid_30, pip, step_bars=30, z_entry=1.45, z_exit=-0.15
            )
            n_mb, wr_mb, net_mb, sh_mb, pf_mb = compute_metrics(pnl_mamba)
            results.append({
                "pair": pair,
                "model": "Mamba Specialist",
                "arch": f"MAMBA ({mamba_ckpt.name} + Hyst)",
                "trades": n_mb,
                "win_rate": wr_mb,
                "net_pips": net_mb,
                "sharpe": sh_mb,
                "pf": pf_mb,
                "horizon": "2.5h (30b)",
            })
            print(
                f"[{pair} - MAMBA    ] Trades: {n_mb:<4} | Win: {wr_mb:4.1f}% | "
                f"Net Pips: {net_mb:+7.1f} | Sharpe: {sh_mb:+5.2f} | PF: {pf_mb:4.2f}",
                flush=True,
            )

        # ---------------------------------------------------------
        # Model 3: Leak-Free 4H Stationary Ensemble + Hysteresis
        # ---------------------------------------------------------
        stat_eng = StationaryEnsembleInferenceEngine(pair=pair)
        preds_bps = stat_eng.booster.predict(X_584[:, stat_eng.curated_indices])
        z_lgb = preds_bps / max(stat_eng.threshold_bps, 1.0)
        z_cs = _cs_policy_z(pair, per_pair_X)
        z_raw = 0.85 * z_cs + 0.15 * z_lgb if z_cs is not None else z_lgb
        z_stat = (z_raw - np.mean(z_raw[:2000])) / (np.std(z_raw[:2000]) + 1e-8)

        pnl_stat = simulate_hysteresis_holdout(
            z_stat, close, cost_pips, valid_48, pip, step_bars=48, z_entry=1.35, z_exit=-0.20
        )
        n_st, wr_st, net_st, sh_st, pf_st = compute_metrics(pnl_stat)
        results.append({
            "pair": pair,
            "model": "Stationary Ensemble (OOS)",
            "arch": "LightGBM (4H + Hysteresis)",
            "trades": n_st,
            "win_rate": wr_st,
            "net_pips": net_st,
            "sharpe": sh_st,
            "pf": pf_st,
            "horizon": "4h (48b)",
        })
        print(
            f"[{pair} - 4H Tree  ] Trades: {n_st:<4} | Win: {wr_st:4.1f}% | "
            f"Net Pips: {net_st:+7.1f} | Sharpe: {sh_st:+5.2f} | PF: {pf_st:4.2f}",
            flush=True,
        )

        # ---------------------------------------------------------
        # Model 4: Multi-Architecture Specialist Zoo Ensemble (HAELT + Mamba + 4H Macro)
        # ---------------------------------------------------------
        if z_mamba is not None:
            if pair == "EURUSD":
                # Combine HAELT + Mamba + 4H Stationary
                z_zoo_raw = 0.45 * z_haelt + 0.15 * z_mamba + 0.40 * z_stat
                zoo_arch = "HAELT + MAMBA + 4H Hyst"
            else:
                z_zoo_raw = 0.50 * z_haelt + 0.10 * z_mamba + 0.40 * z_stat
                zoo_arch = "HAELT + MAMBA + 4H Hyst"
        else:
            # USDJPY (Mamba still queued): blend HAELT Specialist + 4H Stationary
            z_zoo_raw = 0.75 * z_haelt + 0.25 * z_stat if pair == "USDJPY" else 0.60 * z_haelt + 0.40 * z_stat
            zoo_arch = "HAELT + 4H Hyst"

        z_zoo = (z_zoo_raw - np.mean(z_zoo_raw[:2000])) / (np.std(z_zoo_raw[:2000]) + 1e-8)
        z_in_zoo = 1.40 if pair == "USDJPY" else 1.25
        pnl_zoo = simulate_hysteresis_holdout(
            z_zoo, close, cost_pips, valid_30, pip, step_bars=30, z_entry=z_in_zoo, z_exit=-0.20
        )
        n_zoo, wr_zoo, net_zoo, sh_zoo, pf_zoo = compute_metrics(pnl_zoo)
        results.append({
            "pair": pair,
            "model": "Multi-Arch Zoo Ensemble",
            "arch": zoo_arch,
            "trades": n_zoo,
            "win_rate": wr_zoo,
            "net_pips": net_zoo,
            "sharpe": sh_zoo,
            "pf": pf_zoo,
            "horizon": "2.5h (30b)",
        })
        portfolio_zoo_pips += net_zoo
        portfolio_zoo_trades += n_zoo
        portfolio_zoo_pnls.append(pnl_zoo)
        print(
            f"[{pair} - Zoo Ens  ] Trades: {n_zoo:<4} | Win: {wr_zoo:4.1f}% | "
            f"Net Pips: {net_zoo:+7.1f} | Sharpe: {sh_zoo:+5.2f} | PF: {pf_zoo:4.2f}\n",
            flush=True,
        )

    all_zoo_pnl = np.concatenate(portfolio_zoo_pnls) if portfolio_zoo_pnls else np.zeros(0)
    _, port_wr, port_net, port_sh, port_pf = compute_metrics(all_zoo_pnl)

    print("=" * 115, flush=True)
    print("                     FINAL 2025 OUT-OF-SAMPLE HOLDOUT PERFORMANCE SCORECARD", flush=True)
    print("=" * 115, flush=True)
    header = (
        f"{'Pair':<8} | {'Model Tier':<25} | {'Architecture':<28} | "
        f"{'Horizon':<10} | {'Trades':<6} | {'Win%':<6} | {'PF':<5} | {'Net Pips':<10} | {'Sharpe':<7}"
    )
    print(header, flush=True)
    print("-" * 115, flush=True)
    for r in results:
        print(
            f"{r['pair']:<8} | {r['model']:<25} | {r['arch']:<28} | "
            f"{r['horizon']:<10} | {r['trades']:<6} | {r['win_rate']:>5.1f}% | "
            f"{r['pf']:>5.2f} | {r['net_pips']:>+9.1f}p | {r['sharpe']:>+6.2f}",
            flush=True,
        )
    print("-" * 115, flush=True)
    print(
        f"{'PORTFOLIO':<8} | {'3-Pair Zoo Ensemble':<25} | {'HAELT + MAMBA + 4H Hyst':<28} | "
        f"{'2.5h (30b)':<10} | {portfolio_zoo_trades:<6} | {port_wr:>5.1f}% | "
        f"{port_pf:>5.2f} | {port_net:>+9.1f}p | {port_sh:>+6.2f}",
        flush=True,
    )
    print("=" * 115, flush=True)

    out_path = ROOT / "docs" / "holdout_bakeoff_report.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "holdout_start": HOLDOUT_START,
                "n_bars": n_holdout,
                "results": results,
                "portfolio_3pair_zoo": {
                    "trades": int(portfolio_zoo_trades),
                    "win_rate": round(port_wr, 2),
                    "profit_factor": round(port_pf, 2),
                    "net_pips": round(port_net, 1),
                    "sharpe": round(port_sh, 2),
                },
            },
            f,
            indent=2,
        )
    print(f"Saved report -> {out_path}", flush=True)


if __name__ == "__main__":
    run_bakeoff()
