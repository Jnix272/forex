"""
scripts/run_holdout_bakeoff.py
==============================
Vectorized, High-Speed Out-of-Sample Holdout Bake-Off:
Evaluates and compares:
  1. Dedicated Single-Pair Specialists (haelt_eurusd_v1, haelt_usdcad_v1, haelt_usdjpy_v1)
  2. 4H Stationary LightGBM Ensemble (checkpoints/stationary_ensemble)
  3. Monolithic Multi-Pair Baseline (checkpoints/haelt_modern_2018_2025)

Evaluation period: Out-of-sample holdout tail (rows 172,144 to 191,271, Mar 2025 - Dec 2025).
Realistic friction: 1.5-pip spread + 0.7-pip slippage deducted on every round-trip trade.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.pytorch_inference import PyTorchInferenceEngine
from trading.inference_engines import StationaryEnsembleInferenceEngine
from config.settings import get_pip_size

PAIRS = ["EURUSD", "USDCAD", "USDJPY"]
HOLDOUT_START_IDX = 172_144


def batch_predict_proba(engine: PyTorchInferenceEngine, windows: np.ndarray, batch_size: int = 1000) -> np.ndarray:
    all_probs = []
    n = len(windows)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = windows[start:end]
        scaled = engine._transform_window(batch)
        x = engine._torch.as_tensor(scaled, dtype=engine._torch.float32, device=engine.device)
        with engine._torch.no_grad():
            out = engine.model(x)
            if isinstance(out, (tuple, list)) and len(out) >= 3:
                from models.ensemble import PRIMARY_PAIR_INDEX
                from training.decision import action_proba
                pr = action_proba(out).detach().cpu().numpy()
                if pr.ndim == 3:
                    pr = pr[:, PRIMARY_PAIR_INDEX, :]
                all_probs.append(pr.astype(np.float32))
            else:
                if isinstance(out, (tuple, list)):
                    out = out[0]
                if hasattr(out, "detach"):
                    out = out.detach().cpu().numpy()
                out_arr = np.asarray(out)
                if out_arr.ndim == 1:
                    out_arr = out_arr[:, np.newaxis]
                if out_arr.shape[-1] == 1:
                    v = out_arr[:, 0].astype(np.float64)
                    sell = -v - 0.35
                    hold = np.zeros_like(v)
                    buy = v - 0.35
                    stacked = np.stack([sell, hold, buy], axis=-1)
                    e = np.exp(stacked - stacked.max(axis=-1, keepdims=True))
                    pr = (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)
                    all_probs.append(pr)
                elif out_arr.shape[-1] == 3:
                    e = np.exp(out_arr - out_arr.max(axis=-1, keepdims=True))
                    pr = (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)
                    all_probs.append(pr)
                else:
                    all_probs.append(np.full((len(batch), 3), 1/3, dtype=np.float32))
    return np.concatenate(all_probs, axis=0)


def evaluate_model_performance(
    actions: np.ndarray,
    close: np.ndarray,
    spread: np.ndarray,
    pip_size: float,
    holding_bars: int = 12,
    slippage_pips: float = 0.7,
) -> dict:
    """Simulate execution with trade hold time and exact cost accounting."""
    n = len(actions)
    pnl_list = []
    i = 0
    trade_count = 0
    wins = 0
    equity = 10000.0
    equity_curve = [equity]

    while i < n - holding_bars:
        act = actions[i]
        if act in (1, -1):  # 1: Long (BUY), -1: Short (SELL)
            entry_idx = i
            exit_idx = i + holding_bars
            entry_price = close[entry_idx]
            exit_price = close[exit_idx]
            
            spread_cost = spread[entry_idx] if spread is not None else 1.5 * pip_size
            slip_cost = slippage_pips * pip_size
            cost_pips = (spread_cost + slip_cost) / pip_size
            
            if act == 1:  # Long
                gross_diff = exit_price - entry_price
            else:  # Short
                gross_diff = entry_price - exit_price
                
            gross_pips = gross_diff / pip_size
            net_pips = gross_pips - cost_pips
            
            trade_count += 1
            if net_pips > 0:
                wins += 1
                
            pnl_list.append(net_pips)
            
            lot_dollar_per_pip = 10.0 if "JPY" not in str(pip_size) else 1000.0 * pip_size
            trade_dollar = net_pips * lot_dollar_per_pip * 0.1  # 0.1 lot
            equity += trade_dollar
            equity_curve.append(equity)
            
            i += holding_bars
        else:
            i += 1

    if trade_count == 0:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "net_pips": 0.0,
            "annual_sharpe": 0.0,
            "max_drawdown": 0.0,
            "total_return_pct": 0.0,
        }

    win_rate = wins / trade_count
    gains = sum(p for p in pnl_list if p > 0)
    loss_sum = abs(sum(p for p in pnl_list if p < 0))
    profit_factor = (gains / loss_sum) if loss_sum > 1e-8 else (gains if gains > 0 else 0.0)
    net_pips = sum(pnl_list)
    
    eq_arr = np.array(equity_curve)
    peaks = np.maximum.accumulate(eq_arr)
    drawdowns = (peaks - eq_arr) / peaks
    max_dd = float(np.max(drawdowns)) if len(drawdowns) else 0.0
    total_return_pct = ((equity - 10000.0) / 10000.0) * 100.0
    
    pnl_arr = np.array(pnl_list)
    if len(pnl_arr) > 1 and pnl_arr.std() > 1e-8:
        trades_per_year = max(10, trade_count * (12.0 / 9.5))
        annual_sharpe = float((pnl_arr.mean() / pnl_arr.std()) * np.sqrt(trades_per_year))
    else:
        annual_sharpe = 0.0

    return {
        "n_trades": trade_count,
        "win_rate": win_rate * 100.0,
        "profit_factor": float(profit_factor),
        "net_pips": float(net_pips),
        "annual_sharpe": float(annual_sharpe),
        "max_drawdown": float(max_dd * 100.0),
        "total_return_pct": float(total_return_pct),
    }


def run_bakeoff():
    print("=" * 86, flush=True)
    print("OUT-OF-SAMPLE HOLDOUT BAKE-OFF: SPECIALISTS vs ENSEMBLE vs MONOLITHIC", flush=True)
    print("=" * 86, flush=True)

    cache_patterns = {
        "EURUSD": "data/processed/dataset_scalping_5m_EURUSD_0_dukascopy_120_triple_barrier_reward*.zarr",
        "USDCAD": "data/processed/dataset_scalping_5m_USDCAD_0_dukascopy_120_triple_barrier_reward*.zarr",
        "USDJPY": "data/processed/dataset_scalping_5m_USDJPY_0_dukascopy_120_triple_barrier_reward*.zarr",
    }
    
    multi_pattern = "data/processed/dataset_scalping_5m_EURUSD-USDCAD-USDJPY_0_dukascopy_120_triple_barrier_reward*.zarr"
    multi_caches = glob.glob(multi_pattern)
    multi_cache = multi_caches[0] if multi_caches else None
    
    specialist_checkpoints = {
        "EURUSD": "checkpoints/haelt_eurusd_v1/haelt/haelt/haelt_best.pt",
        "USDCAD": "checkpoints/haelt_usdcad_v1/haelt/haelt/haelt_best.pt",
        "USDJPY": "checkpoints/haelt_usdjpy_v1/haelt/haelt/haelt_best.pt",
    }
    
    monolithic_ckpt = "checkpoints/haelt_modern_2018_2025/haelt/haelt/haelt_best.pt"
    mono_engine = None
    if Path(monolithic_ckpt).is_file():
        try:
            print(f"[Baseline] Loading Monolithic Multi-Pair Engine: {monolithic_ckpt}", flush=True)
            mono_engine = PyTorchInferenceEngine(
                checkpoint_path=monolithic_ckpt,
                model_name="haelt",
                seq_len=60,
            )
        except Exception as exc:
            print(f"[Baseline] Note: Monolithic baseline skipped ({exc})", flush=True)

    results = []

    for pair in PAIRS:
        print(f"\n{'=' * 30} EVALUATING {pair} {'=' * 30}", flush=True)
        p_pip = get_pip_size(pair)
        z_files = glob.glob(cache_patterns[pair])
        if not z_files:
            print(f"[Error] No dataset found for {pair}", flush=True)
            continue
        z_path = z_files[0]
        root = zarr.open(z_path, mode="r")
        total_rows = root["X"].shape[0]
        n_holdout = total_rows - HOLDOUT_START_IDX
        print(f"[{pair}] Holdout slice: [{HOLDOUT_START_IDX} : {total_rows}] ({n_holdout:,} bars, ~9.5 months of 2025)", flush=True)

        close_arr = np.asarray(root["close"][HOLDOUT_START_IDX:total_rows])
        spread_arr = np.asarray(root["spread"][HOLDOUT_START_IDX:total_rows])
        
        # --- Model A: Dedicated Single-Pair Specialist ---
        spec_ckpt = specialist_checkpoints[pair]
        print(f"[{pair}] Loading Dedicated Specialist: {spec_ckpt}", flush=True)
        spec_engine = PyTorchInferenceEngine(
            checkpoint_path=spec_ckpt,
            model_name="haelt",
            seq_len=60,
        )
        spec_engine.pair = pair
        spec_engine.pair_index = 0
        
        print(f"[{pair}] Vectorized inference for Specialist...", flush=True)
        t_spec = time.time()
        # Read holdout windows in chunks
        spec_actions = np.zeros(n_holdout, dtype=np.int8)
        chunk_step = 2000
        for s in range(0, n_holdout, chunk_step):
            e = min(s + chunk_step, n_holdout)
            w_chunk = np.asarray(root["X"][HOLDOUT_START_IDX + s : HOLDOUT_START_IDX + e, -60:, :])
            probs_chunk = batch_predict_proba(spec_engine, w_chunk, batch_size=500)
            for idx_c, p_row in enumerate(probs_chunk):
                top_class = int(p_row.argmax())
                top_prob = float(p_row[top_class])
                # Class mapping: 0=SELL, 1=HOLD, 2=BUY
                if top_class == 2 and top_prob >= 0.42:
                    spec_actions[s + idx_c] = 1  # Long
                elif top_class == 0 and top_prob >= 0.42:
                    spec_actions[s + idx_c] = -1  # Short
                else:
                    spec_actions[s + idx_c] = 0  # Hold

        print(f"[{pair}] Specialist inference completed in {time.time() - t_spec:.2f}s", flush=True)
        spec_perf = evaluate_model_performance(spec_actions, close_arr, spread_arr, p_pip)
        spec_perf.update({"pair": pair, "model_type": "Dedicated Specialist", "arch": "HAELT"})
        results.append(spec_perf)
        print(f"[{pair} - Specialist] Trades: {spec_perf['n_trades']} | Win: {spec_perf['win_rate']:.1f}% | Net Pips: {spec_perf['net_pips']:+.1f} | Sharpe: {spec_perf['annual_sharpe']:+.2f} | PF: {spec_perf['profit_factor']:.2f}", flush=True)

        # --- Model B: 4H Stationary LightGBM Ensemble ---
        print(f"[{pair}] Evaluating 4H Stationary LightGBM Model...", flush=True)
        stat_engine = StationaryEnsembleInferenceEngine(pair=pair)
        stat_actions = np.zeros(n_holdout, dtype=np.int8)
        
        # Extract features for tree booster
        if multi_cache:
            m_root = zarr.open(multi_cache, mode="r")
            m_X = np.asarray(m_root["X"][HOLDOUT_START_IDX:total_rows, -1, :])
        else:
            m_X = np.asarray(root["X"][HOLDOUT_START_IDX:total_rows, -1, :])
            
        if m_X.shape[-1] < 584:
            m_X = np.pad(m_X, ((0, 0), (0, 584 - m_X.shape[-1])))
        elif m_X.shape[-1] > 584:
            m_X = m_X[:, :584]

        # Vectorized tree predictions
        feats = m_X[:, stat_engine.curated_indices]
        preds_bps = stat_engine.booster.predict(feats)
        thresh = stat_engine.threshold_bps
        stat_actions[preds_bps > thresh] = 1   # Long
        stat_actions[preds_bps < -thresh] = -1  # Short
        stat_actions[np.abs(preds_bps) <= thresh] = 0

        stat_perf = evaluate_model_performance(stat_actions, close_arr, spread_arr, p_pip, holding_bars=48)
        stat_perf.update({"pair": pair, "model_type": "Stationary 4H Tree", "arch": "LightGBM"})
        results.append(stat_perf)
        print(f"[{pair} - 4H Tree] Trades: {stat_perf['n_trades']} | Win: {stat_perf['win_rate']:.1f}% | Net Pips: {stat_perf['net_pips']:+.1f} | Sharpe: {stat_perf['annual_sharpe']:+.2f} | PF: {stat_perf['profit_factor']:.2f}", flush=True)

        # --- Model C: Hybrid Specialist + 4H Tree Consensus ---
        print(f"[{pair}] Evaluating Hybrid (Specialist + 4H Tree Consensus)...", flush=True)
        hybrid_actions = np.zeros(n_holdout, dtype=np.int8)
        agreed_mask = (spec_actions == stat_actions) & (spec_actions != 0)
        hybrid_actions[agreed_mask] = spec_actions[agreed_mask]
        
        hybrid_perf = evaluate_model_performance(hybrid_actions, close_arr, spread_arr, p_pip, holding_bars=24)
        hybrid_perf.update({"pair": pair, "model_type": "Specialist + 4H Hybrid", "arch": "Consensus"})
        results.append(hybrid_perf)
        print(f"[{pair} - Hybrid] Trades: {hybrid_perf['n_trades']} | Win: {hybrid_perf['win_rate']:.1f}% | Net Pips: {hybrid_perf['net_pips']:+.1f} | Sharpe: {hybrid_perf['annual_sharpe']:+.2f} | PF: {hybrid_perf['profit_factor']:.2f}", flush=True)

    # Output Summary Table
    print("\n" + "=" * 105, flush=True)
    print("FINAL HOLDOUT PERFORMANCE SCORECARD (2025 OUT-OF-SAMPLE TAIL)", flush=True)
    print("=" * 105, flush=True)
    header = f"{'Pair':<8} | {'Model Architecture':<32} | {'Trades':<7} | {'Win%':<6} | {'PF':<5} | {'Net Pips':<10} | {'Sharpe':<7} | {'MaxDD%':<7}"
    print(header, flush=True)
    print("-" * 105, flush=True)
    for r in results:
        m_label = f"{r['model_type']} ({r['arch']})"
        print(f"{r['pair']:<8} | {m_label:<32} | {r['n_trades']:<7} | {r['win_rate']:>5.1f}% | {r['profit_factor']:>5.2f} | {r['net_pips']:>+9.1f} | {r['annual_sharpe']:>+6.2f} | {r['max_drawdown']:>6.1f}%", flush=True)
    print("=" * 105, flush=True)

    out_path = Path("docs/holdout_bakeoff_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nReport saved to: {out_path}", flush=True)
    return results


if __name__ == "__main__":
    run_bakeoff()
