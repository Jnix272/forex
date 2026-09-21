"""
scripts/run_paper_trading_demo.py
=================================
Live Paper Trading Demonstration using the certified ONNX models:
  - Stacking Ensemble Meta-Learner (checkpoints/ensemble/ensemble_meta_best.onnx)
  - Multi-RL Recurrent Execution Policy (checkpoints/ensemble/rl_ensemble_best.onnx)
  - In-Memory PaperBroker execution simulation with real-time mark-to-market
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtesting.backtest import ScalingAction
from trading.live_actions import LiveAction, scaling_action_to_live_action
from trading.live_engine import PaperBroker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class LivePaperTradingSession:
    def __init__(
        self,
        pairs: list[str] = ["EURUSD", "GBPUSD", "USDCAD", "USDJPY"],
        initial_equity: float = 10_000.0,
        ensemble_onnx: str = "checkpoints/ensemble/ensemble_meta_best.onnx",
        rl_onnx: str = "checkpoints/ensemble/rl_ensemble_best.onnx",
    ):
        self.pairs = pairs
        self.initial_equity = initial_equity
        self.broker = PaperBroker(initial_equity=initial_equity)
        self.broker.connect()

        logging.info(f"Initializing PaperBroker | Balance: ${self.broker.balance:,.2f} | Equity: ${self.broker.equity:,.2f}")

        # 1. Load Stacking Ensemble ONNX
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        logging.info(f"Loading Stacking Ensemble Meta-Learner from {ensemble_onnx}...")
        self.ensemble_session = ort.InferenceSession(ensemble_onnx, opts, providers=["CPUExecutionProvider"])
        self.ens_input_name = self.ensemble_session.get_inputs()[0].name

        # 2. Load Multi-RL Policy ONNX
        logging.info(f"Loading Multi-RL Consensus Policy from {rl_onnx}...")
        self.rl_session = ort.InferenceSession(rl_onnx, opts, providers=["CPUExecutionProvider"])
        self.rl_input_name = self.rl_session.get_inputs()[0].name

        # State tracking per pair
        self.pair_states = {
            p: {
                "position_lots": 0.0,
                "entry_price": 0.0,
                "current_price": 1.1000 if "EUR" in p else (1.3000 if "GBP" in p else (1.3500 if "CAD" in p else 150.00)),
                "spread_pips": 1.2,
                "atr": 0.0015 if "JPY" not in p else 0.15,
                "trades_count": 0,
            }
            for p in pairs
        }
        self.trade_journal = []

    def simulate_ticks_and_trade(self, n_ticks: int = 30):
        logging.info(f"Starting Live Paper Trading Stream ({n_ticks} tick events across {len(self.pairs)} pairs)...")
        print("\n" + "=" * 90)
        print(f"{'TICK':<6} | {'PAIR':<7} | {'PRICE':<9} | {'DIRECTION':<9} | {'ACTION':<14} | {'LOTS':<6} | {'EQUITY':<11} | {'UNREALIZED PnL':<14}")
        print("=" * 90)

        rng = np.random.RandomState(42)

        for tick_idx in range(1, n_ticks + 1):
            pair = self.pairs[(tick_idx - 1) % len(self.pairs)]
            state = self.pair_states[pair]

            # 1. Simulate micro market movement
            pip_unit = 0.01 if "JPY" in pair else 0.0001
            price_delta = rng.normal(loc=0.0002 * (1 if tick_idx % 4 != 0 else -1), scale=1.5) * pip_unit
            state["current_price"] += price_delta
            bid = state["current_price"] - (state["spread_pips"] * 0.5 * pip_unit)
            ask = state["current_price"] + (state["spread_pips"] * 0.5 * pip_unit)
            self.broker.update_quote(bid, ask, pair=pair)

            # 2. Extract features & Run Stacking Ensemble ONNX
            # Dummy feature window: (1, 120, 584)
            features = rng.randn(1, 120, 584).astype(np.float32)
            t_start = time.perf_counter()
            ens_logits = self.ensemble_session.run(None, {self.ens_input_name: features})[0][0]
            # Output is 3 logits: [Sell, Neutral, Buy]
            pred_class = int(np.argmax(ens_logits))
            dir_label = ["SELL", "HOLD", "BUY"][pred_class]

            # 3. Construct 591-dim RL Observation Vector
            # [signal (2), market_features (584), portfolio_state (5)]
            signal_feat = np.array([ens_logits[2] - ens_logits[0], float(np.var(ens_logits))], dtype=np.float32)
            market_last = features[0, -1, :]
            # Portfolio state: position_lots, entry_ratio, pnl_pct, atr, spread
            pos_ratio = state["position_lots"] / 1.0
            pnl_pct = ((bid - state["entry_price"]) / (state["entry_price"] + 1e-8)) if state["entry_price"] > 0 else 0.0
            portfolio_state = np.array([pos_ratio, pnl_pct, state["atr"], state["spread_pips"], 1.0], dtype=np.float32)

            rl_obs = np.concatenate([signal_feat, market_last, portfolio_state]).astype(np.float32).reshape(1, -1)

            # 4. Run Multi-RL Consensus Policy ONNX
            rl_logits = self.rl_session.run(None, {self.rl_input_name: rl_obs})[0][0]
            t_infer = (time.perf_counter() - t_start) * 1000.0  # ms

            # Action selection:
            raw_action_id = int(np.argmax(rl_logits))
            scaling_act = ScalingAction(raw_action_id)

            # Execute action on PaperBroker
            executed_lots = 0.0
            action_desc = scaling_act.name

            if scaling_act == ScalingAction.OPEN_LONG and state["position_lots"] == 0.0:
                executed_lots = 0.20
                state["position_lots"] = executed_lots
                state["entry_price"] = ask
                self.broker.market_order(pair, "BUY", executed_lots)
                state["trades_count"] += 1
            elif scaling_act == ScalingAction.OPEN_SHORT and state["position_lots"] == 0.0:
                executed_lots = 0.20
                state["position_lots"] = -executed_lots
                state["entry_price"] = bid
                self.broker.market_order(pair, "SELL", executed_lots)
                state["trades_count"] += 1
            elif scaling_act in (ScalingAction.SCALE_IN_25, ScalingAction.SCALE_IN_50, ScalingAction.SCALE_IN_100) and state["position_lots"] > 0:
                scale_fraction = {ScalingAction.SCALE_IN_25: 0.25, ScalingAction.SCALE_IN_50: 0.50, ScalingAction.SCALE_IN_100: 1.00}[scaling_act]
                add_lots = round(abs(state["position_lots"]) * scale_fraction, 2)
                state["position_lots"] += add_lots
                self.broker.market_order(pair, "BUY", add_lots)
                executed_lots = add_lots
            elif scaling_act in (ScalingAction.SCALE_OUT_25, ScalingAction.SCALE_OUT_50, ScalingAction.SCALE_OUT_100) and abs(state["position_lots"]) > 0:
                scale_fraction = {ScalingAction.SCALE_OUT_25: 0.25, ScalingAction.SCALE_OUT_50: 0.50, ScalingAction.SCALE_OUT_100: 1.00}[scaling_act]
                sub_lots = round(abs(state["position_lots"]) * scale_fraction, 2)
                side = "SELL" if state["position_lots"] > 0 else "BUY"
                self.broker.market_order(pair, side, sub_lots)
                state["position_lots"] -= (sub_lots if state["position_lots"] > 0 else -sub_lots)
                executed_lots = sub_lots
            elif scaling_act == ScalingAction.CLOSE_ALL and abs(state["position_lots"]) > 0:
                side = "SELL" if state["position_lots"] > 0 else "BUY"
                self.broker.market_order(pair, side, abs(state["position_lots"]))
                executed_lots = abs(state["position_lots"])
                state["position_lots"] = 0.0
                state["entry_price"] = 0.0

            acct = self.broker.get_account()
            unrealized = acct["equity"] - acct["balance"]

            print(
                f"#{tick_idx:<5} | {pair:<7} | {state['current_price']:<9.5f} | {dir_label:<9} | "
                f"{action_desc:<14} | {state['position_lots']:<+6.2f} | ${acct['equity']:<10,.2f} | "
                f"${unrealized:<+13,.2f} ({t_infer:.1f}ms)"
            )

            # Record event
            self.trade_journal.append({
                "tick": tick_idx,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pair": pair,
                "price": float(state["current_price"]),
                "signal_dir": dir_label,
                "action": action_desc,
                "lots": float(state["position_lots"]),
                "equity": float(acct["equity"]),
                "balance": float(acct["balance"]),
                "latency_ms": float(t_infer),
            })

        print("=" * 90)
        final_acct = self.broker.get_account()
        pnl = final_acct["equity"] - self.initial_equity
        ret_pct = (pnl / self.initial_equity) * 100.0
        print(f"\n[SUMMARY] Initial Equity : ${self.initial_equity:,.2f}")
        print(f"[SUMMARY] Final Equity   : ${final_acct['equity']:,.2f}")
        print(f"[SUMMARY] Realized P&L   : ${pnl:+,.2f} ({ret_pct:+.2f}%)")
        print(f"[SUMMARY] Total Open Lots: {sum(abs(v['position_lots']) for v in self.pair_states.values()):.2f}")

        # Save journal
        out_log = Path("logs/paper_trading_session.json")
        out_log.parent.mkdir(parents=True, exist_ok=True)
        with open(out_log, "w", encoding="utf-8") as fh:
            json.dump({
                "session_summary": {
                    "initial_equity": self.initial_equity,
                    "final_equity": final_acct["equity"],
                    "pnl": pnl,
                    "return_pct": ret_pct,
                    "ticks_processed": n_ticks,
                    "pairs": self.pairs,
                },
                "journal": self.trade_journal,
            }, fh, indent=2)
        logging.info(f"Live Paper Trading session journal saved to {out_log}")


if __name__ == "__main__":
    session = LivePaperTradingSession()
    session.simulate_ticks_and_trade(n_ticks=24)
