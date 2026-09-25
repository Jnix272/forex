"""
scripts/run_paper_trading_demo.py
=================================
Smoke harness for the live-execution boundary that exercises the certified ONNX
artifacts end to end on top of the in-memory ``PaperBroker``:
  - Stacking Ensemble Meta-Learner (checkpoints/ensemble/ensemble_meta_best.onnx)
  - Multi-RL Recurrent Execution Policy (checkpoints/ensemble/rl_ensemble_best.onnx)

IMPORTANT: the feature window is *synthetic* (Gaussian noise), so the model
outputs are meaningless as a trading signal.  This script verifies plumbing --
observation schema, action mapping, order routing, position/mark-to-market
bookkeeping -- NOT strategy quality.  For real paper trading run
``python trading/live_engine.py --broker paper`` (synthetic market feed) or
``--broker oanda`` (practice account).
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
from models.rl_agents import build_action_mask, build_agent_state
from trading.live_actions import LiveAction, scaling_action_to_live_action
from trading.live_engine import PaperBroker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D logit vector."""
    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    z = z - np.max(z)
    e = np.exp(z)
    return e / max(float(np.sum(e)), 1e-12)


class LivePaperTradingSession:
    """Drives tick events through the ONNX models and a PaperBroker."""

    #: Contract size in units per lot.  MUST match PaperBroker.UNITS_PER_LOT and
    #: ForexTradingEnv.lot_size (scripts/train_rl.py) or P&L and the RL agent
    #: state are both off-scale.
    LOT_SIZE = 10_000.0
    #: Cap used by the RL agent state (ForexTradingEnv max_lots = 3.0 at training).
    MAX_LOTS = 3.0
    #: Per-trade clip applied by this harness so a demo cannot over-leverage.
    TRADE_LOTS = 0.20

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
        ens_shape = self.ensemble_session.get_inputs()[0].shape
        self.market_features = int(ens_shape[-1])

        # 2. Load Multi-RL Policy ONNX
        logging.info(f"Loading Multi-RL Consensus Policy from {rl_onnx}...")
        self.rl_session = ort.InferenceSession(rl_onnx, opts, providers=["CPUExecutionProvider"])
        self.rl_input_name = self.rl_session.get_inputs()[0].name
        self.obs_size = int(self.rl_session.get_inputs()[0].shape[-1])
        expected_obs = 2 + self.market_features + 5
        if self.obs_size != expected_obs:
            raise SystemExit(
                f"RL observation contract mismatch: ONNX expects {self.obs_size} values but the "
                f"trained layout is [signal(2) + market({self.market_features}) + agent_state(5)] "
                f"= {expected_obs} (scripts/train_rl.py:374, models/rl_agents.py:135)."
            )

        # State tracking per pair
        self.pair_states = {
            p: {
                "position_lots": 0.0,
                "entry_price": 0.0,
                "current_price": 1.1000 if "EUR" in p else (1.3000 if "GBP" in p else (1.3500 if "CAD" in p else 150.00)),
                "spread_pips": 1.2,
                "atr": 0.0015 if "JPY" not in p else 0.15,
                "holding_bars": 0,
                "trades_count": 0,
            }
            for p in pairs
        }
        self.trade_journal = []


    def _action_mask(self, position_lots: float) -> np.ndarray:
        """Validity mask for the 10-action space (see ``models.rl_agents``)."""
        return build_action_mask(position_lots, self.MAX_LOTS)

    def simulate_ticks_and_trade(self, n_ticks: int = 30):
        logging.info(f"Starting Paper Trading plumbing smoke test ({n_ticks} tick events across {len(self.pairs)} pairs)...")
        logging.warning(
            "Feature windows are synthetic Gaussian noise; model outputs are NOT a trading "
            "signal. The exported ensemble ONNX graph exposes only `logits`, so the trained "
            "ensemble-disagreement head is approximated by a softmax-confidence proxy."
        )
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
            # Synthetic feature window: (1, 120, market_features)
            features = rng.randn(1, 120, self.market_features).astype(np.float32)
            t_start = time.perf_counter()
            ens_logits = self.ensemble_session.run(None, {self.ens_input_name: features})[0][0]
            # Output is 3 logits: [Sell, Neutral, Buy]
            pred_class = int(np.argmax(ens_logits))
            dir_label = ["SELL", "HOLD", "BUY"][pred_class]
            probs = _softmax(ens_logits)

            # 3. Construct the RL observation using the TRAINED layout
            #    [signal (2), market_features (N), agent_state (5)]
            #    (scripts/train_rl.py:374 concatenates the 2 supervised-signal
            #    columns onto the raw market features; ForexTradingEnv then appends
            #    the 5 agent-state values, giving obs_size == N + 2 + 5.)
            signal_feat = np.array(
                [
                    float(probs[2] - probs[0]),  # p(up) - p(down)
                    float(1.0 - np.max(probs)),  # uncertainty proxy (see warning below)
                ],
                dtype=np.float32,
            )
            market_last = features[0, -1, :]

            mid = (bid + ask) / 2.0
            equity_now = float(self.broker.get_account()["equity"])
            agent_state = build_agent_state(
                position_lots=state["position_lots"],
                max_lots=self.MAX_LOTS,
                current_price=mid,
                entry_price=state["entry_price"],
                lot_size=self.LOT_SIZE,
                holding_bars=state["holding_bars"],
                equity=equity_now,
                initial_equity=self.initial_equity,
            )

            rl_obs = np.concatenate([signal_feat, market_last, agent_state]).astype(np.float32).reshape(1, -1)

            # 4. Run Multi-RL Consensus Policy ONNX
            rl_logits = self.rl_session.run(None, {self.rl_input_name: rl_obs})[0][0]
            t_infer = (time.perf_counter() - t_start) * 1000.0  # ms

            # 5. Action selection using the SAME mask the policy trained under
            #    (ForexTradingEnv.action_mask): scaling/closing is invalid while
            #    flat, and the direction that would double existing exposure is
            #    invalid.  Without the mask the policy emits actions the broker
            #    cannot honour and the journal reports phantom trades.
            mask = self._action_mask(state["position_lots"])
            masked_logits = np.where(mask, np.asarray(rl_logits, dtype=np.float64), -np.inf)
            scaling_act = ScalingAction(int(np.argmax(masked_logits)))

            # 6. Translate the RL action through the live execution contract and
            #    route it.  Going via scaling_action_to_live_action keeps the
            #    BUG-LIVE-001 regression (exits silently mapped to HOLD) covered by
            #    this harness.  An action is only reported as executed when an
            #    order was actually routed.
            pos_before = state["position_lots"]
            live_action = LiveAction(int(scaling_action_to_live_action(int(scaling_act), position_lots=pos_before)))
            action_desc = f"{scaling_act.name}->{live_action.name}"
            order_side: str | None = None
            order_lots = 0.0

            scale_in_frac = {
                LiveAction.SCALE_IN_25: 0.25,
                LiveAction.SCALE_IN_50: 0.50,
                LiveAction.SCALE_IN_100: 1.00,
            }
            scale_out_frac = {
                LiveAction.SCALE_OUT_25: 0.25,
                LiveAction.SCALE_OUT_50: 0.50,
                LiveAction.SCALE_OUT_100: 1.00,
            }

            if live_action is LiveAction.BUY and abs(pos_before) < 1e-12:
                order_side, order_lots = "BUY", self.TRADE_LOTS
                state["position_lots"] = order_lots
                state["entry_price"] = ask
                state["holding_bars"] = 0
                state["trades_count"] += 1
            elif live_action is LiveAction.SELL and abs(pos_before) < 1e-12:
                order_side, order_lots = "SELL", self.TRADE_LOTS
                state["position_lots"] = -order_lots
                state["entry_price"] = bid
                state["holding_bars"] = 0
                state["trades_count"] += 1
            elif live_action in scale_in_frac and abs(pos_before) > 1e-12:
                order_lots = round(min(self.TRADE_LOTS, abs(pos_before)) * scale_in_frac[live_action], 2)
                if order_lots > 0:
                    order_side = "BUY" if pos_before > 0 else "SELL"
                    fill = ask if pos_before > 0 else bid
                    new_abs = abs(pos_before) + order_lots
                    # Volume-weighted average entry, matching PaperBroker.market_order.
                    state["entry_price"] = (state["entry_price"] * abs(pos_before) + fill * order_lots) / new_abs
                    state["position_lots"] = new_abs if pos_before > 0 else -new_abs
            elif live_action in scale_out_frac and abs(pos_before) > 1e-12:
                order_lots = round(abs(pos_before) * scale_out_frac[live_action], 2)
                if order_lots > 0:
                    order_side = "SELL" if pos_before > 0 else "BUY"
                    remaining = abs(pos_before) - order_lots
                    if remaining <= 1e-12:
                        state["position_lots"] = 0.0
                        state["entry_price"] = 0.0
                        state["holding_bars"] = 0
                    else:
                        state["position_lots"] = remaining if pos_before > 0 else -remaining
            elif live_action is LiveAction.CLOSE and abs(pos_before) > 1e-12:
                order_side = "SELL" if pos_before > 0 else "BUY"
                order_lots = abs(pos_before)
                state["position_lots"] = 0.0
                state["entry_price"] = 0.0
                state["holding_bars"] = 0
            else:
                state["holding_bars"] += 1

            if order_side is not None:
                self.broker.market_order(pair, order_side, order_lots)
            else:
                action_desc = f"{action_desc} (noop)"

            acct = self.broker.get_account()
            unrealized = acct["equity"] - acct["balance"]

            print(
                f"#{tick_idx:<5} | {pair:<7} | {state['current_price']:<9.5f} | {dir_label:<9} | "
                f"{action_desc:<14} | {state['position_lots']:<+6.2f} | ${acct['equity']:<10,.2f} | "
                f"${unrealized:<+13,.2f} ({t_infer:.1f}ms)"
            )

            # Record event.  ``action`` is the action the policy requested and
            # ``executed`` is what actually reached the broker.
            self.trade_journal.append({
                "tick": tick_idx,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pair": pair,
                "price": float(state["current_price"]),
                "signal_dir": dir_label,
                "signal_confidence": float(np.max(probs)),
                "action": str(scaling_act.name),
                "executed": order_side is not None,
                "order_side": order_side,
                "order_lots": float(order_lots),
                "lots": float(state["position_lots"]),
                "equity": float(acct["equity"]),
                "balance": float(acct["balance"]),
                "latency_ms": float(t_infer),
            })

        print("=" * 90)
        final_acct = self.broker.get_account()
        pnl = final_acct["equity"] - self.initial_equity
        ret_pct = (pnl / self.initial_equity) * 100.0
        n_orders = sum(1 for e in self.trade_journal if e["executed"])
        print(f"\n[SUMMARY] Initial Equity : ${self.initial_equity:,.2f}")
        print(f"[SUMMARY] Final Equity   : ${final_acct['equity']:,.2f}")
        print(f"[SUMMARY] Mark-to-Market : ${pnl:+,.2f} ({ret_pct:+.2f}%)")
        print(f"[SUMMARY] Orders Routed  : {n_orders} of {n_ticks} tick events")
        print(f"[SUMMARY] Total Open Lots: {sum(abs(v['position_lots']) for v in self.pair_states.values()):.2f}")
        print("[SUMMARY] NOTE: features were synthetic Gaussian noise - this is a plumbing")
        print("[SUMMARY]       smoke test, NOT a strategy result. Use trading/live_engine.py")
        print("[SUMMARY]       --broker paper (or --broker oanda) for actual paper trading.")

        # Save journal
        out_log = Path("logs/paper_trading_session.json")
        out_log.parent.mkdir(parents=True, exist_ok=True)
        with open(out_log, "w", encoding="utf-8") as fh:
            json.dump({
                "disclaimer": (
                    "Synthetic-feature plumbing smoke test. Model outputs are meaningless as "
                    "a trading signal; only order routing and bookkeeping are being verified."
                ),
                "session_summary": {
                    "initial_equity": self.initial_equity,
                    "final_equity": final_acct["equity"],
                    "pnl": pnl,
                    "return_pct": ret_pct,
                    "ticks_processed": n_ticks,
                    "orders_routed": n_orders,
                    "pairs": self.pairs,
                    "lot_size": self.LOT_SIZE,
                    "max_lots": self.MAX_LOTS,
                },
                "journal": self.trade_journal,
            }, fh, indent=2)
        logging.info(f"Live Paper Trading session journal saved to {out_log}")


if __name__ == "__main__":
    session = LivePaperTradingSession()
    session.simulate_ticks_and_trade(n_ticks=24)
