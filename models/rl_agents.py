"""
models/rl_agents.py
====================
PPO and Deep Q-Learning agents with:
  - 10-action ScalingAction space for hold/open/scale/reduce/close execution
  - Decomposable reward (P&L - drawdown - costs - overtrading)
  - Combined pyramiding + martingale scaling strategy
  - Dynamic stop-loss integration
"""

import collections
import random
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH = True
except ImportError:
    TORCH = False


from backtesting.backtest import ScalingAction

ACTION_NAMES = {a.value: a.name for a in ScalingAction}


# ─────────────────────────────────────────────────────────────────────────────
# TRADING ENVIRONMENT  (10-action, combined scaling, dynamic SL)
# ─────────────────────────────────────────────────────────────────────────────

class ForexTradingEnv:
    """
    Forex environment for 10-action RL execution agents.

    Market array units (pass from cache via train_gpu A-C2):
      - prices: absolute FX quote (e.g. 1.0850 for EURUSD), one per sequence/bar
      - atr: average true range in price units (same scale as prices; used for SL/TP)
      - spreads: bid-ask width in price units (spread_pips * pip_size from the pipeline)
      - pip_size (default 0.0001): used only when converting SL/TP hits to pip PnL

    Scaling strategy: BOTH combined (spec requirement)
      - Pyramiding:  add to winning positions (momentum)
      - Martingale:  add to losing positions (mean-reversion, guarded)

    Reward is DECOMPOSABLE into 4 components that can be monitored
    independently in production to diagnose performance decay.
    """

    def __init__(
        self,
        features:       np.ndarray,
        prices:         np.ndarray,
        atr:            np.ndarray,
        spreads:        np.ndarray,
        initial_equity: float = 10_000.0,
        lot_size:       float = 10_000.0,
        max_lots:       float = 3.0,
        commission_per_lot: float = 3.5,
        slippage_pips:  float = 0.5,
        pip_size:       float = 0.0001,
        reward_weights: dict | None = None,
        # Dynamic SL params
        atr_sl_mult:    float = 1.5,
        trail_activation_r: float = 1.0,
        breakeven_at_r: float = 0.5,
        # Scaling params (both)
        pyramid_pct:    float = 0.25,
        martingale_pct: float = 0.25,
        # A-H1: episode sampling — randomise start offset / sub-window each reset
        # so the agent doesn't replay the identical trajectory every episode.
        random_reset:   bool = True,
        episode_len:    int | None = None,
        # Annualised bars for Sharpe: 252 * 24 * 60 for 1-min bars. Pass
        # 252 * 24 * (60 // m) for m-minute bars (e.g. 3024 for 5-min).
        bars_per_year:  int = 252 * 24 * 60,
    ):
        self.features       = features.astype(np.float32)
        self.prices         = prices.astype(np.float32)
        self.atr            = atr.astype(np.float32)
        self.spreads        = spreads.astype(np.float32)
        self.initial_equity = initial_equity
        self.lot_size       = lot_size
        self.max_lots       = max_lots
        self.commission     = commission_per_lot
        self.slippage_pips  = slippage_pips
        self.pip_size       = pip_size
        self.rw             = {
            "pnl": 1.0, "drawdown": 0.5, "tx_cost": 0.3, "overtrade": 0.2, "holding": 0.01
        }
        if reward_weights:
            self.rw.update({
                "pnl": float(reward_weights.get("pnl", reward_weights.get("pnl_weight", self.rw["pnl"]))),
                "drawdown": float(reward_weights.get("drawdown", reward_weights.get("drawdown_penalty", self.rw["drawdown"]))),
                "tx_cost": float(reward_weights.get("tx_cost", reward_weights.get("transaction_cost_penalty", self.rw["tx_cost"]))),
                "overtrade": float(reward_weights.get("overtrade", reward_weights.get("overtrading_penalty", self.rw["overtrade"]))),
                "holding": float(reward_weights.get("holding", reward_weights.get("holding_cost", self.rw["holding"]))),
            })
        self.atr_sl_mult    = atr_sl_mult
        self.trail_act_r    = trail_activation_r
        self.be_r           = breakeven_at_r
        self.pyramid_pct    = pyramid_pct
        self.martingale_pct = martingale_pct
        self.random_reset   = bool(random_reset)
        # Sub-window length per episode. None or >= series length => use full series.
        self.episode_len    = int(episode_len) if episode_len else None

        self.obs_size = features.shape[1] + 5  # + agent state
        self.n_actions = 10
        self.bars_per_year = int(bars_per_year)
        self.reset()

    def reset(self, valid_starts: np.ndarray | None = None) -> np.ndarray:
        # A-H1: sample a random start offset / sub-window so each episode sees a
        # different slice of the price history (otherwise every episode replays
        # the identical trajectory from idx=0 and the agent overfits one path).
        n = len(self.prices)
        if valid_starts is not None and len(valid_starts) > 0:
            if self.episode_len is not None:
                valid = valid_starts[valid_starts <= n - self.episode_len - 1]
                if len(valid) == 0: valid = np.array([0])
            else:
                valid = valid_starts
            self.start_idx = int(np.random.choice(valid)) if self.random_reset else int(valid[0])
            self.end_idx = min(self.start_idx + (self.episode_len if self.episode_len else n - 1), n - 1)
        elif self.episode_len is not None and self.episode_len < n - 1:
            max_start = max(0, n - self.episode_len - 1)  # guard against negative when episode_len >= n
            self.start_idx = int(np.random.randint(0, max_start + 1)) if self.random_reset else 0
            self.end_idx   = self.start_idx + self.episode_len
        else:
            self.start_idx = (int(np.random.randint(0, max(1, (n - 1) // 4)))
                              if self.random_reset else 0)
            self.end_idx   = n - 1
        self.idx        = self.start_idx
        self.equity     = self.initial_equity
        self.peak       = self.initial_equity
        self.position   = 0.0      # lots (+long, -short)
        self.entry_price = 0.0
        # _initial_entry_price is locked at trade open and never updated on
        # scale-in. SL/TP always reference this so stops don't slip when we pyramid.
        self._initial_entry_price = 0.0
        self.stop_loss  = 0.0
        self.take_profit = 0.0
        self.holding    = 0
        self.n_trades   = 0
        self.total_costs = 0.0
        self.episode_pnl = []
        self.done       = False
        self._prev_mtm_equity = self.initial_equity  # Track MTM equity for reward consistency
        self._last_obs  = self._obs()
        return self._last_obs

    def _obs(self) -> np.ndarray:
        mkt = self.features[self.idx]
        p   = self.prices[self.idx]
        upnl = (p - self.entry_price) * self.position * self.lot_size if self.position != 0 else 0.0
        agent = np.array([
            np.clip(self.position / self.max_lots, -1, 1),
            np.clip(upnl / self.initial_equity, -0.5, 0.5),
            min(self.holding / 100, 1.0),
            np.clip((self.equity - self.initial_equity) / self.initial_equity, -0.5, 0.5),
            int(self.position != 0),
        ], dtype=np.float32)
        return np.concatenate([mkt, agent])

    def action_mask(self) -> np.ndarray:
        """
        Returns a boolean array where True means the action is valid.
        0: HOLD, 1: OPEN_LONG, 2: OPEN_SHORT, 3-5: SCALE_IN, 6-8: SCALE_OUT, 9: CLOSE_ALL
        """
        mask = np.ones(self.n_actions, dtype=bool)
        if self.position == 0:
            mask[3:] = False  # Can't scale or close if flat
        else:
            if self.position > 0:
                mask[1] = False
            elif self.position < 0:
                mask[2] = False
            if abs(self.position) >= self.max_lots - 1e-6:
                mask[3:6] = False
        return mask

    def _exec_cost(self, lots: float) -> float:
        cost = abs(lots) * self.commission + abs(lots) * self.slippage_pips * self.pip_size * self.lot_size
        self.equity -= cost; self.total_costs += cost
        return cost

    def _set_dynamic_sl(self, direction: int, entry: float, current_atr: float):
        """Dynamic stop-loss: ATR-based, trails after profit, moves to breakeven.
        Uses _initial_entry_price so stops are always anchored to trade open."""
        self._initial_entry_price = entry  # lock at first fill
        self.stop_loss   = entry - direction * self.atr_sl_mult * current_atr
        self.take_profit = entry + direction * self.atr_sl_mult * 2 * current_atr

    def _update_trailing_sl(self, p: float, direction: int, entry: float, current_atr: float):
        """Trail SL after profit exceeds trail_activation_r × ATR.
        Uses _initial_entry_price so scale-in doesn't reset the trail anchor."""
        if self.position == 0: return
        init_entry = self._initial_entry_price
        profit_r = direction * (p - init_entry) / (current_atr + 1e-9)
        if profit_r >= self.trail_act_r:
            new_sl = p - direction * self.atr_sl_mult * current_atr
            if direction > 0: self.stop_loss = max(self.stop_loss, new_sl)
            else:             self.stop_loss = min(self.stop_loss, new_sl)
        elif profit_r >= self.be_r:
            # Move to breakeven using initial entry, not averaged entry
            if direction > 0: self.stop_loss = max(self.stop_loss, init_entry)
            else:             self.stop_loss = min(self.stop_loss, init_entry)

    def _check_sl_tp(self, p: float, direction: int) -> tuple[bool, float]:
        """Returns (hit, pnl_pips). P&L from avg ``entry_price`` on remaining size."""
        entry = self.entry_price
        if direction > 0:
            if p <= self.stop_loss:   return True, (self.stop_loss - entry) / self.pip_size
            if p >= self.take_profit: return True, (self.take_profit - entry) / self.pip_size
        else:
            if p >= self.stop_loss:   return True, (entry - self.stop_loss) / self.pip_size
            if p <= self.take_profit: return True, (entry - self.take_profit) / self.pip_size
        return False, 0.0

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        assert not self.done
        p   = self.prices[self.idx]
        atr = self.atr[self.idx]
        direction = int(np.sign(self.position)) if self.position != 0 else 0

        realised_pnl = 0.0; cost = 0.0

        # ── Check dynamic stop/TP before executing new action ───────────────
        if self.position != 0:
            self._update_trailing_sl(p, direction, self.entry_price, atr)
            hit, pips = self._check_sl_tp(p, direction)
            if hit:
                pnl_usd = pips * self.pip_size * abs(self.position) * self.lot_size
                realised_pnl += pnl_usd; self.equity += pnl_usd
                self.position = 0.0; self.holding = 0
                self.n_trades += 1

        # Position before the new action (after any SL/TP close above) — used to
        # detect a fresh entry / scale-in / flip for the trade-frequency penalty.
        pos_before = self.position

        # ── Execute action ──────────────────────────────────────────────────
        if action == ScalingAction.OPEN_LONG.value:
            if self.position == 0:
                lots = 1.0; cost = self._exec_cost(lots)
                self.position = lots; self.entry_price = p
                self._set_dynamic_sl(+1, p, atr)
            elif self.position < 0:
                # Close short and open long
                pnl = (self.entry_price - p) * abs(self.position) * self.lot_size
                realised_pnl += pnl; self.equity += pnl
                cost = self._exec_cost(abs(self.position) + 1.0)
                self.position = 1.0; self.entry_price = p; self.holding = 0; self.n_trades += 1
                self._set_dynamic_sl(+1, p, atr)

        elif action == ScalingAction.OPEN_SHORT.value:
            if self.position == 0:
                lots = 1.0; cost = self._exec_cost(lots)
                self.position = -lots; self.entry_price = p
                self._set_dynamic_sl(-1, p, atr)
            elif self.position > 0:
                pnl = (p - self.entry_price) * self.position * self.lot_size
                realised_pnl += pnl; self.equity += pnl
                cost = self._exec_cost(abs(self.position) + 1.0)
                self.position = -1.0; self.entry_price = p; self.holding = 0; self.n_trades += 1
                self._set_dynamic_sl(-1, p, atr)

        elif action == ScalingAction.CLOSE_ALL.value and self.position != 0:
            pnl = (p - self.entry_price) * self.position * self.lot_size if self.position > 0 else (self.entry_price - p) * abs(self.position) * self.lot_size
            realised_pnl += pnl; self.equity += pnl
            cost = self._exec_cost(abs(self.position))
            self.position = 0.0; self.holding = 0; self.n_trades += 1

        elif action in [ScalingAction.SCALE_IN_25.value, ScalingAction.SCALE_IN_50.value, ScalingAction.SCALE_IN_100.value] and self.position != 0:
            frac = {ScalingAction.SCALE_IN_25.value: 0.25, ScalingAction.SCALE_IN_50.value: 0.50, ScalingAction.SCALE_IN_100.value: 1.0}[action]
            add = min(frac, self.max_lots - abs(self.position))
            if add > 0:
                self._update_avg_entry(p, add * direction)
                cost = self._exec_cost(add)

        elif action in [ScalingAction.SCALE_OUT_25.value, ScalingAction.SCALE_OUT_50.value, ScalingAction.SCALE_OUT_100.value] and self.position != 0:
            frac = {ScalingAction.SCALE_OUT_25.value: 0.25, ScalingAction.SCALE_OUT_50.value: 0.50, ScalingAction.SCALE_OUT_100.value: 1.0}[action]
            sub = min(frac, abs(self.position))
            if sub > 0:
                pnl = (p - self.entry_price) * sub * self.lot_size if direction > 0 else (self.entry_price - p) * sub * self.lot_size
                realised_pnl += pnl; self.equity += pnl
                cost = self._exec_cost(sub)
                self.position -= sub * direction
                if abs(self.position) < 1e-6:
                    self.position = 0.0; self.holding = 0; self.n_trades += 1

        # HOLD: do nothing

        if self.position != 0: self.holding += 1
        # Mark-to-market equity: include unrealised PnL so holding losses are penalised
        if self.position != 0:
            _direction = int(np.sign(self.position))
            _unrealised = _direction * (p - self.entry_price) * abs(self.position) * self.lot_size
        else:
            _unrealised = 0.0
        mtm_equity = self.equity + _unrealised
        self.peak = max(self.peak, mtm_equity)
        self.episode_pnl.append(realised_pnl)

        # A-M2: trade-frequency cost. A fresh entry, scale-in, or flip (position
        # magnitude grew or sign changed) incurs a FIXED penalty per event so the
        # agent is discouraged from overtrading. The old `1/n_trades` term was
        # inverted -- it shrank as trades grew, effectively *rewarding* churn.
        opened_or_flipped = (
            abs(self.position) > abs(pos_before) + 1e-9
            or (self.position != 0 and np.sign(self.position) != np.sign(pos_before))
        )

        # -- Decomposable reward (consistent MTM-based) --
        w = self.rw
        dd = max(0, (self.peak - mtm_equity) / max(self.peak, 1e-8))
        # FIX: Use MTM P&L (change in mtm_equity) for consistency with MTM drawdown
        # Previously used realised_pnl which mixed realized P&L with MTM drawdown,
        # creating perverse incentive to hold losers (avoids realized loss but penalized for drawdown)
        mtm_pnl = mtm_equity - getattr(self, "_prev_mtm_equity", mtm_equity)
        self._prev_mtm_equity = mtm_equity
        holding_penalty = w["holding"] * abs(self.position) * min(self.holding / 100, 1.0)
        reward = (
            w["pnl"]       * mtm_pnl / self.initial_equity
          - w["drawdown"]  * dd
          - w["tx_cost"]   * cost / self.initial_equity
          - w["overtrade"] * (1.0 if opened_or_flipped else 0.0)
          - holding_penalty
        )

        self.idx += 1
        self.done = self.idx >= self.end_idx

        # Force-close at episode end — P&L from avg entry on remaining size;
        # include in episode_pnl so Sharpe sees the terminal close.
        # NOTE: do NOT add final_pnl to reward — MTM-based reward above already
        # captured the full unrealised move step-by-step via mtm_pnl.
        # Adding it again would double-count the terminal bar's P&L and create a
        # perverse incentive for the agent to hold positions until end-of-episode.
        if self.done and self.position != 0:
            last_p = self.prices[min(self.idx, len(self.prices) - 1)]
            d = int(np.sign(self.position))
            final_pnl = d * (last_p - self.entry_price) * abs(self.position) * self.lot_size
            self.equity += final_pnl
            self.episode_pnl[-1] = float(self.episode_pnl[-1]) + final_pnl
            # reward is intentionally NOT adjusted — MTM already accounts for this
            self.position = 0.0
            self.holding = 0
            self.n_trades += 1

        # Return last valid obs on done (not zeros) so value bootstrapping isn't confused
        obs = self._obs() if not self.done else self._last_obs
        if not self.done:
            self._last_obs = obs
        return obs, float(reward), self.done, {"equity": self.equity, "pnl": realised_pnl}

    def _update_avg_entry(self, p: float, delta_lots: float):
        total = abs(self.position) + abs(delta_lots)
        self.entry_price = (abs(self.position) * self.entry_price + abs(delta_lots) * p) / total
        self.position += delta_lots

    def summary(self) -> dict:
        rets = np.array(self.episode_pnl)
        bars_per_year = self.bars_per_year
        if len(rets) > 1:
            std = rets.std(ddof=1)
            sharpe = (rets.mean() / (std + 1e-9)) * np.sqrt(bars_per_year) if std > 1e-12 else 0.0
        else:
            sharpe = 0.0
        return {
            "total_return_pct": (self.equity / self.initial_equity - 1) * 100,
            "sharpe": sharpe,
            "n_trades": self.n_trades,
            "total_costs": self.total_costs,
            "max_dd_pct": max(0, (self.peak - self.equity) / self.peak) * 100,
        }


# ─────────────────────────────────────────────────────────────────────────────
# PPO AGENT
# ─────────────────────────────────────────────────────────────────────────────

if TORCH:

    class ActorCritic(nn.Module):
        """Shared backbone -> actor (policy) + critic (value) heads.

        With ``use_lstm=True`` the backbone is an LSTM over a (B, T, D) window
        of past observations; policy/critic read the final hidden state. Single
        obs vectors are broadcast to T=1 so the same interface works.
        """
        def __init__(self, obs_size, n_actions=10, hidden=256,
                     use_lstm=False, lstm_hidden=128, num_layers=1, dropout=0.0):
            super().__init__()
            self.use_lstm = use_lstm
            if use_lstm:
                self.lstm = nn.LSTM(obs_size, lstm_hidden, num_layers=num_layers,
                                    batch_first=True, dropout=dropout)
                self.proj = nn.Sequential(nn.Linear(lstm_hidden, hidden), nn.Tanh())
            else:
                self.backbone = nn.Sequential(
                    nn.Linear(obs_size, hidden), nn.Tanh(),
                    nn.Linear(hidden, hidden),   nn.Tanh(),
                )
            self.actor  = nn.Linear(hidden, n_actions)
            self.critic = nn.Linear(hidden, 1)

        def forward(self, x):
            if self.use_lstm:
                if x.ndim == 2:
                    x = x.unsqueeze(1)
                h, _ = self.lstm(x)
                h = self.proj(h[:, -1, :])
            else:
                h = self.backbone(x)
            return self.actor(h), self.critic(h)

        def act(self, obs, mask=None, greedy: bool = False):
            """Sample an action from the policy.

            Parameters
            ----------
            obs    : (B, ...) observation tensor.
            mask   : optional (B, A) boolean action mask (True = allowed).
            greedy : when True, return the argmax-action instead of sampling.
                     Use during live inference / deterministic evaluation; never
                     during training rollout (PPO exploration needs sampling).
                     Audit finding I3 (2026-08-07): previously the actor was
                     always stochastic, which made live inference nondeterministic
                     and added noise to position-sizing decisions.
            """
            logits, value = self(obs)
            if mask is not None:
                # mask may be a numpy bool array; masked_fill requires a torch bool tensor
                if not isinstance(mask, torch.Tensor):
                    mask = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
                logits = logits.masked_fill(~mask, -1e9)
            if greedy:
                # Greedy: argmax over action logits. log_prob is informational
                # only (the caller shouldn't train on greedy outputs).
                action = logits.argmax(dim=-1)
                # log_prob under the policy: useful for diagnostics
                with torch.no_grad():
                    dist = torch.distributions.Categorical(logits=logits)
                    log_prob = dist.log_prob(action)
                return action, log_prob, value.squeeze(-1)
            dist   = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            return action, dist.log_prob(action), value.squeeze(-1)

        def evaluate(self, obs, actions, mask=None):
            logits, value = self(obs)
            if mask is not None:
                # mask may be a numpy bool array; masked_fill requires a torch bool tensor
                if not isinstance(mask, torch.Tensor):
                    mask = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
                logits = logits.masked_fill(~mask, -1e9)
            dist = torch.distributions.Categorical(logits=logits)
            return dist.log_prob(actions), dist.entropy(), value.squeeze(-1)


    class PPOAgent:
        """
        Proximal Policy Optimisation agent for 10-action forex execution.
        Uses GAE advantage estimation and clip-objective for stable training.
        """
        def __init__(self, obs_size, n_actions=10, hidden=256,
                     lr=3e-4, gamma=0.99, lam=0.95, clip=0.2,
                     entropy_coef=0.01, value_coef=0.5, n_epochs=10, device="cpu",
                     use_lstm=False, lstm_hidden=128, hist_len=32):
            self.gamma = gamma; self.lam = lam; self.clip = clip
            self.ent_c = entropy_coef; self.val_c = value_coef
            self.n_epochs = n_epochs
            self.device = torch.device(device)
            self.use_lstm = bool(use_lstm)
            self.hist_len = int(hist_len)
            self.net = ActorCritic(obs_size, n_actions, hidden,
                                   use_lstm=self.use_lstm, lstm_hidden=lstm_hidden).to(self.device)
            self.opt = optim.Adam(self.net.parameters(), lr=lr)
            self.buffer = []
            self._hist = collections.deque(maxlen=self.hist_len)

        def _make_hist_seq(self) -> np.ndarray:
            """Left-pad the observation deque to ``hist_len`` (causal)."""
            seq = list(self._hist)
            if not seq:
                raise RuntimeError("PPOAgent LSTM history is empty")
            if len(seq) < self.hist_len:
                seq = [seq[0]] * (self.hist_len - len(seq)) + seq
            return np.stack(seq[-self.hist_len :], axis=0)

        def _preview_obs_seq(self, obs: np.ndarray) -> np.ndarray:
            """Sequence ending at ``obs`` without mutating history (for bootstrap)."""
            arr = np.asarray(obs, dtype=np.float32).copy()
            seq = list(self._hist) + [arr]
            if len(seq) < self.hist_len:
                seq = [seq[0]] * (self.hist_len - len(seq)) + seq
            return np.stack(seq[-self.hist_len :], axis=0)

        def select_action(self, obs: np.ndarray, mask: np.ndarray | None = None,
                          greedy: bool = False):
            """Select an action from the policy.

            Parameters
            ----------
            obs    : current observation.
            mask   : optional boolean mask of allowed actions.
            greedy : when True, returns the argmax-action — used by live
                     inference / deterministic eval. Default False = stochastic
                     (training exploration).
            """
            if self.use_lstm:
                x = self._preview_obs_seq(obs)
            else:
                x = obs
            xt = torch.tensor(x, dtype=torch.float32, device=self.device)
            if self.use_lstm:
                xt = xt.unsqueeze(0)
            m = torch.tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0) if mask is not None else None
            with torch.no_grad():
                action, log_prob, value = self.net.act(xt, mask=m, greedy=greedy)
            return action.item(), log_prob.item(), value.item()

        def store(self, obs, action, reward, done, log_prob, value, mask=None):
            if mask is None: mask = np.ones(self.net.actor.out_features, dtype=bool)
            if self.use_lstm:
                self._hist.append(np.asarray(obs, dtype=np.float32).copy())
                seq = self._make_hist_seq()
                self.buffer.append((seq, action, reward, done, log_prob, value, mask))
                if done:
                    self._hist.clear()
            else:
                self.buffer.append((obs, action, reward, done, log_prob, value, mask))

        def update(self, last_value: float = 0.0):
            """Run PPO update on the current buffer.

            last_value : critic value V(s_{T+1}) used to bootstrap the final
                         transition. On a TRUE terminal pass 0.0; on a buffer
                         truncation (n_steps reached mid-episode) pass the critic
                         value of the next observation so the advantage of the
                         last step isn't silently bootstrapped to 0 (A-M4).
            """
            if len(self.buffer) < 64: return {}
            obs_b, act_b, rew_b, done_b, lp_b, val_b, mask_b = zip(*self.buffer)
            self.buffer.clear()

            if self.use_lstm:
                obs_t = torch.tensor(np.stack(obs_b, axis=0), dtype=torch.float32, device=self.device)
            else:
                obs_t = torch.tensor(np.array(obs_b), dtype=torch.float32, device=self.device)
            act_t  = torch.tensor(act_b,            dtype=torch.long,    device=self.device)
            rew_t  = torch.tensor(rew_b,            dtype=torch.float32, device=self.device)
            done_t = torch.tensor(done_b,           dtype=torch.float32, device=self.device)
            old_lp = torch.tensor(lp_b,             dtype=torch.float32, device=self.device)
            val_t  = torch.tensor(val_b,            dtype=torch.float32, device=self.device)
            mask_t = torch.tensor(np.array(mask_b), dtype=torch.bool,    device=self.device)

            # GAE advantage. The final transition bootstraps with `last_value`
            # (still gated by its done flag) instead of a hardcoded 0, which
            # previously truncated the value target whenever the rollout buffer
            # filled mid-episode (A-M4).
            adv = torch.zeros_like(rew_t); gae = 0.0
            for t in reversed(range(len(rew_t))):
                nv  = val_t[t+1] if t < len(rew_t)-1 else float(last_value)
                delta = rew_t[t] + self.gamma * nv * (1 - done_t[t]) - val_t[t]
                gae   = delta + self.gamma * self.lam * (1 - done_t[t]) * gae
                adv[t] = gae
            returns = adv + val_t
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            total_loss = 0.0
            for _ in range(self.n_epochs):
                lp_new, entropy, val_new = self.net.evaluate(obs_t, act_t, mask=mask_t)
                ratio  = (lp_new - old_lp).exp()
                s1 = ratio * adv
                s2 = ratio.clamp(1 - self.clip, 1 + self.clip) * adv
                pol_loss = -torch.min(s1, s2).mean()
                val_loss = F.mse_loss(val_new, returns)
                loss = pol_loss + self.val_c * val_loss - self.ent_c * entropy.mean()
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()
                total_loss += loss.item()

            return {"loss": total_loss / self.n_epochs}


    # ── Deep Q-Network ────────────────────────────────────────────────────

    class DQNetwork(nn.Module):
        """Double DQN network."""
        def __init__(self, obs_size, n_actions=10, hidden=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_size, hidden), nn.ReLU(),
                nn.Linear(hidden,   hidden), nn.ReLU(),
                nn.Linear(hidden,   n_actions),
            )
        def forward(self, x): return self.net(x)


    class ReplayBuffer:
        def __init__(self, capacity=1_000_000):
            self.capacity = capacity
            self.buf = collections.deque(maxlen=capacity)
            self.class_counts = collections.defaultdict(int)
            self._cached_weights = None
            self._cached_len = -1
            self._sample_calls = 0
            # Fix D: track push count to amortize O(N) weight rebuild cost.
            # Previously, _cached_weights=None on every push forced O(N) rebuild
            # every sample() call — crushing DQN throughput with a 1M buffer.
            self._push_count = 0
            self._last_rebuild_push = 0

        def push(self, *args):
            # args = (obs, action, reward, next_obs, done)
            was_full = len(self.buf) == self.capacity
            if was_full:
                old_action = self.buf[0][1]
                self.class_counts[old_action] -= 1
            self.buf.append(args)
            self.class_counts[args[1]] += 1
            self._push_count += 1
            # Fix D: only invalidate cache every 1000 pushes, not on every push.
            # The sampling distribution drifts by at most ~0.1% between rebuilds,
            # which is negligible vs. the O(N) overhead of rebuilding at 1M items.
            # Also invalidate when the buffer transitions from growing to full,
            # because that changes which old entry was evicted.
            became_full = (not was_full) and len(self.buf) == self.capacity
            if became_full or (self._push_count - self._last_rebuild_push) >= 1000:
                self._cached_weights = None
                self._cached_len = -1
                self._last_rebuild_push = self._push_count

        def sample(self, n):
            if len(self.buf) == 0:
                return []
            buf_len = len(self.buf)
            # Rebuild weights only when invalidated (see push() above)
            if self._cached_weights is None or self._cached_len != buf_len:
                # Precompute inverse-frequency weight per action class
                inv_freq = {a: 1.0 / (c + 1e-3) for a, c in self.class_counts.items()}
                weights = np.array([inv_freq.get(t[1], 1.0) for t in self.buf])
                weights /= weights.sum()
                self._cached_weights = weights
                self._cached_len = buf_len
            self._sample_calls += 1
            indices = np.random.choice(buf_len, size=min(n, buf_len), p=self._cached_weights, replace=True)
            return [self.buf[i] for i in indices]

        def __len__(self):     return len(self.buf)


    class DQNAgent:
        """
        Double DQN agent — faster inference (~2ms) than PPO, ideal for TIP-Search
        fast model. Epsilon-greedy exploration decays over training.
        """
        def __init__(self, obs_size, n_actions=10, hidden=256,
                     lr=1e-4, gamma=0.99, eps_start=1.0, eps_end=0.01,
                     eps_decay=0.995, buf_size=1_000_000, batch=64,
                     target_update=100, double_dqn=True, device="cpu"):
            self.n_actions = n_actions; self.gamma = gamma; self.batch = batch
            self.double = double_dqn; self.target_update = target_update
            self.eps = eps_start; self.eps_end = eps_end; self.eps_decay = eps_decay
            self.device = torch.device(device)
            self.policy_net = DQNetwork(obs_size, n_actions, hidden).to(self.device)
            self.target_net = DQNetwork(obs_size, n_actions, hidden).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.opt    = optim.Adam(self.policy_net.parameters(), lr=lr)
            self.buf    = ReplayBuffer(buf_size)
            self.steps  = 0

        def select_action(self, obs: np.ndarray, mask: np.ndarray | None = None) -> int:
            if random.random() < self.eps:
                if mask is not None:
                    valid = np.where(mask)[0]
                    return int(np.random.choice(valid)) if len(valid) > 0 else 0
                return random.randrange(self.n_actions)
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q = self.policy_net(x).squeeze(0)
                if mask is not None:
                    q = q.masked_fill(~torch.tensor(mask, dtype=torch.bool, device=self.device), -1e9)
                return int(q.argmax(dim=0).item())

        def store(self, obs, action, reward, next_obs, done, next_mask=None):
            self.buf.push(obs, action, reward, next_obs, done, next_mask)

        def decay_epsilon(self):
            """A-M1: decay exploration ONCE per episode (call from the training
            loop). Decaying per environment step collapsed epsilon to eps_end in
            ~900 steps (decay**900 ≈ eps_end), killing exploration almost
            immediately on multi-thousand-step trajectories."""
            self.eps = max(self.eps_end, self.eps * self.eps_decay)

        def update(self) -> dict:
            if len(self.buf) < self.batch: return {}
            batch = self.buf.sample(self.batch)
            obs_b, act_b, rew_b, nobs_b, done_b = zip(*[t[:5] for t in batch])
            nmsk_b = [t[5] if len(t) > 5 else None for t in batch]

            obs  = torch.tensor(np.array(obs_b),  dtype=torch.float32, device=self.device)
            acts = torch.tensor(act_b,             dtype=torch.long,    device=self.device)
            rews = torch.tensor(rew_b,             dtype=torch.float32, device=self.device)
            nobs = torch.tensor(np.array(nobs_b),  dtype=torch.float32, device=self.device)
            done = torch.tensor(done_b,             dtype=torch.float32, device=self.device)

            # TM-013: apply next-state action masks so the target network can
            # never bootstrap from an action that is invalid in that state.
            has_masks = any(m is not None for m in nmsk_b)
            if has_masks:
                _ones = np.ones(self.n_actions, dtype=bool)
                nmsk_t = torch.tensor(
                    np.array([m if m is not None else _ones for m in nmsk_b], dtype=bool),
                    dtype=torch.bool, device=self.device,
                )

            q_vals = self.policy_net(obs).gather(1, acts.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                if self.double:
                    policy_nq = self.policy_net(nobs)
                    target_nq = self.target_net(nobs)
                    if has_masks:
                        policy_nq = policy_nq.masked_fill(~nmsk_t, -1e9)
                        target_nq = target_nq.masked_fill(~nmsk_t, -1e9)
                    best_acts = policy_nq.argmax(1)
                    next_q = target_nq.gather(1, best_acts.unsqueeze(1)).squeeze(1)
                else:
                    next_q = self.target_net(nobs)
                    if has_masks:
                        next_q = next_q.masked_fill(~nmsk_t, -1e9)
                    next_q = next_q.max(1)[0]
                target = rews + self.gamma * next_q * (1 - done)

            loss = F.smooth_l1_loss(q_vals, target)
            self.opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.opt.step()

            self.steps += 1
            if self.steps % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            return {"loss": loss.item(), "epsilon": self.eps}

else:
    class PPOAgent:
        def __init__(self, **kw): pass
        def select_action(self, obs): return 0
        def store(self, *a): pass
        def update(self, last_value: float = 0.0): return {}

    class DQNAgent:
        def __init__(self, **kw): pass
        def select_action(self, obs, mask=None): return 0
        def store(self, *a): pass
        def decay_epsilon(self): pass
        def update(self): return {}


# ─────────────────────────────────────────────────────────────────────────────
# REWARD NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

class RunningRewardNormalizer:
    """Online reward normalisation via Welford's algorithm.

    Normalises rewards to ~N(0,1) during training so the agent sees
    consistent reward magnitudes regardless of price scale or episode length.
    """

    def __init__(self, decay: float = 0.999):
        self.decay    = decay
        self.mean     = 0.0
        self.m2       = 0.0  # sum of squared diffs
        self.count    = 1e-4
        self._alpha   = 1.0 - decay

    def __call__(self, reward: float) -> float:
        self.count = self.decay * self.count + 1.0
        delta      = reward - self.mean
        self.mean += delta / self.count
        delta2     = reward - self.mean
        self.m2    = self.decay * self.m2 + delta * delta2
        std        = (self.m2 / self.count) ** 0.5 + 1e-8
        # BUG-008: subtract mean for proper ~N(0,1) normalization
        return float(np.clip((reward - self.mean) / std, -5.0, 5.0))

    def reset_episode(self):
        pass  # keep running stats across episodes


# ── Sharpe-reward wrapper adapter (bridges rl_advanced.SharpeRewardWrapper) ──

class _SharpeRewardAdapter:
    """Wraps SharpeRewardWrapper for use inside train_agent()."""

    def __init__(self, window: int = 100, cost_penalty: float = 0.3, dd_penalty: float = 0.5):
        from models.rl_advanced import SharpeRewardWrapper as _SRW
        self._w = _SRW(window=window, cost_penalty=cost_penalty, dd_penalty=dd_penalty)

    def __call__(self, raw_pnl: float, tx_cost: float, equity: float) -> float:
        return self._w.compute(raw_pnl=raw_pnl, tx_cost=tx_cost, equity=equity)

    def reset_episode(self):
        self._w.reset()

    def rolling_sharpe(self) -> float:
        return self._w.rolling_sharpe()


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_agent(
    agent,
    env: ForexTradingEnv,
    n_episodes: int = 5,
    agent_type: str = "ppo",
    greedy: bool = True,
) -> tuple:
    """Run evaluation episodes; return (episode_returns, env.summary())."""
    old_eps = getattr(agent, "eps", None)
    if greedy and agent_type != "ppo" and old_eps is not None:
        agent.eps = 0.0
    returns = []
    for _ in range(int(n_episodes)):
        obs = env.reset()
        while not env.done:
            if agent_type == "ppo":
                mask = env.action_mask()
                action, _, _ = agent.select_action(obs, mask=mask)
            else:
                action = agent.select_action(obs)
            obs, _, _, _ = env.step(action)
        returns.append(env.summary()["total_return_pct"])
    if greedy and old_eps is not None:
        agent.eps = old_eps
    return returns, env.summary()


def _estimate_off_policy_rewards(agent, obs_list, actions, rewards, episode: int) -> dict | None:
    """
    Diagnostic OPE (Improvement #5): re-estimate episode returns with IPS /
    doubly-robust estimators. Uses a **uniform** target policy — this does
    **not** change the agent's training objective; results are logged on
    ``agent.off_policy_estimates`` for post-hoc evaluation only.
    """
    try:
        import torch as _torch

        from labeling.off_policy_rewards import compute_off_policy_rewards

        if not hasattr(agent, "net") or not hasattr(agent.net, "actor"):
            return None
        obs_t = _torch.tensor(np.asarray(obs_list, dtype=float), dtype=_torch.float32,
                              device=getattr(agent, "device", "cpu"))
        with _torch.no_grad():
            h = agent.net.backbone(obs_t) if hasattr(agent.net, "backbone") else obs_t
            logits = agent.net.actor(h)
            if hasattr(logits, "cpu"):
                logits = logits.cpu().numpy()
        df = compute_off_policy_rewards(
            actions=np.asarray(actions, dtype=int),
            rewards=np.asarray(rewards, dtype=float),
            behavior_logits=logits,
            target_probs=np.full((len(actions), logits.shape[1]), 1.0 / logits.shape[1]),
            seed=int(episode),
        )
        last = df.tail(1).to_dicts()[0]
        return {
            "episode": int(episode),
            "ips_value": float(last["ipw_value"]),
            "dr_value": float(last["dr_value"]),
            "n_steps": len(actions),
            "diagnostic_only": True,
        }
    except Exception:
        return None


def train_agent(
    agent,
    env: ForexTradingEnv,
    n_episodes: int = 100,
    n_steps_ppo: int = 2048,
    agent_type: str = "ppo",
    curriculum = None,
    reward_normalizer: Any | None = None,
    reward_sharpe: Any | None = None,
    her_buffer: Any | None = None,
    off_policy_rewards: bool = False,
) -> list:
    """Generic training loop for PPO or DQN.

    ``off_policy_rewards`` (Improvement #5): when enabled, per-episode
    IPS / doubly-robust estimates are logged on ``agent.off_policy_estimates``.
    This is a **diagnostic OPE metric only** (uniform target policy) — it does
    not alter gradients or checkpoint selection.

    ``her_buffer``: optional ``HERBuffer`` for DQN — stores transitions with
    goal/achieved price and injects hindsight samples into the agent replay.
    """
    returns = []
    if off_policy_rewards:
        agent.off_policy_estimates = []
    avg_atr = float(np.mean(env.atr)) if len(env.atr) > 0 else 0.0005
    for ep in range(n_episodes):
        valid_starts = None
        if curriculum:
            curriculum.step()
            curriculum.log_phase(ep)
            valid_mask = curriculum.filter_bars(env.atr.reshape(-1, 1), atr_col_idx=0, avg_atr=avg_atr)
            valid_starts = np.where(valid_mask)[0]
            if len(valid_starts) == 0: valid_starts = None

        obs = env.reset(valid_starts=valid_starts)
        ep_reward = 0.0
        if reward_sharpe is not None:
            reward_sharpe.reset_episode()
        if her_buffer is not None:
            her_buffer._episode.clear()
        _op_obs, _op_act, _op_rew = [], [], []
        while not env.done:
            if agent_type == "ppo":
                mask = env.action_mask()
                action, log_prob, value = agent.select_action(obs, mask=mask)
                next_obs, reward, done, info = env.step(action)
                if reward_sharpe is not None:
                    reward = reward_sharpe(info.get("pnl", 0.0), tx_cost=0.0, equity=info.get("equity", 1.0))
                if reward_normalizer is not None:
                    reward = reward_normalizer(reward)
                if off_policy_rewards:
                    _op_obs.append(np.asarray(obs, dtype=float))
                    _op_act.append(int(action))
                    _op_rew.append(float(reward))
                agent.store(obs, action, reward, done, log_prob, value, mask=mask)
                ep_reward += reward; obs = next_obs
                if len(agent.buffer) >= n_steps_ppo:
                    if done:
                        last_value = 0.0
                    else:
                        next_mask = env.action_mask()
                        _, _, last_value = agent.select_action(next_obs, mask=next_mask)
                    agent.update(last_value=last_value)
            else:  # dqn
                dqn_mask = env.action_mask()
                action = agent.select_action(obs, mask=dqn_mask)
                next_obs, reward, done, info = env.step(action)
                if reward_sharpe is not None:
                    reward = reward_sharpe(info.get("pnl", 0.0), tx_cost=0.0, equity=info.get("equity", 1.0))
                if reward_normalizer is not None:
                    reward = reward_normalizer(reward)
                if her_buffer is not None:
                    t_idx = min(int(getattr(env, "idx", 0)), len(env.prices) - 1)
                    price = float(env.prices[t_idx])
                    goal_px = float(getattr(env, "entry_price", 0.0) or price)
                    her_buffer.store_transition(
                        obs=np.asarray(obs, dtype=np.float32),
                        action=int(action),
                        reward=float(reward),
                        next_obs=np.asarray(next_obs, dtype=np.float32),
                        done=bool(done),
                        goal=np.asarray([goal_px], dtype=np.float32),
                        achieved=np.asarray([price], dtype=np.float32),
                    )
                agent.store(obs, action, reward, next_obs, done, next_mask=env.action_mask())
                agent.update()
                ep_reward += reward; obs = next_obs

        if her_buffer is not None and agent_type != "ppo":
            her_buffer.end_episode()
            # Push hindsight samples into DQN replay (strip goal concat to keep obs dim)
            try:
                obs_dim = int(np.asarray(obs).shape[-1])
                batch = her_buffer.sample(min(32, len(her_buffer)))
                for tr in batch:
                    if not tr.get("info", {}).get("her"):
                        continue
                    # Keep full obs (including goal dimensions) — truncating to
                    # obs_dim strips the goal and injects conflicting Q-targets
                    # for identical states (HER requires a UVFA-style obs+goal input).
                    o  = np.asarray(tr["obs"],      dtype=np.float32).reshape(-1)
                    no = np.asarray(tr["next_obs"], dtype=np.float32).reshape(-1)
                    if o.shape[0] < obs_dim or no.shape[0] < obs_dim:
                        continue
                    agent.store(o, int(tr["action"]), float(tr["reward"]), no, bool(tr["done"]))
            except Exception:
                pass

        if agent_type == "ppo" and len(getattr(agent, "buffer", [])) > 0:
            agent.update(last_value=0.0)
        if agent_type != "ppo" and hasattr(agent, "decay_epsilon"):
            agent.decay_epsilon()
        if off_policy_rewards and _op_act:
            _estimate = _estimate_off_policy_rewards(agent, _op_obs, _op_act, _op_rew, ep)
            if _estimate is not None:
                agent.off_policy_estimates.append(_estimate)

        summary = env.summary()
        returns.append(summary["total_return_pct"])
        if (ep + 1) % 10 == 0:
            avg = np.mean(returns[-10:])
            print(f"  Ep {ep+1:3d}/{n_episodes} | "
                  f"Return: {summary['total_return_pct']:+.2f}% | "
                  f"Avg10: {avg:+.2f}% | Trades: {summary['n_trades']}")

    return returns


if __name__ == "__main__":
    print("RL Agents smoke test (10-action ScalingAction space)")
    import sys; sys.path.insert(0, "..")
    from data.data_ingestion import ForexDataPipeline, generate_synthetic_tick_data
    from features.feature_engineering import FeatureEngineer

    ticks = generate_synthetic_tick_data(n_rows=200_000)
    bars  = ForexDataPipeline(bar_freq="5min", session_filter=False, apply_frac_diff=False).run(ticks)
    fe    = FeatureEngineer()
    feats = fe.build(bars)
    bars_a = bars.reindex(feats.index).dropna()

    prices  = bars_a["close"].values.astype(np.float32)
    atr     = feats["atr_6"].values.astype(np.float32)
    spreads = np.full(len(prices), 0.00005, dtype=np.float32)
    feat_arr = feats.values.astype(np.float32)

    env = ForexTradingEnv(feat_arr, prices, atr, spreads)
    print(f"Env obs_size={env.obs_size}, n_actions={env.n_actions}")

    if TORCH:
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        dqn = DQNAgent(obs_size=env.obs_size, n_actions=env.n_actions, device=device)
        returns = train_agent(dqn, env, n_episodes=5, agent_type="dqn")
        print(f"\nDQN 5-episode returns: {[f'{r:+.2f}%' for r in returns]}")
    else:
        print("Install torch for full RL training.")
