"""
models/rl_agents.py
====================
PPO and Deep Q-Learning agents with:
  - 10-action ScalingAction space for hold/open/scale/reduce/close execution
  - Decomposable reward (P&L - drawdown - costs - overtrading)
  - Combined pyramiding + martingale scaling strategy
  - Dynamic stop-loss integration
"""

import numpy as np
import collections
import random
from typing import Any, Optional, Tuple, Dict

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
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
        reward_weights: Optional[Dict] = None,
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
        episode_len:    Optional[int] = None,
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
        self.reset()

    def reset(self, valid_starts: Optional[np.ndarray] = None) -> np.ndarray:
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
            max_start = n - self.episode_len - 1
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

    def _check_sl_tp(self, p: float, direction: int) -> Tuple[bool, float]:
        """Returns (hit, pnl_pips). P&L calculated from _initial_entry_price."""
        init_entry = self._initial_entry_price
        if direction > 0:
            if p <= self.stop_loss:   return True, (self.stop_loss - init_entry) / self.pip_size
            if p >= self.take_profit: return True, (self.take_profit - init_entry) / self.pip_size
        else:
            if p >= self.stop_loss:   return True, (init_entry - self.stop_loss) / self.pip_size
            if p <= self.take_profit: return True, (init_entry - self.take_profit) / self.pip_size
        return False, 0.0

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
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
        self.peak = max(self.peak, self.equity)
        self.episode_pnl.append(realised_pnl)

        # A-M2: trade-frequency cost. A fresh entry, scale-in, or flip (position
        # magnitude grew or sign changed) incurs a FIXED penalty per event so the
        # agent is discouraged from overtrading. The old `1/n_trades` term was
        # inverted — it shrank as trades grew, effectively *rewarding* churn.
        opened_or_flipped = (
            abs(self.position) > abs(pos_before) + 1e-9
            or (self.position != 0 and np.sign(self.position) != np.sign(pos_before))
        )

        # ── Decomposable reward ──────────────────────────────────────────────
        w = self.rw
        dd = max(0, (self.peak - self.equity) / self.peak)
        holding_penalty = w["holding"] * abs(self.position) * min(self.holding / 100, 1.0)
        reward = (
            w["pnl"]       * realised_pnl / self.initial_equity
          - w["drawdown"]  * dd
          - w["tx_cost"]   * cost / self.initial_equity
          - w["overtrade"] * (1.0 if opened_or_flipped else 0.0)
          - holding_penalty
        )

        self.idx += 1
        self.done = self.idx >= self.end_idx

        # Force-close at episode end — P&L from _initial_entry_price, not avg entry
        if self.done and self.position != 0:
            last_p = self.prices[min(self.idx, len(self.prices) - 1)]
            d = int(np.sign(self.position))
            final_pnl = d * (last_p - self._initial_entry_price) * abs(self.position) * self.lot_size
            self.equity += final_pnl

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
        bars_per_year = 252 * 24 * 60  # 1-min bars default
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
        """Shared backbone -> actor (policy) + critic (value) heads."""
        def __init__(self, obs_size, n_actions=10, hidden=256):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(obs_size, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden),   nn.Tanh(),
            )
            self.actor  = nn.Linear(hidden, n_actions)
            self.critic = nn.Linear(hidden, 1)

        def forward(self, x):
            h = self.backbone(x)
            return self.actor(h), self.critic(h)

        def act(self, obs, mask=None):
            logits, value = self(obs)
            if mask is not None:
                logits = logits.masked_fill(~mask, -1e9)
            dist   = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            return action, dist.log_prob(action), value.squeeze(-1)

        def evaluate(self, obs, actions, mask=None):
            logits, value = self(obs)
            if mask is not None:
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
                     entropy_coef=0.01, value_coef=0.5, n_epochs=10, device="cpu"):
            self.gamma = gamma; self.lam = lam; self.clip = clip
            self.ent_c = entropy_coef; self.val_c = value_coef
            self.n_epochs = n_epochs
            self.device = torch.device(device)
            self.net = ActorCritic(obs_size, n_actions, hidden).to(self.device)
            self.opt = optim.Adam(self.net.parameters(), lr=lr)
            self.buffer = []

        def select_action(self, obs: np.ndarray, mask: Optional[np.ndarray] = None):
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            m = torch.tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0) if mask is not None else None
            with torch.no_grad():
                action, log_prob, value = self.net.act(x, mask=m)
            return action.item(), log_prob.item(), value.item()

        def store(self, obs, action, reward, done, log_prob, value, mask=None):
            if mask is None: mask = np.ones(self.net.actor.out_features, dtype=bool)
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

            obs_t  = torch.tensor(np.array(obs_b),  dtype=torch.float32, device=self.device)
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

        def push(self, *args):
            # args = (obs, action, reward, next_obs, done)
            if len(self.buf) == self.capacity:
                old_action = self.buf[0][1]
                self.class_counts[old_action] -= 1
            self.buf.append(args)
            self.class_counts[args[1]] += 1

        def sample(self, n):
            # BUG-012: Maintain cached weight array instead of O(N) rebuild every call.
            # Weights are invalidated only when buffer composition changes significantly.
            if len(self.buf) == 0:
                return []
            buf_len = len(self.buf)
            # Rebuild weights only if buffer size changed or every 100 calls
            if (not hasattr(self, '_cached_weights') or
                    self._cached_len != buf_len or
                    getattr(self, '_sample_calls', 0) % 100 == 0):
                # Precompute inverse-frequency weight per action class
                inv_freq = {a: 1.0 / (c + 1e-3) for a, c in self.class_counts.items()}
                weights = np.array([inv_freq.get(t[1], 1.0) for t in self.buf])
                weights /= weights.sum()
                self._cached_weights = weights
                self._cached_len = buf_len
            self._sample_calls = getattr(self, '_sample_calls', 0) + 1
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

        def select_action(self, obs: np.ndarray, mask: Optional[np.ndarray] = None) -> int:
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

        def store(self, obs, action, reward, next_obs, done):
            self.buf.push(obs, action, reward, next_obs, done)

        def decay_epsilon(self):
            """A-M1: decay exploration ONCE per episode (call from the training
            loop). Decaying per environment step collapsed epsilon to eps_end in
            ~900 steps (decay**900 ≈ eps_end), killing exploration almost
            immediately on multi-thousand-step trajectories."""
            self.eps = max(self.eps_end, self.eps * self.eps_decay)

        def update(self) -> dict:
            if len(self.buf) < self.batch: return {}
            batch = self.buf.sample(self.batch)
            obs_b, act_b, rew_b, nobs_b, done_b = zip(*batch)

            obs  = torch.tensor(np.array(obs_b),  dtype=torch.float32, device=self.device)
            acts = torch.tensor(act_b,             dtype=torch.long,    device=self.device)
            rews = torch.tensor(rew_b,             dtype=torch.float32, device=self.device)
            nobs = torch.tensor(np.array(nobs_b),  dtype=torch.float32, device=self.device)
            done = torch.tensor(done_b,             dtype=torch.float32, device=self.device)

            q_vals = self.policy_net(obs).gather(1, acts.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                if self.double:
                    best_acts = self.policy_net(nobs).argmax(1)
                    next_q = self.target_net(nobs).gather(1, best_acts.unsqueeze(1)).squeeze(1)
                else:
                    next_q = self.target_net(nobs).max(1)[0]
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
        self.mean += self._alpha * delta / self.count
        delta2     = reward - self.mean
        self.m2   += self._alpha * delta * delta2
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


def _estimate_off_policy_rewards(agent, obs_list, actions, rewards, episode: int) -> Optional[dict]:
    """
    Re-estimate an episode's rewards with IPS / doubly-robust estimators
    (Improvement #5) using the agent's current policy as the behavior policy.
    Returns a dict (or None when the agent exposes no logits, e.g. DQN).
    """
    try:
        from labeling.off_policy_rewards import compute_off_policy_rewards
        import torch as _torch

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
            "n_steps": int(len(actions)),
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
    reward_normalizer: Optional[Any] = None,
    reward_sharpe: Optional[Any] = None,
    off_policy_rewards: bool = False,
) -> list:
    """Generic training loop for PPO or DQN.

    ``off_policy_rewards`` (Improvement #5): when enabled, per-episode
    (actions, rewards, behavior_logits) are logged for PPO agents and
    re-estimated with IPS / doubly-robust estimators. Estimates are stored on
    ``agent.off_policy_estimates`` (list of dicts) so the return signature is
    unchanged.
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
                agent.store(obs, action, reward, next_obs, done)
                agent.update()
                ep_reward += reward; obs = next_obs

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
    from data.data_ingestion import generate_synthetic_tick_data, ForexDataPipeline
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
