import numpy as np

from models.rl_agents import ForexTradingEnv, ScalingAction


def test_churning_on_flat_prices_has_negative_sharpe():
    # Flat market: the only P&L is execution cost, so a churning agent must
    # show a negative Sharpe (it used to report the gross, cost-free Sharpe).
    n = 400
    env = ForexTradingEnv(
        features=np.zeros((n, 4), dtype=np.float32),
        prices=np.full(n, 1.1000),
        atr=np.full(n, 0.0010),
        spreads=np.full(n, 0.0001),
        random_reset=False,
    )
    env.reset()
    step = 0
    while not env.done:
        # open / hold / close rhythm so per-bar P&L is not constant (std > 0)
        phase = step % 3
        action = (
            ScalingAction.OPEN_LONG.value if phase == 0
            else ScalingAction.HOLD.value if phase == 1
            else ScalingAction.CLOSE_ALL.value
        )
        env.step(action)
        step += 1
    s = env.summary()
    assert s["total_costs"] > 0
    assert s["total_return_pct"] < 0
    assert s["sharpe"] < 0
    # Per-bar P&L now sums to the equity change (costs included).
    assert np.isclose(sum(env.episode_pnl), env.equity - env.initial_equity)
