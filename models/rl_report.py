"""
models/rl_report.py
===================
Generates a comprehensive rl_report.json after every RL run (PPO/DQN) to
mathematically prove whether the RL policy outperforms the supervised baseline.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path


class RLReporter:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.logger = logging.getLogger(__name__)

    def generate_report(
        self,
        algorithm: str,
        observation_mode: str,
        train_return: float,
        validation_return: float,
        validation_sharpe: float,
        max_drawdown: float,
        profit_factor: float,
        trade_count: int,
        turnover: float,
        average_hold_time: float,
        action_distribution: dict,
        reward_component_breakdown: dict,
        rl_best_updated: bool,
        comparison_vs_supervised: dict,
    ) -> None:
        """
        Compiles all critical reinforcement learning metrics and writes them to rl_report.json.
        """

        report_data = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "configuration": {"algorithm": algorithm.upper(), "observation_mode": observation_mode},
            "performance_metrics": {
                "train_return": train_return,
                "validation_return": validation_return,
                "validation_sharpe": validation_sharpe,
                "max_drawdown": max_drawdown,
                "profit_factor": profit_factor,
            },
            "trading_behavior": {
                "trade_count": trade_count,
                "turnover": turnover,
                "average_hold_time_steps": average_hold_time,
                "action_distribution": action_distribution,
            },
            "reward_mechanics": {"component_breakdown": reward_component_breakdown},
            "promotion_status": {
                "rl_best_updated": rl_best_updated,
                "comparison_vs_supervised_baseline": comparison_vs_supervised,
            },
        }

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.checkpoint_dir / "rl_report.json"

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        self.logger.info(f"RL Report generated successfully at {report_path}")

        if rl_best_updated:
            self.logger.info(f"SUCCESS: The RL Policy ({algorithm}) crushed the supervised baseline and was promoted!")
        else:
            self.logger.warning("REJECTED: The RL Policy failed to beat the supervised baseline. Discarding weights.")
