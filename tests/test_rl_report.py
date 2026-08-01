import unittest
import tempfile
import json
from pathlib import Path
import shutil

# Importing the module we just created
from models.rl_report import RLReporter

class TestRLReport(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.reporter = RLReporter(checkpoint_dir=self.temp_dir)
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_rl_report_generation(self):
        """Validates that rl_report.json is correctly generated with all required metrics."""
        
        # Mock data representing a successful PPO run
        self.reporter.generate_report(
            algorithm="PPO",
            observation_mode="frozen_encoder",
            train_return=0.15,
            validation_return=0.12,
            validation_sharpe=2.1,
            max_drawdown=-0.04,
            profit_factor=1.8,
            trade_count=150,
            turnover=0.45,
            average_hold_time=12.5,
            action_distribution={"Buy": 0.3, "Sell": 0.3, "Hold": 0.4},
            reward_component_breakdown={"pnl": 0.9, "spread_penalty": -0.05, "hold_penalty": -0.05},
            rl_best_updated=True,
            comparison_vs_supervised={"delta_sharpe": 0.3, "delta_return": 0.05}
        )
        
        report_path = Path(self.temp_dir) / "rl_report.json"
        self.assertTrue(report_path.exists(), "rl_report.json was not created.")
        
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Verify schema structure
        self.assertEqual(data["configuration"]["algorithm"], "PPO")
        self.assertEqual(data["configuration"]["observation_mode"], "frozen_encoder")
        
        self.assertEqual(data["performance_metrics"]["validation_sharpe"], 2.1)
        self.assertEqual(data["performance_metrics"]["max_drawdown"], -0.04)
        
        self.assertEqual(data["trading_behavior"]["trade_count"], 150)
        self.assertEqual(data["trading_behavior"]["action_distribution"]["Hold"], 0.4)
        
        self.assertTrue(data["promotion_status"]["rl_best_updated"])
        self.assertEqual(data["promotion_status"]["comparison_vs_supervised_baseline"]["delta_sharpe"], 0.3)
        
    def test_rl_report_rejection(self):
        """Validates that the report handles rejection gracefully."""
        self.reporter.generate_report(
            algorithm="DQN",
            observation_mode="raw_features",
            train_return=-0.05,
            validation_return=-0.10,
            validation_sharpe=-1.0,
            max_drawdown=-0.25,
            profit_factor=0.6,
            trade_count=10,
            turnover=0.01,
            average_hold_time=2.0,
            action_distribution={"Buy": 0.1, "Sell": 0.1, "Hold": 0.8},
            reward_component_breakdown={"pnl": -0.5, "spread_penalty": -0.2, "hold_penalty": -0.1},
            rl_best_updated=False,
            comparison_vs_supervised={"delta_sharpe": -2.0, "delta_return": -0.15}
        )
        
        report_path = Path(self.temp_dir) / "rl_report.json"
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        self.assertFalse(data["promotion_status"]["rl_best_updated"])
        self.assertEqual(data["configuration"]["algorithm"], "DQN")

if __name__ == '__main__':
    unittest.main()
