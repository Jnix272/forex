import argparse
import json

from training.training_memory import TrainingMemory


def test_apply_to_model_args_accepts_train_gpu_call_signature(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text(
        json.dumps(
            {
                "total_runs": 2,
                "recommended_lr": 0.00005,
                "recommended_dropout": 0.30,
                "recommended_patience": 6,
                "recommended_max_epochs": 24,
                "best_epoch_pattern": "plateau",
            }
        ),
        encoding="utf-8",
    )
    memory = TrainingMemory(path)
    model_args = argparse.Namespace(lr=0.001, dropout=0.1, patience=10, epochs=50)
    base_args = argparse.Namespace(lr=0.001)

    memory.apply_to_model_args(model_args, "mamba", base_args=base_args)

    assert model_args.lr < 0.001
    assert model_args.dropout > 0.1
    assert model_args.patience == 6
    assert base_args.lr == 0.001


def test_stale_one_epoch_recommendation_is_floored(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text(
        json.dumps({"total_runs": 12, "recommended_max_epochs": 1, "best_epoch_pattern": "early_peak"}),
        encoding="utf-8",
    )
    args = argparse.Namespace(epochs=40)

    TrainingMemory(path).apply_to_args(args)

    assert args.epochs == 12


def test_one_epoch_run_does_not_lower_future_epoch_cap(tmp_path):
    memory = TrainingMemory(tmp_path / "memory.json")
    memory.update(
        {
            "model_name": "haelt",
            "run_name": "smoke",
            "best_sharpe": 0.1,
            "best_epoch": 0,
            "total_epochs": 1,
            "history": {"val_sharpe": [0.1], "train_loss": [1.0], "val_loss": [1.0]},
            "args_snapshot": {"epochs": 1, "lr": 3e-5, "dropout": 0.3, "patience": 10},
        }
    )

    assert memory.get("recommended_max_epochs") >= 12
    assert memory.get("best_epoch_pattern") != "early_peak"
    args = argparse.Namespace(epochs=40)
    memory.apply_to_args(args)
    assert args.epochs == 40
