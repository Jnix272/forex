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
