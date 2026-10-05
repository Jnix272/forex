import math
import torch
import pytest
from unittest.mock import MagicMock
from training.training_controller import TrainingController
from training.loop_epochs import validate_epoch


def test_training_controller_3_class_and_deadband_guard():
    # 1. Verify default threshold is 0.33 for 3 classes
    ctrl = TrainingController()
    assert ctrl.dir_acc_random_threshold == 0.33

    # 2. When val_sharpe is 0.0 (deadband hold with 0 trades),
    # even if dir_acc is low (0.22, the true hold prior),
    # it must NOT trigger lower_lr or increase_dropout.
    ctrl.evaluate_epoch(1, train_loss=1.5, val_loss=1.5, val_sharpe=0.0, dir_acc=0.2206)
    resp = ctrl.evaluate_epoch(2, train_loss=1.4, val_loss=1.4, val_sharpe=0.0, dir_acc=0.2206)
    assert not resp.get("lower_lr", False)
    assert not resp.get("increase_dropout", False)
    assert not resp.get("hold_curriculum", False)

    # 3. When trading IS active (val_sharpe != 0.0), if dir_acc drops below 0.33
    # for 2 consecutive epochs, it should trigger
    ctrl.evaluate_epoch(3, train_loss=1.3, val_loss=1.3, val_sharpe=0.5, dir_acc=0.25)
    resp = ctrl.evaluate_epoch(4, train_loss=1.2, val_loss=1.2, val_sharpe=0.4, dir_acc=0.25)
    assert resp.get("lower_lr", False)
    assert resp.get("increase_dropout", False)


def test_validate_epoch_exposes_last_n_trades():
    assert hasattr(validate_epoch, "last_n_trades")
