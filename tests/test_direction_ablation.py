import copy

import numpy as np
import pytest

from scripts.research_direction_ablation import direction_metrics, paired_configs


def test_good_direction_can_be_hidden_by_confidence_gate():
    result = direction_metrics([-3, 3, -3, 3], [-8]*4, [-1, 1, -1, 1])
    assert result['raw_direction_accuracy'] == 1
    assert result['raw_direction_balanced_accuracy'] == 1
    assert result['policy_pred_counts'] == [0, 4, 0]
    assert result['policy_nonhold_fraction'] == 0
    assert result['policy_accuracy'] == 0


def test_raw_direction_excludes_hold_and_handles_class_imbalance():
    result = direction_metrics([3, 3, 3, 3], [3]*4, [-1, -1, 1, 0])
    assert result['n_direction_rows'] == 3
    assert result['raw_direction_accuracy'] == pytest.approx(1/3)
    assert result['raw_direction_balanced_accuracy'] == .5
    assert sum(sum(row) for row in result['policy_confusion_true_rows_pred_columns']) == 4


@pytest.mark.parametrize('logits,conf,labels', [([], [], []), ([0], [0, 0], [1]),
                                                  ([np.nan], [0], [1]), ([0], [0], [2])])
def test_bad_diagnostics_fail_closed(logits, conf, labels):
    with pytest.raises(ValueError):
        direction_metrics(logits, conf, labels)


def test_paired_configs_change_only_confidence_loss_and_output_identity(tmp_path):
    base = {'paths': {'data_cache': 'corrected-cache'}, 'tracking': {},
            'training': {'seed': 1337, 'epochs': 12, 'label_smoothing': .1},
            'data': {'pairs': ['EURUSD']}}
    untouched = copy.deepcopy(base)
    arms = paired_configs(base, tmp_path)
    assert base == untouched
    a, b = copy.deepcopy(arms['direction_confidence']), copy.deepcopy(arms['direction_only'])
    assert a['multitask'].pop('w_conf') == .3
    assert b['multitask'].pop('w_conf') == 0
    for cfg in (a, b):
        cfg['paths'].pop('checkpoint_dir')
        cfg['tracking'].pop('run_name')
    assert a == b
    assert a['training']['early_stop_metric'] == 'val_loss'
    assert a['training']['label_smoothing'] == .1


def test_direction_only_objective_does_not_train_confidence_or_returns():
    import torch
    from models.architectures import MultiTaskLoss
    for weight in (0., .3):
        logits = torch.tensor([-1., 1.], requires_grad=True)
        returns = torch.zeros(2, requires_grad=True)
        confidence = torch.zeros(2, requires_grad=True)
        loss = MultiTaskLoss(w_ret=0., w_conf=weight, w_quantile=0., label_smoothing=.1)
        value = loss(logits, returns, confidence, torch.tensor([0, 2]), torch.tensor([-1., 1.]))
        value.backward()
        assert logits.grad.abs().sum() > 0
        assert returns.grad.abs().sum() == 0
        assert bool(confidence.grad.abs().sum() > 0) == bool(weight)
