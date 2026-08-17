import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from labeling.triple_barrier_labeling import compute_triple_barrier_labels
from models.architectures import MultiTaskHead, MultiTaskLoss, MultiTaskWrapper
from monitoring.discord_alerts import DiscordAlerter
from training.train_gpu import validate_epoch


def test_triple_barrier_accepts_trainer_execution_keywords():
    idx = pd.date_range("2025-01-01", periods=30, freq="min", tz="UTC")
    close = 1.1 + np.arange(30) * 0.00001
    bars = pd.DataFrame({"close": close, "bid_close": close - 0.00005, "ask_close": close + 0.00005}, index=idx)
    feats = pd.DataFrame({"atr_6": 0.0002, "spread_pips": 1.0}, index=idx)
    out = compute_triple_barrier_labels(
        bars,
        feats,
        vertical_bars=5,
        execution_delay_bars=2,
        pair="EURUSD",
        use_numba=False,
        parallel=False,
    )
    assert not out.empty
    assert len(out) == len(idx) - 7


def test_discord_methods_accept_active_trainer_payload(monkeypatch):
    alerter = DiscordAlerter(webhook_url="", verbose=False)
    sent = []
    monkeypatch.setattr(
        alerter,
        "send",
        lambda kind, fields=None, force=False, **kwargs: sent.append((kind, fields)) or True,
    )
    alerter.send_training_started(model="MAMBA", run_name="run-1", pairs=["EURUSD"], data_window="2020 to 2025")
    alerter.send_training_completed(model="mamba", fold=2, metric="sharpe", score=0.42)
    assert sent[0][1]["Model"] == "MAMBA"
    assert sent[1][1]["Score"] == "0.4200"


def test_multitask_head_preserves_three_output_default_and_opt_in_aux():
    h = torch.randn(3, 8)
    assert len(MultiTaskHead(8)(h)) == 3
    assert len(MultiTaskHead(8, return_aux=True)(h)) == 5


def test_multitask_loss_skips_incompatible_reconstruction_width():
    batch = 4
    logits = torch.randn(batch, 3)
    ret_hat = torch.randn(batch)
    conf = torch.randn(batch)
    y_cls = torch.randint(0, 3, (batch,))
    y_cont = torch.randn(batch)
    recon_hat = torch.randn(batch, 128)
    recon_tgt = torch.randn(batch, 966)

    loss = MultiTaskLoss()(
        logits,
        ret_hat,
        conf,
        y_cls,
        y_cont,
        recon_hat=recon_hat,
        recon_tgt=recon_tgt,
    )

    assert torch.isfinite(loss)


def test_multitask_aux_reconstruction_can_target_feature_width():
    class TinyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = torch.nn.Linear(5, 8)
            self.head = torch.nn.Linear(8, 1)

        def forward(self, x):
            return self.head(self.trunk(x[:, -1, :]))

    x = torch.randn(4, 3, 5)
    wrapped = MultiTaskWrapper(
        TinyBackbone(),
        head_in=8,
        hidden=16,
        return_aux=True,
        recon_out_features=5,
    )
    out = wrapped(x)

    assert len(out) == 5
    assert out[3].shape == (4, 5)


def test_validate_epoch_rl_reward_uses_direction_sidecar_for_sharpe():
    class ShortModel(torch.nn.Module):
        def forward(self, x):
            batch = x.shape[0]
            logits = torch.tensor([[5.0, 0.0, -5.0]], dtype=x.dtype, device=x.device).repeat(batch, 1)
            ret_hat = torch.ones(batch, dtype=x.dtype, device=x.device)
            conf_logit = torch.zeros(batch, dtype=x.dtype, device=x.device)
            return logits, ret_hat, conf_logit

    x = torch.zeros(2, 4, 3)
    y_reward = torch.tensor([2.0, 4.0])
    y_cls = torch.tensor([-1.0, -1.0])
    path_quality = torch.ones(2)
    loader = DataLoader(TensorDataset(x, y_reward, y_cls, path_quality), batch_size=2)

    _, _, sharpe = validate_epoch(
        ShortModel(),
        loader,
        MultiTaskLoss(),
        torch.device("cpu"),
        classification=True,
        multitask=True,
        rl_mode=True,
        sharpe_ann_factor=1.0,
    )

    assert sharpe > 0


def test_strict_load_report_returns_load_summary_dict():
    from training.model_factory import _strict_load_report

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(2, 2)

    model = TinyModel()
    state = {
        "layer.weight": model.layer.weight.detach().clone(),
        "layer.bias": model.layer.bias.detach().clone(),
    }

    report = _strict_load_report(model, state, context_name="TestLoad", min_frac_loaded=0.6)

    assert isinstance(report, dict)
    assert report["missing"] == []
    assert report["unexpected"] == []
    assert report["frac_loaded"] >= 0.6
    assert report["n_target"] > 0
