"""Tests for ``sickling.engineering.duration.DurationCallback``."""
from __future__ import annotations

import json

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from sickling.rbc_classification.py_modules.engineering.duration import DurationCallback, _batch_size_of


class _TinyRegressor(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Linear(4, 1)

    def training_step(self, batch, _idx):
        x, y = batch
        return torch.nn.functional.mse_loss(self.net(x).squeeze(-1), y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.net.parameters(), lr=1e-2)


def test_duration_callback_writes_payload(tmp_path):
    ds = TensorDataset(torch.randn(32, 4), torch.randn(32))
    loader = DataLoader(ds, batch_size=8)
    out_dir = tmp_path / "ckpt_run"
    cb = DurationCallback(output_dir=out_dir, run_name="unit-test")
    trainer = pl.Trainer(
        max_epochs=2,
        callbacks=[cb],
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
        accelerator="cpu",
        devices=1,
    )
    trainer.fit(_TinyRegressor(), loader)

    path = out_dir / "duration.json"
    assert path.exists()
    data = json.loads(path.read_text())

    assert data["run_name"] == "unit-test"
    assert data["completed_epochs"] == 2
    assert len(data["epoch_seconds"]) == 2
    assert data["fit_seconds"] > 0.0
    assert data["samples_seen_total"] == 32 * 2  # 2 epochs over 32 samples
    assert data["imgs_per_second_mean"] > 0.0
    assert data["max_epochs"] == 2
    assert "precision" in data
    assert "device_name" in data
    assert "n_devices" in data
    assert "strategy" in data


def test_batch_size_of_handles_three_shapes():
    # bare tensor
    assert _batch_size_of(torch.zeros(7, 3, 4)) == 7
    # (x, y) tuple
    assert _batch_size_of((torch.zeros(11, 4), torch.zeros(11))) == 11
    # ({modality: tensor}, label) — multimodal pattern
    assert _batch_size_of(({"image": torch.zeros(5, 3, 96, 96), "morphology": torch.zeros(5, 30)}, torch.zeros(5))) == 5
    # weird stuff falls back to 0 without raising
    assert _batch_size_of(None) == 0
    assert _batch_size_of(42) == 0
