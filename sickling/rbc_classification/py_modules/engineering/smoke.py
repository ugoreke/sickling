"""End-to-end scaffolding sanity check.

A trivial ``LightningModule`` (one linear layer, MSE on random tensors) is
trained for one epoch on CPU using the project's ``WandbLogger`` +
``ModelCheckpoint`` factories. Exits 0 if the entire stack works.

Run via ``sickling smoke`` or ``make smoke``.
"""
from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sickling.rbc_classification.py_modules.config import Config
from sickling.rbc_classification.py_modules.engineering.lightning_utils import build_trainer


class _DummyModule(pl.LightningModule):
    def __init__(self, input_dim: int, lr: float):
        super().__init__()
        self.save_hyperparameters()
        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.mse_loss(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = F.mse_loss(self(x), y)
        self.log("val_pr_auc", -loss.detach())  # bigger == better, hooks ModelCheckpoint
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


def _build_smoke_loaders(cfg: Config) -> tuple[DataLoader, DataLoader]:
    n, d = cfg.smoke.num_samples, cfg.smoke.input_dim
    g = torch.Generator().manual_seed(cfg.project.seed)
    x = torch.randn(n, d, generator=g)
    true_w = torch.randn(d, generator=g)
    y = x @ true_w + 0.1 * torch.randn(n, generator=g)

    n_train = int(0.8 * n)
    train = TensorDataset(x[:n_train], y[:n_train])
    val = TensorDataset(x[n_train:], y[n_train:])
    return (
        DataLoader(train, batch_size=cfg.training.batch_size, shuffle=True),
        DataLoader(val, batch_size=cfg.training.batch_size),
    )


def run_smoke(cfg: Config) -> None:
    """Train the dummy module for one epoch on CPU. Raises if anything is broken."""
    train_loader, val_loader = _build_smoke_loaders(cfg)
    module = _DummyModule(input_dim=cfg.smoke.input_dim, lr=cfg.training.lr)

    trainer = build_trainer(
        cfg,
        run_name="smoke",
        max_epochs=cfg.training.max_epochs,
        devices=1,
        precision=cfg.training.precision,
        offline_wandb=True,
        tags=["smoke", "scaffolding"],
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"Smoke test OK. {trainer.current_epoch} epoch(s) completed.")
