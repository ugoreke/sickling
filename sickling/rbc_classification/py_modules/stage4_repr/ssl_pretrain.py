"""MAE continuation pretraining (Model C, SSL phase).

Trains an ``MAEReconstructor`` on the unlabeled cell-crop corpus. After the
run, the encoder weights are loaded into ``MAEViTEncoder`` for fine-tuning
via ``MAEViTEncoder.load_mae_checkpoint``.
"""
from __future__ import annotations

import pytorch_lightning as pl
import torch

from sickling.rbc_classification.py_modules.stage4_repr.linear_probe import _cosine_with_warmup
from sickling.rbc_classification.py_modules.stage4_repr.mae_encoder import MAEReconstructor


class MAEPretrainModule(pl.LightningModule):
    def __init__(
        self,
        reconstructor: MAEReconstructor,
        mask_ratio: float = 0.75,
        lr: float = 1.5e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 10,
        max_epochs: int = 300,
    ) -> None:
        super().__init__()
        self.reconstructor = reconstructor
        # Lightning hooks `encoder.X` keys when saving — keep a direct ref so
        # `MAEViTEncoder.load_mae_checkpoint` can find them.
        self.encoder = reconstructor.encoder
        self.mask_ratio = mask_ratio
        self.save_hyperparameters(ignore=["reconstructor"])

    def training_step(self, batch, batch_idx):
        x = batch if torch.is_tensor(batch) else batch[0]
        loss, _, _, _ = self.reconstructor(x, self.mask_ratio)
        self.log("train_recon_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch if torch.is_tensor(batch) else batch[0]
        loss, _, _, _ = self.reconstructor(x, self.mask_ratio)
        self.log("val_recon_loss", loss, prog_bar=True)
        # Log negative loss as `val_pr_auc` so existing ModelCheckpoint with
        # monitor=val_pr_auc / mode=max keeps the best (lowest-loss) epoch.
        self.log("val_pr_auc", -loss.detach())

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.95),  # MAE-paper recipe
        )
        sched = _cosine_with_warmup(opt, self.hparams.warmup_epochs, self.hparams.max_epochs)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
