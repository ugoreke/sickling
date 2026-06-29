"""Full fine-tune LightningModule used for Models B and C.

Encoder unfrozen, layer-wise LR decay (per ``ImageEncoder.trainable_param_groups``),
linear warmup + cosine schedule. Headline metric ``val_pr_auc`` for checkpointing.
"""
from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from sickling.rbc_classification.py_modules.stage4_repr._metrics import mcc, pr_auc
from sickling.rbc_classification.py_modules.stage4_repr.encoder import ImageEncoder
from sickling.rbc_classification.py_modules.stage4_repr.linear_probe import _cosine_with_warmup


class FinetuneModule(pl.LightningModule):
    def __init__(
        self,
        encoder: ImageEncoder,
        num_classes: int = 2,
        head_lr: float = 1.0e-3,
        backbone_lr: float = 1.0e-4,
        llrd: float = 0.65,
        weight_decay: float = 0.05,
        warmup_epochs: int = 5,
        max_epochs: int = 30,
        sickle_class_index: int = 1,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.embed_dim, num_classes)
        self.save_hyperparameters(ignore=["encoder"])

        self._val_logits: list[torch.Tensor] = []
        self._val_targets: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.head(z)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(
            logits, y, label_smoothing=self.hparams.label_smoothing
        )
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        self._val_logits.append(logits.detach())
        self._val_targets.append(y.detach())
        loss = F.cross_entropy(logits, y)
        self.log("val_loss", loss, prog_bar=True)

    def on_validation_epoch_end(self):
        if not self._val_logits:
            return
        logits = torch.cat(self._val_logits, dim=0)
        targets = torch.cat(self._val_targets, dim=0)
        scores = F.softmax(logits, dim=-1)[:, self.hparams.sickle_class_index]
        preds = logits.argmax(dim=-1)
        is_sickle = (targets == self.hparams.sickle_class_index).long()
        self.log("val_pr_auc", pr_auc(is_sickle, scores), prog_bar=True)
        self.log("val_mcc", mcc(is_sickle, (preds == self.hparams.sickle_class_index).long()))
        self._val_logits.clear()
        self._val_targets.clear()

    def configure_optimizers(self):
        backbone_groups = self.encoder.trainable_param_groups(
            base_lr=self.hparams.backbone_lr, llrd=self.hparams.llrd
        )
        head_group = {
            "params": self.head.parameters(),
            "lr": self.hparams.head_lr,
        }
        opt = torch.optim.AdamW(
            backbone_groups + [head_group],
            weight_decay=self.hparams.weight_decay,
        )
        sched = _cosine_with_warmup(opt, self.hparams.warmup_epochs, self.hparams.max_epochs)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
