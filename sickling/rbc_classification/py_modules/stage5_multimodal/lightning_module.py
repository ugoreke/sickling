"""Multimodal fine-tune LightningModule.

Wraps a ``MultimodalClassifier``. Optimizer uses LLRD on the image tower
(via ``ImageEncoder.trainable_param_groups``) and a flat LR on every other
tower + the fusion head.
"""
from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from sickling.rbc_classification.py_modules.stage4_repr._metrics import mcc, pr_auc
from sickling.rbc_classification.py_modules.stage4_repr.linear_probe import _cosine_with_warmup
from sickling.rbc_classification.py_modules.stage5_multimodal.classifier import MultimodalClassifier


class MultimodalFinetuneModule(pl.LightningModule):
    def __init__(
        self,
        classifier: MultimodalClassifier,
        backbone_lr: float = 1.0e-4,
        morphology_lr: float = 1.0e-3,
        head_lr: float = 1.0e-3,
        llrd: float = 0.65,
        weight_decay: float = 0.05,
        warmup_epochs: int = 5,
        max_epochs: int = 30,
        sickle_class_index: int = 1,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.classifier = classifier
        self.save_hyperparameters(ignore=["classifier"])
        self._val_logits: list[torch.Tensor] = []
        self._val_targets: list[torch.Tensor] = []

    def forward(self, inputs):
        return self.classifier(inputs)

    def training_step(self, batch, batch_idx):
        inputs, y = batch
        logits = self.forward(inputs)
        loss = F.cross_entropy(logits, y, label_smoothing=self.hparams.label_smoothing)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, y = batch
        logits = self.forward(inputs)
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
        # Per-tower LRs: image gets backbone_lr (with LLRD); morphology + any
        # future tower default to morphology_lr unless caller has registered
        # something else. Head gets head_lr.
        base_lrs: dict[str, float] = {}
        for name in self.classifier.modalities:
            base_lrs[name] = self.hparams.backbone_lr if name == "image" else self.hparams.morphology_lr

        groups = self.classifier.trainable_param_groups(
            base_lrs=base_lrs,
            llrd=self.hparams.llrd,
            head_lr=self.hparams.head_lr,
        )
        opt = torch.optim.AdamW(groups, weight_decay=self.hparams.weight_decay)
        sched = _cosine_with_warmup(opt, self.hparams.warmup_epochs, self.hparams.max_epochs)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
