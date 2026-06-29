"""``MultimodalClassifier`` — late fusion over an arbitrary dict of towers.

Implementation contract (PIPELINE_PLAN §2 Stage 5): adding a modality is
literally registering a new tower in this dict. The constructor never
hard-codes any modality name.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from sickling.rbc_classification.py_modules.stage5_multimodal.tower import Tower


class MultimodalClassifier(nn.Module):
    """Concat tower outputs, run through a small fusion MLP, predict logits.

    Args:
        towers: ``{modality_name: Tower}``. The Dataset must yield a dict with
            exactly these keys. Iteration order is the dict insertion order
            (Python 3.7+) — that determines the concat order, which matters
            only if you care about debug-print stability.
        num_classes: 2 for sickle / non_sickle.
        hidden: width of the fusion MLP hidden layer.
        dropout: pre-head dropout.
    """

    def __init__(
        self,
        towers: Mapping[str, Tower],
        num_classes: int = 2,
        hidden: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if not towers:
            raise ValueError("MultimodalClassifier requires at least one tower.")
        self.towers = nn.ModuleDict(dict(towers))
        self.modalities = list(towers.keys())
        total_dim = sum(int(t.D) for t in towers.values())

        self.fusion = nn.Sequential(
            nn.Linear(total_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    @property
    def total_embed_dim(self) -> int:
        return sum(int(t.D) for t in self.towers.values())

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        missing = [m for m in self.modalities if m not in inputs]
        if missing:
            raise KeyError(f"MultimodalClassifier: missing modality input(s) {missing}.")
        embeddings = [self.towers[m](inputs[m]) for m in self.modalities]
        z = torch.cat(embeddings, dim=-1)
        return self.fusion(z)

    def trainable_param_groups(
        self,
        base_lrs: Mapping[str, float],
        llrd: float | None = None,
        head_lr: float | None = None,
    ) -> list[dict]:
        """Return per-tower param groups + a head group.

        Args:
            base_lrs: ``{modality_name: lr}``. Each tower receives its own LR;
                LLRD only applies to towers that override ``trainable_param_groups``
                (today, only ``ImageTower``).
            llrd: forwarded to every tower's ``trainable_param_groups``.
            head_lr: LR for the fusion MLP. Defaults to ``max(base_lrs.values())``.
        """
        groups: list[dict] = []
        for name, tower in self.towers.items():
            lr = base_lrs.get(name)
            if lr is None:
                continue  # tower frozen at the caller level
            groups.extend(tower.trainable_param_groups(base_lr=lr, llrd=llrd))
        head_lr = head_lr if head_lr is not None else max(base_lrs.values())
        groups.append({"params": list(self.fusion.parameters()), "lr": head_lr})
        return groups
