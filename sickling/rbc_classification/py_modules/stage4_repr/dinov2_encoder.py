"""Model A — frozen DINOv2 ViT-S/14, [CLS] embedding, linear probe downstream.

Weights are loaded once via ``torch.hub`` and cached under ``~/.cache/torch/hub/``.
The encoder is set ``eval()`` and frozen at construction; do not unfreeze.
"""
from __future__ import annotations

import torch

from sickling.rbc_classification.py_modules.stage4_repr.encoder import ImageEncoder


class DinoV2Encoder(ImageEncoder):
    """Frozen DINOv2. Returns the [CLS] (=norm) embedding from the final layer."""

    embed_dim = 384

    def __init__(self, repo: str = "facebookresearch/dinov2", model: str = "dinov2_vits14") -> None:
        super().__init__()
        self.backbone = torch.hub.load(repo, model)
        self.backbone.eval()
        self.freeze_backbone()

    def train(self, mode: bool = True):  # noqa: D401 — keep frozen even in train()
        super().train(mode)
        # Keep BN/dropout in eval mode (DINOv2 has none, but be defensive).
        self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.standardize(x)
        with torch.no_grad():
            # DINOv2 hub model exposes ``forward()`` returning the CLS token directly.
            cls = self.backbone(x)
        return cls

    def trainable_param_groups(self, base_lr, llrd=None) -> list[dict]:
        # No trainable params — caller will optimize only the downstream head.
        return []
