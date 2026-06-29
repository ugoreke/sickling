"""Tower wrapper around any ``ImageEncoder`` from Stage 4.

The encoder retains its layer-wise LR decay machinery; the tower simply
proxies ``trainable_param_groups`` so the multimodal Lightning module can
optimize the image branch with LLRD while everything else (morphology MLP,
fusion head) gets a flat LR.
"""
from __future__ import annotations

import torch

from sickling.rbc_classification.py_modules.stage4_repr.encoder import ImageEncoder
from sickling.rbc_classification.py_modules.stage5_multimodal.tower import Tower


class ImageTower(Tower):
    def __init__(self, encoder: ImageEncoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.D = int(encoder.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def trainable_param_groups(
        self, base_lr: float, llrd: float | None = None
    ) -> list[dict]:
        return self.encoder.trainable_param_groups(base_lr=base_lr, llrd=llrd)
