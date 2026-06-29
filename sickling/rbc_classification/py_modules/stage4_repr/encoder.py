"""ImageEncoder ABC.

Every encoder in the bake-off (DINOv2 frozen, timm-supervised ViT-S, MAE
continuation) implements this interface. Adding a fourth backbone later
means writing one new file — the Lightning modules don't need to change.

Contract:
    forward(x: float[B, 3, H, W]) -> float[B, embed_dim]
        x is in ``[0, 1]`` (Stage 3 percentile-clipped) before standardization.
        Each encoder applies its own pretraining-statistics standardization
        inside ``standardize`` (called from ``forward`` or from the LM).

    embed_dim: int — dimensionality of the [CLS] embedding.

    freeze_backbone(): set requires_grad=False on encoder params (head
        excluded if a head exists).

    trainable_param_groups(base_lr, llrd): list of param-group dicts suitable
        for ``torch.optim.AdamW``, with optional layer-wise LR decay.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

# ImageNet means/stds — used by DINOv2, timm-supervised, and MAE alike.
# Even though our ch1/ch2 are binary masks we still standardize the whole
# (3, H, W) tensor with these — the mask channels become slightly off-zero
# but the pretrained patch_embed has seen broadly-distributed input.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_standardize(x: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet mean/std standardization to a (B, 3, H, W) tensor."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


class ImageEncoder(nn.Module, ABC):
    embed_dim: int

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, 3, H, W] -> [B, embed_dim]
        ...

    def standardize(self, x: torch.Tensor) -> torch.Tensor:
        return imagenet_standardize(x)

    def freeze_backbone(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)

    def trainable_param_groups(
        self, base_lr: float, llrd: float | None = None
    ) -> list[dict]:
        """Default: every trainable param at ``base_lr``. Override in subclasses
        that want per-block LR scaling."""
        params = [p for p in self.parameters() if p.requires_grad]
        return [{"params": params, "lr": base_lr}]
