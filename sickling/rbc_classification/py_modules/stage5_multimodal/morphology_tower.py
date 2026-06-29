"""Small MLP over the hand-crafted shape descriptors.

Architecture: input → BN → Linear → GELU → Dropout → Linear → GELU → Linear (D)
The first BatchNorm is *not* a substitute for proper standardization — call
``set_feature_stats(mean, std)`` once with train-set statistics so the tower
operates on standardized inputs at inference time too. The mean/std are
registered as buffers and travel with the checkpoint.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from sickling.rbc_classification.py_modules.stage5_multimodal.tower import Tower


class MorphologyTower(Tower):
    def __init__(
        self,
        in_features: int,
        hidden: int = 64,
        out_features: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.D = out_features
        self.in_features = in_features
        self.register_buffer("feature_mean", torch.zeros(in_features))
        self.register_buffer("feature_std", torch.ones(in_features))

        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_features),
        )

    def set_feature_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.numel() != self.in_features or std.numel() != self.in_features:
            raise ValueError(
                f"Stats must be length {self.in_features}; got mean={mean.numel()}, "
                f"std={std.numel()}."
            )
        eps = 1e-6
        self.feature_mean.copy_(mean.detach())
        self.feature_std.copy_(std.detach().clamp(min=eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.feature_mean) / self.feature_std
        return self.mlp(x)
