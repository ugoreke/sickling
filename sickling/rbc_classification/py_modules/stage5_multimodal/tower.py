"""Tower ABC.

Per PIPELINE_PLAN §2 Stage 5, every tower exposes:

    forward(input) -> Tensor[B, D]
    D : int  (or self.D)

Adding a new modality means writing one ``Tower`` subclass — the
``MultimodalClassifier`` accepts ``{name: Tower}`` and concatenates outputs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Tower(nn.Module, ABC):
    D: int

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``Tensor[B, D]`` for any modality-specific input."""
        ...

    def trainable_param_groups(self, base_lr: float, llrd: float | None = None) -> list[dict]:
        """Default: single param group at ``base_lr`` over all trainable params."""
        params = [p for p in self.parameters() if p.requires_grad]
        return [{"params": params, "lr": base_lr}]
