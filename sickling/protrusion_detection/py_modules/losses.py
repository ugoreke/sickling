"""Segmentation losses preserved from ``training_2.ipynb``.

Composite training loss (see ARCHITECTURE.md §10):

    L = WeightedDice + CE + cfg.CONFUSION_WEIGHT * directed_confusion
        + cfg.FN_WEIGHT * directed_fn
        + cfg.TVERSKY_WEIGHT * Tversky(over TVERSKY_CLASSES)

All terms honour ``cfg.IGNORE_INDEX`` so partial tile labels train safely.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import cfg


def _one_hot_with_ignore(targets: torch.Tensor, n_classes: int, ignore_index: int):
    """Return (one_hot[N,C,H,W], valid_mask[N,1,H,W])."""
    valid = (targets != ignore_index)
    safe = torch.where(valid, targets, torch.zeros_like(targets))
    oh = F.one_hot(safe, num_classes=n_classes).permute(0, 3, 1, 2).float()
    valid_f = valid.unsqueeze(1).float()
    return oh * valid_f, valid_f


class WeightedDiceLoss(nn.Module):
    """Class-weighted soft Dice loss with ignore-index support."""

    def __init__(self, weights: torch.Tensor | None = None, smooth: float = 1.0,
                 ignore_index: int = cfg.IGNORE_INDEX) -> None:
        super().__init__()
        self.weights = weights
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        n_classes = probs.shape[1]
        targets_oh, valid_f = _one_hot_with_ignore(targets, n_classes, self.ignore_index)
        probs = probs * valid_f

        total = probs.new_tensor(0.0)
        for c in range(n_classes):
            p = probs[:, c].reshape(-1)
            t = targets_oh[:, c].reshape(-1)
            inter = (p * t).sum()
            dice = (2.0 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth)
            w = self.weights[c] if self.weights is not None else 1.0
            total = total + (1.0 - dice) * w
        denom = self.weights.sum() if self.weights is not None else n_classes
        return total / denom


class TverskyLoss(nn.Module):
    """Per-class Tversky T = TP / (TP + alpha*FP + beta*FN). alpha < beta => recall."""

    def __init__(self, class_indices: Sequence[int], alpha: float = 0.4, beta: float = 0.6,
                 smooth: float = 1.0, ignore_index: int = cfg.IGNORE_INDEX) -> None:
        super().__init__()
        self.class_indices = list(class_indices)
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        n_classes = probs.shape[1]
        targets_oh, valid_f = _one_hot_with_ignore(targets, n_classes, self.ignore_index)
        probs = probs * valid_f

        loss = probs.new_tensor(0.0)
        for c in self.class_indices:
            p = probs[:, c].reshape(-1)
            t = targets_oh[:, c].reshape(-1)
            tp = (p * t).sum()
            fp = (p * (1 - t)).sum()
            fn = ((1 - p) * t).sum()
            tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            loss = loss + (1.0 - tversky)
        return loss / max(len(self.class_indices), 1)


def directed_confusion_loss(
    probs: torch.Tensor,
    targets_oh: torch.Tensor,
    penalties: Iterable[Tuple[int, int, float]],
) -> torch.Tensor:
    """Sum of w * mean(probs[pred_c] * one_hot[true_c]) over (pred_c, true_c, w)."""
    out = probs.new_tensor(0.0)
    for pred_c, true_c, w in penalties:
        out = out + w * (probs[:, pred_c] * targets_oh[:, true_c]).mean()
    return out


def directed_fn_loss(
    probs: torch.Tensor,
    targets_oh: torch.Tensor,
    pairs: Iterable[Tuple[int, int]],
) -> torch.Tensor:
    """Sum of mean(probs[pred_c] * one_hot[true_c]) over (pred_c, true_c)."""
    out = probs.new_tensor(0.0)
    for pred_c, true_c in pairs:
        out = out + (probs[:, pred_c] * targets_oh[:, true_c]).mean()
    return out


def build_class_weights(device: torch.device) -> torch.Tensor:
    """Class-weight vector for Dice + CE driven by cfg.BOOSTED_CLASSES."""
    w = torch.ones(cfg.N_CLASSES, dtype=torch.float32, device=device)
    for c, v in cfg.BOOSTED_CLASSES.items():
        if c < cfg.N_CLASSES:
            w[c] = v
    return w


class CompositeLoss(nn.Module):
    """Bundles the five terms into one callable so the training loop stays small."""

    def __init__(self, class_weights: torch.Tensor) -> None:
        super().__init__()
        self.dice = WeightedDiceLoss(weights=class_weights, ignore_index=cfg.IGNORE_INDEX)
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=cfg.IGNORE_INDEX)
        self.tversky = TverskyLoss(
            class_indices=cfg.TVERSKY_CLASSES,
            alpha=cfg.TVERSKY_ALPHA,
            beta=cfg.TVERSKY_BETA,
            ignore_index=cfg.IGNORE_INDEX,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        targets_oh, _ = _one_hot_with_ignore(targets, cfg.N_CLASSES, cfg.IGNORE_INDEX)
        return (
            self.dice(logits, targets)
            + self.ce(logits, targets)
            + cfg.CONFUSION_WEIGHT * directed_confusion_loss(probs, targets_oh, cfg.DIRECTED_CONFUSION_PENALTY)
            + cfg.FN_WEIGHT * directed_fn_loss(probs, targets_oh, cfg.DIRECTED_FN_PENALTY)
            + cfg.TVERSKY_WEIGHT * self.tversky(logits, targets)
        )
