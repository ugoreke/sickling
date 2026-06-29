"""Class-balanced sampling for the imbalanced labeled fine-tune."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def make_weighted_sampler(
    labels: np.ndarray, minority_frac: float = 0.5
) -> WeightedRandomSampler:
    """Return a sampler that draws roughly ``minority_frac`` minority samples
    per batch and ``1 - minority_frac`` majority samples.

    Implementation: per-sample weight is class-frequency-inverse, then scaled so
    expected per-batch ratios match. Length defaults to the dataset size so the
    sampler emits one epoch's worth of samples per call.
    """
    labels = np.asarray(labels).astype(np.int64)
    if labels.size == 0:
        raise ValueError("Cannot build a weighted sampler from an empty label set.")
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) != 2:
        raise ValueError(f"Expected binary labels for the sampler, got classes {classes.tolist()}")

    # Identify minority. Classes are 0/1; minority = the rarer one.
    minority_class = int(classes[counts.argmin()])

    n = labels.size
    n_min = int((labels == minority_class).sum())
    n_maj = n - n_min
    w_min = minority_frac / max(n_min, 1)
    w_maj = (1.0 - minority_frac) / max(n_maj, 1)

    weights = np.where(labels == minority_class, w_min, w_maj).astype(np.float64)
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=n,
        replacement=True,
    )
