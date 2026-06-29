"""Target-class-aware crop sampler (generalised from CLASS0_*).

Replaces the old single-class ``CLASS0_TARGET`` / ``CLASS0_CROP_PROB``. With
``cfg.TARGET_CROP_PROB`` we use a target-biased crop; the class is picked
from ``cfg.TARGET_CLASSES`` weighted by **inverse** in-image pixel
frequency so the rarer class is sampled harder when multiple targets are
configured. Otherwise a uniform random crop is drawn.

Each image precomputes its eligible centers per target class so sampling
is O(1) at training time.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .config import cfg


@dataclass
class CenterIndex:
    """Per-image precomputed centers for the configured target classes.

    Centers are restricted to pixels that can act as the *centre* of a tile
    fully inside the image — saves clipping at sample time.
    """
    per_class: Dict[int, Tuple[np.ndarray, np.ndarray]]   # class -> (rows, cols)
    inv_freq_weights: Dict[int, float]                    # class -> weight (rarer = bigger)


def build_center_index(mask: np.ndarray, target_classes: List[int], tile_size: int) -> CenterIndex:
    """Index target-class centers in `mask` for tile-sized crops.

    Pixels closer than ``tile_size//2`` to the border are dropped if the
    image is at least one tile in each dimension (so the resulting crop lies
    entirely inside the array).
    """
    h, w = mask.shape
    half = tile_size // 2
    edge_safe = (h > tile_size) and (w > tile_size)

    per_class: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    counts: Dict[int, int] = {}
    for c in target_classes:
        hits = (mask == c)
        if edge_safe:
            hits[:half, :] = False
            hits[h - half:, :] = False
            hits[:, :half] = False
            hits[:, w - half:] = False
        rows, cols = np.where(hits)
        per_class[c] = (rows, cols)
        counts[c] = int(rows.size)

    total = sum(counts.values())
    inv_freq: Dict[int, float] = {}
    if total > 0:
        # weight ~ 1 / freq, normalised so the rarest present class gets the largest weight.
        for c, n in counts.items():
            inv_freq[c] = (total / n) if n > 0 else 0.0
        s = sum(inv_freq.values()) or 1.0
        inv_freq = {c: v / s for c, v in inv_freq.items()}
    else:
        inv_freq = {c: 0.0 for c in target_classes}

    return CenterIndex(per_class=per_class, inv_freq_weights=inv_freq)


def _pick_target_class(idx: CenterIndex) -> int | None:
    """Sample a target class proportional to its inverse-frequency weight.

    Skips classes with zero pixels. Returns None when no target class has any
    eligible centers.
    """
    weights = []
    classes = []
    for c, (rows, _) in idx.per_class.items():
        if rows.size == 0:
            continue
        weights.append(idx.inv_freq_weights.get(c, 1.0))
        classes.append(c)
    if not classes:
        return None
    total = sum(weights) or 1.0
    r = random.random() * total
    acc = 0.0
    for c, wt in zip(classes, weights):
        acc += wt
        if r <= acc:
            return c
    return classes[-1]


def sample_crop(
    img_h: int,
    img_w: int,
    idx: CenterIndex,
    tile_size: int,
    target_prob: float,
) -> Tuple[int, int]:
    """Return (top, left) for a tile-sized crop.

    With probability ``target_prob`` and if any target pixels exist, biases
    the crop toward a target-class centre; otherwise picks uniformly.
    Coordinates are clipped so the crop lies entirely inside the image.
    """
    use_biased = (random.random() < target_prob) and any(
        rows.size > 0 for rows, _ in idx.per_class.values()
    )

    half = tile_size // 2

    if use_biased:
        c = _pick_target_class(idx)
        if c is not None:
            rows, cols = idx.per_class[c]
            k = random.randint(0, rows.size - 1)
            cy, cx = int(rows[k]), int(cols[k])
            top = max(0, min(cy - half, img_h - tile_size))
            left = max(0, min(cx - half, img_w - tile_size))
            return top, left

    top = random.randint(0, max(0, img_h - tile_size))
    left = random.randint(0, max(0, img_w - tile_size))
    return top, left
