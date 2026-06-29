"""Raw-image loading + percentile normalization.

Single source of truth for ch0 of every per-cell crop. Mirrors the
``normalize_image`` function in ``training 2.ipynb`` so the same preprocessing
applies in Stage 1 (U-Net training) and Stage 3 (crop extraction).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

RAW_EXTS = ("jpg", "jpeg", "png", "tif", "tiff")


def normalize_image(img: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """Percentile-clip + scale to ``[0, 1]``. Returns float32.

    The 99th-percentile is the brightest non-saturated value used as the
    upper bound, which is robust to a handful of dust specks or hot pixels
    that would otherwise compress the dynamic range.
    """
    a = img.astype(np.float32)
    p = np.percentile(a, percentile)
    denom = p if p > 0 else (a.max() if a.max() > 0 else 1.0)
    return np.clip(a / denom, 0, 1).astype(np.float32)


def load_raw_greyscale(path: str | Path) -> np.ndarray:
    """Load a raw image as a 2-D float32 array (no normalization applied)."""
    return np.array(Image.open(path).convert("L"), dtype=np.float32)


def find_raw_image(stem: str, raw_dir: Path) -> Path | None:
    """Return the first ``stem.<ext>`` match in ``raw_dir`` for any extension
    in :data:`RAW_EXTS`, or ``None`` if no file matches."""
    for ext in RAW_EXTS:
        candidate = raw_dir / f"{stem}.{ext}"
        if candidate.exists():
            return candidate
    return None
