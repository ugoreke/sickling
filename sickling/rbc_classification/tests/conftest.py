"""Shared synthetic fixtures used across the test suite."""
from __future__ import annotations

import numpy as np
import pytest

from sickling.rbc_classification.py_modules.config import ClassesConfig

# All classes default — match `configs/base.yaml`.
CLASSES = ClassesConfig()


def _draw_disk(arr: np.ndarray, cy: int, cx: int, r: float, value: int) -> None:
    """Paint a filled disk into ``arr`` (in place). Out-of-bounds is clipped."""
    h, w = arr.shape
    y0 = max(0, int(np.floor(cy - r)))
    y1 = min(h, int(np.ceil(cy + r)) + 1)
    x0 = max(0, int(np.floor(cx - r)))
    x1 = min(w, int(np.ceil(cx + r)) + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    arr[y0:y1, x0:x1][mask] = value


def _draw_ring(arr: np.ndarray, cy: int, cx: int, r_outer: float, r_inner: float, value: int) -> None:
    h, w = arr.shape
    y0 = max(0, int(np.floor(cy - r_outer)))
    y1 = min(h, int(np.ceil(cy + r_outer)) + 1)
    x0 = max(0, int(np.floor(cx - r_outer)))
    x1 = min(w, int(np.ceil(cx + r_outer)) + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    ring = (d2 <= r_outer * r_outer) & (d2 > r_inner * r_inner)
    arr[y0:y1, x0:x1][ring] = value


@pytest.fixture
def synth_label_map() -> np.ndarray:
    """256x256 4-class label map with five simulated cells.

    Layout:
      * cell A (centroid 64,64, r=24): plain cell body, isolated.
      * cell B (centroid 64,160, r=24): cell body surrounded by a polymer ring
        (tests that polymer-ringed cells survive watershed and stay attached).
      * cells C+D (centroids 192,80 and 192,128, r=22 each): touching pair,
        watershed must split them.
      * cell E (centroid 32,232, r=22): touches the right edge → must be dropped
        when ``drop_edge_touching=True``.

    Pixel values follow ``ClassesConfig`` defaults:
      0 = polymer, 1 = background, 2 = cell_body, 3 = cell_border.
    """
    h = w = 256
    arr = np.full((h, w), CLASSES.background, dtype=np.int16)

    # Cell A — plain.
    _draw_disk(arr, 64, 64, 24, CLASSES.cell_body)

    # Cell B — polymer ring (outer 32, inner 24), then body inside.
    _draw_ring(arr, 64, 160, 32, 24, CLASSES.polymer)
    _draw_disk(arr, 64, 160, 24, CLASSES.cell_body)

    # Cells C + D — touching at row 192.
    _draw_disk(arr, 192, 80, 22, CLASSES.cell_body)
    _draw_disk(arr, 192, 128, 22, CLASSES.cell_body)

    # Cell E — partially off the right edge (will be edge-touching).
    _draw_disk(arr, 32, 232, 22, CLASSES.cell_body)

    return arr


@pytest.fixture
def synth_label_map_with_blobs(synth_label_map: np.ndarray) -> np.ndarray:
    """``synth_label_map`` plus a tiny 50-px speck and a 12 000-px blob.

    Used to exercise the min/max area filters.
    """
    arr = synth_label_map.copy()
    _draw_disk(arr, 130, 30, 4, CLASSES.cell_body)         # ~50 px → below min_area=800
    _draw_disk(arr, 130, 200, 65, CLASSES.cell_body)        # ~13 000 px → above max_area=6000
    return arr
