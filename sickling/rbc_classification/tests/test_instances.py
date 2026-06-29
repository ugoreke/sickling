"""Tests for ``sickling.stage2_instances.watershed.mask_to_instances``."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sickling.rbc_classification.py_modules.config import ClassesConfig, InstancesConfig
from sickling.rbc_classification.py_modules.io.h5 import load_label_map
from sickling.rbc_classification.py_modules.stage2_instances.watershed import mask_to_instances

CLASSES = ClassesConfig()


def _default_cfg(**overrides) -> InstancesConfig:
    return InstancesConfig(**{**InstancesConfig().model_dump(), **overrides})


def test_isolated_and_touching_cells_separate(synth_label_map):
    """Five cells in the synthetic FOV; cell E touches the right edge and is dropped."""
    cfg = _default_cfg()
    instances, stats = mask_to_instances(synth_label_map, cfg, CLASSES)

    assert instances.dtype == np.uint16
    assert stats.n_kept == 4
    assert stats.n_dropped_edge == 1
    assert stats.n_dropped_min_area == 0
    assert stats.n_dropped_max_area == 0

    # Sequentially relabeled.
    unique = np.unique(instances)
    assert unique.tolist() == list(range(stats.n_kept + 1))


def test_polymer_ring_kept_in_foreground(synth_label_map):
    """Cell B has a polymer ring around the body. The watershed foreground is
    polymer ∪ cell_body, so cell B's instance should occupy *both* the ring
    and the inner disk pixels."""
    cfg = _default_cfg()
    instances, _ = mask_to_instances(synth_label_map, cfg, CLASSES)

    # Inner-disk centroid → cell B's instance id.
    iid_inner = int(instances[64, 160])
    assert iid_inner != 0, "Cell B body should be foreground."

    # The polymer-ring pixels (e.g. (64, 188) is ~28 px from center → inside outer ring).
    iid_ring = int(instances[64, 188])
    assert iid_ring == iid_inner, (
        "Polymer ring of cell B must belong to the same instance as its body."
    )


def test_min_area_drops_small_blob():
    """One ~315-px disk in the middle of an empty FOV. A low ``threshold_rel``
    forces a marker on it; the default ``min_area=800`` then drops it."""
    arr = np.full((256, 256), CLASSES.background, dtype=np.int16)
    # Disk r=10 → area ~314.
    for y in range(256):
        for x in range(256):
            if (y - 128) ** 2 + (x - 128) ** 2 <= 10 * 10:
                arr[y, x] = CLASSES.cell_body
    cfg = _default_cfg(peak_threshold_rel=0.0, max_area=200_000)
    _, stats = mask_to_instances(arr, cfg, CLASSES)
    assert stats.n_total == 1
    assert stats.n_dropped_min_area == 1
    assert stats.n_kept == 0


def test_max_area_drops_merged_blob():
    """One ~7850-px disk centered well inside the FOV. ``max_area=6000`` drops it."""
    arr = np.full((512, 512), CLASSES.background, dtype=np.int16)
    cy, cx, r = 256, 256, 50  # area ≈ π·50² ≈ 7854
    for y in range(cy - r, cy + r + 1):
        for x in range(cx - r, cx + r + 1):
            if (y - cy) ** 2 + (x - cx) ** 2 <= r * r:
                arr[y, x] = CLASSES.cell_body
    cfg = _default_cfg(min_area=10)
    _, stats = mask_to_instances(arr, cfg, CLASSES)
    assert stats.n_total == 1
    assert stats.n_dropped_max_area == 1
    assert stats.n_kept == 0


def test_disable_edge_drop_keeps_edge_cell(synth_label_map):
    cfg = _default_cfg(drop_edge_touching=False)
    _, stats = mask_to_instances(synth_label_map, cfg, CLASSES)
    assert stats.n_dropped_edge == 0
    assert stats.n_kept == 5


def test_empty_label_map_returns_empty():
    arr = np.full((128, 128), CLASSES.background, dtype=np.int16)
    cfg = _default_cfg()
    instances, stats = mask_to_instances(arr, cfg, CLASSES)
    assert instances.sum() == 0
    assert stats == type(stats)(0, 0, 0, 0, 0)


def test_2d_required():
    arr = np.zeros((4, 8, 8), dtype=np.int16)
    with pytest.raises(ValueError):
        mask_to_instances(arr, _default_cfg(), CLASSES)


REAL_H5 = Path("unet_predictions/PRED_D16_03_1_1_Bright Field_001.h5")


@pytest.mark.skipif(not REAL_H5.exists(), reason=f"Real fixture {REAL_H5} not found.")
def test_real_h5_smoke():
    """Sanity floor on the real 1992x1992 FOV: more than 100 cells survive."""
    label_map = load_label_map(REAL_H5, n_classes=4)
    cfg = InstancesConfig()
    instances, stats = mask_to_instances(label_map, cfg, CLASSES)
    assert stats.n_kept > 100, f"Expected >100 cells in real FOV, got {stats.n_kept}: {stats}"
    assert instances.shape == label_map.shape
    assert instances.max() == stats.n_kept
