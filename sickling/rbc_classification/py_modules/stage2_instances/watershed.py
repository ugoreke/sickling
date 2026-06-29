"""Stage 2 — convert a 4-class semantic label map into an integer instance
label image via marker-seeded watershed.

The pipeline (mirrors PIPELINE_PLAN §2 Stage 2):

    1. foreground = (==polymer) | (==cell_body)        # polymer is part of the
                                                        # sickle cell's morphology
    2. morphological closing on foreground             # bridge thin gaps
    3. Euclidean distance transform on closed fg
    4. peak_local_max on distance transform → markers   # 1 per cell center
    5. watershed on -distance, masked to closed fg
    6. drop edge-touching, < min_area, > max_area instances; relabel sequentially

The ``mask_to_instances`` function is pure (no IO) and deterministic given the
config. It returns the instance label image and a small stats dataclass that
the CLI accumulates per FOV for QA.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import binary_closing, disk
from skimage.segmentation import watershed

from sickling.rbc_classification.py_modules.config import ClassesConfig, InstancesConfig


@dataclass
class InstanceStats:
    """Per-FOV summary written alongside the instance label image."""
    n_total: int                # peaks the marker step proposed
    n_kept: int                 # instances surviving all filters
    n_dropped_edge: int
    n_dropped_min_area: int
    n_dropped_max_area: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


# Drop reason codes — used by the QA path and by Stage 3 when joining labels.
DROP_KEPT = "kept"
DROP_EDGE = "edge"
DROP_MIN = "min_area"
DROP_MAX = "max_area"
DROP_EMPTY = "empty"  # marker swallowed by another basin during watershed
DROP_REASONS = (DROP_KEPT, DROP_EDGE, DROP_MIN, DROP_MAX, DROP_EMPTY)


def _foreground_mask(label_map: np.ndarray, classes: ClassesConfig) -> np.ndarray:
    """Foreground = polymer ∪ cell_body. Polymer is intentionally included so
    sickle-shaped polymer extensions stay attached to the cell."""
    return (label_map == classes.polymer) | (label_map == classes.cell_body)


def _touches_edge(props_bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = props_bbox
    h, w = shape
    return r0 == 0 or c0 == 0 or r1 == h or c1 == w


def _run_watershed(
    label_map: np.ndarray,
    cfg: InstancesConfig,
    classes: ClassesConfig,
) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[int, int, int, int]]]:
    """Internal: run the deterministic CV pipeline and return
    ``(pre_instance_image, areas_array, bbox_per_id)``.

    ``pre_instance_image`` has values 1..N for the N markers found, 0 elsewhere.
    No filtering is applied here.
    """
    if label_map.ndim != 2:
        raise ValueError(f"Expected 2-D label map, got shape {label_map.shape}")

    foreground = _foreground_mask(label_map, classes)
    if cfg.closing_radius > 0:
        foreground = binary_closing(foreground, disk(cfg.closing_radius))

    distance = ndi.distance_transform_edt(foreground)

    peak_coords = peak_local_max(
        distance,
        min_distance=cfg.peak_min_distance,
        threshold_rel=cfg.peak_threshold_rel,
        labels=foreground.astype(np.uint8),
        exclude_border=False,
    )
    n_total = int(peak_coords.shape[0])

    if n_total == 0:
        empty = np.zeros_like(label_map, dtype=np.int64)
        return empty, np.zeros(1, dtype=np.int64), {}

    markers = np.zeros(distance.shape, dtype=np.int32)
    markers[tuple(peak_coords.T)] = np.arange(1, n_total + 1, dtype=np.int32)

    instances = watershed(-distance, markers, mask=foreground).astype(np.int64)

    areas = np.bincount(instances.ravel(), minlength=n_total + 1)

    rows, cols = np.where(instances > 0)
    ids_at = instances[rows, cols]
    bbox: dict[int, tuple[int, int, int, int]] = {}
    if ids_at.size > 0:
        order = np.argsort(ids_at, kind="stable")
        ids_sorted = ids_at[order]
        rows_sorted = rows[order]
        cols_sorted = cols[order]
        boundaries = np.where(np.diff(ids_sorted, prepend=ids_sorted[0] - 1) != 0)[0]
        boundaries = np.append(boundaries, len(ids_sorted))
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            iid = int(ids_sorted[start])
            rr = rows_sorted[start:end]
            cc = cols_sorted[start:end]
            bbox[iid] = (int(rr.min()), int(cc.min()), int(rr.max()) + 1, int(cc.max()) + 1)

    return instances, areas, bbox


def _classify_drops(
    n_total: int,
    areas: np.ndarray,
    bbox: dict[int, tuple[int, int, int, int]],
    shape: tuple[int, int],
    cfg: InstancesConfig,
) -> dict[int, str]:
    """Return ``{pre_filter_id: reason}`` for ids 1..n_total."""
    reasons: dict[int, str] = {}
    for iid in range(1, n_total + 1):
        area = int(areas[iid]) if iid < len(areas) else 0
        if area == 0:
            reasons[iid] = DROP_EMPTY
        elif cfg.drop_edge_touching and _touches_edge(bbox[iid], shape):
            reasons[iid] = DROP_EDGE
        elif area < cfg.min_area:
            reasons[iid] = DROP_MIN
        elif area > cfg.max_area:
            reasons[iid] = DROP_MAX
        else:
            reasons[iid] = DROP_KEPT
    return reasons


def _apply_filters(
    pre_instance_image: np.ndarray,
    reasons: dict[int, str],
    n_total: int,
) -> tuple[np.ndarray, InstanceStats]:
    keep_lookup = np.zeros(n_total + 1, dtype=np.int64)
    next_id = 1
    counts = {r: 0 for r in DROP_REASONS}
    for iid in range(1, n_total + 1):
        reason = reasons[iid]
        counts[reason] += 1
        if reason == DROP_KEPT:
            keep_lookup[iid] = next_id
            next_id += 1

    instance_image = keep_lookup[pre_instance_image].astype(np.uint16)
    n_kept = next_id - 1
    stats = InstanceStats(
        n_total=n_total,
        n_kept=n_kept,
        n_dropped_edge=counts[DROP_EDGE],
        # Empty-basin markers are reported under min_area for backward compat
        # (dropped-because-no-pixels-survived behaves like an under-area drop).
        n_dropped_min_area=counts[DROP_MIN] + counts[DROP_EMPTY],
        n_dropped_max_area=counts[DROP_MAX],
    )
    return instance_image, stats


def mask_to_instances(
    label_map: np.ndarray,
    cfg: InstancesConfig,
    classes: ClassesConfig,
) -> tuple[np.ndarray, InstanceStats]:
    """Return ``(instance_image, stats)``.

    See module docstring. ``instance_image`` is uint16, 0 = background,
    1..N = sequential survivor IDs.
    """
    pre_instance_image, areas, bbox = _run_watershed(label_map, cfg, classes)
    n_total = int(pre_instance_image.max(initial=0))
    if n_total == 0:
        return np.zeros_like(label_map, dtype=np.uint16), InstanceStats(0, 0, 0, 0, 0)
    reasons = _classify_drops(n_total, areas, bbox, pre_instance_image.shape, cfg)
    return _apply_filters(pre_instance_image, reasons, n_total)


def mask_to_instances_with_reasons(
    label_map: np.ndarray,
    cfg: InstancesConfig,
    classes: ClassesConfig,
) -> tuple[np.ndarray, InstanceStats, np.ndarray, dict[int, str]]:
    """Like ``mask_to_instances`` but also returns the pre-filter watershed
    image and a ``{pre_filter_id: reason}`` mapping.

    Used by Stage 2 QA visualizations and (later) by Stage 3 when a label
    coordinate falls inside an instance that was dropped — we want to log
    *why* it was dropped, not just "dropped".
    """
    pre_instance_image, areas, bbox = _run_watershed(label_map, cfg, classes)
    n_total = int(pre_instance_image.max(initial=0))
    if n_total == 0:
        return (
            np.zeros_like(label_map, dtype=np.uint16),
            InstanceStats(0, 0, 0, 0, 0),
            np.zeros_like(label_map, dtype=np.int64),
            {},
        )
    reasons = _classify_drops(n_total, areas, bbox, pre_instance_image.shape, cfg)
    instance_image, stats = _apply_filters(pre_instance_image, reasons, n_total)
    return instance_image, stats, pre_instance_image, reasons
