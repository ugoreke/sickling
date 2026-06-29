"""Per-cell crop extraction.

Each crop is a ``(3, H, W)`` float32 tensor with the channel layout fixed
across the project:

    ch0 = full-FOV percentile-normalized greyscale, cropped to the window.
    ch1 = binary mask of *this* instance's ``cell_body`` pixels.
    ch2 = binary mask of *this* instance's ``polymer`` pixels.

Polymer and body are kept in separate channels (rather than merged into one
foreground mask) because polymer extent is morphologically diagnostic of
sickling — the multimodal classifier's morphology tower computes a
polymer-area-to-cell-area ratio from ch2 / ch1.
"""
from __future__ import annotations

import numpy as np
import torch

from sickling.rbc_classification.py_modules.config import ClassesConfig, CropConfig


def _bbox_for_instance(instance_image: np.ndarray, instance_id: int) -> tuple[int, int, int, int]:
    rows, cols = np.where(instance_image == instance_id)
    if rows.size == 0:
        raise ValueError(f"instance_id={instance_id} not present in instance_image.")
    return int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1


def _centroid(instance_image: np.ndarray, instance_id: int) -> tuple[float, float]:
    rows, cols = np.where(instance_image == instance_id)
    return float(cols.mean()), float(rows.mean())  # (x, y)


def _crop_window(
    cy: int,
    cx: int,
    size: int,
    h: int,
    w: int,
) -> tuple[int, int, int, int] | None:
    """Window centered at ``(cy, cx)``. Returns ``None`` if it would clip."""
    half = size // 2
    y0 = cy - half
    x0 = cx - half
    y1 = y0 + size
    x1 = x0 + size
    if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
        return None
    return y0, x0, y1, x1


def extract_one(
    raw_norm: np.ndarray,
    label_map: np.ndarray,
    instance_image: np.ndarray,
    instance_id: int,
    cfg: CropConfig,
    classes: ClassesConfig,
) -> tuple[torch.Tensor | None, dict]:
    """Build the 3-channel crop for one instance.

    Returns ``(tensor, meta)`` where ``meta`` always contains:
        ``centroid_x``, ``centroid_y``, ``area``, ``bbox_x0/y0/x1/y1``.
    If the 96×96 window would clip the FOV and ``cfg.drop_if_clipped`` is True,
    returns ``(None, meta)`` and the caller appends meta to ``failed.jsonl``.
    """
    if instance_image.shape != raw_norm.shape != label_map.shape:
        raise ValueError("raw_norm, label_map, and instance_image must share shape.")

    h, w = instance_image.shape
    instance_mask = instance_image == instance_id
    area = int(instance_mask.sum())
    if area == 0:
        raise ValueError(f"instance_id={instance_id} has zero pixels.")

    # Centroid in pixel coords; round to int for window placement.
    rows, cols = np.where(instance_mask)
    cy = int(round(float(rows.mean())))
    cx = int(round(float(cols.mean())))

    bbox = (
        int(rows.min()), int(cols.min()),
        int(rows.max()) + 1, int(cols.max()) + 1,
    )

    meta = {
        "centroid_x": float(cols.mean()),
        "centroid_y": float(rows.mean()),
        "area": area,
        "bbox_y0": bbox[0],
        "bbox_x0": bbox[1],
        "bbox_y1": bbox[2],
        "bbox_x1": bbox[3],
    }

    window = _crop_window(cy, cx, cfg.size, h, w)
    if window is None:
        if cfg.drop_if_clipped:
            return None, meta
        # Pad-and-shift: place the cell off-center and zero-pad outside.
        # (Not the default; only used when drop_if_clipped=False.)
        pad = cfg.size // 2
        padded_raw = np.pad(raw_norm, pad, mode="constant", constant_values=0.0)
        padded_lbl = np.pad(label_map, pad, mode="constant", constant_values=classes.background)
        padded_inst = np.pad(instance_image, pad, mode="constant", constant_values=0)
        cy_p, cx_p = cy + pad, cx + pad
        y0, x0 = cy_p - cfg.size // 2, cx_p - cfg.size // 2
        ch0 = padded_raw[y0:y0 + cfg.size, x0:x0 + cfg.size]
        instance_window = padded_inst[y0:y0 + cfg.size, x0:x0 + cfg.size] == instance_id
        ch1 = (padded_lbl[y0:y0 + cfg.size, x0:x0 + cfg.size] == classes.cell_body) & instance_window
        ch2 = (padded_lbl[y0:y0 + cfg.size, x0:x0 + cfg.size] == classes.polymer) & instance_window
    else:
        y0, x0, y1, x1 = window
        ch0 = raw_norm[y0:y1, x0:x1]
        instance_window = instance_image[y0:y1, x0:x1] == instance_id
        ch1 = (label_map[y0:y1, x0:x1] == classes.cell_body) & instance_window
        ch2 = (label_map[y0:y1, x0:x1] == classes.polymer) & instance_window

    tensor = np.stack([ch0.astype(np.float32),
                       ch1.astype(np.float32),
                       ch2.astype(np.float32)], axis=0)
    return torch.from_numpy(tensor), meta


def extract_for_fov(
    raw_norm: np.ndarray,
    label_map: np.ndarray,
    instance_image: np.ndarray,
    cfg: CropConfig,
    classes: ClassesConfig,
) -> tuple[torch.Tensor, list[int], list[dict], list[dict]]:
    """Run :func:`extract_one` over every kept instance in ``instance_image``.

    Returns:
        tensors: ``[N, 3, H, W]`` float32 — one row per surviving crop.
        instance_ids: parallel list of N kept ``instance_id`` values.
        kept_meta: parallel list of N metadata dicts (centroid, bbox, area).
        failed_meta: list of metadata dicts for instances dropped at this stage
            (e.g. clipped). Each carries an extra ``'instance_id'`` and
            ``'reason'`` field for ``failed.jsonl``.
    """
    n = int(instance_image.max(initial=0))
    tensors: list[torch.Tensor] = []
    instance_ids: list[int] = []
    kept_meta: list[dict] = []
    failed_meta: list[dict] = []

    for iid in range(1, n + 1):
        tensor, meta = extract_one(
            raw_norm=raw_norm,
            label_map=label_map,
            instance_image=instance_image,
            instance_id=iid,
            cfg=cfg,
            classes=classes,
        )
        if tensor is None:
            failed_meta.append({**meta, "instance_id": iid, "reason": "clipped"})
            continue
        tensors.append(tensor)
        instance_ids.append(iid)
        kept_meta.append(meta)

    if tensors:
        stacked = torch.stack(tensors, dim=0)
    else:
        stacked = torch.empty((0, 3, cfg.size, cfg.size), dtype=torch.float32)
    return stacked, instance_ids, kept_meta, failed_meta
