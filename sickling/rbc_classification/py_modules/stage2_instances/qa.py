"""Stage 2 QA visualization.

Renders a 4-panel figure for one FOV:

    A. Raw image (greyscale) — or the 4-class label map if no raw is supplied.
    B. Kept instances colorized (random per-id palette over raw greyscale).
    C. Drop-reason overlay over raw greyscale: kept = grey,
       edge = orange, min_area = red, max_area = magenta, empty-basin = cyan.
    D. Histogram of pre-filter instance areas with min_area / max_area
       cutoffs as dashed lines.

The point is to eyeball whether the dropped fragments are *real cells the
U-Net under-segmented* (= a problem) or *speckle noise* (= correct behavior).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from PIL import Image

from sickling.rbc_classification.py_modules.config import ClassesConfig, InstancesConfig
from sickling.rbc_classification.py_modules.io.h5 import load_label_map
from sickling.rbc_classification.py_modules.stage2_instances.watershed import (
    DROP_EDGE,
    DROP_EMPTY,
    DROP_KEPT,
    DROP_MAX,
    DROP_MIN,
    mask_to_instances_with_reasons,
)

# Palette for drop reasons. Greys for kept; saturated colors for drops.
_REASON_RGBA = {
    DROP_KEPT: (0.55, 0.55, 0.55, 0.55),
    DROP_EDGE: (1.00, 0.55, 0.10, 0.85),   # orange
    DROP_MIN:  (0.95, 0.10, 0.10, 0.85),   # red
    DROP_MAX:  (0.85, 0.10, 0.85, 0.85),   # magenta
    DROP_EMPTY:(0.10, 0.85, 0.85, 0.85),   # cyan
}


def _norm_grey(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    lo, hi = np.percentile(a, [1.0, 99.0])
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def _random_color_image(instance_image: np.ndarray, seed: int = 0) -> np.ndarray:
    """Map each instance id (1..N) to a stable random RGB color."""
    n = int(instance_image.max(initial=0))
    rng = np.random.default_rng(seed)
    palette = rng.uniform(0.25, 1.0, size=(n + 1, 3)).astype(np.float32)
    palette[0] = 0.0
    return palette[instance_image]


def _drop_reason_image(
    pre_instance_image: np.ndarray,
    reasons: dict[int, str],
) -> np.ndarray:
    """RGBA overlay where each pre-filter instance is painted by its drop reason."""
    h, w = pre_instance_image.shape
    out = np.zeros((h, w, 4), dtype=np.float32)
    for iid, reason in reasons.items():
        rgba = _REASON_RGBA[reason]
        mask = pre_instance_image == iid
        out[mask] = rgba
    return out


def make_qa_figure(
    label_map: np.ndarray,
    instance_image: np.ndarray,
    pre_instance_image: np.ndarray,
    reasons: dict[int, str],
    cfg: InstancesConfig,
    raw_image: np.ndarray | None = None,
    title: str | None = None,
) -> Figure:
    """Build the 4-panel QA figure. Returns an unsaved matplotlib ``Figure``."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 13), constrained_layout=True)

    # ---- A. raw or label map ----
    ax = axes[0, 0]
    if raw_image is not None:
        ax.imshow(_norm_grey(raw_image), cmap="gray")
        ax.set_title("A. Raw image (1–99% percentile clip)")
    else:
        ax.imshow(label_map, cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
        ax.set_title("A. 4-class label map (raw image unavailable)")
    ax.set_xticks([]); ax.set_yticks([])

    # ---- B. kept instances colorized ----
    ax = axes[0, 1]
    if raw_image is not None:
        ax.imshow(_norm_grey(raw_image), cmap="gray")
    color_kept = _random_color_image(instance_image)
    alpha = (instance_image > 0).astype(np.float32) * 0.55
    rgba_kept = np.concatenate([color_kept, alpha[..., None]], axis=-1)
    ax.imshow(rgba_kept)
    n_kept = int(instance_image.max(initial=0))
    ax.set_title(f"B. Kept instances (n = {n_kept})")
    ax.set_xticks([]); ax.set_yticks([])

    # ---- C. drop-reason overlay ----
    ax = axes[1, 0]
    if raw_image is not None:
        ax.imshow(_norm_grey(raw_image), cmap="gray")
    ax.imshow(_drop_reason_image(pre_instance_image, reasons))
    counts = {r: 0 for r in _REASON_RGBA}
    for r in reasons.values():
        counts[r] = counts.get(r, 0) + 1
    legend = [
        Patch(color=_REASON_RGBA[DROP_KEPT][:3], label=f"kept ({counts[DROP_KEPT]})"),
        Patch(color=_REASON_RGBA[DROP_EDGE][:3], label=f"edge ({counts[DROP_EDGE]})"),
        Patch(color=_REASON_RGBA[DROP_MIN][:3], label=f"min_area ({counts[DROP_MIN]})"),
        Patch(color=_REASON_RGBA[DROP_MAX][:3], label=f"max_area ({counts[DROP_MAX]})"),
        Patch(color=_REASON_RGBA[DROP_EMPTY][:3], label=f"empty_basin ({counts[DROP_EMPTY]})"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=9, framealpha=0.85)
    ax.set_title("C. Drop reasons (over raw)")
    ax.set_xticks([]); ax.set_yticks([])

    # ---- D. area histogram ----
    ax = axes[1, 1]
    areas = np.array(
        [int((pre_instance_image == iid).sum()) for iid in reasons],
        dtype=np.int64,
    )
    if areas.size > 0:
        # Log-x to make small/large blobs both visible.
        bins = np.logspace(np.log10(max(areas.min(), 1)), np.log10(max(areas.max(), 1) * 1.05), 40)
        ax.hist(areas, bins=bins, color="#3a7ca5", edgecolor="black", alpha=0.85)
        ax.axvline(cfg.min_area, color="red", linestyle="--", label=f"min_area={cfg.min_area}")
        ax.axvline(cfg.max_area, color="magenta", linestyle="--", label=f"max_area={cfg.max_area}")
        ax.set_xscale("log")
        ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("Pre-filter instance area (px)")
    ax.set_ylabel("Count")
    ax.set_title("D. Pre-filter area distribution")

    if title:
        fig.suptitle(title, fontsize=14)
    return fig


def render_qa_for_h5(
    h5_path: Path,
    cfg: InstancesConfig,
    classes: ClassesConfig,
    raw_image: np.ndarray | None = None,
) -> Figure:
    """Convenience wrapper: load the U-Net 4-class h5 at ``h5_path`` and build
    the QA figure."""
    label_map = load_label_map(h5_path, n_classes=4)
    instance_image, _stats, pre_inst, reasons = mask_to_instances_with_reasons(
        label_map, cfg, classes
    )
    return make_qa_figure(
        label_map=label_map,
        instance_image=instance_image,
        pre_instance_image=pre_inst,
        reasons=reasons,
        cfg=cfg,
        raw_image=raw_image,
        title=h5_path.name,
    )


def load_raw_image(stem: str, raw_dir: Path) -> np.ndarray | None:
    """Find ``stem.{jpg,jpeg,png,tif,tiff}`` in ``raw_dir`` and return it as a
    greyscale numpy array. Returns None if no match exists."""
    for ext in ("jpg", "jpeg", "png", "tif", "tiff"):
        candidate = raw_dir / f"{stem}.{ext}"
        if candidate.exists():
            return np.array(Image.open(candidate).convert("L"))
    return None


def save_qa_figure(fig: Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
