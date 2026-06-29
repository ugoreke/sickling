"""Quick-scan visual overlays for labels and predictions.

Renders ``viz/<source-dir>/<stem>.png`` — the raw grayscale image with a
semi-transparent class overlay on top — so you can flip through a folder in
the file explorer and *see* what the labels (or predictions) look like
without opening ilastik.

Used to answer "is the model finding polymer in the right places?" and
"are the InitialLabels actually clean?" in seconds, which matters a lot
once the val metrics flatten and you have to decide whether more labelling
will help.

One module-level function per source kind plus a master
``render_all_overlays()`` that calls them all. Output paths mirror the
source layout under ``cfg.VIZ_DIR``.
"""

from __future__ import annotations

import os
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .config import cfg
from .masks import load_bootstrap_label, load_ilastik_mask
from .paths import (
    list_h5,
    parse_tile_filename,
    stem_of,
)


# class 0 (polymer): red.
# class 1 (background): no overlay — leave raw visible.
# class 2 (cell body):  dark teal.
# class 3 (cell boundary): green.
# ignore (255): no overlay; the raw shows through so you can spot
# untouched regions in sparse tiles.
CLASS_COLORS: dict[int, Optional[Tuple[int, int, int]]] = {
    0: (230, 57, 70),
    1: None,
    2: (38, 70, 83),
    3: (42, 157, 143),
}

ALPHA = 0.45
MAX_DIM = 1024            # downsample anything larger before writing PNG
LEGEND_TEXT = "0=polymer(red)  2=body(teal)  3=boundary(green)"


def _raw_to_rgb_base(raw_np: np.ndarray) -> np.ndarray:
    """Normalize a 1-ch raw image to a (H, W, 3) uint8 background for overlay."""
    raw = raw_np.astype(np.float32)
    m = raw.max() if raw.max() > 0 else 1.0
    raw = (raw / m) * 255.0
    return np.stack([raw, raw, raw], axis=-1)


def _overlay(raw_np: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
    """Blend a class-colored overlay onto a grayscale raw."""
    base = _raw_to_rgb_base(raw_np)
    out = base.copy()
    for c, color in CLASS_COLORS.items():
        if color is None:
            continue
        sel = (mask_np == c)
        if not sel.any():
            continue
        color_arr = np.array(color, dtype=np.float32)
        out[sel] = (1.0 - ALPHA) * out[sel] + ALPHA * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def _maybe_downsample(img: np.ndarray, max_dim: int = MAX_DIM) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    pil = Image.fromarray(img)
    return np.array(pil.resize((new_w, new_h), Image.LANCZOS))


def render_overlay_to(out_path: str, raw_path: str, mask_path: str,
                      mask_loader: Callable[[str], np.ndarray]) -> bool:
    """Render one overlay PNG. Returns True on success, False on a soft skip."""
    if not os.path.exists(raw_path):
        return False
    raw = np.array(Image.open(raw_path).convert("L"), dtype=np.uint8)
    try:
        mask = mask_loader(mask_path)
    except OSError:
        return False
    if mask.shape != raw.shape:
        # Some PREDs are written 5D or padded; squeeze if needed.
        mask = np.squeeze(mask)
        if mask.shape != raw.shape:
            return False
    rgb = _overlay(raw, mask)
    rgb = _maybe_downsample(rgb)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(rgb).save(out_path, format="PNG", optimize=False)
    return True


def _viz_subdir(name: str) -> str:
    return os.path.join(cfg.VIZ_DIR, name)


# ---- source-specific renderers ---------------------------------------------

def render_initial_labels() -> int:
    """Overlays for InitialLabels (dense, 0..N-1 / 255)."""
    out_dir = _viz_subdir("InitialLabels")
    n = 0
    for lp in list_h5(cfg.INITIAL_LABELS_DIR):
        stem = stem_of(lp)
        if stem.endswith("_labels"):
            stem = stem[: -len("_labels")]
        raw = os.path.join(cfg.CORRECTION_POOL_DIR, f"{stem}.jpg")
        out = os.path.join(out_dir, f"{stem}.png")
        if render_overlay_to(out, raw, lp, load_bootstrap_label):
            n += 1
    return n


def render_bootstrap_labels() -> int:
    """Overlays for BootstrappedLabels (dense, 0..N-1 / 255)."""
    out_dir = _viz_subdir("BootstrappedLabels")
    n = 0
    for lp in list_h5(cfg.BOOTSTRAP_LABELS_DIR):
        stem = stem_of(lp)
        if stem.endswith("_labels"):
            stem = stem[: -len("_labels")]
        raw = os.path.join(cfg.CORRECTION_POOL_DIR, f"{stem}.jpg")
        out = os.path.join(out_dir, f"{stem}.png")
        if render_overlay_to(out, raw, lp, load_bootstrap_label):
            n += 1
    return n


def render_corrected_tiles() -> int:
    """Overlays for CorrectedTiles (ilastik 1-based; subtract-1 / 255 ignore)."""
    out_dir = _viz_subdir("CorrectedTiles")
    n = 0
    for lp in list_h5(cfg.CORRECTED_TILES_DIR):
        # only the genuine painted labels (mirror build_tile_pairs hardening)
        base = os.path.basename(lp)
        if base.startswith("PRED_") or not base.endswith("_labels.h5"):
            continue
        prov = parse_tile_filename(lp)
        if prov is None:
            continue
        raw = os.path.join(cfg.CORRECTED_TILES_DIR,
                           f"{prov.stem}__y{prov.top}_x{prov.left}.jpg")
        out_stem = f"{prov.stem}__y{prov.top}_x{prov.left}"
        out = os.path.join(out_dir, f"{out_stem}.png")
        if render_overlay_to(out, raw, lp, load_ilastik_mask):
            n += 1
    return n


def render_staged_tiles() -> int:
    """Overlays for TilesToBeCorrected (the not-yet-painted staged tiles).

    Each staged tile has a raw ``<stem>__y..._x...jpg`` plus its
    ``PRED_<stem>__y..._x....h5`` (the model's prediction we shipped to
    ilastik). Overlay shows the model's PRED on the raw so you can scan the
    queue visually before painting and triage / skip the ones it got
    obviously wrong.
    """
    out_dir = _viz_subdir("TilesToBeCorrected")
    n = 0
    for pp in list_h5(cfg.TILES_TODO_DIR):
        base = os.path.basename(pp)
        # Only the PRED files carry the model's overlay; the painted-label
        # convention isn't expected to appear in the to-be-corrected folder.
        if not base.startswith("PRED_"):
            continue
        prov = parse_tile_filename(pp)
        if prov is None:
            continue
        raw = os.path.join(cfg.TILES_TODO_DIR,
                           f"{prov.stem}__y{prov.top}_x{prov.left}.jpg")
        out_stem = f"PRED_{prov.stem}__y{prov.top}_x{prov.left}"
        out = os.path.join(out_dir, f"{out_stem}.png")
        if render_overlay_to(out, raw, pp, load_ilastik_mask):
            n += 1
    return n


def render_pool_preds(max_n: Optional[int] = 100) -> int:
    """Overlays for ``PRED_<stem>.h5`` in CorrectionPool. ``max_n`` caps work
    at the N most-recently-written PREDs; pass None to render every one."""
    src = cfg.CORRECTION_POOL_DIR
    pred_paths = [p for p in list_h5(src) if os.path.basename(p).startswith("PRED_")]
    # most recently regenerated first
    pred_paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    if max_n is not None:
        pred_paths = pred_paths[:max_n]
    out_dir = _viz_subdir("CorrectionPool")
    n = 0
    for pp in pred_paths:
        s = stem_of(pp)
        if s.startswith("PRED_"):
            s = s[len("PRED_"):]
        raw = os.path.join(src, f"{s}.jpg")
        out = os.path.join(out_dir, f"PRED_{s}.png")
        if render_overlay_to(out, raw, pp, load_ilastik_mask):
            n += 1
    return n


# ---- master ---------------------------------------------------------------

def render_all_overlays(
    initial: bool = True,
    bootstrap: bool = True,
    corrected_tiles: bool = True,
    staged_tiles: bool = True,
    pool_preds: bool = True,
    pool_pred_max_n: Optional[int] = 100,
) -> dict[str, int]:
    """Rebuild overlay PNGs for every source folder you want to inspect.

    Returns a per-source count of files written. Output paths mirror the
    source layout under ``cfg.VIZ_DIR`` (e.g. ``viz/InitialLabels/*.png``).
    Each call OVERWRITES existing PNGs — re-run after a label edit or a new
    PRED refresh to see the change. ``pool_pred_max_n=None`` renders every
    PRED in the pool (slow: one PNG per pool image).

    All overlays use the same color map (polymer=red, body=dark teal,
    boundary=green, bg/ignore=raw shows through) so identical *colors* mean
    identical *classes* across InitialLabels, BootstrappedLabels, painted
    tiles, staged tiles (model PREDs), and pool PREDs.
    """
    cfg.ensure_dirs()
    os.makedirs(cfg.VIZ_DIR, exist_ok=True)
    counts: dict[str, int] = {}
    if initial:         counts["InitialLabels"]      = render_initial_labels()
    if bootstrap:       counts["BootstrappedLabels"] = render_bootstrap_labels()
    if corrected_tiles: counts["CorrectedTiles"]     = render_corrected_tiles()
    if staged_tiles:    counts["TilesToBeCorrected"] = render_staged_tiles()
    if pool_preds:      counts["CorrectionPool"]     = render_pool_preds(pool_pred_max_n)
    return counts
