"""Folder discovery, filename conventions, and provenance parsing.

The double underscore (`__`) before `y<top>_x<left>` is the provenance
delimiter — stems can themselves contain spaces and single underscores
(e.g. ``D16_03_1_1_Bright Field_001``), so we never split on a single ``_``.

See ARCHITECTURE.md §4 for the filename table.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from .config import cfg


# Tile provenance: <stem>__y<top>_x<left>[_labels].h5  (or PRED_<stem>__y..._x...h5)
_TILE_COORDS_RE = re.compile(r"__y(?P<y>\d+)_x(?P<x>\d+)$")


def stem_of(path: str) -> str:
    """Filename without directory or extension. Keeps spaces and single underscores."""
    return os.path.splitext(os.path.basename(path))[0]


def well_of(stem: str) -> str:
    """First 3 chars of a stem = the well code (e.g. 'D16', 'G21', 'H20').

    Project naming convention: every source image's stem begins with the
    physical well it came from. Crossing well boundaries tests model
    generalization across samples — the same model that's great on
    its-training-wells can be terrible on a novel well.
    """
    return stem[:3]


# --- Filename builders --------------------------------------------------------

def raw_jpg(stem: str, raw_dir: Optional[str] = None) -> str:
    return os.path.join(raw_dir or cfg.CORRECTION_POOL_DIR, f"{stem}.jpg")


def whole_label_path(stem: str, labels_dir: str) -> str:
    return os.path.join(labels_dir, f"{stem}_labels.h5")


def whole_pred_path(stem: str, pool_dir: Optional[str] = None) -> str:
    return os.path.join(pool_dir or cfg.CORRECTION_POOL_DIR, f"PRED_{stem}.h5")


def tile_pred_path(stem: str, top: int, left: int, dest_dir: Optional[str] = None) -> str:
    return os.path.join(dest_dir or cfg.TILES_TODO_DIR, f"PRED_{stem}__y{top}_x{left}.h5")


def tile_label_path(stem: str, top: int, left: int, dest_dir: Optional[str] = None) -> str:
    return os.path.join(dest_dir or cfg.CORRECTED_TILES_DIR, f"{stem}__y{top}_x{left}_labels.h5")


def tile_raw_path(stem: str, top: int, left: int, dest_dir: Optional[str] = None) -> str:
    """Raw crop of the tile, written alongside the tile PRED for ilastik."""
    return os.path.join(dest_dir or cfg.TILES_TODO_DIR, f"{stem}__y{top}_x{left}.jpg")


# --- Provenance parsing ------------------------------------------------------

@dataclass(frozen=True)
class TileProvenance:
    """Origin of a tile: which source image, and where in it."""
    stem: str   # source-image stem (may contain spaces / single underscores)
    top: int
    left: int


def parse_tile_filename(path: str) -> Optional[TileProvenance]:
    """Recover (stem, top, left) from any tile-style filename.

    Returns None when the filename has no ``__y<...>_x<...>`` suffix
    (e.g. a whole-image file). The leading ``PRED_`` prefix and a trailing
    ``_labels`` are both stripped before matching.
    """
    base = stem_of(path)
    if base.startswith("PRED_"):
        base = base[len("PRED_"):]
    if base.endswith("_labels"):
        base = base[: -len("_labels")]

    m = _TILE_COORDS_RE.search(base)
    if not m:
        return None
    stem = base[: m.start()]
    return TileProvenance(stem=stem, top=int(m.group("y")), left=int(m.group("x")))


def list_h5(folder: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, "*.h5")))


def list_raw(folder: str) -> List[str]:
    out: List[str] = []
    for ext in cfg.RAW_EXTS:
        out.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(out)


def labeled_stems() -> Tuple[set, set]:
    """Return (initial_label_stems, bootstrap_label_stems)."""
    init = {stem_of(p).removesuffix("_labels") for p in list_h5(cfg.INITIAL_LABELS_DIR)}
    boot = {stem_of(p).removesuffix("_labels") for p in list_h5(cfg.BOOTSTRAP_LABELS_DIR)}
    return init, boot


def staged_tile_provenances() -> List[TileProvenance]:
    """Provenances of every tile currently in TilesToBeCorrected or CorrectedTiles.

    Drives the duplicate-crop guard and (when promotion to val is on) the
    leakage barrier.
    """
    out: List[TileProvenance] = []
    for folder in (cfg.TILES_TODO_DIR, cfg.CORRECTED_TILES_DIR):
        for p in list_h5(folder) + list_raw(folder):
            prov = parse_tile_filename(p)
            if prov is not None:
                out.append(prov)
    return out


# --- Geometry helpers --------------------------------------------------------

def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """IoU of two axis-aligned boxes in (top, left, h, w) form."""
    ay, ax, ah, aw = a
    by, bx, bh, bw = b
    y1, x1 = max(ay, by), max(ax, bx)
    y2, x2 = min(ay + ah, by + bh), min(ax + aw, bx + bw)
    inter_h = max(0, y2 - y1)
    inter_w = max(0, x2 - x1)
    inter = inter_h * inter_w
    if inter == 0:
        return 0.0
    union = ah * aw + bh * bw - inter
    return inter / union


def overlaps_any(
    candidate: Tuple[int, int, int, int],
    others: Iterable[Tuple[int, int, int, int]],
    threshold: float,
) -> bool:
    return any(iou(candidate, o) > threshold for o in others)
