"""Build the 10x10 grid image for manual polymer-length measurement.

Takes the variable-size eval crops in ``MiniTilesForEval/``, pads each into a
fixed cell, draws a dashed dotted frame just outside the eval region inside
each cell so the operator can see exactly what region is graded, and tiles
them into a single grid image.

Output (next to ``MiniTilesForEval/``):

  - ``grid_<n>x<n>.png``        — the assembled image
  - ``grid_<n>x<n>_layout.json`` — for each cell:
      * source stem of the crop
      * source-pool-image coordinates of the eval region
      * eval region position **in grid coordinates**
      * the crop's filename

The operator opens the PNG in Photoshop, drops Count-tool markers at the
endpoints of polymers inside each dashed frame (start, end, start, end, ...)
and exports the coordinates via ``count2csv.jsx``. Downstream
(``sickling/notebooks/polymer_length_grid.ipynb``) takes the resulting CSV,
the layout JSON, and the UNet checkpoint, maps each manual polymer back to
its source crop, computes manual length as the Euclidean distance between
the paired endpoints, computes model length as the skeleton length plus
regionprops major-axis of the matched polymer CC inside the same eval
region, and writes the comparison figure.

CLI defaults match the v3 eval-crop layout (2x context, eval region at the
center) and the user's instinct on the dashed frame: 5-pixel-square dots at
32 px spacing, brightness 200 (just above typical bright-field background so
the dot is discernible against the noisy background but well clear of the
polymer-dark range).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# Allow running from sickling/ root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sickling.protrusion_detection.config import cfg
from sickling.protrusion_detection.minicrops import parse_mini_filename


def _draw_dashed_frame(
    img: np.ndarray,
    eval_top: int, eval_left: int, eval_h: int, eval_w: int,
    dot_color: int = 200,
    dot_spacing: int = 32,
    dot_size: int = 5,
) -> None:
    """Draw 4 dotted edges JUST OUTSIDE the eval region.

    Modifies ``img`` in place. Dot rows sit entirely outside the eval region
    so no graded pixel is altered.
    """
    H, W = img.shape[:2]
    half_pre = dot_size // 2
    er_right_col = eval_left + eval_w - 1
    er_bot_row   = eval_top  + eval_h - 1

    # The row containing the top-edge dots: dot's bottom row lands at
    # (eval_top - 1) — one row above the eval region.
    top_dot_row    = eval_top - dot_size
    bot_dot_row    = er_bot_row + 1
    left_dot_col   = eval_left - dot_size
    right_dot_col  = er_right_col + 1

    def _place(dot_top: int, dot_left: int) -> None:
        y0 = max(0, dot_top);  y1 = min(H, dot_top + dot_size)
        x0 = max(0, dot_left); x1 = min(W, dot_left + dot_size)
        if y0 < y1 and x0 < x1:
            img[y0:y1, x0:x1] = dot_color

    # Horizontal edges (top and bottom): dot columns step across the eval
    # region's width. Anchor each dot's leftmost column relative to the
    # eval-region left, so dots line up visually with the eval corners.
    for x in range(0, eval_w + 1, dot_spacing):
        col = eval_left + x - half_pre
        _place(top_dot_row, col)
        _place(bot_dot_row, col)

    # Vertical edges (left and right): same scheme on the row axis.
    for y in range(0, eval_h + 1, dot_spacing):
        row = eval_top + y - half_pre
        _place(row, left_dot_col)
        _place(row, right_dot_col)


def build_grid(
    src_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    cell_size: int = 256,
    grid_side: int = 10,
    bg_value: int = 180,
    dot_color: int = 200,
    dot_spacing: int = 32,
    dot_size: int = 5,
) -> tuple[Path, Path]:
    """Build a ``grid_side`` × ``grid_side`` grid of eval crops.

    Parameters
    ----------
    src_dir : crop source folder. Default ``cfg.MINI_TILES_FOR_EVAL_DIR``.
    out_dir : where the grid + layout are written. Default ``src_dir``.
    cell_size : pixel side of each grid cell. Default 256 = 2 *
        ``cfg.MINI_EVAL_CROP_MAX``.
    grid_side : crops per row/col. Default 10 (= 100 crops, matches default
        ``cfg.MINI_EVAL_N_CROPS``).
    bg_value : gray padding around each variable-size crop.
    dot_color, dot_spacing, dot_size : dashed-frame style.

    Returns the (grid_png_path, layout_json_path) pair.
    """
    src_dir = Path(src_dir or cfg.MINI_TILES_FOR_EVAL_DIR)
    out_dir = Path(out_dir or src_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jpgs = sorted(p for p in src_dir.glob("*.jpg"))
    n_target = grid_side * grid_side
    if len(jpgs) < n_target:
        print(f"⚠️  only {len(jpgs)} crops; {n_target - len(jpgs)} cells will be blank.")

    grid_dim = cell_size * grid_side
    grid_img = np.full((grid_dim, grid_dim), bg_value, dtype=np.uint8)

    layout = {
        "cell_size": cell_size,
        "grid_side": grid_side,
        "bg_value": bg_value,
        "dot_color": dot_color,
        "dot_spacing": dot_spacing,
        "dot_size": dot_size,
        "n_cells": 0,
        "cells": [],
    }

    for idx in range(min(len(jpgs), n_target)):
        jpg = jpgs[idx]
        prov = parse_mini_filename(str(jpg))
        if prov is None:
            print(f"⚠️  no provenance for {jpg.name}, skipping cell {idx}.")
            continue

        crop_img = np.array(Image.open(jpg).convert("L"))
        cropH, cropW = crop_img.shape
        evalH, evalW = prov.h, prov.w

        row = idx // grid_side
        col = idx % grid_side
        cell_y = row * cell_size
        cell_x = col * cell_size

        if cropH > cell_size or cropW > cell_size:
            print(f"⚠️  crop {jpg.name} is {cropH}x{cropW} > {cell_size}, clipping.")
            cropH = min(cropH, cell_size)
            cropW = min(cropW, cell_size)
            crop_img = crop_img[:cropH, :cropW]

        pad_top  = (cell_size - cropH) // 2
        pad_left = (cell_size - cropW) // 2

        # Drop the crop into the grid (centered with bg padding).
        grid_img[
            cell_y + pad_top  : cell_y + pad_top  + cropH,
            cell_x + pad_left : cell_x + pad_left + cropW,
        ] = crop_img

        # Eval region inside the crop is centered (v3 convention: crop is
        # 2x eval size). Position it in grid coords:
        eval_top_grid  = cell_y + pad_top  + (cropH - evalH) // 2
        eval_left_grid = cell_x + pad_left + (cropW - evalW) // 2

        _draw_dashed_frame(
            grid_img,
            eval_top_grid, eval_left_grid, evalH, evalW,
            dot_color=dot_color, dot_spacing=dot_spacing, dot_size=dot_size,
        )

        layout["cells"].append({
            "grid_row": row,
            "grid_col": col,
            "cell_origin_yx": [cell_y, cell_x],
            "crop_origin_in_grid_yx": [cell_y + pad_top, cell_x + pad_left],
            "crop_hw": [cropH, cropW],
            "eval_region_in_grid_yx": [eval_top_grid, eval_left_grid],
            "eval_region_hw": [evalH, evalW],
            "source_stem": prov.stem,
            "source_image_eval_top_left_yx": [prov.top, prov.left],
            "jpg_filename": jpg.name,
        })

    layout["n_cells"] = len(layout["cells"])
    grid_png    = out_dir / f"grid_{grid_side}x{grid_side}.png"
    layout_json = out_dir / f"grid_{grid_side}x{grid_side}_layout.json"
    Image.fromarray(grid_img).save(grid_png)
    with open(layout_json, "w") as f:
        json.dump(layout, f, indent=2)

    print(f"wrote: {grid_png}  ({grid_dim}x{grid_dim}, {layout['n_cells']} cells filled)")
    print(f"wrote: {layout_json}")
    return grid_png, layout_json


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the polymer-length grid image.")
    ap.add_argument("--cell-size",   type=int, default=256, help="px side of each cell (default 256)")
    ap.add_argument("--grid-side",   type=int, default=10,  help="cells per row/col (default 10)")
    ap.add_argument("--bg-value",    type=int, default=180, help="gray pad value around each crop (default 180)")
    ap.add_argument("--dot-color",   type=int, default=200, help="dashed-frame dot brightness (default 200)")
    ap.add_argument("--dot-spacing", type=int, default=32,  help="px between dot positions (default 32)")
    ap.add_argument("--dot-size",    type=int, default=5,   help="dot side in px (default 5)")
    args = ap.parse_args()

    build_grid(
        cell_size=args.cell_size,
        grid_side=args.grid_side,
        bg_value=args.bg_value,
        dot_color=args.dot_color,
        dot_spacing=args.dot_spacing,
        dot_size=args.dot_size,
    )


if __name__ == "__main__":
    main()
