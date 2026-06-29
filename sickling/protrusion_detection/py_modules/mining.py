"""FN-aware tile mining (selection-stage only).

Rank candidate 512-px tiles by:
- **Soft target probability** — mean of ``P(target | x)`` inside the tile.
  Picks tiles where the target class is plausible *even where the argmax says
  background*, which is exactly where the model misses (ARCHITECTURE.md §7).
- **Fold disagreement** — across-fold standard deviation of the per-pixel
  target probability, averaged over the tile. Only active in ``kfold`` mode
  with multiple checkpoints; in ``single`` mode the score collapses to the
  soft-prob term.

Mining is **not** a loss weight. It only chooses which tiles the human
labels; training itself remains uniform over the pool.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from .config import cfg
from .inference import (
    load_unet,
    predict_probs,
    predict_probs_per_fold,
)
from .masks import normalize_image
from .paths import (
    TileProvenance,
    overlaps_any,
    parse_tile_filename,
    raw_jpg,
    staged_tile_provenances,
    stem_of,
    whole_pred_path,
)


@dataclass(frozen=True)
class TileCandidate:
    """A scored 512-px crop proposed for labeling."""
    stem: str
    top: int
    left: int
    score: float          # soft_prob + lambda * disagreement
    soft_prob: float
    disagreement: float

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return (self.top, self.left, cfg.CORRECTION_TILE_SIZE, cfg.CORRECTION_TILE_SIZE)


# --- per-pixel target-prob map ----------------------------------------------

def _target_prob(probs: torch.Tensor, target_classes: Sequence[int]) -> torch.Tensor:
    """Sum P(c) over the target class indices. probs is (C, H, W)."""
    if len(target_classes) == 1:
        return probs[target_classes[0]]
    return probs[list(target_classes)].sum(dim=0)


def disagreement_map(
    per_fold_probs: Sequence[torch.Tensor],
    target_classes: Sequence[int],
) -> torch.Tensor:
    """Per-pixel std of target-prob across folds. Returns (H, W)."""
    stack = torch.stack([_target_prob(p, target_classes) for p in per_fold_probs], dim=0)
    return stack.std(dim=0, unbiased=False)


# --- tile-level scoring ------------------------------------------------------

def _score_tiles_on_map(
    soft: torch.Tensor,           # (H, W) target-prob map
    dis: Optional[torch.Tensor],  # (H, W) disagreement map, or None
    tile_size: int,
    stride: int,
    lam: float,
) -> List[Tuple[int, int, float, float, float]]:
    """Slide tiles over the maps; return (top, left, score, soft, dis) per candidate."""
    h, w = soft.shape
    out: List[Tuple[int, int, float, float, float]] = []
    if h < tile_size or w < tile_size:
        return out

    ys = list(range(0, h - tile_size + 1, stride))
    if ys and ys[-1] + tile_size < h:
        ys.append(h - tile_size)
    xs = list(range(0, w - tile_size + 1, stride))
    if xs and xs[-1] + tile_size < w:
        xs.append(w - tile_size)

    for y in ys:
        for x in xs:
            sp = float(soft[y:y + tile_size, x:x + tile_size].mean().item())
            dv = float(dis[y:y + tile_size, x:x + tile_size].mean().item()) if dis is not None else 0.0
            out.append((y, x, sp + lam * dv, sp, dv))
    return out


# --- per-image entry point ---------------------------------------------------

def score_pool_image(
    jpg_path: str,
    best_model: torch.nn.Module,
    fold_ckpts: Optional[Sequence[str]] = None,
    target_classes: Optional[Sequence[int]] = None,
    write_pred: bool = True,
) -> Tuple[List[TileCandidate], torch.Tensor]:
    """Predict on one pool image; score every candidate tile; optionally save PRED.

    Returns
    -------
    (candidates, probs)
        ``candidates`` is the per-image list of scored 512-px crops.
        ``probs`` is the (C, H, W) softmax map from the best model — caller
        decides whether to save the argmax PRED via :func:`save_whole_pred`.
    """
    target_classes = list(target_classes) if target_classes is not None else list(cfg.TARGET_CLASSES)
    stem = stem_of(jpg_path)

    img_np = np.array(Image.open(jpg_path).convert("L"), dtype=np.float32)
    img_t = torch.from_numpy(normalize_image(img_np)).float().unsqueeze(0).to(cfg.DEVICE)

    probs = predict_probs(best_model, img_t)             # (C, H, W)
    soft = _target_prob(probs, target_classes)            # (H, W)

    dis_map: Optional[torch.Tensor] = None
    if fold_ckpts and len(fold_ckpts) > 1 and cfg.MINING_SCORE == "softprob+disagreement":
        per_fold = predict_probs_per_fold(fold_ckpts, img_t)
        dis_map = disagreement_map(per_fold, target_classes)
        del per_fold

    raw_tiles = _score_tiles_on_map(
        soft=soft,
        dis=dis_map,
        tile_size=cfg.CORRECTION_TILE_SIZE,
        stride=cfg.MINING_STRIDE,
        lam=cfg.MINING_LAMBDA,
    )
    candidates = [
        TileCandidate(stem=stem, top=t, left=l, score=s, soft_prob=sp, disagreement=dv)
        for (t, l, s, sp, dv) in raw_tiles
    ]
    return candidates, probs


# --- top-K with duplicate-crop guard -----------------------------------------

def pick_top_k(
    candidates: Sequence[TileCandidate],
    k: int,
    existing_tiles: Optional[Sequence[TileProvenance]] = None,
    iou_threshold: Optional[float] = None,
) -> List[TileCandidate]:
    """Greedy top-K: highest score first, skip if IoU > threshold with any
    already-accepted or already-staged tile from the same source stem.

    Cross-image candidates can never overlap (different stems), so the IoU
    test only applies within a stem.
    """
    iou_threshold = iou_threshold if iou_threshold is not None else cfg.MINING_DUP_IOU
    by_stem_existing: dict[str, List[Tuple[int, int, int, int]]] = {}
    for p in (existing_tiles or []):
        by_stem_existing.setdefault(p.stem, []).append(
            (p.top, p.left, cfg.CORRECTION_TILE_SIZE, cfg.CORRECTION_TILE_SIZE)
        )

    chosen: List[TileCandidate] = []
    by_stem_chosen: dict[str, List[Tuple[int, int, int, int]]] = {}
    for cand in sorted(candidates, key=lambda c: c.score, reverse=True):
        if len(chosen) >= k:
            break
        already = by_stem_existing.get(cand.stem, []) + by_stem_chosen.get(cand.stem, [])
        if overlaps_any(cand.box, already, iou_threshold):
            continue
        chosen.append(cand)
        by_stem_chosen.setdefault(cand.stem, []).append(cand.box)
    return chosen


# --- staging tiles to disk ---------------------------------------------------

def stage_tile(
    cand: TileCandidate,
    raw_image_np: np.ndarray,
    pred_argmax_np: np.ndarray,
    dest_dir: Optional[str] = None,
) -> Tuple[str, str]:
    """Write the raw 512-crop (jpg) + the corresponding PRED 512-crop (h5) for
    one mined tile. Returns ``(raw_path, pred_path)``.

    The PRED tile is saved in the ilastik 5D Labels format so it can be
    imported in the painting UI.
    """
    from .masks import save_ilastik_mask
    from .paths import tile_pred_path, tile_raw_path

    dest_dir = dest_dir or cfg.TILES_TODO_DIR
    os.makedirs(dest_dir, exist_ok=True)
    ts = cfg.CORRECTION_TILE_SIZE
    y, x = cand.top, cand.left

    raw_crop = raw_image_np[y:y + ts, x:x + ts]
    pred_crop = pred_argmax_np[y:y + ts, x:x + ts]

    raw_path = tile_raw_path(cand.stem, y, x, dest_dir=dest_dir)
    pred_path = tile_pred_path(cand.stem, y, x, dest_dir=dest_dir)

    Image.fromarray(raw_crop.astype(np.uint8)).save(raw_path, quality=95)
    save_ilastik_mask(pred_path, pred_crop.astype(np.uint8))
    return raw_path, pred_path
