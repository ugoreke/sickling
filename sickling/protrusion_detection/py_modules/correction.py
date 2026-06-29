"""Correction-mode round orchestrator.

Run order (see GUIDE.md §3):

1. Pick ``cfg.PRED_BATCH_SIZE`` pool images: prefer those without a PRED yet,
   then fill with the oldest-PRED images. InitialLabels stems are always
   excluded; BootstrappedLabels stems are excluded when
   ``cfg.MINING_EXCLUDE_LABELED_STEMS`` is True.
2. For each selected image, run the best model (single-fold mode:
   ``BEST_FOLD``; kfold mode: best-by-val-dice fold) to write a fresh
   ``PRED_<stem>.h5`` and score every 512-px candidate tile in it. In kfold
   mode, also collect per-fold probability maps to compute disagreement.
3. Globally rank all scored candidates and pick ``cfg.MINING_BATCH_SIZE``
   with the IoU duplicate-crop guard against tiles already in
   ``TilesToBeCorrected`` ∪ ``CorrectedTiles``.
4. Stage each pick as ``<stem>__y<top>_x<left>.jpg`` + ``PRED_<stem>__y<top>_x<left>.h5``
   in ``TilesToBeCorrected`` for the human to paint.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from .config import cfg
from .inference import (
    best_ckpt_for_inference,
    discover_fold_checkpoints,
    load_unet,
)
from .masks import normalize_image, save_ilastik_mask
from .mining import (
    TileCandidate,
    pick_top_k,
    score_pool_image,
    stage_tile,
)
from .paths import (
    labeled_stems,
    list_raw,
    staged_tile_provenances,
    stem_of,
    whole_pred_path,
)


# --- pool selection ----------------------------------------------------------

def _pred_mtime(stem: str) -> float:
    p = whole_pred_path(stem)
    try:
        return os.path.getmtime(p)
    except OSError:
        return -1.0  # missing PRED -> sorted first


def regenerate_pool_preds(
    val_pairs: Sequence[Tuple[str, str]],
    stems: Optional[Sequence[str]] = None,
    exclude_labeled: Optional[bool] = None,
) -> int:
    """Rewrite ``PRED_<stem>.h5`` for every eligible pool image with the
    current best model. One-shot, manual — call it when you want fresh
    predictions across the whole pool (e.g. just after a retrain), then go
    back to the cheap per-round refresh in ``run_correction_round`` which
    only touches ``cfg.PRED_BATCH_SIZE`` images.

    Parameters
    ----------
    val_pairs : used to pick the inference checkpoint in kfold mode; ignored
        in single mode.
    stems : limit the refresh to this set of stems. Default = every raw
        ``.jpg`` in ``CORRECTION_POOL_DIR`` not excluded as labeled.
    exclude_labeled : same semantics as ``select_pool_batch``.

    Returns the number of PREDs written.
    """
    exclude_labeled = exclude_labeled if exclude_labeled is not None else cfg.MINING_EXCLUDE_LABELED_STEMS
    cfg.ensure_dirs()

    init, boot = labeled_stems()
    blocked = set(init) | (set(boot) if exclude_labeled else set())

    all_raw = list_raw(cfg.CORRECTION_POOL_DIR)
    eligible = [p for p in all_raw if stem_of(p) not in blocked]
    if stems is not None:
        wanted = set(stems)
        eligible = [p for p in eligible if stem_of(p) in wanted]

    best_ckpt = best_ckpt_for_inference(val_pairs)
    best_model = load_unet(best_ckpt, device=cfg.DEVICE).eval()
    print(f"🔄 Full pool PRED refresh: {len(eligible)} images  | model: {os.path.basename(best_ckpt)}")

    n_written = 0
    for jpg_path in tqdm(eligible, desc="Pool PRED refresh"):
        # score_pool_image both predicts probs and (when write_pred=True) writes
        # an argmax PRED via save_ilastik_mask — reuse it so we don't drift.
        _, probs = score_pool_image(
            jpg_path,
            best_model=best_model,
            fold_ckpts=None,  # disagreement irrelevant here; we just want the PRED
            target_classes=cfg.TARGET_CLASSES,
            write_pred=True,
        )
        argmax_np = torch.argmax(probs, dim=0).cpu().numpy().astype(np.uint8)
        save_ilastik_mask(whole_pred_path(stem_of(jpg_path)), argmax_np)
        del probs
        n_written += 1

    del best_model
    return n_written


def select_pool_batch(
    n: Optional[int] = None,
    exclude_labeled: Optional[bool] = None,
) -> List[str]:
    """Return up to ``n`` raw .jpg paths from CorrectionPool, missing-PRED first.

    InitialLabels stems are unconditionally excluded (val/test). Optionally
    excludes BootstrappedLabels stems too.
    """
    n = n if n is not None else cfg.PRED_BATCH_SIZE
    exclude_labeled = exclude_labeled if exclude_labeled is not None else cfg.MINING_EXCLUDE_LABELED_STEMS

    init, boot = labeled_stems()
    blocked = set(init)
    if exclude_labeled:
        blocked |= boot

    all_raw = list_raw(cfg.CORRECTION_POOL_DIR)
    eligible = [p for p in all_raw if stem_of(p) not in blocked]

    # Stable sort: PRED missing (mtime=-1) goes first; otherwise oldest first.
    eligible.sort(key=lambda p: _pred_mtime(stem_of(p)))
    return eligible[:n]


# --- a round -----------------------------------------------------------------

@dataclass
class RoundReport:
    n_predicted: int
    n_candidates: int
    n_staged: int
    staged_paths: List[Tuple[str, str]]   # (raw_path, pred_path) pairs
    picks: List[TileCandidate]
    best_ckpt: str


def run_correction_round(
    val_pairs: Sequence[Tuple[str, str]],
    n_pool: Optional[int] = None,
    n_mine: Optional[int] = None,
) -> RoundReport:
    """End-to-end correction round (predict -> score -> stage).

    Parameters
    ----------
    val_pairs : (raw_jpg, dense_label_h5) pairs used to pick the inference
        checkpoint in kfold mode (typically the first ``TRUTH_VAL_COUNT``
        items from InitialLabels). Ignored in single-fold mode.
    n_pool : how many pool images to refresh (default ``cfg.PRED_BATCH_SIZE``).
    n_mine : how many tiles to stage (default ``cfg.MINING_BATCH_SIZE``).
    """
    cfg.ensure_dirs()

    n_pool = n_pool if n_pool is not None else cfg.PRED_BATCH_SIZE
    n_mine = n_mine if n_mine is not None else cfg.MINING_BATCH_SIZE

    best_ckpt = best_ckpt_for_inference(val_pairs)
    best_model = load_unet(best_ckpt, device=cfg.DEVICE).eval()

    fold_ckpts: Optional[List[str]] = None
    if cfg.FOLD_MODE == "kfold":
        fold_ckpts = discover_fold_checkpoints()
        # Disagreement is only meaningful with >= 2 folds.
        if len(fold_ckpts) < 2:
            fold_ckpts = None

    batch = select_pool_batch(n=n_pool)
    all_candidates: List[TileCandidate] = []
    # Keep the raw image arrays + argmax PRED maps in a cache so the final
    # staging step doesn't have to re-read them from disk.
    cache: dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for jpg_path in tqdm(batch, desc="Pool predict + score"):
        cands, probs = score_pool_image(
            jpg_path,
            best_model=best_model,
            fold_ckpts=fold_ckpts,
            target_classes=cfg.TARGET_CLASSES,
            write_pred=True,
        )
        all_candidates.extend(cands)

        argmax_np = torch.argmax(probs, dim=0).cpu().numpy().astype(np.uint8)
        del probs
        save_ilastik_mask(whole_pred_path(stem_of(jpg_path)), argmax_np)

        raw_np = np.array(Image.open(jpg_path).convert("L"))
        cache[stem_of(jpg_path)] = (raw_np, argmax_np)

    picks = pick_top_k(
        all_candidates,
        k=n_mine,
        existing_tiles=staged_tile_provenances(),
    )

    staged: List[Tuple[str, str]] = []
    for cand in tqdm(picks, desc="Stage tiles"):
        raw_np, argmax_np = cache[cand.stem]
        raw_path, pred_path = stage_tile(cand, raw_np, argmax_np)
        staged.append((raw_path, pred_path))

    return RoundReport(
        n_predicted=len(batch),
        n_candidates=len(all_candidates),
        n_staged=len(staged),
        staged_paths=staged,
        picks=picks,
        best_ckpt=best_ckpt,
    )
