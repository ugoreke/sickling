"""Sliding-window inference, TTA, ensembling, and best-fold selection.

Single source of truth for "predict on a whole image": always goes through
``predict_probs`` (returns the softmax map) and ``predict_mask`` (argmax of
that map). Mining and PRED export both need probability maps, so they live
in the same module.

ARCHITECTURE.md §10 keeps sliding-window + eval-time TTA; PRED export does
**not** TTA (rejected in §13 unless we revisit).
"""

from __future__ import annotations

import glob
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

from .config import cfg
from .masks import load_bootstrap_label, normalize_image
from .model import UNet


# --- core sliding window -----------------------------------------------------

def _pad_to_tile(img: torch.Tensor, tile_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Reflect-pad a (C, H, W) image so each axis is at least ``tile_size``."""
    _, h, w = img.shape
    pad_h = max(0, tile_size - h)
    pad_w = max(0, tile_size - w)
    if pad_h or pad_w:
        img = F.pad(img.unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect").squeeze(0)
    return img, (pad_h, pad_w)


@torch.no_grad()
def predict_probs(
    model: torch.nn.Module,
    img_t: torch.Tensor,
    tile_size: Optional[int] = None,
    overlap: float = 0.5,
    num_classes: Optional[int] = None,
) -> torch.Tensor:
    """Sliding-window softmax probability map for one image.

    Parameters
    ----------
    model : module already in ``eval()`` mode (caller's responsibility).
    img_t : (C, H, W) float tensor on the device used for inference.
    tile_size, overlap, num_classes : default to cfg values.

    Returns
    -------
    probs : (num_classes, H, W) float tensor on the same device as ``img_t``.
    """
    tile_size = tile_size or cfg.TILE_SIZE
    num_classes = num_classes or cfg.N_CLASSES

    orig_h, orig_w = img_t.shape[-2], img_t.shape[-1]
    img_t, (pad_h, pad_w) = _pad_to_tile(img_t, tile_size)
    _, h, w = img_t.shape

    stride = int(tile_size * (1 - overlap))
    if stride <= 0:
        stride = tile_size

    prob_map = torch.zeros((num_classes, h, w), device=img_t.device)
    count_map = torch.zeros((num_classes, h, w), device=img_t.device)

    ys = list(range(0, max(1, h - tile_size + 1), stride))
    if ys[-1] + tile_size < h:
        ys.append(h - tile_size)
    xs = list(range(0, max(1, w - tile_size + 1), stride))
    if xs[-1] + tile_size < w:
        xs.append(w - tile_size)

    for y in ys:
        for x in xs:
            ys_, xs_ = y, x
            crop = img_t[:, ys_:ys_ + tile_size, xs_:xs_ + tile_size].unsqueeze(0)
            probs = torch.softmax(model(crop), dim=1).squeeze(0)
            prob_map[:, ys_:ys_ + tile_size, xs_:xs_ + tile_size] += probs
            count_map[:, ys_:ys_ + tile_size, xs_:xs_ + tile_size] += 1

    probs = prob_map / count_map.clamp(min=1)
    return probs[:, :orig_h, :orig_w]


def predict_mask(model: torch.nn.Module, img_t: torch.Tensor, **kw) -> torch.Tensor:
    """Argmax of ``predict_probs``. Returns a (H, W) int64 tensor."""
    return torch.argmax(predict_probs(model, img_t, **kw), dim=0)


# --- TTA ---------------------------------------------------------------------

@torch.no_grad()
def predict_probs_tta(
    model: torch.nn.Module,
    img_t: torch.Tensor,
    tile_size: Optional[int] = None,
    overlap: float = 0.5,
    num_classes: Optional[int] = None,
) -> torch.Tensor:
    """8-fold TTA: average ``predict_probs`` over flip + 90deg-rotation symmetries.

    The geometric transform is inverted on the probability map before
    averaging so classes stay aligned with the input image.
    """
    tile_size = tile_size or cfg.TILE_SIZE
    num_classes = num_classes or cfg.N_CLASSES

    accum: Optional[torch.Tensor] = None
    n = 0
    for k in range(4):
        for flip in (False, True):
            x = torch.rot90(img_t, k=k, dims=(1, 2))
            if flip:
                x = torch.flip(x, dims=(2,))
            p = predict_probs(model, x, tile_size=tile_size, overlap=overlap, num_classes=num_classes)
            if flip:
                p = torch.flip(p, dims=(2,))
            p = torch.rot90(p, k=-k, dims=(1, 2))
            accum = p if accum is None else accum + p
            n += 1
    return accum / n


# --- Ensemble over folds -----------------------------------------------------

def load_unet(ckpt_path: str, device: torch.device | None = None,
              backbone: Optional[str] = None) -> torch.nn.Module:
    """Restore a checkpoint into the matching backbone.

    The backbone is parsed from the filename (``<backbone>_fold_*_best_loop_*.pth``)
    so the same loader works for UNet, SMP-EfficientNet, etc., and the
    fallback chain in ``best_ckpt_for_inference`` can mix and match across
    backbones without the caller knowing. Override with ``backbone=...`` to
    force a specific architecture (rare).

    Name kept for backward compatibility — it's a misnomer once SMP backbones
    are in play, but every callsite already uses it.
    """
    from .model import build_model, parse_backbone_from_ckpt_path
    device = device or cfg.DEVICE
    bk = backbone if backbone is not None else parse_backbone_from_ckpt_path(ckpt_path)
    model = build_model(bk).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model


@torch.no_grad()
def predict_probs_ensemble(
    ckpt_paths: Sequence[str],
    img_t: torch.Tensor,
    tile_size: Optional[int] = None,
    overlap: float = 0.5,
    tta: bool = False,
) -> torch.Tensor:
    """Mean softmax over a list of fold checkpoints."""
    accum: Optional[torch.Tensor] = None
    for cp in ckpt_paths:
        model = load_unet(cp, device=img_t.device).eval()
        p = (predict_probs_tta if tta else predict_probs)(
            model, img_t, tile_size=tile_size, overlap=overlap
        )
        accum = p if accum is None else accum + p
        del model
    return accum / len(ckpt_paths)


@torch.no_grad()
def predict_probs_per_fold(
    ckpt_paths: Sequence[str],
    img_t: torch.Tensor,
    tile_size: Optional[int] = None,
    overlap: float = 0.5,
) -> List[torch.Tensor]:
    """Return one probability map per fold checkpoint (for disagreement scoring)."""
    out: List[torch.Tensor] = []
    for cp in ckpt_paths:
        model = load_unet(cp, device=img_t.device).eval()
        out.append(predict_probs(model, img_t, tile_size=tile_size, overlap=overlap))
        del model
    return out


# --- Best-fold selection -----------------------------------------------------

def discover_fold_checkpoints(loop: Optional[int] = None,
                              backbone: Optional[str] = None) -> List[str]:
    """Fold checkpoints for one loop under cfg.MODELS_DIR, sorted by fold index.

    Filters by ``backbone`` (default ``cfg.MODEL_BACKBONE``) so a UNet run
    and an SMP run can coexist in the same models folder without confusing
    fold discovery. ``loop`` defaults to the latest trained loop *of that
    backbone* (``cfg.latest_loop_index(backbone)``).
    """
    bk = backbone if backbone is not None else cfg.MODEL_BACKBONE
    loop = loop if loop is not None else cfg.latest_loop_index(backbone=bk)
    paths = glob.glob(os.path.join(cfg.MODELS_DIR, f"{bk}_fold_*_best_loop_{loop}.pth"))
    # Parse "<backbone>_fold_<N>_best_loop_<L>.pth" → N.
    import re
    fold_rx = re.compile(rf"^{re.escape(bk)}_fold_(\d+)_best_loop_\d+\.pth$")
    def _idx(p: str) -> int:
        m = fold_rx.search(os.path.basename(p))
        return int(m.group(1)) if m else 99
    return sorted(paths, key=_idx)


def latest_available_fold_ckpt(fold_1based: int,
                               backbone: Optional[str] = None) -> Optional[str]:
    """Highest-loop checkpoint for ``fold_1based`` under ``backbone``.

    Walks back across loops, so a missing top-of-stack file (e.g. the user
    deleted ``unet_fold_2_best_loop_3.pth``) falls back to the previous loop
    rather than crashing. Returns ``None`` if no checkpoint exists for this
    fold at any loop *for this backbone*.
    """
    import re
    bk = backbone if backbone is not None else cfg.MODEL_BACKBONE
    pattern = os.path.join(cfg.MODELS_DIR, f"{bk}_fold_{fold_1based}_best_loop_*.pth")
    paths = glob.glob(pattern)
    if not paths:
        return None
    rx = re.compile(r"_best_loop_(\d+)\.pth$")
    def _loop(p: str) -> int:
        m = rx.search(os.path.basename(p))
        return int(m.group(1)) if m else -1
    paths.sort(key=_loop, reverse=True)
    return paths[0]


def _load_image_tensor(jpg_path: str, device: torch.device) -> torch.Tensor:
    img = np.array(Image.open(jpg_path).convert("L"), dtype=np.float32)
    img = normalize_image(img)
    return torch.from_numpy(img).float().unsqueeze(0).to(device)


def _per_class_dice(pred: torch.Tensor, gt: torch.Tensor, n_classes: int, ignore_index: int) -> np.ndarray:
    dices = np.zeros(n_classes, dtype=np.float64)
    valid = (gt != ignore_index).float()
    for c in range(n_classes):
        p = (pred == c).float() * valid
        t = (gt == c).float() * valid
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        if union == 0:
            dices[c] = 1.0 if t.sum() == 0 else 0.0
        else:
            dices[c] = (2.0 * inter / union).item()
    return dices


def select_best_fold(
    val_pairs: Sequence[Tuple[str, str]],
    ckpt_paths: Optional[Sequence[str]] = None,
    target_classes: Optional[Sequence[int]] = None,
    tta: bool = True,
) -> Tuple[str, np.ndarray]:
    """Pick the fold with the highest mean target-class dice on val.

    Parameters
    ----------
    val_pairs : (raw_jpg_path, dense_mask_path) pairs — typically the
        first ``TRUTH_VAL_COUNT`` items of InitialLabels.
    ckpt_paths : fold checkpoints to consider. Defaults to whatever
        ``discover_fold_checkpoints()`` finds.
    target_classes : average dice over these classes; defaults to
        ``cfg.TARGET_CLASSES``.
    tta : 8-fold TTA on the val pass (cheap, more honest).

    Returns
    -------
    (best_ckpt_path, per_fold_target_dice)
    """
    ckpt_paths = list(ckpt_paths) if ckpt_paths is not None else discover_fold_checkpoints()
    if not ckpt_paths:
        raise FileNotFoundError(f"No fold checkpoints found under {cfg.MODELS_DIR}")
    target_classes = list(target_classes) if target_classes is not None else list(cfg.TARGET_CLASSES)
    device = cfg.DEVICE

    scores = np.zeros(len(ckpt_paths), dtype=np.float64)
    for fi, cp in enumerate(ckpt_paths):
        model = load_unet(cp, device=device).eval()
        accum = np.zeros(cfg.N_CLASSES, dtype=np.float64)
        for raw_p, mask_p in val_pairs:
            img_t = _load_image_tensor(raw_p, device)
            probs = (predict_probs_tta if tta else predict_probs)(model, img_t)
            pred = torch.argmax(probs, dim=0)
            gt = torch.from_numpy(load_bootstrap_label(mask_p).astype(np.int64)).to(device)
            accum += _per_class_dice(pred, gt, cfg.N_CLASSES, cfg.IGNORE_INDEX)
        accum /= max(len(val_pairs), 1)
        scores[fi] = float(np.mean([accum[c] for c in target_classes if c < cfg.N_CLASSES]))
        del model

    best = int(np.argmax(scores))
    return ckpt_paths[best], scores


def best_ckpt_for_inference(val_pairs: Sequence[Tuple[str, str]]) -> str:
    """Return the checkpoint path for the inference passes (mining, PRED export).

    Resolution order:

    * **Single mode** — start with the carried-over best fold (the winner of
      the most recent kfold loop, or ``cfg.BEST_FOLD`` if the log has no
      kfold). If that fold's latest-loop checkpoint is missing (deleted by
      hand, for example), walk back through prior loops for that fold. If
      that fold has *no* checkpoints anywhere, fall back to ``cfg.BEST_FOLD``,
      then to the latest available checkpoint of any fold.
    * **K-fold mode** — re-score the current loop's fold checkpoints on val
      and pick the highest target-class dice; if no checkpoints exist for
      the latest loop yet (e.g. mining before retrain), fall back to the
      single-mode resolution.
    """
    # Local import to dodge a metrics <-> inference cycle.
    from .metrics import latest_best_fold

    def _single_resolve() -> str:
        chain: List[int] = []
        carried = latest_best_fold(default=cfg.BEST_FOLD)
        for f in (carried, cfg.BEST_FOLD):
            if f not in chain:
                chain.append(f)
        for f in range(1, cfg.N_FOLDS + 1):
            if f not in chain:
                chain.append(f)
        for f in chain:
            ckpt = latest_available_fold_ckpt(f)
            if ckpt is not None:
                return ckpt
        raise FileNotFoundError(
            f"No fold checkpoints found under {cfg.MODELS_DIR} for any fold."
        )

    if cfg.FOLD_MODE == "single":
        return _single_resolve()

    # K-fold: prefer the current loop's best-by-dice, fall back to single-mode
    # resolution if the latest loop has no fold checkpoints yet.
    ckpts = discover_fold_checkpoints()
    if not ckpts:
        return _single_resolve()
    best, _ = select_best_fold(val_pairs, ckpt_paths=ckpts)
    return best
