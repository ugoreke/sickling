"""Mini-crop mining: many small adaptive crops focused on one target class.

Use this once dense 512-px tile mining plateaus and your labelling time is
the bottleneck. Each mini-crop is sized to fit a predicted target-class
connected component plus a small context margin — small enough to scan and
paint in seconds, big enough to verify it visually.

Workflow:

1. **Filter** pool stems by their disk PRED — skip any image whose current
   PRED has no target-class pixels (saves the inference cost on negatives).
   Stems with no PRED on disk fall through to full inference. Run §3.0
   ``regenerate_pool_preds`` first if you want the filter to be current.
2. **Score** the survivors: run inference on each, find connected
   components of the target class in the argmax, and score each CC by
   ``mean target soft-prob + MINING_LAMBDA * cross-fold disagreement``.
   Argmax gets written back to disk (so the next mining call's filter is
   current without an extra §3.0 pass).
3. **Rank + dedupe** globally; the IoU dedup checks against everything
   currently staged, already painted, *and* previously skipped.
4. **Stage** the top ``MINI_CROP_BATCH_SIZE`` as
   ``MiniTilesToBeCorrected/<stem>__y..._x..._h..._w....{jpg, h5}``.

Skip semantics: any staged crop older than the latest model checkpoint is
considered stale and moved to ``MiniTilesToBeCorrected/_skipped/`` at the
start of the next mining call. So "skip a crop" = "don't paint it and let
the next retrain advance the model"; mining auto-cleans on the next call.

Painting workflow in ilastik (per crop):
- Open ``<stem>__y..._x..._h..._w....jpg``.
- Import ``PRED_<stem>__y..._x..._h..._w....h5`` as Labels.
- Paint **only the target class** where it's wrong / missing. Background and
  other classes stay un-painted; the densify-on-retrain step uses the clean
  model to fill them in so there's no positive-only-tile over-firing.
- Save the painted label as
  ``MiniTilesCorrected/<stem>__y..._x..._h..._w..._labels.h5``.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
from tqdm.auto import tqdm

from .config import cfg
from .inference import (
    best_ckpt_for_inference,
    discover_fold_checkpoints,
    load_unet,
    predict_probs,
    predict_probs_per_fold,
)
from .masks import (
    load_ilastik_mask,
    normalize_image,
    save_ilastik_mask,
)
from .paths import (
    labeled_stems,
    list_h5,
    list_raw,
    overlaps_any,
    stem_of,
    whole_pred_path,
)


# ---- filename convention --------------------------------------------------

# Mini-crops carry size in the filename so the parser can recover the bbox.
# parse_tile_filename() (in paths.py) doesn't understand this — that's
# intentional, mini-crops live in their own folders and don't touch the
# 512-px tile pipeline.
_MINI_RE = re.compile(r"__y(?P<y>\d+)_x(?P<x>\d+)_h(?P<h>\d+)_w(?P<w>\d+)$")


@dataclass(frozen=True)
class MiniProv:
    stem: str
    top: int
    left: int
    h: int
    w: int

    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.top, self.left, self.h, self.w)

    def base(self) -> str:
        return f"{self.stem}__y{self.top}_x{self.left}_h{self.h}_w{self.w}"


def parse_mini_filename(path: str) -> Optional[MiniProv]:
    base = stem_of(path)
    if base.startswith("PRED_"):
        base = base[len("PRED_"):]
    if base.endswith("_labels"):
        base = base[: -len("_labels")]
    m = _MINI_RE.search(base)
    if not m:
        return None
    stem = base[: m.start()]
    return MiniProv(
        stem=stem,
        top=int(m["y"]), left=int(m["x"]),
        h=int(m["h"]), w=int(m["w"]),
    )


def mini_raw_path(prov: MiniProv, *, in_corrected: bool = False) -> str:
    return os.path.join(
        cfg.MINI_TILES_CORRECTED_DIR if in_corrected else cfg.MINI_TILES_TODO_DIR,
        f"{prov.base()}.jpg",
    )


def mini_pred_path(prov: MiniProv) -> str:
    return os.path.join(cfg.MINI_TILES_TODO_DIR, f"PRED_{prov.base()}.h5")


def mini_label_path(prov: MiniProv) -> str:
    return os.path.join(cfg.MINI_TILES_CORRECTED_DIR, f"{prov.base()}_labels.h5")


# ---- crop geometry --------------------------------------------------------

def _crop_around_cc(cc_mask: np.ndarray, img_h: int, img_w: int) -> Tuple[int, int, int, int]:
    """Centered square crop around a CC, padded and clamped to image bounds.

    Returns ``(top, left, side, side)``.
    """
    ys, xs = np.where(cc_mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    bbox_side = max(y1 - y0 + 1, x1 - x0 + 1)
    side = bbox_side + 2 * cfg.MINI_CROP_PADDING
    side = int(np.clip(side, cfg.MINI_CROP_MIN, cfg.MINI_CROP_MAX))
    side = min(side, img_h, img_w)
    cy = int(round((y0 + y1) / 2))
    cx = int(round((x0 + x1) / 2))
    top = max(0, min(img_h - side, cy - side // 2))
    left = max(0, min(img_w - side, cx - side // 2))
    return top, left, side, side


# ---- scoring --------------------------------------------------------------

@dataclass(frozen=True)
class MiniCandidate:
    stem: str
    top: int
    left: int
    h: int
    w: int
    score: float
    soft_prob: float
    disagreement: float
    cc_area: int

    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.top, self.left, self.h, self.w)


def _score_components(
    pred_argmax: np.ndarray,
    target_probs: np.ndarray,
    target_class: int,
    fold_target_probs: Optional[List[np.ndarray]],
) -> List[Tuple[np.ndarray, float, float, float, int]]:
    """Return per-CC tuples: ``(cc_mask, score, soft_prob, disagreement, area)``."""
    binary = (pred_argmax == target_class)
    labelled, n = ndi.label(binary)
    out: List[Tuple[np.ndarray, float, float, float, int]] = []
    for cc_id in range(1, n + 1):
        cc = (labelled == cc_id)
        area = int(cc.sum())
        if area < cfg.MINI_CROP_MIN_CC_AREA:
            continue
        sp = float(target_probs[cc].mean())
        if fold_target_probs is not None and len(fold_target_probs) > 1:
            per_fold = np.array([float(fp[cc].mean()) for fp in fold_target_probs])
            dis = float(per_fold.std())
        else:
            dis = 0.0
        score = sp + cfg.MINING_LAMBDA * dis
        out.append((cc, score, sp, dis, area))
    return out


# ---- per-image pipeline ---------------------------------------------------

def _maybe_skip_via_disk_pred(jpg_path: str, target_class: int) -> bool:
    """Return True if the disk PRED says no target-class pixels (skip inference)."""
    pred_p = whole_pred_path(stem_of(jpg_path))
    if not os.path.exists(pred_p):
        return False
    try:
        disk = load_ilastik_mask(pred_p)
    except OSError:
        return False
    return not (disk == target_class).any()


def _candidates_from_pool_image(
    jpg_path: str,
    best_model: torch.nn.Module,
    fold_ckpts: Optional[Sequence[str]],
    target_class: int,
    refresh_disk_pred: bool,
) -> Tuple[List[MiniCandidate], Optional[np.ndarray], Optional[np.ndarray]]:
    """Score every CC of the target class in one pool image.

    Returns ``(candidates, raw_uint8, pred_argmax)``. If the disk PRED says
    this image has no target-class pixels, returns immediately with empty
    candidates so we skip the inference cost.
    """
    if _maybe_skip_via_disk_pred(jpg_path, target_class):
        return [], None, None

    raw_uint8 = np.array(Image.open(jpg_path).convert("L"), dtype=np.uint8)
    img_t = torch.from_numpy(normalize_image(raw_uint8.astype(np.float32))).float()
    img_t = img_t.unsqueeze(0).to(cfg.DEVICE)

    with torch.no_grad():
        probs = predict_probs(best_model, img_t)
    pred_argmax = torch.argmax(probs, dim=0).cpu().numpy().astype(np.uint8)
    target_probs = probs[target_class].cpu().numpy()
    del probs

    fold_target_probs: Optional[List[np.ndarray]] = None
    if fold_ckpts is not None and len(fold_ckpts) >= 2:
        fold_probs = predict_probs_per_fold(fold_ckpts, img_t)
        fold_target_probs = [fp[target_class].cpu().numpy() for fp in fold_probs]
        del fold_probs

    if refresh_disk_pred:
        try:
            save_ilastik_mask(whole_pred_path(stem_of(jpg_path)), pred_argmax)
        except OSError:
            pass

    H, W = pred_argmax.shape
    stem = stem_of(jpg_path)
    cands: List[MiniCandidate] = []
    for cc, score, sp, dis, area in _score_components(
        pred_argmax, target_probs, target_class, fold_target_probs,
    ):
        top, left, h, w = _crop_around_cc(cc, H, W)
        cands.append(MiniCandidate(
            stem=stem, top=top, left=left, h=h, w=w,
            score=score, soft_prob=sp, disagreement=dis, cc_area=area,
        ))
    return cands, raw_uint8, pred_argmax


# ---- dedup ----------------------------------------------------------------

def existing_mini_bboxes_by_stem() -> Dict[str, List[Tuple[int, int, int, int]]]:
    """Bboxes of every mini-crop already staged / corrected / skipped,
    keyed by source stem — fed into the IoU dedup so we never re-stage
    something the user already saw."""
    out: Dict[str, List[Tuple[int, int, int, int]]] = {}
    folders = [
        cfg.MINI_TILES_TODO_DIR,
        cfg.MINI_TILES_CORRECTED_DIR,
        os.path.join(cfg.MINI_TILES_TODO_DIR, "_skipped"),
    ]
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for fn in os.listdir(folder):
            path = os.path.join(folder, fn)
            if not os.path.isfile(path):
                continue
            prov = parse_mini_filename(path)
            if prov is None:
                continue
            out.setdefault(prov.stem, []).append(prov.bbox())
    return out


def _pick_top_k(
    cands: List[MiniCandidate],
    k: int,
    existing: Dict[str, List[Tuple[int, int, int, int]]],
) -> List[MiniCandidate]:
    cands = sorted(cands, key=lambda c: c.score, reverse=True)
    picks: List[MiniCandidate] = []
    for c in cands:
        if len(picks) >= k:
            break
        others = list(existing.get(c.stem, []))
        others.extend(p.bbox() for p in picks if p.stem == c.stem)
        if overlaps_any(c.bbox(), others, cfg.MINI_CROP_DUP_IOU):
            continue
        picks.append(c)
    return picks


# ---- staging --------------------------------------------------------------

def _stage_one(
    cand: MiniCandidate,
    raw_uint8: np.ndarray,
    pred_argmax: np.ndarray,
) -> Tuple[str, str]:
    prov = MiniProv(cand.stem, cand.top, cand.left, cand.h, cand.w)
    raw_crop = raw_uint8[cand.top:cand.top + cand.h, cand.left:cand.left + cand.w]
    pred_crop = pred_argmax[cand.top:cand.top + cand.h, cand.left:cand.left + cand.w]
    raw_path = mini_raw_path(prov)
    pred_path = mini_pred_path(prov)
    os.makedirs(cfg.MINI_TILES_TODO_DIR, exist_ok=True)
    Image.fromarray(raw_crop).save(raw_path)
    save_ilastik_mask(pred_path, pred_crop)
    return raw_path, pred_path


# ---- skip-on-stale + painted cleanup -------------------------------------

def _latest_ckpt_mtime() -> Optional[float]:
    ckpts = glob.glob(os.path.join(cfg.MODELS_DIR, "unet_fold_*_best_loop_*.pth"))
    if not ckpts:
        return None
    return max(os.path.getmtime(p) for p in ckpts)


def _sweep_stale_to_skipped() -> int:
    """Move staged crops older than the latest checkpoint to ``_skipped/``.

    A staged crop's mtime is when *that* round mined it. If a retrain has
    happened since (i.e. the newest checkpoint is newer than the file), the
    crop is "from a previous epoch of the model" — it would have ranked
    differently against the new model, and the user implicitly skipped it by
    not painting before the retrain. So we step it aside.
    """
    todo = cfg.MINI_TILES_TODO_DIR
    if not os.path.isdir(todo):
        return 0
    ckpt_mtime = _latest_ckpt_mtime()
    if ckpt_mtime is None:
        return 0
    skipped_dir = os.path.join(todo, "_skipped")
    moved = 0
    for fn in os.listdir(todo):
        path = os.path.join(todo, fn)
        if not os.path.isfile(path):
            continue
        if os.path.getmtime(path) < ckpt_mtime:
            os.makedirs(skipped_dir, exist_ok=True)
            os.replace(path, os.path.join(skipped_dir, fn))
            moved += 1
    return moved


def _cleanup_painted() -> int:
    """For each ``MiniTilesCorrected/*_labels.h5``, remove the matching raw +
    PRED from ``MiniTilesToBeCorrected`` so the staging folder shows only
    what's still pending."""
    corrected = cfg.MINI_TILES_CORRECTED_DIR
    if not os.path.isdir(corrected):
        return 0
    n = 0
    for lp in list_h5(corrected):
        if not os.path.basename(lp).endswith("_labels.h5"):
            continue
        prov = parse_mini_filename(lp)
        if prov is None:
            continue
        for q in (mini_raw_path(prov), mini_pred_path(prov)):
            if os.path.exists(q):
                os.remove(q)
                n += 1
    return n


# ---- top-level orchestrator -----------------------------------------------

@dataclass
class MiniRoundReport:
    n_swept_stale: int
    n_cleaned_painted: int
    n_eligible: int
    n_inferred: int
    n_candidates: int
    n_staged: int
    best_ckpt: str
    picks: List[MiniCandidate]


# ---- Held-out eval set ----------------------------------------------------

def _stems_with_training_crops() -> set:
    """Source stems that already contribute at least one crop to the
    training pool (``CorrectedTiles`` or ``MiniTilesCorrected``). Used by
    ``stage_eval_minicrops`` to avoid picking them as eval candidates."""
    from .paths import list_h5, parse_tile_filename
    out: set = set()
    for folder in (cfg.CORRECTED_TILES_DIR, cfg.MINI_TILES_CORRECTED_DIR):
        if not os.path.isdir(folder):
            continue
        for fn in os.listdir(folder):
            path = os.path.join(folder, fn)
            if not os.path.isfile(path):
                continue
            prov = parse_tile_filename(path)
            if prov is None:
                prov_mini = parse_mini_filename(path)
                if prov_mini is None:
                    continue
                out.add(prov_mini.stem)
            else:
                out.add(prov.stem)
    return out


def _crop_around_cc_eval(cc_mask: np.ndarray, img_h: int, img_w: int) -> Tuple[int, int, int, int]:
    """Same logic as _crop_around_cc but reads the eval-specific config
    (smaller / faster-to-paint defaults). Returns (top, left, side, side)."""
    ys, xs = np.where(cc_mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    bbox_side = max(y1 - y0 + 1, x1 - x0 + 1)
    side = bbox_side + 2 * cfg.MINI_EVAL_CROP_PADDING
    side = int(np.clip(side, cfg.MINI_EVAL_CROP_MIN, cfg.MINI_EVAL_CROP_MAX))
    side = min(side, img_h, img_w)
    cy = int(round((y0 + y1) / 2))
    cx = int(round((x0 + x1) / 2))
    top = max(0, min(img_h - side, cy - side // 2))
    left = max(0, min(img_w - side, cx - side // 2))
    return top, left, side, side


def stage_eval_minicrops(
    n_crops: Optional[int] = None,
    seed: int = 0,
    balance_by_well: bool = True,
    val_pairs: Optional[Sequence[Tuple[str, str]]] = None,
) -> List[str]:
    """Stage polymer-centered, adaptively-sized eval crops, sealed from training.

    Each crop is centered on a **predicted polymer connected component**
    in its source image, with size = component bbox + ``MINI_EVAL_CROP_PADDING``
    clamped to ``[MINI_EVAL_CROP_MIN, MINI_EVAL_CROP_MAX]``. Same convention
    as the §3.1 training crops, just with smaller defaults so eval paints
    in 30–60 s per crop. This is the part where I diverged from your
    intent in the v1 implementation: random fixed-size crops mostly miss
    polymer and waste paint time. CC-centered crops *guarantee* that every
    painted crop measures the polymer recall question, not the "is there
    polymer at all in this random patch?" question.

    Selection rules:
      - one crop per source image (diversity),
      - the highest-scoring CC per image (most confident polymer prediction),
      - round-robin across wells so dominant wells don't soak up the budget,
      - sealed leakage: any source stem in ``InitialLabels``,
        ``BootstrappedLabels``, or contributing to
        ``CorrectedTiles``/``MiniTilesCorrected`` is dropped before mining.

    No PRED is written; only the raw ``.jpg`` crop. The operator opens it
    in ilastik, dense-labels all four classes from scratch, and exports
    ``<base>_labels.h5`` in the same folder.

    Runs inference on every eligible pool image — slow on the first call
    (a few minutes on CUDA for ~300 stems), fast on rerun if you keep the
    existing crops (they re-seal those stems and skip the inference path
    for them).
    """
    import random
    from .paths import labeled_stems, list_raw, well_of

    n_crops = n_crops if n_crops is not None else cfg.MINI_EVAL_N_CROPS
    cfg.ensure_dirs()
    rng = random.Random(seed)

    init, boot = labeled_stems()
    # Eval crops in MiniTilesForEval/ already block their own stems via the
    # dedup pass downstream (one crop per source image), so we don't need a
    # separate eval-stems set here. Sealing is enforced at training time —
    # the eval surface is for paper figures / external comparison, not for
    # training, so no need to bar the eval stems from training-pool inputs.
    blocked = init | boot | _stems_with_training_crops()
    # Also skip stems already present in MiniTilesForEval/ (idempotence: a
    # second call with a higher n_crops only adds new stems).
    if os.path.isdir(cfg.MINI_TILES_FOR_EVAL_DIR):
        for fn in os.listdir(cfg.MINI_TILES_FOR_EVAL_DIR):
            path = os.path.join(cfg.MINI_TILES_FOR_EVAL_DIR, fn)
            if not os.path.isfile(path):
                continue
            prov = parse_mini_filename(path)
            if prov is not None:
                blocked.add(prov.stem)
    pool_jpgs = [p for p in list_raw(cfg.CORRECTION_POOL_DIR) if stem_of(p) not in blocked]

    if not pool_jpgs:
        print("⚠️  no eligible pool images for eval staging.")
        return []

    # Resolve inference checkpoint(s) before the per-image loop.
    if val_pairs is None:
        from .splits import build_val_test_pairs
        val_pairs, _ = build_val_test_pairs()
    best_ckpt = best_ckpt_for_inference(val_pairs)
    best_model = load_unet(best_ckpt, device=cfg.DEVICE).eval()
    fold_ckpts: Optional[List[str]] = None
    if cfg.FOLD_MODE == "kfold":
        fold_ckpts = discover_fold_checkpoints()
        if len(fold_ckpts) < 2:
            fold_ckpts = None

    target_class = cfg.MINI_CROP_TARGET_CLASS

    # Pre-print: well distribution of eligible pool stems
    elig_by_well: dict = {}
    for jpg in pool_jpgs:
        elig_by_well.setdefault(well_of(stem_of(jpg)), []).append(jpg)
    well_pre = ", ".join(f"{w}={len(elig_by_well[w])}" for w in sorted(elig_by_well))
    print(f"🎯 Staging eval mini-crops: target {n_crops}, adaptive size "
          f"[{cfg.MINI_EVAL_CROP_MIN}, {cfg.MINI_EVAL_CROP_MAX}]px, "
          f"{len(pool_jpgs)} eligible stems across {len(elig_by_well)} wells "
          f"({well_pre}). Scoring with {os.path.basename(best_ckpt)}...")

    # Score every eligible pool image: keep the top-1 polymer CC per image.
    # Uses the per-image disk-PRED pre-filter (no polymer in disk PRED →
    # skip inference), same as the §3.1 mining flow.
    per_image_best: dict = {}  # stem -> (cand, raw_uint8)
    for jpg in tqdm(pool_jpgs, desc="Eval-mine scoring"):
        cands, raw_np, pred_np = _candidates_from_pool_image(
            jpg, best_model, fold_ckpts, target_class, refresh_disk_pred=False,
        )
        if not cands or raw_np is None:
            continue
        # Re-derive crop bbox using the *eval-specific* size config (the
        # candidate generator used the training MIN/MAX, but we want the
        # smaller eval clamps to limit paint time).
        top_cand = max(cands, key=lambda c: c.score)
        # Reconstruct the CC mask from the source pred_np to re-size around
        # the same blob with the eval size config.
        from scipy import ndimage as ndi
        binary = (pred_np == target_class)
        labelled, _ = ndi.label(binary)
        # Find the CC by looking for any target-class pixel inside the
        # candidate's bbox (more robust than reading the bbox center —
        # the center can easily land on a non-CC pixel for irregular blobs).
        bbox_pred = pred_np[top_cand.top:top_cand.top + top_cand.h,
                            top_cand.left:top_cand.left + top_cand.w]
        target_ys, target_xs = np.where(bbox_pred == target_class)
        if target_ys.size == 0:
            # No target pixel in the bbox (shouldn't happen for a polymer-CC
            # candidate, but be defensive). Re-clamp the original bbox to
            # the eval size range — we still want size ≤ MINI_EVAL_CROP_MAX.
            side = int(np.clip(max(top_cand.h, top_cand.w),
                               cfg.MINI_EVAL_CROP_MIN, cfg.MINI_EVAL_CROP_MAX))
            side = min(side, pred_np.shape[0], pred_np.shape[1])
            cy = top_cand.top + top_cand.h // 2
            cx = top_cand.left + top_cand.w // 2
            top  = max(0, min(pred_np.shape[0] - side, cy - side // 2))
            left = max(0, min(pred_np.shape[1] - side, cx - side // 2))
            h, w = side, side
        else:
            cc_id = labelled[target_ys[0] + top_cand.top, target_xs[0] + top_cand.left]
            cc_mask = (labelled == cc_id)
            top, left, h, w = _crop_around_cc_eval(cc_mask, *pred_np.shape)
        sized = MiniCandidate(
            stem=top_cand.stem, top=top, left=left, h=h, w=w,
            score=top_cand.score, soft_prob=top_cand.soft_prob,
            disagreement=top_cand.disagreement, cc_area=top_cand.cc_area,
        )
        per_image_best[stem_of(jpg)] = (sized, raw_np)
    del best_model

    if not per_image_best:
        print("⚠️  no pool image yielded a polymer CC. "
              "Run §3.0 to refresh disk PREDs, then retry.")
        return []

    # Build the staging order: round-robin by well so the first N crops
    # cover every well at least once before any well gets a second crop.
    stems = list(per_image_best.keys())
    if balance_by_well:
        by_well: dict = {}
        for s in stems:
            by_well.setdefault(well_of(s), []).append(s)
        for w in by_well:
            # Highest-scoring first within well (gives the strongest polymer
            # signal per well), with rng-driven tiebreak.
            by_well[w].sort(
                key=lambda s: (per_image_best[s][0].score, rng.random()),
                reverse=True,
            )
        well_keys = sorted(by_well.keys())
        rng.shuffle(well_keys)
        max_per_well = max(len(v) for v in by_well.values())
        ordered_stems = []
        for round_idx in range(max_per_well):
            for w in well_keys:
                if round_idx < len(by_well[w]):
                    ordered_stems.append(by_well[w][round_idx])
    else:
        ordered_stems = sorted(stems,
                               key=lambda s: per_image_best[s][0].score,
                               reverse=True)

    # Stage the top n_crops, each as a **2x context image** with the eval
    # region centered and marked by a 1-px white frame just outside it.
    written: List[str] = []
    sizes: List[int] = []
    skipped_no_context = 0
    for s in ordered_stems:
        if len(written) >= n_crops:
            break
        cand, raw_uint8 = per_image_best[s]
        H, W = raw_uint8.shape

        # Compute the context bounds: 2x eval size, centered on the eval region.
        context_top = cand.top - cand.h // 2
        context_left = cand.left - cand.w // 2
        context_h = cand.h * 2
        context_w = cand.w * 2

        # Only stage if a *centered* 2x context fits inside the source image.
        # Otherwise the eval region wouldn't be at the convention-defined
        # center of the saved image and the downstream evaluator would
        # mis-locate it. (CC mining already biases toward image interior so
        # this skip rarely fires in practice.)
        if (context_top < 0 or context_left < 0 or
                context_top + context_h > H or context_left + context_w > W):
            skipped_no_context += 1
            continue

        context_crop = raw_uint8[
            context_top:context_top + context_h,
            context_left:context_left + context_w,
        ]
        # Filename describes the eval region on the *source* image. The
        # saved image is the 2x context crop (eval region is the centered
        # cand.h × cand.w block within it). Downstream tools (the 10x10
        # grid builder, the length-comparison script) need both the eval
        # region size (from filename) and the context-vs-eval mapping
        # (convention: eval region is centered, context is 2x).
        prov = MiniProv(stem=cand.stem, top=cand.top, left=cand.left, h=cand.h, w=cand.w)
        out_path = os.path.join(cfg.MINI_TILES_FOR_EVAL_DIR, f"{prov.base()}.jpg")
        Image.fromarray(context_crop).save(out_path)
        written.append(out_path)
        sizes.append(cand.h)

    if len(written) < n_crops:
        msg = (f"⚠️  staged only {len(written)} crops (asked for {n_crops}). "
               f"Eligible pool ran out of polymer-CC-containing images")
        if skipped_no_context:
            msg += (f"; also skipped {skipped_no_context} CCs too close to the "
                    f"image edge for a centered 2x context to fit")
        print(msg + ".")
    else:
        print(f"✅ staged {len(written)} eval crops in {cfg.MINI_TILES_FOR_EVAL_DIR}.")

    import collections
    by_well_post = collections.Counter(well_of(stem_of(p)) for p in written)
    if by_well_post:
        breakdown = ", ".join(f"{w}={by_well_post[w]}" for w in sorted(by_well_post))
        print(f"   well distribution : {breakdown}")
    if sizes:
        print(f"   eval region (px)  : min={min(sizes)}  med={int(np.median(sizes))}  "
              f"max={max(sizes)}  mean={int(np.mean(sizes))}")
        print(f"   image saved (px)  : 2x of the eval region (eval region "
              f"centered inside the saved image).")
    return written


def mine_mini_crops(
    val_pairs: Sequence[Tuple[str, str]],
    n_stage: Optional[int] = None,
    target_class: Optional[int] = None,
    refresh_disk_pred: bool = True,
) -> MiniRoundReport:
    """End-to-end mini-crops mining round.

    Parameters
    ----------
    val_pairs : used to pick the inference checkpoint in kfold mode.
    n_stage : how many crops to stage (default ``cfg.MINI_CROP_BATCH_SIZE``).
    target_class : single class to focus on (default
        ``cfg.MINI_CROP_TARGET_CLASS``).
    refresh_disk_pred : when True (default), the argmax we compute during
        scoring also gets written to ``CorrectionPool/PRED_<stem>.h5``, so
        the *next* mining round's disk-PRED filter is current without a
        separate §3.0 pass. Turn off if you want to preserve a manually
        chosen PRED state.
    """
    cfg.ensure_dirs()
    n_stage = n_stage if n_stage is not None else cfg.MINI_CROP_BATCH_SIZE
    target_class = target_class if target_class is not None else cfg.MINI_CROP_TARGET_CLASS

    n_swept = _sweep_stale_to_skipped()
    n_cleaned = _cleanup_painted()
    print(f"🧹 stale -> _skipped: {n_swept}   painted -> cleaned: {n_cleaned}")

    init, boot = labeled_stems()
    blocked = set(init)
    if cfg.MINING_EXCLUDE_LABELED_STEMS:
        blocked |= boot
    raws = [p for p in list_raw(cfg.CORRECTION_POOL_DIR) if stem_of(p) not in blocked]

    best_ckpt = best_ckpt_for_inference(val_pairs)
    best_model = load_unet(best_ckpt, device=cfg.DEVICE).eval()
    fold_ckpts: Optional[List[str]] = None
    if cfg.FOLD_MODE == "kfold":
        fold_ckpts = discover_fold_checkpoints()
        if len(fold_ckpts) < 2:
            fold_ckpts = None

    all_cands: List[MiniCandidate] = []
    raw_cache: Dict[str, np.ndarray] = {}
    pred_cache: Dict[str, np.ndarray] = {}
    n_inferred = 0
    for jpg in tqdm(raws, desc=f"Score (target={target_class})"):
        cands, raw_np, pred_np = _candidates_from_pool_image(
            jpg, best_model, fold_ckpts, target_class, refresh_disk_pred,
        )
        if raw_np is not None:
            n_inferred += 1
            raw_cache[stem_of(jpg)] = raw_np
            pred_cache[stem_of(jpg)] = pred_np
        all_cands.extend(cands)
    del best_model

    existing = existing_mini_bboxes_by_stem()
    picks = _pick_top_k(all_cands, n_stage, existing)

    for p in tqdm(picks, desc="Stage"):
        _stage_one(p, raw_cache[p.stem], pred_cache[p.stem])

    return MiniRoundReport(
        n_swept_stale=n_swept,
        n_cleaned_painted=n_cleaned,
        n_eligible=len(raws),
        n_inferred=n_inferred,
        n_candidates=len(all_cands),
        n_staged=len(picks),
        best_ckpt=best_ckpt,
        picks=picks,
    )
