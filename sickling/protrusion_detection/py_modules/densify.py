"""One-time repair: densify positive-only corrected tiles.

When a correction round is painted with **only** the target class (e.g. polymer)
and every other pixel is left unannotated, the tiles carry no negative signal:
training sees "fire polymer here" and nothing that says "do not fire polymer
there", so the model over-fires the target. See ARCHITECTURE.md §5/§12 for the
sparse-label convention this works around.

This tool reconstructs a *dense* label for each such tile:

  - the human-painted **target** pixels are kept verbatim (100 % human), and
  - every other pixel is filled with a **clean** model's prediction over the
    NON-target classes only. The target class is removed from the model's
    argmax, so the model can never invent target labels — the one class we do
    not trust it on stays entirely human. This is what stops the densification
    from reinforcing the model's own target bias.

Reads ``PRED_<stem>__y..._x...h5`` (positive-only ilastik exports) in
``CorrectedTiles``, writes dense ``<stem>__y..._x..._labels.h5`` files, and
moves each source into a backup subfolder so the step is fully reversible.

Run once after a positive-only round, before retraining:

    python -m sickling.protrusion_detection.densify            # uses loop 0 (clean) model
    python -m sickling.protrusion_detection.densify --dry-run   # report only, write nothing
"""

from __future__ import annotations

import argparse
import os
import shutil
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from .config import cfg
from .inference import load_unet, predict_probs, predict_probs_tta
from .masks import load_ilastik_mask, normalize_image, save_ilastik_mask
from .paths import (
    list_h5,
    list_raw,
    parse_tile_filename,
    stem_of,
    tile_label_path,
)


def _raw_crop_for(label_path: str, raw_by_key: dict) -> Optional[str]:
    prov = parse_tile_filename(label_path)
    if prov is None:
        return None
    return raw_by_key.get(f"{prov.stem}__y{prov.top}_x{prov.left}")


def densify_corrected_tiles(
    loop: int = 0,
    target_classes: Optional[Sequence[int]] = None,
    backup_subdir: str = "_pre_densify_backup",
    tta: bool = True,
    dry_run: bool = False,
) -> List[str]:
    """Densify every positive-only tile in ``CorrectedTiles``.

    Parameters
    ----------
    loop : checkpoint generation to fill non-target pixels from. Default 0 —
        the clean pre-correction model, which never saw the positive-only
        tiles and so carries none of the over-firing bias.
    target_classes : classes kept 100 % human (defaults to ``cfg.TARGET_CLASSES``).
    backup_subdir : source files are moved here (under ``CorrectedTiles``) after
        a dense label is written — reversible.
    tta : 8-fold TTA when predicting non-target classes (better label quality).
    dry_run : report what would happen; write/move nothing.

    Returns the list of dense label paths written (or that would be written).
    """
    target_classes = list(target_classes) if target_classes is not None else list(cfg.TARGET_CLASSES)
    tile_dir = cfg.CORRECTED_TILES_DIR
    backup_dir = os.path.join(tile_dir, backup_subdir)

    # Source files = every .h5 directly in CorrectedTiles that is NOT already a
    # dense "_labels.h5" we wrote. list_h5 is non-recursive, so the backup
    # subfolder is never seen here.
    sources = [p for p in list_h5(tile_dir) if not os.path.basename(p).endswith("_labels.h5")]

    raw_by_key = {stem_of(p): p for p in list_raw(tile_dir)}

    # Densify is decoupled from training backbone — always uses the configured
    # DENSIFY_BACKBONE/FOLD/LOOP (default: unet/2/0, the user's clean baseline)
    # so SMP experiments can still densify against the original clean UNet.
    ckpt = cfg.fold_ckpt_path(cfg.DENSIFY_FOLD, loop, backbone=cfg.DENSIFY_BACKBONE)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"Clean fill model not found: {ckpt}. "
            f"Densify needs a {cfg.DENSIFY_BACKBONE} loop-{loop} fold-{cfg.DENSIFY_FOLD} "
            f"checkpoint to fill non-target pixels."
        )

    print(f"🧱 Densify: {len(sources)} positive-only tiles  | target_classes={target_classes}")
    print(f"   fill model: {os.path.basename(ckpt)}  (TTA={tta})")
    if dry_run:
        print("   DRY RUN — nothing will be written or moved.")

    device = cfg.DEVICE
    model = load_unet(ckpt, device=device).eval()
    predict = predict_probs_tta if tta else predict_probs

    written: List[str] = []
    if not dry_run:
        os.makedirs(backup_dir, exist_ok=True)

    for src in sources:
        prov = parse_tile_filename(src)
        if prov is None:
            print(f"⚠️  {os.path.basename(src)} has no provenance — skipping.")
            continue
        raw_p = _raw_crop_for(src, raw_by_key)
        if raw_p is None:
            print(f"⚠️  {os.path.basename(src)} has no raw crop alongside — skipping.")
            continue

        human = load_ilastik_mask(src)  # 0..N-1 valid, 255 = unannotated
        painted = int(np.isin(human, target_classes).sum())

        img = np.array(Image.open(raw_p).convert("L"), dtype=np.float32)
        img_t = torch.from_numpy(normalize_image(img)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            probs = predict(model, img_t).cpu().numpy()  # (N, H, W), >= 0

        # Argmax over NON-target classes only: the fill model never emits a
        # target label, so the target class stays 100 % human.
        masked = probs.copy()
        for tc in target_classes:
            if tc < masked.shape[0]:
                masked[tc] = -1.0
        dense = masked.argmax(axis=0).astype(np.int64)

        # Stamp the human target paint on top.
        for tc in target_classes:
            dense[human == tc] = tc

        out_path = tile_label_path(prov.stem, prov.top, prov.left, dest_dir=tile_dir)
        present = sorted(int(c) for c in np.unique(dense))
        print(
            f"   {os.path.basename(src)}  painted_target={painted}px  "
            f"-> classes {present}  -> {os.path.basename(out_path)}"
        )
        if not dry_run:
            save_ilastik_mask(out_path, dense)
            # Move the positive-only source out of the training folder.
            shutil.move(src, os.path.join(backup_dir, os.path.basename(src)))
        written.append(out_path)

    del model
    print(f"✅ Densified {len(written)} tiles." + ("" if dry_run else f"  Sources backed up to {backup_dir}"))
    return written


# ---- Mini-crops variant ---------------------------------------------------
#
# Mini-crops in MiniTilesCorrected/ are painted polymer-only labels saved by
# the user (filename ``<base>_labels.h5``). They share the positive-only-tile
# problem — only target pixels carry signal, everything else is "ignore" —
# but we don't want to overwrite the user's paint. Instead, write a dense
# sibling ``<base>_dense.h5`` whose target pixels are 100% the user's paint
# and whose non-target pixels come from a clean (loop_0) model. Training
# reads ``_dense.h5`` exclusively. Idempotent + cached: a dense sibling is
# only rebuilt when the painted label is newer.


def densify_mini_crops_pending(
    loop: Optional[int] = None,
    target_classes: Optional[Sequence[int]] = None,
    tta: bool = False,
) -> List[str]:
    """Build or refresh dense siblings for every painted mini-crop.

    Scans ``cfg.MINI_TILES_CORRECTED_DIR`` for ``<base>_labels.h5`` files;
    for each one whose ``<base>_dense.h5`` sibling is missing or older than
    the painted label, runs the clean fill model and writes a fresh dense
    sibling. Cheap to re-run — no work for crops that are already up to date.

    Returns the list of dense paths that exist after the call (so
    ``build_train_pool`` can just consume them).
    """
    target_classes = list(target_classes) if target_classes is not None else list(cfg.TARGET_CLASSES)
    loop = loop if loop is not None else cfg.DENSIFY_LOOP

    src_dir = cfg.MINI_TILES_CORRECTED_DIR
    if not os.path.isdir(src_dir):
        return []

    # Local imports keep this module independent of minicrops when not used.
    from .minicrops import parse_mini_filename, mini_raw_path

    painted: List[str] = []
    dense_for: dict[str, str] = {}
    for fn in os.listdir(src_dir):
        if not fn.endswith("_labels.h5"):
            continue
        # Skip our own outputs (defensive — shouldn't be possible with the
        # naming, but better safe than crashing).
        if fn.endswith("_dense.h5"):
            continue
        full = os.path.join(src_dir, fn)
        base = full[: -len("_labels.h5")]
        dense = base + "_dense.h5"
        painted.append(full)
        dense_for[full] = dense

    pending: List[tuple[str, str]] = []
    for lp in painted:
        dp = dense_for[lp]
        if os.path.exists(dp) and os.path.getmtime(dp) >= os.path.getmtime(lp):
            continue   # cached fresh
        pending.append((lp, dp))

    if not pending:
        # nothing to do — return all known dense paths
        return [dense_for[lp] for lp in painted if os.path.exists(dense_for[lp])]

    # Resolve the fill model. Pinned to cfg.DENSIFY_BACKBONE/FOLD/LOOP so
    # this stays the clean baseline even when MODEL_BACKBONE is switched to
    # SMP. If the exact file is missing (deleted by hand), walk back across
    # loops for the same backbone-and-fold. The fill model never sees
    # target-class outputs (we mask them in argmax below) so it can't
    # reinforce target bias regardless of what we end up loading.
    from .inference import latest_available_fold_ckpt
    ckpt = cfg.fold_ckpt_path(cfg.DENSIFY_FOLD, loop, backbone=cfg.DENSIFY_BACKBONE)
    if not os.path.exists(ckpt):
        fallback = latest_available_fold_ckpt(cfg.DENSIFY_FOLD, backbone=cfg.DENSIFY_BACKBONE)
        if fallback is None:
            raise FileNotFoundError(
                f"Mini-crops densify needs a fill model and none was found. "
                f"Tried {ckpt!r}; no {cfg.DENSIFY_BACKBONE} fold-{cfg.DENSIFY_FOLD} "
                f"checkpoint exists at any loop."
            )
        print(f"⚠️  mini-crops densify: loop-{loop} ckpt missing, "
              f"falling back to {os.path.basename(fallback)}")
        ckpt = fallback

    model = load_unet(ckpt, device=cfg.DEVICE).eval()
    predict = predict_probs_tta if tta else predict_probs

    print(f"🧱 Mini-crops densify: {len(pending)} pending  |  fill={os.path.basename(ckpt)}  TTA={tta}")
    for lp, dp in pending:
        prov = parse_mini_filename(lp)
        if prov is None:
            print(f"⚠️  no provenance for {os.path.basename(lp)}, skipping")
            continue
        # Raw crop is usually next to the label in MiniTilesCorrected (user
        # may have moved it there); fall back to the staging folder where it
        # was originally written.
        raw_p = mini_raw_path(prov, in_corrected=True)
        if not os.path.exists(raw_p):
            raw_p = mini_raw_path(prov, in_corrected=False)
        if not os.path.exists(raw_p):
            print(f"⚠️  no raw crop for {os.path.basename(lp)} (looked in MiniTilesCorrected/ then MiniTilesToBeCorrected/), skipping")
            continue

        img = np.array(Image.open(raw_p).convert("L"), dtype=np.float32)
        img_t = torch.from_numpy(normalize_image(img)).float().unsqueeze(0).to(cfg.DEVICE)
        # Mini-crops (64–192 px) are smaller than cfg.TILE_SIZE (256), so
        # predict_probs' reflect-pad would error. Run sliding-window with a
        # tile_size capped to the crop's smaller dimension instead — that's
        # one window covering the whole crop, no padding needed.
        ts = int(min(img_t.shape[-2], img_t.shape[-1], cfg.TILE_SIZE))
        with torch.no_grad():
            probs = predict(model, img_t, tile_size=ts).cpu().numpy()
        masked = probs.copy()
        for tc in target_classes:
            if tc < masked.shape[0]:
                masked[tc] = -1.0   # exclude target from argmax → model never invents it
        dense = masked.argmax(axis=0).astype(np.int64)

        # Overlay human target paint.
        human = load_ilastik_mask(lp)
        for tc in target_classes:
            dense[human == tc] = tc

        save_ilastik_mask(dp, dense)

    del model
    return [dense_for[lp] for lp in painted if os.path.exists(dense_for[lp])]


def _main() -> None:
    ap = argparse.ArgumentParser(description="Densify positive-only corrected tiles.")
    ap.add_argument("--loop", type=int, default=0, help="Clean model loop index to fill non-target pixels (default 0).")
    ap.add_argument("--no-tta", action="store_true", help="Disable 8-fold TTA on the fill model.")
    ap.add_argument("--dry-run", action="store_true", help="Report only; write/move nothing.")
    ap.add_argument("--mini", action="store_true", help="Densify mini-crops (MiniTilesCorrected) instead of the 512-px tile workflow.")
    args = ap.parse_args()
    if args.mini:
        densify_mini_crops_pending(loop=args.loop, tta=not args.no_tta)
    else:
        densify_corrected_tiles(loop=args.loop, tta=not args.no_tta, dry_run=args.dry_run)


if __name__ == "__main__":
    _main()
