"""Train / val / test split construction.

Centralised so the leakage barrier (ARCHITECTURE.md §8) lives in one place.

- ``build_val_test_pairs`` splits ``InitialLabels`` into the first
  ``cfg.TRUTH_VAL_COUNT`` files for per-epoch checkpoint selection and the
  remainder for Panel B/C final test.
- ``build_train_pool`` returns the whole-image and tile file pairs to train
  on. When ``cfg.PROMOTE_TILES_TO_VAL`` is True the caller supplies a list of
  source stems used for val/test; any whole image or tile sharing one of
  those stems is excluded from training.
"""

from __future__ import annotations

import glob
import os
from typing import List, Optional, Sequence, Set, Tuple

from .config import cfg
from .paths import (
    list_h5,
    list_raw,
    parse_tile_filename,
    raw_jpg,
    stem_of,
)


FilePair = Tuple[str, str]


def _stem_of_label(label_path: str) -> str:
    """``<stem>_labels.h5`` -> ``<stem>``. Falls back to plain stem_of."""
    s = stem_of(label_path)
    return s[: -len("_labels")] if s.endswith("_labels") else s


def _pair_raw_label(label_paths: Sequence[str], raw_dir: str) -> List[FilePair]:
    """For each label, find its raw .jpg in ``raw_dir``."""
    by_stem = {stem_of(p): p for p in list_raw(raw_dir)}
    pairs: List[FilePair] = []
    for lp in label_paths:
        stem = _stem_of_label(lp)
        if stem in by_stem:
            pairs.append((by_stem[stem], lp))
        else:
            print(f"⚠️  {os.path.basename(lp)} has no raw image (stem {stem!r}) — skipping.")
    return pairs


def build_val_test_pairs() -> Tuple[List[FilePair], List[FilePair]]:
    """Split InitialLabels into (val, test).

    If ``cfg.VAL_STEMS`` is non-empty, those exact source stems form val and
    every other ``InitialLabels`` file is test. Otherwise val is the first
    ``cfg.TRUTH_VAL_COUNT`` files by sorted name and test is the remainder
    (legacy behavior).
    """
    label_paths = list_h5(cfg.INITIAL_LABELS_DIR)
    pairs = _pair_raw_label(label_paths, cfg.CORRECTION_POOL_DIR)

    if cfg.VAL_STEMS:
        wanted = set(cfg.VAL_STEMS)
        present = {stem_of(raw) for raw, _ in pairs}
        unknown = wanted - present
        if unknown:
            print(f"⚠️  cfg.VAL_STEMS contains stems not found in InitialLabels: "
                  f"{sorted(unknown)} (ignored).")
        val = [p for p in pairs if stem_of(p[0]) in wanted]
        test = [p for p in pairs if stem_of(p[0]) not in wanted]
        return val, test

    n_val = min(cfg.TRUTH_VAL_COUNT, len(pairs))
    return pairs[:n_val], pairs[n_val:]


def build_bootstrap_pairs() -> List[FilePair]:
    """Whole-image (raw, dense_label) pairs in BootstrappedLabels."""
    label_paths = list_h5(cfg.BOOTSTRAP_LABELS_DIR)
    return _pair_raw_label(label_paths, cfg.CORRECTION_POOL_DIR)


def build_tile_pairs(barred_stems: Optional[Set[str]] = None) -> List[FilePair]:
    """All (raw_crop, label_crop) pairs in ``CorrectedTiles``.

    Tiles are stored as ``<stem>__y<top>_x<left>.jpg`` (raw) and
    ``<stem>__y<top>_x<left>_labels.h5`` (painted). If ``barred_stems`` is
    given, any tile whose provenance stem is in the set is dropped (leakage
    barrier).
    """
    barred = barred_stems or set()
    label_paths = list_h5(cfg.CORRECTED_TILES_DIR)
    raw_paths = {stem_of(p): p for p in list_raw(cfg.CORRECTED_TILES_DIR)}

    out: List[FilePair] = []
    for lp in label_paths:
        # Only genuine painted labels train. A corrected tile label is
        # "<stem>__y<top>_x<left>_labels.h5"; the PRED_*.h5 crops shipped into
        # TilesToBeCorrected are *targets to fix*, never labels. If a PRED file
        # is left in CorrectedTiles, parse_tile_filename would strip its
        # "PRED_" prefix and silently train on it (positive-only over-firing),
        # so reject anything that isn't a "_labels.h5" file outright.
        name = os.path.basename(lp)
        if name.startswith("PRED_") or not name.endswith("_labels.h5"):
            print(f"⚠️  {name} is not a *_labels.h5 tile — skipping (not a training label).")
            continue
        # Label name is "<stem>__y<top>_x<left>_labels.h5"
        # Raw name is   "<stem>__y<top>_x<left>.jpg"
        prov = parse_tile_filename(lp)
        if prov is None:
            print(f"⚠️  {name} has no parseable provenance — skipping.")
            continue
        if prov.stem in barred:
            continue
        raw_key = f"{prov.stem}__y{prov.top}_x{prov.left}"
        rp = raw_paths.get(raw_key)
        if rp is None:
            print(f"⚠️  Tile {os.path.basename(lp)} has no raw crop alongside — skipping.")
            continue
        out.append((rp, lp))
    return out


def build_mini_tile_pairs(barred_stems: Optional[Set[str]] = None) -> List[FilePair]:
    """``(raw_crop, dense_label)`` pairs for every painted mini-crop.

    Each painted ``MiniTilesCorrected/<base>_labels.h5`` is densified once
    via ``densify.densify_mini_crops_pending`` (clean model fills non-target
    pixels; the human's target paint stays verbatim) and trained against the
    resulting ``<base>_dense.h5`` sibling. ``load_ilastik_mask`` reads them
    the same way it reads densified 512-px tiles, so the existing dataset
    plumbing handles them without changes.
    """
    barred = barred_stems or set()
    # Local import dodges any chance of densify <-> splits cycles at module load.
    from .densify import densify_mini_crops_pending
    from .minicrops import parse_mini_filename, mini_raw_path

    try:
        dense_paths = densify_mini_crops_pending()
    except FileNotFoundError as e:
        # No fill model — surface the cause and treat as "no mini-crops" so
        # training doesn't fail just because mini-crops aren't wired up.
        print(f"⚠️  skipping mini-crops in train pool ({e})")
        return []

    out: List[FilePair] = []
    for dp in dense_paths:
        # Recover the base label path to drive parsing + raw lookup.
        base = dp[: -len("_dense.h5")] if dp.endswith("_dense.h5") else dp
        label_for_parse = base + "_labels.h5"
        prov = parse_mini_filename(label_for_parse)
        if prov is None:
            print(f"⚠️  no provenance for {os.path.basename(dp)} — skipping mini-crop.")
            continue
        if prov.stem in barred:
            continue
        raw = mini_raw_path(prov, in_corrected=True)
        if not os.path.exists(raw):
            raw = mini_raw_path(prov, in_corrected=False)
        if not os.path.exists(raw):
            print(f"⚠️  no raw crop for {os.path.basename(dp)} — skipping mini-crop.")
            continue
        out.append((raw, dp))
    return out


def build_train_pool(barred_stems: Optional[Set[str]] = None) -> Tuple[List[FilePair], List[FilePair]]:
    """Return (whole_image_pairs, tile_pairs) for training.

    Whole-image pairs come from ``BootstrappedLabels``; tile pairs come from
    ``CorrectedTiles`` **plus densified mini-crops** from
    ``MiniTilesCorrected``. Mini-crops are auto-densified on demand (cached
    siblings) so the user paints only the target class.

    When ``barred_stems`` is provided explicitly (e.g. when tile promotion to
    val/test is on), those stems are excluded from training.
    """
    barred = set(barred_stems or set())
    whole = [pp for pp in build_bootstrap_pairs() if stem_of(pp[0]) not in barred]
    tiles = build_tile_pairs(barred)
    minis = build_mini_tile_pairs(barred)
    return whole, tiles + minis
