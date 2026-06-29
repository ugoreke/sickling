"""Bootstrap mode (ARCHITECTURE.md §6).

Cold-start helper, mostly dormant once ``BootstrappedLabels`` is populated.

Phases:

A. **Throwaway generator** — short training run from scratch on the 5
   ``InitialLabels``. No rigorous checkpointing (you can't meaningfully
   validate on 5 images). Used solely to make starting predictions for new
   cold-start raws.

B. **PRED gen for new cold-start raws** — walks ``cfg.BOOTSTRAP_RAW_DIR``
   and finds raw ``.jpg``\s whose stem is in
   ``InitialLabels ∪ BootstrappedLabels`` but lacks any dense label. For each,
   writes a whole-image ``PRED_<stem>.h5`` into ``BootstrappedLabels`` so the
   operator can paint it into a full dense mask.

   *InitialLabels stems are never eligible as a PRED-gen destination* — they
   are permanent val/test, not future training labels.

C. **Final bootstrap train** — a thin wrapper around ``train.run_training``
   that uses the current pool (``BootstrappedLabels`` + ``CorrectedTiles``,
   though tiles are normally empty during bootstrap) and validates on
   ``InitialLabels``.

In the steady state (12 BootstrappedLabels exist) phases A and B are a
no-op; only C runs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm

from .config import cfg
from .datasets import MicroscopyDataset, TileDataset
from .inference import predict_probs
from .losses import CompositeLoss, build_class_weights
from .masks import (
    load_bootstrap_label,
    load_ilastik_mask,
    normalize_image,
    save_ilastik_mask,
)
from .model import UNet, build_model  # noqa: F401
from .paths import (
    labeled_stems,
    list_raw,
    stem_of,
    whole_label_path,
)
from .splits import build_val_test_pairs
from .train import TrainReport, run_training


THROWAWAY_CKPT = "unet_throwaway_bootstrap.pth"


@dataclass
class BootstrapReport:
    throwaway_trained: bool
    n_generated_preds: int
    generated_pred_paths: List[str]
    final_train: Optional[TrainReport]


# --- Phase A: throwaway generator -------------------------------------------

def train_throwaway_generator(epochs: int = 10, include_train_pool: bool = False) -> str:
    """Minimal training on ``InitialLabels`` (returns checkpoint path).

    All ``InitialLabels`` are used as the throwaway training set (no held-out
    split — five images can't be split meaningfully). Setting
    ``include_train_pool=True`` additionally folds in the real training pool
    (``BootstrappedLabels`` + densified ``CorrectedTiles`` + mini-crops),
    which produces sharper starting PREDs at the cost of a slower throwaway
    run. Recommended once the pool has accumulated good labels (those labels
    have far more polymer signal than the 5 InitialLabels alone, so the
    starting PRED for a new bootstrap image is much closer to "correct"
    before the operator opens ilastik).
    """
    cfg.ensure_dirs()
    val_pairs, test_pairs = build_val_test_pairs()
    initial_pairs = val_pairs + test_pairs
    if not initial_pairs:
        raise RuntimeError(f"No InitialLabels found in {cfg.INITIAL_LABELS_DIR}.")

    parts = [MicroscopyDataset(initial_pairs, is_train=True, mask_loader=load_bootstrap_label)]
    summary = f"InitialLabels ({len(initial_pairs)})"
    if include_train_pool:
        from .splits import build_train_pool
        whole_pairs, tile_pairs = build_train_pool()
        if whole_pairs:
            parts.append(MicroscopyDataset(whole_pairs, is_train=True, mask_loader=load_bootstrap_label))
        if tile_pairs:
            parts.append(TileDataset(tile_pairs, mask_loader=load_ilastik_mask))
        summary += f" + BootstrappedLabels ({len(whole_pairs)}) + tiles/mini ({len(tile_pairs)})"

    print(f"🅰 Throwaway generator: {summary} for {epochs} epochs.")

    device = cfg.DEVICE
    ds = parts[0] if len(parts) == 1 else ConcatDataset(parts)
    loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=True)

    model = build_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
    loss_fn = CompositeLoss(build_class_weights(device)).to(device)

    for ep in range(epochs):
        model.train()
        running = 0.0
        for imgs, masks in loader:
            imgs = imgs.to(device); masks = masks.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(imgs), masks)
            loss.backward()
            optimizer.step()
            running += loss.item()
        if (ep + 1) % 2 == 0:
            print(f"  throwaway ep {ep+1}/{epochs}  loss={running/len(loader):.4f}")

    ckpt = os.path.join(cfg.MODELS_DIR, THROWAWAY_CKPT)
    torch.save(model.state_dict(), ckpt)
    print(f"💾 throwaway -> {ckpt}")
    return ckpt


# --- Phase B: pred-gen for new cold-start raws ------------------------------

def _stems_needing_pred(force_stems: Optional[Sequence[str]] = None) -> List[str]:
    """Bootstrap stems with no dense label anywhere (skip InitialLabels stems).

    The candidate set is normally ``raw_stems ∩ (InitialLabels ∪
    BootstrappedLabels)`` minus anything that's already labelled. That keeps
    the steady-state run from firing on every raw image in the pool.

    Pass ``force_stems`` to explicitly add raws that aren't in that
    intersection — e.g. you deleted a corrupt ``BootstrappedLabels`` entry
    and want a fresh PRED. ``InitialLabels`` stems are still refused (they
    are permanent val/test and must not become paint destinations).
    """
    init, boot = labeled_stems()
    raw_stems = {stem_of(p) for p in list_raw(cfg.BOOTSTRAP_RAW_DIR)}
    bootstrap_set = raw_stems & (init | boot)

    needs: List[str] = []
    for s in sorted(bootstrap_set):
        if s in init:
            continue   # never paint val/test stems into BootstrappedLabels
        if s in boot:
            continue   # already has a dense label
        needs.append(s)

    if force_stems:
        for s in force_stems:
            if s in init:
                print(f"⚠️  refusing to force-bootstrap InitialLabels stem {s!r} "
                      f"— that folder is permanently held out (ARCHITECTURE.md §6/§8).")
                continue
            if s not in raw_stems:
                print(f"⚠️  no raw image for forced stem {s!r} in {cfg.BOOTSTRAP_RAW_DIR}.")
                continue
            if s in boot:
                print(f"ℹ️  forced stem {s!r} already has a BootstrappedLabels entry — "
                      f"re-generating PRED will overwrite it.")
            if s not in needs:
                needs.append(s)
    return needs


def generate_bootstrap_preds(
    model: torch.nn.Module,
    tta: bool = False,
    force_stems: Optional[Sequence[str]] = None,
) -> List[str]:
    """For each stem needing a starting PRED, write it into ``BootstrappedLabels``.

    The destination is ``BootstrappedLabels/<stem>_labels.h5`` (the ilastik
    Labels format) so the operator can open the painting UI directly on it.

    ``force_stems`` is forwarded to :func:`_stems_needing_pred` so a caller
    that decided to re-bootstrap a specific stem in ``run_bootstrap`` doesn't
    silently drop that intent here.
    """
    cfg.ensure_dirs()
    os.makedirs(cfg.BOOTSTRAP_LABELS_DIR, exist_ok=True)
    needs = _stems_needing_pred(force_stems=force_stems)
    if not needs:
        print("🅑 No bootstrap stems need a starting PRED.")
        return []

    print(f"🅑 Generating starting PREDs for {len(needs)} bootstrap stem(s).")
    written: List[str] = []
    model.eval()
    for stem in tqdm(needs, desc="Bootstrap PRED-gen"):
        raw_p = os.path.join(cfg.BOOTSTRAP_RAW_DIR, f"{stem}.jpg")
        if not os.path.exists(raw_p):
            print(f"  ⚠️  No raw .jpg for {stem!r} in {cfg.BOOTSTRAP_RAW_DIR}; skipping.")
            continue
        img = np.array(Image.open(raw_p).convert("L"), dtype=np.float32)
        img_t = torch.from_numpy(normalize_image(img)).float().unsqueeze(0).to(cfg.DEVICE)
        with torch.no_grad():
            probs = predict_probs(model, img_t)
        argmax = torch.argmax(probs, dim=0).cpu().numpy().astype(np.uint8)
        out = whole_label_path(stem, cfg.BOOTSTRAP_LABELS_DIR)
        save_ilastik_mask(out, argmax)
        written.append(out)
    return written


# --- Phase C: final bootstrap train -----------------------------------------

def train_final_bootstrap(**kw) -> Optional[TrainReport]:
    """Thin wrapper around :func:`train.run_training`.

    Refuses with a clear explanation if the training pool is empty — i.e.
    you're at cold start with no ``BootstrappedLabels`` painted yet (and no
    ``CorrectedTiles`` / mini-crops either). In that case the **throwaway
    generator from Phase A is your current model** — by design, the final
    train doesn't fire until at least one bootstrap label has been painted.
    ``InitialLabels`` stays held out (ARCHITECTURE.md §12).
    """
    from .splits import build_train_pool
    whole, tiles = build_train_pool()
    if not whole and not tiles:
        print("🅒 Empty training pool — no BootstrappedLabels, no CorrectedTiles, no mini-crops.")
        print("   At cold start the throwaway generator from Phase A IS your current model.")
        print("   Paint at least one PRED into BootstrappedLabels (or mine + paint tiles), "
              "then re-run with do_final_train=True.")
        print("   InitialLabels is permanently held out as val/test by design "
              "(ARCHITECTURE.md §12) — it never becomes a final-train pool.")
        return None
    print("🅒 Final bootstrap train: from scratch on the current pool.")
    return run_training(**kw)


# --- Orchestrator ------------------------------------------------------------

def run_bootstrap(
    throwaway_epochs: int = 10,
    do_final_train: bool = True,
    force_stems: Optional[Sequence[str]] = None,
    include_train_pool: bool = False,
    **train_kw,
) -> BootstrapReport:
    """Full bootstrap pipeline.

    1. If any bootstrap stem needs a starting PRED (or any in ``force_stems``
       qualifies), train the throwaway generator and write the PREDs.
       Otherwise skip A and B entirely.
    2. If ``do_final_train``, kick off the final from-scratch training on
       ``BootstrappedLabels`` (+ tiles, normally empty during bootstrap).
       Refuses with a clear message if the pool is empty (see
       :func:`train_final_bootstrap`).

    Parameters
    ----------
    force_stems : Optional[Sequence[str]]
        Source stems to unconditionally include in Phase B's PRED generation,
        even if they aren't currently in ``BootstrappedLabels``. Use this to
        re-bootstrap a label you deleted (e.g. the user spotted a wrong-image
        bug in ``BootstrappedLabels/D16_03_1_14_Bright Field_001_labels.h5``,
        removed the file, and now wants a fresh PRED). ``InitialLabels`` stems
        are still refused.
    include_train_pool : bool
        When True, the Phase-A throwaway also trains on the current
        ``BootstrappedLabels`` + ``CorrectedTiles`` + mini-crops pool, not
        just ``InitialLabels``. Slower (multi-minute throwaway) but yields
        a much sharper starting PRED than 5 InitialLabels alone — typically
        worth it once a healthy pool exists.
    """
    needs = _stems_needing_pred(force_stems=force_stems)
    throwaway_trained = False
    generated: List[str] = []
    if needs:
        ckpt = train_throwaway_generator(
            epochs=throwaway_epochs,
            include_train_pool=include_train_pool,
        )
        # The throwaway ckpt name doesn't carry a backbone tag, but the
        # currently-active backbone is what was just trained — so build that.
        model = build_model().to(cfg.DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=cfg.DEVICE))
        generated = generate_bootstrap_preds(model, force_stems=force_stems)
        del model
        throwaway_trained = True
        print(f"⏸  Stop here, paint {len(generated)} new mask(s) in ilastik, save back to "
              f"BootstrappedLabels, then re-run with do_final_train=True.")
        if not do_final_train:
            return BootstrapReport(throwaway_trained, len(generated), generated, None)

    final = train_final_bootstrap(**train_kw) if do_final_train else None
    return BootstrapReport(throwaway_trained, len(generated), generated, final)
