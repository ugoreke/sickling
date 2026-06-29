"""Training loop — always from scratch, kfold or single (BEST_FOLD only).

ARCHITECTURE.md §6: every retrain starts from a freshly initialised UNet; no
warm-start, no fold sample reweighting. The split is the same in both modes
(``KFold(n_splits=N_FOLDS, random_state=42)``); single mode just skips all
folds except ``BEST_FOLD``.

Validation is **always** the first ``cfg.TRUTH_VAL_COUNT`` files in
``InitialLabels``. Per-class dice over ``cfg.TARGET_CLASSES`` (or ``cfg.BOOSTED_CLASSES`` if target is empty)
drives checkpoint selection. The remaining InitialLabels are the held-out
test set used by ``metrics.append_iteration_row`` to log the trajectory.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset, DataLoader

from .config import cfg
from .datasets import MicroscopyDataset, TileDataset
from .inference import predict_probs_tta
from .losses import CompositeLoss, build_class_weights
from .masks import load_bootstrap_label, load_ilastik_mask
from .metrics import (
    append_iteration_row,
    confusion_matrix,
    per_class_dice,
    polymer_monitor,
    target_recall_precision,
)
from .model import UNet, build_model  # noqa: F401  (UNet retained for type/back-compat)
from .paths import list_h5, stem_of
from .splits import (
    build_bootstrap_pairs,
    build_tile_pairs,
    build_train_pool,
    build_val_test_pairs,
)


FilePair = Tuple[str, str]


# ---- val pass ---------------------------------------------------------------

@torch.no_grad()
def _val_pass(model: nn.Module, val_pairs: Sequence[FilePair], tta: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Run one model over val_pairs. Returns (avg_per_class_dice, summed_confusion)."""
    from PIL import Image
    from .masks import normalize_image
    from .inference import predict_probs

    model.eval()
    dice_sum = np.zeros(cfg.N_CLASSES, dtype=np.float64)
    cm = np.zeros((cfg.N_CLASSES, cfg.N_CLASSES), dtype=np.int64)
    for raw_p, mask_p in val_pairs:
        img = np.array(Image.open(raw_p).convert("L"), dtype=np.float32)
        img_t = torch.from_numpy(normalize_image(img)).float().unsqueeze(0).to(cfg.DEVICE)
        probs = predict_probs_tta(model, img_t) if tta else predict_probs(model, img_t)
        pred = torch.argmax(probs, dim=0)
        gt = torch.from_numpy(load_bootstrap_label(mask_p).astype(np.int64)).to(pred.device)
        dice_sum += per_class_dice(pred, gt)
        cm += confusion_matrix(pred, gt)
    return dice_sum / max(len(val_pairs), 1), cm


def _target_metric(per_class: np.ndarray) -> float:
    classes = [c for c in cfg.TARGET_CLASSES if c < cfg.N_CLASSES]
    if not classes:
        classes = [c for c in cfg.BOOSTED_CLASSES.keys() if c < cfg.N_CLASSES]
    if not classes:
        return float(per_class.mean())
    return float(np.mean([per_class[c] for c in classes]))


# ---- train one fold ---------------------------------------------------------

def _build_train_loader(
    whole_pairs: Sequence[FilePair],
    tile_pairs: Sequence[FilePair],
    indices: Optional[Sequence[int]] = None,
) -> Optional[DataLoader]:
    """Indices reference the *concatenated* (whole + tile) pool. None = all."""
    n_whole = len(whole_pairs)
    n_total = n_whole + len(tile_pairs)
    if n_total == 0:
        return None
    idx_set = set(indices) if indices is not None else set(range(n_total))

    sel_whole = [whole_pairs[i] for i in range(n_whole) if i in idx_set]
    sel_tile = [tile_pairs[i - n_whole] for i in range(n_whole, n_total) if i in idx_set]

    parts = []
    if sel_whole:
        parts.append(MicroscopyDataset(sel_whole, is_train=True, mask_loader=load_bootstrap_label))
    if sel_tile:
        parts.append(TileDataset(sel_tile, mask_loader=load_ilastik_mask))
    if not parts:
        return None
    ds = parts[0] if len(parts) == 1 else ConcatDataset(parts)
    return DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=True)


def train_one_fold(
    fold_1based: int,
    train_indices: Sequence[int],
    whole_pairs: Sequence[FilePair],
    tile_pairs: Sequence[FilePair],
    val_pairs: Sequence[FilePair],
    loop: int,
    epochs: Optional[int] = None,
    tta_val: bool = True,
) -> Tuple[str, float, np.ndarray]:
    """Train one fold from scratch. Returns (ckpt_path, best_target, last_dice)."""
    epochs = epochs or cfg.EPOCHS
    cfg.ensure_dirs()

    train_loader = _build_train_loader(whole_pairs, tile_pairs, train_indices)
    if train_loader is None:
        raise RuntimeError("No training data after splitting — refusing to train an empty fold.")

    device = cfg.DEVICE
    model = build_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
    loss_fn = CompositeLoss(build_class_weights(device)).to(device)

    ckpt_path = cfg.fold_ckpt_path(fold_1based, loop)
    best_target = -1.0
    last_dice = np.zeros(cfg.N_CLASSES, dtype=np.float64)

    for ep in range(epochs):
        model.train()
        running = 0.0
        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()
            running += loss.item()

        last_dice, _ = _val_pass(model, val_pairs, tta=tta_val)
        target = _target_metric(last_dice)

        if (ep + 1) % 5 == 0:
            cls = " ".join(f"c{c}={last_dice[c]:.3f}" for c in range(cfg.N_CLASSES))
            print(f"  fold {fold_1based} ep {ep+1}/{epochs}  loss={running/len(train_loader):.4f}  {cls}  target={target:.3f}")

        if target > best_target:
            best_target = target
            torch.save(model.state_dict(), ckpt_path)

    return ckpt_path, best_target, last_dice


# ---- driver -----------------------------------------------------------------

@dataclass
class TrainReport:
    folds: List[Tuple[int, str, float, np.ndarray]]   # (fold, ckpt, best_target, last_dice)
    val_pairs: List[FilePair]
    test_pairs: List[FilePair]
    n_whole: int
    n_tiles: int


def run_training(
    folds: Optional[Sequence[int]] = None,
    epochs: Optional[int] = None,
    barred_stems: Optional[set] = None,
    tta_val: bool = True,
) -> TrainReport:
    """Train one or more folds from scratch on the current training pool.

    ``folds`` defaults to:

    * **single mode** — the fold that won the most recent kfold loop (read
      from ``iteration_log.csv``), or ``cfg.BEST_FOLD`` if the log has no
      kfold rows yet. So after a kfold round where fold 1 won, switching
      ``FOLD_MODE`` to ``'single'`` automatically trains fold 1 next.
      Override by passing ``folds=[N]`` explicitly.
    * **kfold mode** — ``[1..N_FOLDS]``.

    The same KFold seed is used in both modes so single-mode and the
    corresponding fold of kfold-mode see identical training subsets.
    """
    if folds is None:
        if cfg.FOLD_MODE == "single":
            from .metrics import latest_best_fold
            carried = latest_best_fold(default=cfg.BEST_FOLD)
            if carried != cfg.BEST_FOLD:
                print(f"🔁 Single mode: carrying over best fold {carried} "
                      f"from the latest kfold loop (cfg.BEST_FOLD={cfg.BEST_FOLD}).")
            folds = [carried]
        else:
            folds = list(range(1, cfg.N_FOLDS + 1))

    val_pairs, test_pairs = build_val_test_pairs()
    if not val_pairs:
        raise RuntimeError(f"No InitialLabels found in {cfg.INITIAL_LABELS_DIR}.")

    whole_pairs, tile_pairs = build_train_pool(barred_stems=barred_stems)
    pool_size = len(whole_pairs) + len(tile_pairs)
    if pool_size == 0:
        raise RuntimeError("Empty training pool (no BootstrappedLabels and no CorrectedTiles).")

    train_loop = cfg.next_loop_index()

    print(f"📦 Train pool: {len(whole_pairs)} whole images + {len(tile_pairs)} tiles = {pool_size}")
    print(f"🔬 Val: {len(val_pairs)} | Test: {len(test_pairs)} (from InitialLabels)")
    print(f"🎯 FOLD_MODE={cfg.FOLD_MODE}  folds_to_train={list(folds)}  -> writing loop {train_loop}")

    kf = KFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=42)
    splits = list(kf.split(range(pool_size)))

    results: List[Tuple[int, str, float, np.ndarray]] = []
    for f in folds:
        train_idx, _ = splits[f - 1]
        print(f"\n{'='*16} Fold {f}/{cfg.N_FOLDS} (train size {len(train_idx)}) {'='*16}")
        ckpt, best, last = train_one_fold(
            fold_1based=f,
            train_indices=train_idx,
            whole_pairs=whole_pairs,
            tile_pairs=tile_pairs,
            val_pairs=val_pairs,
            loop=train_loop,
            epochs=epochs,
            tta_val=tta_val,
        )
        print(f"✅ Fold {f} best target dice = {best:.4f}  ckpt -> {ckpt}")
        results.append((f, ckpt, best, last))

    _log_iteration(results, val_pairs)

    return TrainReport(
        folds=results,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        n_whole=len(whole_pairs),
        n_tiles=len(tile_pairs),
    )


# ---- trajectory writeback ---------------------------------------------------

def _log_iteration(
    fold_results: Sequence[Tuple[int, str, float, np.ndarray]],
    val_pairs: Sequence[FilePair],
) -> None:
    """Append a single trajectory row averaging across the trained folds."""
    if not fold_results:
        return
    last_dices = np.stack([d for _, _, _, d in fold_results], axis=0)
    mean_dice = last_dices.mean(axis=0)

    # Pull confusion + polymer monitor from the best fold's checkpoint.
    best_fold, best_ckpt, _, _ = max(fold_results, key=lambda r: r[2])
    # Use the project's load helper so the right backbone is built from the
    # ckpt filename (UNet for legacy files, SMP for new-backbone runs).
    from .inference import load_unet
    model = load_unet(best_ckpt, device=cfg.DEVICE)
    _, cm = _val_pass(model, val_pairs, tta=True)
    # Log recall/precision for every class — not just TARGET_CLASSES — so
    # historical trajectories survive a change of target focus.
    tr = target_recall_precision(cm, list(range(cfg.N_CLASSES)))

    # Polymer monitor on BootstrappedLabels.
    poly_recalls: List[float] = []
    poly_precs: List[float] = []
    boot = build_bootstrap_pairs()
    if boot:
        from PIL import Image
        from .masks import normalize_image
        from .inference import predict_probs
        model.eval()
        with torch.no_grad():
            for raw_p, mask_p in boot:
                img = np.array(Image.open(raw_p).convert("L"), dtype=np.float32)
                img_t = torch.from_numpy(normalize_image(img)).float().unsqueeze(0).to(cfg.DEVICE)
                pred = torch.argmax(predict_probs(model, img_t), dim=0)
                gt = torch.from_numpy(load_bootstrap_label(mask_p).astype(np.int64)).to(pred.device)
                r, p = polymer_monitor(pred, gt)
                if not np.isnan(r): poly_recalls.append(r)
                if not np.isnan(p): poly_precs.append(p)
    del model

    row = {
        "fold_mode": cfg.FOLD_MODE,
        "trained_folds": ",".join(str(f) for f, _, _, _ in fold_results),
        "best_fold": best_fold,
        "n_corrected_tiles": len(list_h5(cfg.CORRECTED_TILES_DIR)),
    }
    for c in range(cfg.N_CLASSES):
        row[f"val_dice_class_{c}"] = float(mean_dice[c])
    for c, rp in tr.items():
        row[f"val_recall_class_{c}"] = float(rp["recall"])
        row[f"val_precision_class_{c}"] = float(rp["precision"])
    row["boot_polymer_recall_mean"] = float(np.mean(poly_recalls)) if poly_recalls else float("nan")
    row["boot_polymer_precision_mean"] = float(np.mean(poly_precs)) if poly_precs else float("nan")

    append_iteration_row(row)
