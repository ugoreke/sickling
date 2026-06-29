# GUIDE.md

> How to actually run the workflow, day to day. For the *why* and the data-role
> definitions, see ARCHITECTURE.md.

---

## 0. One-time setup

- ilastik installed (used **only** for painting — its classifier is unused).
- `Config.BASE_DIR` points at the project root; the five data folders exist:
  `InitialLabels`, `BootstrappedLabels`, `CorrectionPool`, `TilesToBeCorrected`,
  `CorrectedTiles`.
- `InitialLabels` holds your 5 perfect dense masks (`<stem>_labels.h5`) and their
  raw `<stem>.jpg`.
- `CorrectionPool` holds your 354 raw `.jpg`. Predictions are generated on first
  run (see §2), so you don't need to place `PRED_*.h5` there yourself.

A note on painting tiles: open the tile's raw crop in ilastik, import the
matching `PRED_..._labels.h5` as Labels, and **only paint pixels you're sure
about**. Anything you leave untouched stays "unannotated" and is ignored in
training — partial labels are expected and fine. Save back into `CorrectedTiles`
keeping the exact filename.

---

## 1. Pick your mode

Set in `Config`:

- `RUN_MODE = 'bootstrap'` — cold start, makes dense anchor labels.
- `RUN_MODE = 'correction'` — the tile HITL loop on the pool.
- `FOLD_MODE = 'kfold'` — train 5 folds (honest metrics, slower).
- `FOLD_MODE = 'single'` — train one model (fold 2, fast turnaround).
- `TARGET_CLASSES = [0]` for polymer, `[3]` for cell boundary, `[0, 3]` for both.

---

## 2. Bootstrap mode (only needed for cold start / adding new anchor images)

1. Run Bootstrap. It first trains a throwaway generator on `InitialLabels`, then
   writes dense `PRED_<stem>.h5` for each bootstrap raw image **one by one**.
2. For each, paint a **full dense** correction in ilastik (all four classes,
   complete coverage) and save to `BootstrappedLabels` as `<stem>_labels.h5`.
3. When every image is corrected, run the final bootstrap training. It trains
   **from scratch** on `BootstrappedLabels`, validating on `InitialLabels`.

You likely already have your 11 `BootstrappedLabels`, so you only return here
when adding fresh cold-start images.

---

## 3. Correction mode (the main loop)

Each round:

1. **Predict.** Run correction mode; it writes `PRED_<stem>.h5` into
   `CorrectionPool` for any raw image missing one (using the current best model).
2. **Mine.** It ranks tiles FN-aware (soft target-prob, plus fold disagreement
   in `kfold` mode) and moves a batch of `MINING_BATCH_SIZE` crops +
   `PRED_..._labels.h5` into `TilesToBeCorrected`.
3. **Paint.** In ilastik, correct each tile in `TilesToBeCorrected`. Paint only
   what you trust; leave the rest unannotated. Save to `CorrectedTiles` with the
   same filename.
4. **Retrain.** Run training; it trains **from scratch** on
   `BootstrappedLabels` (11) + everything in `CorrectedTiles`.
5. **Repeat.** Re-running prediction/mining now re-ranks with the improved model,
   so the next batch targets the new weak spots.

Tip: start with `MINING_BATCH_SIZE` around 20–40 and adjust to your appetite per
round. Smaller batches = the model re-ranks more often = your labeling time goes
where it helps most.

---

## 4. Targeting cell boundary instead of (or alongside) polymer

Set `TARGET_CLASSES = [3]` (boundary only) or `[0, 3]` (both). Mining and the
within-tile sampler both follow this list. Everything else is identical — boundary
is treated the same as polymer (thin, under-fired). When `[0, 3]`, tiles weak in
either class surface, and you fix whatever's wrong in each crop.

---

## 5. Reading the metrics / when to stop

After training, check the per-iteration trajectory:

- **Primary:** per-class dice on `InitialLabels` (the 5).
- **Polymer monitor:** polymer-only recall/precision on the 11.
- Watch the **target class recall** climb across rounds. When it flattens and a
  round of labeling barely moves it, you've hit diminishing returns — stop or
  switch `TARGET_CLASSES` to the next weak class.

---

## 6. Promoting corrected tiles to test/val (optional, off by default)

`PROMOTE_TILES_TO_VAL = False` by default. If you turn it on, the **strict
leakage rule** fires: once any tile from source image `X` is used for val/test,
**all** tiles from `X` are barred from training. Filenames carry the source stem
and coords so this is automatic — don't rename tiles.

---

## 7. Common gotchas

- Don't drop `CorrectionPool` predictions into `BootstrappedLabels` or any
  training folder — pool predictions are **targets to fix**, never labels.
- Don't rename tiles; provenance (`__y<top>_x<left>`, source stem) is what makes
  leakage protection and de-duplication work.
- Stems contain spaces and underscores; the `__` (double underscore) before the
  coordinates is the delimiter — keep it intact.
- Every retrain is from scratch by design. If a run seems to "remember" a prior
  model, something is loading an old checkpoint — check the mode wiring.
