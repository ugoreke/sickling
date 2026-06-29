# CHANGELOG.md

> Append a dated entry whenever behavior, data layout, configs, or decisions
> change. Newest on top. Keep entries terse; the *why* lives in ARCHITECTURE.md
> §12.

---

## [0.2.0] — 2026-06-11

Plateau-buster release: mini-crop fast-labelling workflow + ceiling-test
hooks for the val_dice flatline. Most of the new infrastructure exists so
the operator can decide whether the bottleneck is GT noise, label volume,
or model capacity — not to "automatically improve" anything.

### Added
- **Loop-versioned checkpoints.** Filenames now
  `<backbone>_fold_<f>_best_loop_<N>.pth`. Each retrain writes a fresh loop
  index (`cfg.next_loop_index`) instead of overwriting; inference reads the
  latest (`cfg.latest_loop_index`). Existing `unet_fold_*_best_loop_*.pth`
  files keep working out of the box.
- **K-fold → single carryover.** Switching `FOLD_MODE` to `'single'` after
  a kfold round now auto-trains the fold that won the last kfold loop
  (`latest_best_fold()` reads `best_fold` from the most recent
  `iteration_log.csv` row). Fallback chain in `best_ckpt_for_inference`
  walks back across loops when a top-of-stack ckpt is missing, then tries
  `cfg.BEST_FOLD`, then any fold.
- **SMP model factory** (`sickling.protrusion_detection.model.build_model`) with the
  `cfg.MODEL_BACKBONE` switch: `'unet'` (default, ~4M params) |
  `'smp_unet_efficientnet-b0'` (~6M) | `'smp_unet_efficientnet-b7'` (~67M).
  ImageNet-pretrained encoders, 1-channel input. Checkpoints carry the
  backbone tag so backbones coexist across loops without colliding.
- **Densify positive-only tiles** (`sickling.protrusion_detection.densify`,
  `python -m sickling.protrusion_detection.densify`). One-time repair for rounds painted
  positive-only: the human target paint is kept verbatim, non-target pixels
  are filled by a clean fill model (default `cfg.DENSIFY_BACKBONE`/`FOLD`/
  `LOOP` = `unet`/2/0). Sources backed up under `_pre_densify_backup/`.
- **Mini-crops mining + workflow** (`sickling.protrusion_detection.minicrops`). Adaptive
  small crops sized to predicted target-class connected components
  (bbox + `MINI_CROP_PADDING`, clamped to `[MINI_CROP_MIN, MINI_CROP_MAX]`),
  ranked by soft-prob + disagreement. Pre-filters by disk PRED to skip
  inference on empty pool images. New folders `MiniTilesToBeCorrected/`,
  `MiniTilesCorrected/`; filename `<stem>__y<top>_x<left>_h<h>_w<w>.<ext>`
  carries the explicit crop size. Skip semantics: leave unpainted; next
  mining call sweeps stale crops (older than the latest checkpoint) to
  `MiniTilesToBeCorrected/_skipped/`. Painted crops auto-cleaned from the
  staging folder.
- **Densify mini-crops at retrain time**
  (`densify.densify_mini_crops_pending`). Each painted polymer-only
  `<base>_labels.h5` in `MiniTilesCorrected/` gets a cached `<base>_dense.h5`
  sibling on demand. Idempotent + mtime-based; `build_train_pool` calls it
  automatically so mini-crops never reintroduce the loop-1 positive-only-tile
  over-firing.
- **Full pool PRED refresh** (`correction.regenerate_pool_preds` + §3.0
  notebook cell). On-demand cell that rewrites every eligible pool PRED
  against the current best model; the per-round §3 mining stays cheap.
- **Visual overlays** (`sickling.protrusion_detection.viz` + §5.0 notebook cell).
  Renders raw-plus-class-overlay PNGs that mirror the source layout under
  `cfg.VIZ_DIR/<source>/`. Covers InitialLabels, BootstrappedLabels,
  CorrectedTiles, TilesToBeCorrected, and CorrectionPool PREDs with the same
  color map across folders (polymer = red, body = dark teal, boundary =
  green, bg/ignore = raw shows through). Pool sweep capped at
  `pool_pred_max_n` (default 100) to keep wall-clock sane.
- **TP / FP trajectory plot** (`metrics.plot_tp_fp_trajectory`,
  `metrics/tp_fp_trajectory.png`). Per-target-class TP rate (recall) and
  FP fraction (`1 − precision`) over `n_corrected_tiles`. Per-class
  recall/precision now logged for **every** class (not just
  `TARGET_CLASSES`) so historical rows survive a target switch.

### Changed
- **`build_tile_pairs` hardened.** Only files ending in `_labels.h5`
  contribute to training; stray `PRED_*.h5` left in `CorrectedTiles` are
  rejected (root cause of the loop-1 polymer over-firing: positive-only
  `PRED_*.h5` were silently picked up by `parse_tile_filename` stripping
  the `PRED_` prefix).
- `cfg.fold_ckpt_path`, `cfg.latest_loop_index`, `cfg.next_loop_index`,
  `discover_fold_checkpoints`, and `latest_available_fold_ckpt` all accept
  an optional `backbone` argument and default to `cfg.MODEL_BACKBONE` so
  UNet and SMP runs coexist in the same `models/` folder.
- `_log_iteration` records `val_recall_class_<c>` / `val_precision_class_<c>`
  for **all** classes (was: only `cfg.TARGET_CLASSES`).
- Pipeline notebook adds §3.0 (full pool PRED refresh), §3.1 (mini-crops
  mining), and §5.0 (rebuild overlays). §1 surfaces the new knobs
  (`MODEL_BACKBONE`, `MINI_CROP_*`, `DENSIFY_*`) in the printed sanity line.

### Fixed
- **`InitialLabels/D16_03_1_17_Bright Field_001_labels.h5`** had every
  class value shifted by +1 (values `{1,2,3,4}` instead of `{0,1,2,3}`),
  so polymer mapped to "bg" and boundary fell out of the class range
  entirely. Fixed in place; original backed up to
  `InitialLabels/_label_fix_backup/`. Mean dice on this image against the
  loop-3 fold-1 ckpt went from `0.016 → 0.880`. The file is in the **test**
  split, so this does *not* explain the val_dice plateau, but it was
  silently destroying the held-out test signal whenever someone looked at
  it.
- **`run_bootstrap`'s PRED-gen candidate set required the stem to already
  be in `(InitialLabels ∪ BootstrappedLabels)`** — so deleting a corrupt
  bootstrap label (e.g. `D16_03_1_14_Bright Field_001` pointed at the wrong
  source) silently dropped the stem off the candidate list, and Phase B
  did nothing instead of re-generating a fresh PRED. Added `force_stems`
  argument to `run_bootstrap()` / `_stems_needing_pred()`: stems in the
  list bypass the intersection requirement (still subject to "has a raw
  in `BOOTSTRAP_RAW_DIR`" and "not an `InitialLabels` stem").
  `generate_bootstrap_preds()` now also accepts and honors `force_stems` —
  the initial fix routed it through `run_bootstrap` and `_stems_needing_pred`
  but not through `generate_bootstrap_preds`, which re-computed `needs`
  internally and got an empty list, so `n_generated_preds` stayed 0 even
  though the throwaway trained.
- **`generate_bootstrap_preds` writes 1-based ilastik files** (so the operator
  can render and paint the starting PRED in ilastik — class 0 = polymer
  would otherwise be invisible) **but `BootstrappedLabels` is read with
  `load_dense_mask` which assumed 0-based** (matching the legacy hand-curated
  files). The two conventions met when `force_stems` actually fired and
  produced a `D16_03_1_14_Bright Field_001_labels.h5` with values
  `{1, 2, 3, 4}` — the same +1 shift that wrecked
  `InitialLabels/D16_03_1_17` earlier. Fixed with a new smart loader
  `load_bootstrap_label` in `masks.py` that inspects each file's value
  distribution and shifts on the fly: legacy 0-based files (have `255` or
  `max < N_CLASSES`) pass through; new 1-based files (`max == N_CLASSES`,
  no `255`) get shifted to 0-based with raw 0 → `IGNORE_INDEX`. Swapped
  every `BootstrappedLabels`/`InitialLabels` read across `bootstrap`,
  `train`, `inference`, and `viz` to the smart loader. The operator no
  longer needs a manual conversion step after painting.
- **Throwaway can now optionally train on the full pool**
  (`include_train_pool=True` on `run_bootstrap` and
  `train_throwaway_generator`). When set, the Phase-A throwaway folds in
  `BootstrappedLabels` + `CorrectedTiles` + densified mini-crops on top of
  `InitialLabels`. Slower, but the starting PRED is much sharper — useful
  when re-bootstrapping a single image once the pool already has good
  polymer signal.
- **Sealed held-out eval set on mini-crops**
  (`minicrops.stage_eval_minicrops`,
  `minicrops.evaluate_on_eval_set`, new §6 of the notebook). One crop per
  random pool image goes into `MiniTilesForEval/`; the operator dense-
  labels all four classes from scratch (no PRED on disk for these). At
  retrain time, `_log_iteration` auto-aggregates per-class
  TP/FP/FN/recall/precision across the union of crops and appends
  `eval_n_crops`, `eval_n_polymer_present`, `eval_recall_class_<c>`,
  `eval_precision_class_<c>`, `eval_total_{tp,fp,fn}_class_0` to
  `iteration_log.csv`. The intent is "robust polymer recall estimate
  across the pool distribution" — answers "is FN still high?" with
  statistical power that the 2-image InitialLabels test surface can't
  match. **Sealed leakage both directions:** staging skips any source
  stem in `InitialLabels`, `BootstrappedLabels`, or `CorrectedTiles ∪
  MiniTilesCorrected`; `build_train_pool` automatically bars every source
  stem present in `MiniTilesForEval/`, so once a crop is staged for eval,
  no other crop from that source image can enter training.
- **Well-balanced sampling + per-well metrics on the eval set**
  (`paths.well_of`, `stage_eval_minicrops(balance_by_well=True)` default,
  per-well columns appended by `evaluate_on_eval_set`). Source stems
  encode a physical well in their first three characters (e.g. `D16`,
  `G21`, `H20`); each well is a different physical sample location, so
  crossing well boundaries is the real generalization test. Staging now
  picks crops round-robin across wells so the first 100 crops cover every
  eligible well at least once before any well gets a second crop —
  critical when one well dominates the pool numerically (e.g. for the
  current data, D16=148 stems out of 301 eligible). `evaluate_on_eval_set`
  reports per-well polymer recall/precision plus the worst- and best-well
  summary so the iteration log captures a pool-wide aggregate AND the
  worst-well figure that the aggregate hides.
- **Eval crops are now polymer-centered + adaptive size** (v2 of the v1
  fixed-size random staging). v1 sampled a fixed 192-px crop at a random
  position inside each pool image, then filtered "image has any polymer
  in its disk PRED" — but a random crop in a polymer-containing image
  rarely lands ON the polymer, so the staged crops were mostly empty and
  painting them taught the recall measurement nothing. v2 centers every
  crop on a predicted-polymer connected component (same distribution as
  the §3.1 training crops), with adaptive sizing
  (bbox + `MINI_EVAL_CROP_PADDING`, clamped to `[MINI_EVAL_CROP_MIN,
  MINI_EVAL_CROP_MAX]`). Defaults intentionally smaller than training
  (`MIN=64`, `MAX=128`, `PADDING=24`) so each eval crop paints in
  30–60 s. Config knobs renamed accordingly:
  removed `MINI_EVAL_CROP_SIZE`, added `MINI_EVAL_CROP_MIN`,
  `MINI_EVAL_CROP_MAX`, `MINI_EVAL_CROP_PADDING`.
- **Eval crops now include 2x context + a 1-px white frame** marking
  the eval region (v3 of the staging). The saved image on disk is twice
  the eval region on each side; the eval region itself sits at the
  center, with a thin white rectangle just outside its outer boundary so
  the operator (and visually-comparing reader) can see exactly what
  region is graded. The 1-px frame is geometrically and tonally distinct
  from polymer fibers (thin curving dark structures), so the model's
  predictions near the boundary aren't biased by mistaking it for
  polymer. Solves "cells / polymer near the eval-region edge are
  contextless and the model can't judge them properly" — the model now
  has at least one full eval-region worth of surrounding context on
  every side, while the metric region itself is identical to v2.
  Evaluator auto-detects v2 (image == eval region) vs v3 (image == 2x,
  eval region centered) at load time so mixing is safe.
- **`train_final_bootstrap` now refuses cleanly when the train pool is
  empty** instead of crashing with `Empty training pool`. Prints a clear
  message explaining that at cold start the **throwaway generator from
  Phase A is the current model** — by design `InitialLabels` stays held
  out, so Phase C is gated on at least one `BootstrappedLabels`/tile being
  painted.

### Config keys (added since the body above)
- **`VAL_STEMS: List[str] = []`** — when non-empty, those exact stems form
  val (rest of `InitialLabels` = test). Default behavior unchanged (first
  `TRUTH_VAL_COUNT` sorted = val). Lets you rebalance val toward
  polymer-heavy images instead of taking what `sorted()` happens to give
  you.
- **`MINI_TILES_FOR_EVAL_DIR`** (default `<BASE_DIR>/MiniTilesForEval`),
  **`MINI_EVAL_N_CROPS=100`** — the held-out mini-crop eval set knobs.
  **`MINI_EVAL_CROP_MIN=64`**, **`MINI_EVAL_CROP_MAX=128`**,
  **`MINI_EVAL_CROP_PADDING=24`** — adaptive size clamp + bbox padding
  for the polymer-centered eval crops (replaces the v1 fixed-size knob).

### Config keys (added)
- **Backbone:** `MODEL_BACKBONE='unet'`, `DENSIFY_BACKBONE='unet'`,
  `DENSIFY_FOLD=2`, `DENSIFY_LOOP=0`.
- **Mini-crops:** `MINI_TILES_TODO_DIR`, `MINI_TILES_CORRECTED_DIR`,
  `MINI_CROP_TARGET_CLASS=0`, `MINI_CROP_PADDING=32`, `MINI_CROP_MIN=64`,
  `MINI_CROP_MAX=192`, `MINI_CROP_BATCH_SIZE=200`,
  `MINI_CROP_MIN_CC_AREA=3`, `MINI_CROP_DUP_IOU=0.25`.
- **Default `PRED_BATCH_SIZE`** bumped from 50 to 150 (notebook override
  matches).

### Decisions (see ARCHITECTURE.md §12)
- **Densify-on-retrain over sparse positive-only training.** Painted-only
  tiles carry zero negative gradient; class weights/Tversky tuning can't
  manufacture the missing signal. Filling non-target pixels from the clean
  baseline model (never from the current model) supplies negatives without
  feeding the model its own predictions on the target class.
- **Backbone tagged in the checkpoint filename.** Lets the user run a UNet
  vs SMP ceiling test in the same `models/` folder without filename
  collisions. The fallback chain works across backbones (the loader parses
  the tag).
- **Densify uses its own backbone/fold/loop knob.** Decoupled from
  `MODEL_BACKBONE` so an SMP experiment can still reuse the original clean
  UNet for non-target fills.
- **Mini-crop "skip" = leave unpainted; sweep is mtime-vs-latest-ckpt.**
  No explicit UI signal needed; "I closed it without saving" + "the next
  retrain landed" is sufficient evidence the operator triaged this crop.
- **Mini-crops centered on predicted target CCs, not on
  below-threshold/uncertain regions.** Trades off some FN coverage for
  simplicity; if recall stalls on a class the model genuinely can't see,
  add a complementary FN-aware crop stream later.

---

## [0.1.0] — 2026-06-09

HITL refactor. Replaces `training_2.ipynb` as the entry point with a thin
orchestration notebook (`pipeline.ipynb`) on top of a `polymer_detection`
package.

### Added
- `polymer_detection` package: `config`, `paths`, `masks`, `model`, `losses`,
  `sampler`, `datasets`, `inference`, `mining`, `splits`, `metrics`, `train`,
  `bootstrap`, `correction`. Thin notebook in `pipeline.ipynb`.
- Two run modes: `bootstrap` (dense, whole-image, cold start) and `correction`
  (sparse, tiled HITL loop), each with `FOLD_MODE` ∈ {`kfold`, `single`}.
- FN-aware tile mining (`MINING_SCORE='softprob+disagreement'`,
  `MINING_LAMBDA=1.0`), selection-stage only. Across-fold standard deviation
  of target probability as the disagreement signal; collapses to soft-prob in
  single mode.
- Tile correction workflow: `CorrectionPool` → `TilesToBeCorrected` →
  `CorrectedTiles`, with `__y<top>_x<left>` provenance in filenames and an
  IoU > 0.25 duplicate-crop guard against staged/corrected tiles.
- `PRED_BATCH_SIZE=50` rotation per correction round (missing-PRED first,
  then oldest by mtime) — keeps wall-clock manageable on a laptop without
  freshness-staleness across retrains.
- Per-iteration trajectory log: `metrics/iteration_log.csv` + `trajectory.png`,
  indexed by `len(CorrectedTiles)` at training time. Polymer-only recall /
  precision monitor on `BootstrappedLabels` (binary per image, averaged).
- Strict whole-image leakage barrier when `PROMOTE_TILES_TO_VAL=True`.
- Folder roles: `InitialLabels`, `BootstrappedLabels`, `CorrectionPool`,
  `TilesToBeCorrected`, `CorrectedTiles`, plus `BOOTSTRAP_RAW_DIR`
  (defaults to `CorrectionPool`), `MODELS_DIR`, `METRICS_DIR`.

### Changed
- `CLASS0_TARGET` / `CLASS0_CROP_PROB` generalised to `TARGET_CLASSES` /
  `TARGET_CROP_PROB`. With multiple target classes the sampler picks one
  weighted by **inverse pixel frequency** in the image.
- Training pool is explicitly `BootstrappedLabels` (whole-image) +
  `CorrectedTiles` (sparse tiles, sub-tiled 512 → 256). Pool predictions in
  `CorrectionPool` are never auto-used as training labels.
- `InitialLabels` val/test split kept as `TRUTH_VAL_COUNT=2` + remainder for
  test (instead of "all 5 as val"); the held-out test feeds Panel B/C.
- In `kfold` correction mode, PRED export uses the **best single fold** by
  target-class val dice (not a 5-fold ensemble) — chosen for wall-clock
  reasons; disagreement scoring still uses every fold's probability map.
- Mask I/O explicit: `load_dense_mask` (already 0-based + 255-ignore on disk:
  `InitialLabels`, `BootstrappedLabels`) vs `load_ilastik_mask` (1-based with
  `0`-unannotated: freshly-painted `CorrectedTiles` and PRED files we write).
- Bootstrap raw set is `BOOTSTRAP_RAW_DIR ∩ (InitialLabels ∪ BootstrappedLabels)`
  stems; PRED is generated only for stems missing a dense label anywhere,
  and `InitialLabels` stems are unconditionally excluded as a paint
  destination. Steady state ⇒ phases A and B are a no-op.

### Preserved
- U-Net architecture, weighted Dice + Tversky + directed confusion / FN
  penalties, 99th-percentile normalization, sliding-window + 8-fold TTA
  inference.

### Removed (legacy `training_2.ipynb` keys)
`RAW_DIR`, `SEG_DIR`, `RAW_SEG_DIR`, `TRUTH_DIR`, `EXPORT_DIR`, `LABEL_DIR`,
`PROJECT_FILE`, `ILASTIK_PATH`, `CLASS0_TARGET`, `CLASS0_CROP_PROB`.

### Decisions (see ARCHITECTURE.md §12)
- Still rejected: confidence-gated export, copy-paste augmentation,
  inference-threshold tuning, warm-start from previous checkpoint,
  differential / recency / fold sample weighting, loose per-tile leakage.

### Deviations from `CODE_SESSION_PROMPT.md`
- PRED rotation policy (50 images/round) added on top of "regenerate every
  round" — operator request, laptop wall-clock.
- `TARGET_CROP_PROB` sampler uses inverse-frequency class pick (not uniform)
  when multiple target classes are configured.
- `InitialLabels` keeps the 2-val / 3-test split rather than collapsing to
  "all 5 as val" — operator request, preserves Panel B/C as honest test.
- In `kfold` PRED export uses the best single fold (not an ensemble) for
  wall-clock; mining still uses every fold for disagreement.

---

## [0.0.0] — baseline (`training_2.ipynb`)

Existing notebook: Config, `MicroscopyDataset`, U-Net, weighted Dice + Tversky,
directed confusion/FN penalties, k-fold training, sliding-window inference,
ilastik-format export, held-out truth evaluation (per-class dice + confusion
matrix). Known issue: polymer (class 0) missed ~34% of the time (→ background);
cell boundary (class 3) under-fired (→ bg ~0.11, → cell body ~0.16).
