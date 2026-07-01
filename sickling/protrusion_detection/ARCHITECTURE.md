# ARCHITECTURE.md (protrusion_detection arm)

> Arm-specific design doc for the HITL UNet pixel-segmentation arm.
> For the top-level project map covering both arms (protrusion_detection +
> rbc_classification), see [`../ARCHITECTURE.md`](../ARCHITECTURE.md).
> Keep this up to date when behavior, data roles, configs, or decisions change.
> Last updated: 2026-06-12 (0.3.0: merged into sickling; painted-label
> eval surface retired in favour of cross-arm notebooks; see CHANGELOG)

---

## 1. Goal

Human-in-the-loop (HITL) active-learning pipeline to improve a semantic
segmentation model on bright-field microscopy images of sickling cells. The
default model is a **vanilla U-Net** (1 grayscale input channel, 4 output
classes); the `cfg.MODEL_BACKBONE` switch swaps it for SMP backbones
(EfficientNet-B0 / B7) without changing the rest of the pipeline. ilastik is
used **only as a painting/correction UI** — its random-forest classifier is
not used. Predictions are exported into ilastik's label format so corrections
can be painted by hand and fed back into training.

## 2. Classes (fixed order — do not reorder)

| Index | Name          | Notes |
|-------|---------------|-------|
| 0     | Protrusion       | **Primary target.** Faint, rare, under-represented. |
| 1     | Background    | (`bg`) |
| 2     | Cell body     | |
| 3     | Cell boundary | Secondary target. Thin, under-fired. |

## 3. The problem (drives every design choice)

The failure mode is **false-negative / under-representation**, not class
confusion:

- Protrusion (0) → Background (1): **34%** of polymer pixels missed. TP ≈ 62%.
- Cell boundary (3) → Background: ~0.11; → Cell body: ~0.16. TP ≈ 0.73.

Both problem classes are thin/faint structures the model *omits* rather than
*mislabels into a sibling*. Protrusion is also genuinely rare in the data, so the
root cause is "not enough examples seen," addressed by mining and labeling more
of it — not by changing the decision rule.

## 4. Data roles & folders

All under `Config.BASE_DIR`. Filename `<stem>` example:
`D16_03_1_1_Bright Field_001` (note: stems contain spaces and single
underscores — provenance uses a **double underscore** delimiter to avoid
collisions).

| Folder | Contents | Role |
|--------|----------|------|
| `InitialLabels`     | 5 handcrafted, fully-dense, all-classes-perfect masks | **Permanent test/val.** Never trained on (except the throwaway Phase-A generator, see §6). Primary checkpoint selector. |
| `BootstrappedLabels`| 11 dense masks, polymer-perfect, other classes imperfect (accepted) | **Training.** Also a secondary polymer-only monitor. |
| `CorrectionPool`    | 355 raw `.jpg` + their whole-image `PRED_<stem>.h5` predictions | **Correction queue only.** Predictions are *targets to fix*, never auto-used as training labels. |
| `TilesToBeCorrected`| Mined 512-px crops: raw tile + `PRED_<stem>__y<top>_x<left>.h5` | Staging area handed to the human to paint. |
| `CorrectedTiles`    | Human-painted tile labels: `<stem>__y<top>_x<left>_labels.h5` | **Training** (added to the 11). Partial labels OK (untouched = ignore). |
| `MiniTilesToBeCorrected` | Mined adaptive small crops (64–192 px) + matching PREDs. Fast single-class labelling. | Staging area for the mini-crops workflow (§7.5). |
| `MiniTilesCorrected`     | Polymer-only painted mini-crop labels + cached `_dense.h5` densified siblings | **Training** (auto-included via `build_train_pool`; mini-crops are densified on demand from a clean fill model). |
| `MiniTilesForEval`       | 100 polymer-centered adaptive crops with 2× context, well-balanced across wells; sealed from training | **Held-out external test surface.** Consumed by `../notebooks/protrusion_length_grid.ipynb` for the manual polymer-length test (not graded inside this arm). |
| `models/`           | Checkpoints: `<backbone>_fold_<f>_best_loop_<N>.pth` | One file per (backbone, fold, loop). Multiple backbones and loops coexist. |
| `metrics/`          | `iteration_log.csv`, `trajectory.png`, `tp_fp_trajectory.png` | Per-retrain progress; viewed via §5 of the notebook. |
| `viz/`              | Raw-plus-class-overlay PNGs mirroring the source folder layout | Quick visual sanity-check of labels and PREDs (§9). On-demand rebuild via the §5.0 cell. |

### Filename conventions

| Kind | Pattern | Example |
|------|---------|---------|
| Raw image            | `<stem>.jpg`                         | `D16_03_1_1_Bright Field_001.jpg` |
| Whole-image label    | `<stem>_labels.h5`                   | `D16_03_1_1_Bright Field_001_labels.h5` |
| Whole-image pred     | `PRED_<stem>.h5`                     | `PRED_D16_03_1_1_Bright Field_001.h5` |
| Tile pred            | `PRED_<stem>__y<top>_x<left>.h5`     | `PRED_..._001__y1234_x0567.h5` |
| Corrected tile label | `<stem>__y<top>_x<left>_labels.h5`   | `..._001__y1234_x0567_labels.h5` |
| Mini-crop raw        | `<stem>__y<top>_x<left>_h<h>_w<w>.jpg` | `..._001__y1234_x0567_h128_w128.jpg` |
| Mini-crop pred       | `PRED_<stem>__y<top>_x<left>_h<h>_w<w>.h5` | `PRED_..._001__y1234_x0567_h128_w128.h5` |
| Mini-crop label      | `<stem>__y<top>_x<left>_h<h>_w<w>_labels.h5` | `..._001__y1234_x0567_h128_w128_labels.h5` |
| Mini-crop dense (cached) | `<stem>__y<top>_x<left>_h<h>_w<w>_dense.h5` | densified sibling rebuilt by `densify.densify_mini_crops_pending` |
| Checkpoint           | `<backbone>_fold_<f>_best_loop_<N>.pth` | `unet_fold_2_best_loop_3.pth`, `smp_unet_efficientnet-b0_fold_1_best_loop_4.pth` |

`<top>`/`<left>` are the crop's top-left pixel coords in the source image; for
mini-crops the explicit `<h>`/`<w>` keeps the bbox recoverable from the
filename alone (adaptive sizes). This provenance is **load-bearing**: it powers
the leakage barrier (§8), the duplicate-crop guard, and the mini-crop dedup
across staged/painted/skipped folders. Mini-crops live in their own folders so
the 512-px parser (`parse_tile_filename`) does not see them and never mistakes
one for the other.

## 5. Label convention

There are two storage flavours on disk; the in-memory training convention is
always **`0..N-1`** valid + **`255`** = `IGNORE_INDEX`. The loader picks the
right reader per folder:

1. **Training-ready** (`InitialLabels`, `BootstrappedLabels`). Mixed
   storage — early hand-curated files are 0-based + 255 ignore; newer files
   produced by `generate_bootstrap_preds` are 1-based ilastik exports (so
   ilastik can render and paint them). Loaded by
   `masks.load_bootstrap_label`, which auto-detects:

   - any pixel == 255                       → 0-based, return as-is.
   - max non-ignore value < `N_CLASSES`     → 0-based, return as-is.
   - else                                   → 1-based, subtract one and map
     raw 0 → `IGNORE_INDEX`.

2. **Ilastik round-trip** (`CorrectedTiles/*.h5` straight from ilastik,
   `MiniTilesCorrected/*_labels.h5`, and `CorrectionPool/PRED_*.h5` we write
   for ilastik to import). Always 1-based, `0` = unannotated. Loaded by
   `masks.load_ilastik_mask` which unconditionally subtracts one and marks
   raw 0 as `IGNORE_INDEX`.

The dual-storage handling for category 1 exists because the on-disk format
ilastik needs for painting (1-based, so class 0 doesn't render as
"unannotated") is incompatible with the original 0-based reader; running
through the smart loader removes the manual `subtract 1` step the operator
previously had to do after each painting round.

## 6. Run modes

A single switch selects the mode; an orthogonal switch (`FOLD_MODE`) selects
`kfold` (5-fold CV) vs `single`. In `single` mode the fold to train defaults
to **the carryover winner** — the fold with the highest `best_fold` in the
most recent `iteration_log.csv` row, or `cfg.BEST_FOLD` if the log is empty
(so a kfold round where fold 1 won is automatically picked up by the next
single retrain).

Across loops, every retrain writes a **new checkpoint generation** —
`<backbone>_fold_<f>_best_loop_<N>.pth` — instead of overwriting the previous
one. Training writes `cfg.next_loop_index(backbone)`; inference reads
`cfg.latest_loop_index(backbone)`; `latest_available_fold_ckpt` walks back
across loops if a top-of-stack file is missing. Backbones (UNet, SMP-B0,
SMP-B7) coexist in the same `models/` folder because the tag is part of the
filename.

### Bootstrap mode (cold start — the factory that produces `BootstrappedLabels`)
1. Train a **throwaway** generator from scratch on `InitialLabels` (the 5).
   Minimal training, no rigorous checkpointing (5 images can't be split
   meaningfully) — it only needs to make usable starting predictions. At
   cold start, **the throwaway IS your current model** until a final train
   becomes feasible. Pass `include_train_pool=True` to also fold the current
   `BootstrappedLabels` + `CorrectedTiles` + mini-crops into the throwaway —
   slower but the starting PRED is much sharper, which is the right tradeoff
   once the pool has accumulated good polymer signal (e.g. re-bootstrapping
   one image mid-project).
2. For each raw image in the bootstrap input set lacking a `BootstrappedLabels`
   entry, generate a **dense, whole-image** `PRED_<stem>.h5` for one-by-one
   human correction.
3. Human corrects each to full dense coverage → `BootstrappedLabels`.
4. Once at least one `BootstrappedLabels` entry exists, train the **final**
   bootstrap model **from scratch** on `BootstrappedLabels` (plus
   `CorrectedTiles` / mini-crops if any have accumulated), with
   `InitialLabels` as test/val.

Bootstrap is intentionally whole-image + dense: anchor images warrant the
effort, and full coverage is what makes them good anchors.

`BOOTSTRAP_RAW_DIR` defaults to `CorrectionPool` (the canonical raw-`.jpg`
store in this layout). The candidate stem set is `raw_dir ∩ (InitialLabels ∪
BootstrappedLabels)`; PRED generation only fires for stems in that set that
lack a dense label anywhere, and `InitialLabels` stems are unconditionally
excluded as a paint destination. Pass `force_stems=[...]` to
`run_bootstrap()` to bypass the intersection rule — useful when you've
deleted a corrupt label and want a fresh PRED for an image whose stem is
no longer in the candidate set. `InitialLabels` stems are still refused as
paint destinations even with `force_stems`. In the steady state phases A
and B are a no-op and `run_bootstrap()` is equivalent to "train from scratch
on the current pool".

Phase C refuses with a helpful message (not a crash) when the train pool
is empty — i.e. no `BootstrappedLabels`, no `CorrectedTiles`, no mini-crops.
At cold start that's the expected state and the throwaway from Phase A is
your model; `InitialLabels` does not become a final-train pool by design
(§12 "permanently held out").

### Correction mode (the tile HITL loop on `CorrectionPool`)
1. **(Optional) Full pool PRED refresh** (§3.0 cell,
   `correction.regenerate_pool_preds`) — run after a retrain to bring every
   eligible pool image's PRED up to date with the current best model. Skip
   in rounds where staleness is acceptable.
2. **Predict:** select `PRED_BATCH_SIZE` (default 150) pool images — missing
   PRED first, then oldest by mtime — and (re)generate their `PRED_<stem>.h5`
   with the current best model. (Predicting all 355 every round is slow;
   step 1 covers the "refresh everything" case explicitly.)
3. **Mine:** FN-aware tile proposal (§7) over those `PRED_BATCH_SIZE` images'
   probability maps. Rank globally, take `MINING_BATCH_SIZE` with the IoU
   duplicate-crop guard.
4. **Stage:** copy the chosen 512-px raw crops + their tile-sized PRED label
   crops into `TilesToBeCorrected`. Stems in `InitialLabels` (always) and
   `BootstrappedLabels` (when `MINING_EXCLUDE_LABELED_STEMS=True`) are skipped.
5. **Human paints** in ilastik — only trusted pixels; everything else stays
   unannotated/ignore. Saves to `CorrectedTiles`.
6. **Retrain from scratch** on `BootstrappedLabels` + `CorrectedTiles` (+
   densified mini-crops if any — see §7.5). The new checkpoint is written
   as a fresh loop generation; prior loops are preserved.
7. Re-rank with the improved model, repeat.

The "current best model" used at steps 1-3 is resolved by
`best_ckpt_for_inference`: single mode → the latest available checkpoint of
the carryover fold (with fallback chain across loops/folds); kfold mode →
the fold with the highest target-class dice on the `InitialLabels` val split
at the current loop. Disagreement scoring uses every fold checkpoint that
exists at the current loop for the current backbone.

Correction is intentionally tiled + sparse: small crops solve the
"image too big → sloppy labels" problem and keep label quality high.

### Mini-crops correction (alternative HITL loop, §7.5)

When dense 512-px tile mining plateaus and the bottleneck is the operator's
time, switch to the mini-crops workflow: many small adaptive crops focused
on a single target class. The flow is the same as steps 1-7 above but with
the §3.1 cell (`minicrops.mine_mini_crops`) and `MiniTilesToBeCorrected/` →
`MiniTilesCorrected/` instead of the 512-px folders. The operator paints
only the target class; densification on retrain (§7.5) fills in
non-target pixels from the clean fill model.

## 7. Mining (FN-aware, selection-stage only)

Because the model **omits** the target, **do not rank by where the model
predicts the target** — that ranking is blind to exactly the missed pixels.
Instead rank tiles by:
- **Soft target probability** — tiles where target-class prob is non-trivial
  even when argmax says background; plus/or
- **Fold disagreement** (query-by-committee) when `FOLD_MODE == kfold`. In
  `single` mode this is unavailable, so the score falls back to soft-prob /
  entropy only (weaker ranking, still functional).

Mining happens **only at the selection stage** (which tiles the human labels).
It is **not** a loss weight and does **not** reweight samples — training runs
uniformly; the pool simply ends up target-heavy because that is what we
proposed. Within-tile class balance is handled by the existing
target-aware crop sampler (`TARGET_CROP_PROB`, generalized from the old
`CLASS0_CROP_PROB`).

`TARGET_CLASSES` (config list) selects what to mine: `[0]` polymer, `[3]`
boundary, or `[0, 3]` both. The same list drives the within-tile sampler. The
whole loop is class-agnostic — boundary behaves like polymer (thin, under-fired)
and benefits from the same treatment. While correcting a tile mined for one
class, obvious errors in any other class are free to fix (partial labels).

## 7.5 Mini-crops mining (single-class, adaptive-bbox)

Once dense 512-px tile labelling plateaus, mini-crops trade label
completeness for **labels per minute**. Each mini-crop is sized to fit a
predicted target-class connected component plus a small context margin
(bbox + `MINI_CROP_PADDING`, clamped to `[MINI_CROP_MIN, MINI_CROP_MAX]`),
ranked by the same FN-aware score (soft-prob + disagreement) over the CC's
pixels. The operator paints only `MINI_CROP_TARGET_CLASS` in each crop;
the densify-on-retrain step (§7.6) replaces the missing negative signal so
this never reintroduces the loop-1 positive-only-tile over-firing.

Disk-PRED pre-filter — mini-crops mining skips inference on any pool image
whose current `PRED_<stem>.h5` has zero target-class pixels (saves the
forward pass on negatives). The argmax computed during scoring is also
written back to disk, so the next mining round's pre-filter is current
without an extra full pool refresh.

Skip-by-leave-unpainted — at the start of every mining round, any staged
mini-crop with mtime older than the latest checkpoint is moved to
`MiniTilesToBeCorrected/_skipped/`. "Skip" needs no UI signal: closing the
crop without saving + a retrain happening is sufficient evidence the
operator triaged this one. Painted crops that have a matching
`*_labels.h5` in `MiniTilesCorrected/` are auto-cleaned from the staging
folder at the same point.

## 7.6 Densify on retrain (positive-only fix)

When a round is painted with only the target class (e.g. only polymer in
each crop), the resulting `_labels.h5` files carry no negative signal:
training sees "fire polymer here" and nothing that says "do not fire
polymer there", so the model over-fires the target on the pool. The fix —
applied automatically in `build_train_pool` for mini-crops, and on demand
via `python -m sickling.protrusion_detection.densify` for the 512-px tiles — fills
non-target pixels from a **clean** fill model (default
`cfg.DENSIFY_BACKBONE/FOLD/LOOP = unet/2/0`, the user's pre-correction
baseline). The fill model's target-class probabilities are masked out of
the argmax, so the model cannot invent target labels and there is no
self-training bias loop. The operator's target paint is kept verbatim.

The densified mini-crop labels are cached as `<base>_dense.h5` siblings;
they are only rebuilt when the painted `_labels.h5` is newer. This makes
re-running `run_training()` cheap when only a few mini-crops have changed.

## 8. Leakage rule (strict)

Once **any** tile from source image `X` is promoted to val/test, the **entire**
image `X` is barred from training. Implemented by parsing the source `<stem>`
from tile provenance and excluding all tiles sharing that stem from the training
pool. The looser "non-overlapping tiles from the same image in both sets"
option was **rejected** — context bleeds across a 512 px boundary for faint
signals, inviting subtle leakage. Tile promotion to val/test is **off by
default** (`PROMOTE_TILES_TO_VAL = False`).

## 9. Evaluation & metrics

- **Primary checkpoint selector:** per-class dice on the `InitialLabels`
  **val** split. By default that's the first `cfg.TRUTH_VAL_COUNT=2` files
  by sorted name; the remaining files form the held-out **test** split.
  Set `cfg.VAL_STEMS` to an explicit list of source stems to override the
  sorted-first rule — used to keep val polymer-balanced (sorted-first can
  land you on the polymer-light end purely by alphanumeric accident).
  Test = whatever's left in `InitialLabels` after val. Both val and test
  are used only for Panel B/C figures and for the iteration log; test
  metrics never enter per-epoch checkpoint selection.
- **Per-iteration tracking** across the HITL loop: per-class dice on val,
  per-class recall/precision (now for **every** class, not just
  `TARGET_CLASSES`), confusion matrix. Plotted as a trajectory
  (`metrics/trajectory.png`) indexed by `len(CorrectedTiles)` at training
  time so the operator can spot diminishing returns and stop. Raw rows in
  `metrics/iteration_log.csv`.
- **TP / FP trajectory** (`metrics/tp_fp_trajectory.png`,
  `plot_tp_fp_trajectory`). Per-target-class TP rate (recall) + FP fraction
  (`1 − precision`) vs `n_corrected_tiles`. Lower FP + higher TP = the
  desired direction; flat TP across rounds = diminishing returns. FP
  fraction is used (not FPR vs true negatives) because for sparse classes
  like polymer the negative count dominates FPR and it barely moves.
- **Secondary monitor:** polymer-only recall/precision on `BootstrappedLabels`
  — binary (polymer vs not-polymer) per image, averaged across the set.
  Trustworthy for polymer even though their other classes are imperfect.
  Soft monitor only; never the checkpoint selector.
- **Visual overlays under `viz/`** (`sickling.protrusion_detection.viz`). Raw-plus-
  class-overlay PNGs that mirror the source layout; same color map across
  every folder (polymer = red, body = dark teal, boundary = green, bg /
  ignore = raw shows through). Sources covered: InitialLabels,
  BootstrappedLabels, CorrectedTiles, TilesToBeCorrected, and a capped
  subset of CorrectionPool PREDs. Used to triage labels before sinking
  more time into them (GT bugs like the `D16_03_1_17` +1 shift are eye-
  visible) and to spot-check what the model is doing on the pool.
- **Polymer-centered eval crops under `MiniTilesForEval/`**
  (`minicrops.stage_eval_minicrops`). One adaptive crop per pool source
  image (bbox + `MINI_EVAL_CROP_PADDING`, clamped to `[MINI_EVAL_CROP_MIN,
  MINI_EVAL_CROP_MAX]`), saved as a **2× context** image with the eval
  region centered. Sampled round-robin across wells (first 3 chars of
  the stem). Not graded inside this arm — these crops are consumed by
  the cross-arm `notebooks/protrusion_length_grid.ipynb` for the manual
  polymer-length test against the model's predictions.
- Promoted corrected tiles (if enabled) are a supplementary, **noisy** monitor —
  a single small tile must not yank checkpoint selection.

### Carryover semantics for the inference checkpoint

`best_ckpt_for_inference` resolves the model used for mining and PRED
export, with a strict fallback chain so a deleted file or a fresh switch of
`FOLD_MODE` never crashes:

1. **Single mode:** carryover fold first (`latest_best_fold`); if no
   checkpoint for that fold at the latest loop, walk back across loops;
   then try `cfg.BEST_FOLD`; then any fold.
2. **Kfold mode:** re-score the current loop's fold checkpoints on val and
   pick the highest target-class dice. If the current loop has no
   checkpoints yet, fall back to the single-mode chain.

The same chain runs **per backbone** — UNet and SMP-B0 checkpoints never
get confused because the tag is part of every filename pattern.

## 10. Loss & preprocessing (preserved from `training_2.ipynb`)

- Weighted Dice + Tversky on target class(es) + directed confusion / FN
  penalties. Kept as-is; the FN bias of Tversky complements the mining strategy.
- Percentile-clip normalization (99th pct) is the single source of truth, applied
  identically at train and inference time. Do not introduce a second
  normalization path.
- Sliding-window inference with overlap; TTA used at evaluation.

## 11. Config keys (as built, 0.1.0)

Single source of truth is `sickling.protrusion_detection.config.Config`. Defaults below.

**Mode switches.** `RUN_MODE` (`bootstrap` | `correction`), `FOLD_MODE`
(`kfold` | `single`), `BEST_FOLD=2`.

**Backbone.** `MODEL_BACKBONE='unet'` — also `'smp_unet_efficientnet-b0'` or
`'smp_unet_efficientnet-b7'`. Built by `model.build_model()`; checkpoints
carry the backbone tag in the filename so backbones coexist. `DENSIFY_BACKBONE`
(`'unet'`), `DENSIFY_FOLD` (`2`), `DENSIFY_LOOP` (`0`) pin the **clean fill
model** used by the positive-only-tile densify path — decoupled from
`MODEL_BACKBONE` so SMP experiments still reuse the original clean UNet for
non-target fills.

**Folders.** `INITIAL_LABELS_DIR`, `BOOTSTRAP_LABELS_DIR`,
`CORRECTION_POOL_DIR`, `TILES_TODO_DIR`, `CORRECTED_TILES_DIR`,
`MINI_TILES_TODO_DIR`, `MINI_TILES_CORRECTED_DIR`,
`BOOTSTRAP_RAW_DIR` (defaults to `CORRECTION_POOL_DIR`), `MODELS_DIR`,
`METRICS_DIR`, `VIZ_DIR`, `EVAL_DIR`.

**Training (preserved).** `TILE_SIZE=256`, `BATCH_SIZE=16`, `EPOCHS=50`,
`STEPS_PER_EPOCH=100`, `LR=1e-4`, `N_CLASSES=4`, `IGNORE_INDEX=255`,
`N_FOLDS=5`, `NORM_PERCENTILE=99`, `TRUTH_VAL_COUNT=2` (val/test split inside
InitialLabels — only used when `VAL_STEMS` is empty). `VAL_STEMS: List[str]`
(default `[]`) overrides the sorted-first-N rule with an explicit val
list — used to balance polymer density in val. Drop `BATCH_SIZE` if you
hit OOM on `smp_unet_efficientnet-b7` (~67M params).

**Loss (preserved).** `BOOSTED_CLASSES={0: 5.0, 3: 3.0}`,
`DIRECTED_CONFUSION_PENALTY`, `DIRECTED_FN_PENALTY`, `CONFUSION_WEIGHT=0.3`,
`FN_WEIGHT=0.1`, `TVERSKY_CLASSES=[0]`, `TVERSKY_WEIGHT=0.3`,
`TVERSKY_ALPHA=0.4`, `TVERSKY_BETA=0.6`.

**Sampler (generalized).** `TARGET_CLASSES=[0]`, `TARGET_CROP_PROB=0.5`.
With multiple targets, the sampler picks a class weighted by
**inverse pixel frequency** in the image (rarer class sampled harder).

**Mining (correction mode, 512-px).** `CORRECTION_TILE_SIZE=512`,
`MINING_BATCH_SIZE=30`, `MINING_STRIDE=256` (50%-overlap candidate grid),
`MINING_DUP_IOU=0.25` (IoU dup guard against staged/corrected tiles),
`MINING_SCORE='softprob+disagreement'`, `MINING_LAMBDA=1.0` (disagreement
weight), `MINING_EXCLUDE_LABELED_STEMS=True` (skip BootstrappedLabels stems
when proposing tiles; InitialLabels stems are *always* excluded).

**Mini-crops (§7.5).** `MINI_CROP_TARGET_CLASS=0` (the one class painted per
crop — usually polymer), `MINI_CROP_PADDING=32` (context around the
predicted CC bbox), `MINI_CROP_MIN=64` / `MINI_CROP_MAX=192` (clamp on the
square crop side), `MINI_CROP_BATCH_SIZE=200` (crops staged per call),
`MINI_CROP_MIN_CC_AREA=3` (discard predicted blobs smaller than this — noise),
`MINI_CROP_DUP_IOU=0.25` (per-stem IoU dup guard, runs against
staged + painted + skipped).

**PRED rotation.** `PRED_BATCH_SIZE=150` — pool images refreshed per
correction round. Missing-PRED first, then oldest by mtime. Run §3.0 of the
notebook (`correction.regenerate_pool_preds`) for a one-shot full-pool
refresh.

**Promotion / leakage.** `PROMOTE_TILES_TO_VAL=False`. When True, the strict
whole-image leakage barrier (§8) fires automatically.

### Removed (legacy `training_2.ipynb` keys)

Pruned with the 0.1.0 refactor: `RAW_DIR`, `SEG_DIR`, `RAW_SEG_DIR`,
`TRUTH_DIR`, `EXPORT_DIR`, `LABEL_DIR`, `PROJECT_FILE`, `ILASTIK_PATH`,
`CLASS0_TARGET`, `CLASS0_CROP_PROB`. Replaced by the new folder + target-class
keys above. `EVAL_DIR` and `VIZ_DIR` kept (Panel B/C output + colored-jpg
helper).

## 12. Design decisions & rejected alternatives (do not re-litigate)

| Decision | Rationale |
|----------|-----------|
| **No** confidence-gated export | Protrusion is easily eye-visible; the value was low and added complexity. |
| **No** copy-paste augmentation | Operator prefers real labels over synthetic polymer. |
| **No** tunable inference threshold / logit bias | Under-detection is acceptable; operator would rather paint polymer than tune the decision rule. |
| **No** warm-start from previous checkpoint | Operator preference; every retrain is from scratch. |
| **No** differential / recency / fold sample weighting | Operator preference; training is uniform over the pool. |
| **FN-aware mining** (soft-prob + disagreement), selection-stage only | The failure is omission; predicted-target ranking would be blind to it. |
| **Strict** whole-image leakage barrier | Context bleed across tile borders would contaminate a faint signal. |
| Bootstrap = dense whole-image; Correction = sparse tiles | Anchors need full coverage; corrections need small crops for label quality. |
| `InitialLabels` permanently held out | A fixed pristine multi-class test set is worth more than 5 extra training images. |
| **Loop-versioned checkpoints** (filename embeds loop index) | Every retrain keeps prior models for rollback / A-B comparison / densify fill, instead of overwriting. |
| **Backbone tag in checkpoint filename** | Lets UNet and SMP runs share `models/` without collisions; the fallback chain works across backbones because the loader parses the tag. |
| **Densify positive-only tiles with the clean fill model** | Class weights / Tversky tuning can't manufacture the negative gradient that an all-target-or-ignore label is missing. Filling non-target from a clean model that the *current* training has not seen avoids the self-training bias. |
| **Densify fill backbone decoupled from training backbone** | An SMP ceiling-test run should still reuse the original clean UNet for fills, not be blocked by "no SMP loop_0 exists yet." |
| **Mini-crops centered on predicted target CCs** (FP/TP correction) | Single-class painted small crops scale labelling throughput. Centring on predictions covers FP/TP correction directly; FN-aware mini-crops (centred on uncertain regions) are deferred until evidence shows it matters. |
| **Mini-crop "skip" = leave unpainted + sweep on next round** | No explicit UI signal; "I closed without saving" + "the next retrain happened" is sufficient evidence the operator triaged the crop. Crops staged after the retrain stay so multiple mining passes between retrains work. |
| **K-fold → single carryover via `iteration_log`** | "The fold that won last round" is a cheap, mechanical rule that frees the operator from remembering which fold to set as `BEST_FOLD` after a switch. |

## 13. Open questions / deferred

- TTA at the *export* step (currently eval-only) — suggested, not adopted.
  Revisit if starting PREDs (bootstrap + correction-round) look noisy.
- Mining-score weighting (`MINING_LAMBDA`, default 1.0). Both terms are in
  `[0, 1]` so equal weighting is a defensible default; bump up if mining gets
  stuck on similar regions or down if disagreement is overpowering soft-prob.
- PRED rotation strategy is naive missing-first / oldest-mtime. A smarter
  policy (e.g., score-driven re-prediction) might cut wall-clock further.
