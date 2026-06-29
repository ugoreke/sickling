# Changelog

All notable changes to the sickle cell classification pipeline.
Format: [version] — date — summary; bullet list of changes.

## [0.9.1] — 2026-05-17 — Rare-class balance fix + orchestrate preview cell

### Bug fix
- **`balanced_group_kfold` ignored the rare class under prevalence gating.** When the labeled corpus is gated to ~10% sickle, sickle cells live in many small FOVs while non-sickle cells live in ~23 heavy FOVs. The v0.9.0 bin-packer minimised `max(sickle_load, non_sickle_load)` in *raw counts*, which is dominated by the larger class — non-sickle ended up perfectly balanced while sickle clumped catastrophically (`62 / 2 / 56 / 2 / 1` across folds in `figures/ablation/ablation_20260517_154137/`). Three of the five folds ended up with effectively zero minority class and PR-AUC ≈ 0.06–0.20.
- **Fix.** Normalise the cost by the global per-class budget (`max(sickle_load / total_sickle, non_sickle_load / total_non_sickle)`) and tie-break on the total normalised share. The rare class now dominates the placement decision exactly when it should. On the real corpus at 10% gate, sickle distribution improves from `62 / 2 / 56 / 2 / 1` to `27 / 27 / 27 / 27 / 15` across the five folds; non-sickle stays balanced at `197 / 223 / 195 / 246 / 247`.
- **Test.** New regression `test_balanced_kfold_keeps_rare_class_spread_at_10pct_prevalence` builds a synthetic mirror of the real-data FOV layout (23 heavy non-sickle FOVs + 96 light sickle FOVs at 1–3 sickles each), runs the gate at 10%, and asserts (a) every fold has at least one sickle val cell, (b) no fold owns more than 40% of the rare class, and (c) non-sickle imbalance stays under 1.5×.

### Orchestrate notebook
- **Pre-ablation composition preview.** New `ho-fold-preview` cell runs immediately before `RUN_ABLATION` and prints the exact gate + fold-strategy combination the ablation will use. Output is a `fold_diagnostics` table with `n_val`, `n_sickle_val`, `n_non_sickle_val`, `% sickle`, and `n_fovs_val` per fold, plus a friendly warning if any fold has zero sickle or fewer than five.
- Uses `make_kfold_splits` so the preview is exactly what the ablation cells will see — no drift between preview and reality.

## [0.9.0] — 2026-05-17 — Address ablation-discussion limitations: balanced folds, prevalence gate, per-tower mask test, longer MAE, run timing

Direct follow-up to `figures/ablation/ablation_20260516_003426/discussion.tex`. Each change here closes a specific limitation called out in that document.

### Validation harness

- **Balanced FOV-grouped k-fold splitter.** `sickling/eval/splits.py` now exposes `balanced_group_kfold(cells_df, n_splits, seed)` alongside the original `group_stratified_kfold`. The new splitter is a greedy multi-key bin-packer (Karmarkar–Karp-style): FOVs are ordered by descending total labelled-cell count and dropped into whichever fold minimises the max per-class load. Maintains the no-FOV-leakage invariant; equalises per-fold sickle and non-sickle counts across folds. Addresses discussion-section limitation 2 (single super-FOV dominating fold 4).
- **Dispatcher.** New `make_kfold_splits(cells_df, strategy=...)` reads `cfg.validation.fold_strategy` (`"balanced"` default, `"stratified"` for reproducing the old ablation). Stage-4 (`run_finetune`), Stage-5 (`run_multimodal_finetune`), and `eval.cli.run_evaluate` all route through the dispatcher.
- **Per-fold diagnostics helper.** `fold_diagnostics(cells_df, splits) -> DataFrame` for tests and post-hoc verification.

### Label-prevalence gate

- **Natural-prevalence simulation.** `sickling/io/labels.py:gate_labels_to_prevalence(cells_df, target_sickle_frac, *, seed, policy)` down-samples whichever class is in excess so the labelled subset hits the target rate (e.g. `0.10` for natural prevalence). Unlabelled and ambiguous rows pass through. Returns a stats dict for the run log. Applied at fold-build time in all three CLIs, so different prevalence targets can be tried without re-extracting Stage-3 crops.
- **PIPELINE_PLAN.md** updated by the user to reflect a `~10%` natural sickle prevalence (was `~5%`). The gate target is `cfg.validation.target_sickle_frac` (default `null` = no gating); `gate_seed` defaults to 42.
- **CLI hooks.** `sickling finetune --target-sickle-frac 0.10` and `sickling ablate --target-sickle-frac 0.10 --fold-strategy balanced` override the config inline.

### Per-tower mask channel zeroing (discussion limitation 5)

- **`MultimodalCropDataset(zero_image_masks_only=True)`** new flag — zeros ch1/ch2 on the image tensor sent to the image tower while the morphology cache is still computed from the un-zeroed `.pt` tensors via `CropDataset._load_fov`. The previous `zero_mask_channels=True` is preserved for back-compat (global zero). The two flags are mutually exclusive.
- **New `AblationRow` field** `zero_image_masks_only=False` plumbed through `run_multimodal_finetune` → `MultimodalCropDataset` and through `run_evaluate`.
- **New default ablation row:** `"- mask channels (image tower only, morphology kept)"`. Lets the next ablation table separate "the image tower would prefer fewer channels" from "the morphology cache is being secretly suppressed."

### Longer MAE single-GPU schedule (discussion limitation 6)

- **`configs/pretrain_mae_long.yaml`** new — 800-epoch single-GPU schedule, `batch_size=192`, `grad_accum=2` (effective 384), 40-epoch warmup, `bf16-mixed`. The previous `pretrain_mae.yaml` (300 epochs) is preserved for reproducibility.
- **`MAEConfig.pretrain_epochs`** documentation default bumped to 800 to match. Trainer continues to read from `cfg.training.max_epochs`; the field is documentation-only.

### Run-duration tracking

- **`sickling/engineering/duration.py`** new module. `DurationCallback` records per-epoch wall-clock, total fit time, device name, n_devices, strategy, precision, samples seen, and mean images/sec. Writes `duration.json` next to checkpoints on `on_fit_end`. Installed by default in `build_trainer` so every Stage-4 / Stage-5 / MAE-pretrain run produces one without further effort.
- **DDP comparison contract.** A single-GPU vs `--devices 4 --strategy ddp` MAE comparison can now be diffed from two `duration.json` files; no rerun of either side required to read the speed-up.
- `_batch_size_of` handles the three batch shapes the project uses (bare tensor, `(x, y)`, `({modality: tensor}, label)`); returns 0 on unknown shapes so the timer is failure-only-informational.

### Tests

- `tests/test_splits.py`: five new tests for `balanced_group_kfold` (no FOV leakage, per-class count range ≤ stratified on a heavily-skewed synthetic FOV corpus, unlabeled rows train-only, rejects too-few-FOVs, `make_kfold_splits` dispatch).
- `tests/test_labels.py`: six new tests for `gate_labels_to_prevalence`.
- `tests/test_dataset.py`: three new tests for `MultimodalCropDataset` mask-zeroing semantics — image-only-zero leaves morphology cache intact, global zero leaves morphology cache intact (legacy invariant), mutually-exclusive flags rejected.
- `tests/test_duration.py` new: end-to-end CPU smoke that writes `duration.json` with the expected schema; `_batch_size_of` shape handling.

### Config / CLI surface

- `cfg.validation.fold_strategy: str = "balanced"`.
- `cfg.validation.target_sickle_frac: float | None = None`.
- `cfg.validation.gate_seed: int = 42`.
- `MAEConfig.pretrain_epochs` default bumped 300 → 800 (documentation only).
- `sickling finetune --target-sickle-frac --fold-strategy` new flags.
- `sickling ablate --target-sickle-frac --fold-strategy` new flags.

Verification: `pytest` results recorded in the matching commit.

## [0.8.2] — 2026-05-16 — bf16-mixed numpy crash in Stage-4/5 validation

### Library bugs
- **`TypeError('Got unsupported ScalarType BFloat16')` killed every ablation cell.** Under `Trainer(precision="bf16-mixed")` the model's logits / softmax probabilities stay in `torch.bfloat16` through validation. The Stage-4 in-Lightning metrics in `sickling/stage4_repr/_metrics.py` called `.cpu().numpy()` directly on those tensors, which NumPy rejects (no BFloat16 dtype). The exception propagated out of `validation_epoch_end`, `_train_one` caught it, marked the ablation cell `FAILED`, and moved on — so all 120 rows in `orchestrate.ipynb` reported failure even though training proceeded normally. Fix: introduced `_to_numpy(x)` helper that casts to float32 via `.detach().float().cpu().numpy()` before handing the tensor to scikit-learn. `pr_auc` and `mcc` now route through it.
- **Same issue, second site.** `sickling/eval/cli.py` scored each batch with `probs[:, sickle_idx].cpu().numpy()`. Same fix — added `.detach().float()` before `.cpu().numpy()` so `sickling evaluate` does not crash when reloading a bf16-trained checkpoint.

Verification: full ablation (8 rows × 3 seeds × 5 folds = 120 runs) under `Trainer(precision="bf16-mixed")` completed without raising `TypeError`. Results written to `figures/ablation/ablation_20260516_003426/` and analyzed in `figures/ablation/ablation_20260516_003426/discussion.tex`.

## [0.8.1] — 2026-05-07 — Bug fixes + annotator UX

### Library bugs
- **Pickle error in dataloaders.** `train_transform` / `eval_transform` / `ssl_transform` now return module-level callable classes (`_TrainTransform`, `_EvalTransform`) instead of local closures. Closures crashed `DataLoader(num_workers > 0)` worker spawn on Windows with `AttributeError: Can't pickle local object 'eval_transform.<locals>._t'`. Affected every ablation row in non-fast-mode.
- **`KeyError: 'notes'` in `render_markdown_table`.** `aggregate_results` returns an empty DataFrame when every run failed (e.g. due to the pickle bug), so the renderer fell over on the missing column. Now defensively handles empty frames + missing `notes` column in both Markdown and LaTeX renderers.
- **WandbLogger reuse warning between ablation runs.** `_train_one` and `_eval_one` now wrap the trainer call in `try/finally` and call `wandb.finish()` so each cell starts a fresh wandb run instead of recycling the previous one.
- **Notebook progress bars duplicating.** Switched every `from tqdm import tqdm` / `from tqdm.notebook import tqdm` to `from tqdm.auto import tqdm` across `stage1_unet/bulk.py`, `stage2_instances/cli.py`, `stage3_crops/cli.py`, `stage5_multimodal/dataset.py`, and the annotation notebook.
- **Silenced two by-design Lightning warnings** in `engineering/lightning_utils.py`: "Found N module(s) in eval mode" (DINOv2 is intentionally frozen and overrides `train()`) and "number of training batches is smaller than the logging interval" (fires for any small fast-mode run).

### Annotator (`notebooks/02_annotate.ipynb`)
- **New segmentation cache cell.** U-Net + watershed runs once at session start and caches `(raw_norm, label_map, instance_image, pre_instance_image, drop_reasons)` per FOV. `build_queue` and `build_redo_queue` now read from the cache, so changing `MIN_POLYMER_SCORE`, `MAX_PER_IMAGE`, `REDO_MODE`, etc. and re-running the build cell does **not** re-segment.
- **`REDO_CLASSES` config.** Set this to a subset like `{"ambiguous"}` to triage only borderline calls.
- **Note hotkeys 1/2/3** insert canned strings into the notes entry: `1 → nearby debris`, `2 → exogenous polymer`, `3 → multiple sickle`.
- **Redo display improvements.** GUI now pre-populates the notes entry with the prior note (so you can keep / edit / clear it) and color-codes the existing label in red (`#d33`) in the legend. The status line shows `current=<label>`.
- **Hotkey-leak fix.** Pressing `8/9/0/1/2/3/z/s` no longer types the digit into the focused notes entry. Bindings are installed on both root *and* the entry widget; each handler returns `"break"` to prevent Tk's default `<KeyPress>` from inserting the character.

Verification: 71/71 non-network pytests passing, `make lint` clean.

## [0.8.0] — 2026-05-07 — Milestone 8: engineering polish + reproduce

- `make reproduce` end-to-end target chaining `predict → instances → crops → ablate → figures`. Overridable on the command line:
  - `make reproduce LABELS_CSV=labels/labels.csv SEEDS=42,43,44 FOLDS=0,1,2,3,4`
  - shortcut `make reproduce-full` for production seeds × full k-fold on the full labels.
- `--labels-csv` flag added to `sickling crops` so the labels source can be swapped without editing config files. Wired into the Make targets as `LABELS_CSV ?= labels/labels_trimmed.csv`.
- `bench-ddp` Make target (one-line wrapper around `python -m sickling.engineering.ddp_benchmark`).
- Ruff lint config tightened: project-wide ignores for `B008` (Typer idiom), `E702` (intentional compact `def`), and `UP015` (explicit `"r"` mode). 41 auto-fixable issues cleaned up across the package; 4 manual fixes (unused var in `sampler.py`, `zip(..., strict=True)` in DDP benchmark, `stacklevel=2` on `warnings.warn`, SIM108 ternary in `morphology_features.py`). **`make lint`** is now green.
- README rewritten: one-line reproduce instructions; full CLI table; notebook index; DDP / sweeps / configuration notes.

Verification: **71/71 non-network tests passing, `make lint` clean.** Stage states on disk at this point: 19 FOVs predicted, 11,584 cells extracted, 89 labels resolved (48 non-sickle + 32 sickle + 9 ambiguous on 6 FOVs) from the trimmed CSV. The full ablation table can now be kicked off with `make reproduce-full` whenever the user is ready.

## [0.7.1] — 2026-05-07 — Orchestration notebook

- New `orchestrate.ipynb` at the repo root — single notebook that drives every stage with flag-gated cells (`RUN_PREDICT`, `RUN_INSTANCES`, `RUN_CROPS`, `RUN_SMOKE_FIT`, `RUN_ABLATION`, `RUN_FIGURES`). All per-run config knobs (trimmed-vs-full labels, fast-mode hyperparameters, seeds, folds) live as Python variables — no YAML strings in the notebook. Each stage call mutates `cfg` directly via attribute assignment.
- The labels filename is `labels/labels_trimmed.csv` (no space) — toggled via `USE_TRIMMED_LABELS = True`.
- Includes an inline Markdown render of the latest `figures/ablation/<ts>/table_markdown.md` so results are visible right after the ablation cell finishes.

## [0.7.0] — 2026-05-07 — Milestone 7: ablation runner + sweeps + DDP benchmark

- `sickling/ablation/` package:
  - `AblationRow` dataclass + `DEFAULT_ABLATION` table (full multimodal, − morphology, − image tower, − mask channels, − weighted sampler, A vs B vs C).
  - `run_ablation_table(cfg, rows, seeds, folds, output_dir, synth_labels, skip_existing)` — crash-resumable orchestrator. Persists `raw_results.json` after every (row, seed, fold) cell.
  - `aggregate_results` (mean ± std across seeds × folds), `render_markdown_table`, `render_latex_table`, `write_tables`.
- `sickling ablate` Typer command live: `--seeds 42,43,44 --folds 0,1,2,3,4 --synth-labels --no-resume`.
- `zero_mask_channels` flag plumbed through `CropDataset`, `MultimodalCropDataset`, `run_finetune`, `run_multimodal_finetune`, `run_evaluate` — enables the "− mask channels" ablation row.
- `cfg.paths.labels_csv` field added so the labels source can be swapped (e.g. `labels/labels_trimmed.csv` for fast iteration). Stage 3 now reads from this field instead of the hardcoded `labels/labels.csv`.
- `sickling/stage1_unet/bulk.py` — `bulk_predict(cfg, input_dir, model_path, copy_raws, overwrite, n_classes)`. New `sickling predict` Typer command runs the U-Net over `to_be_labeled/`, persists predictions to `unet_predictions/PRED_<stem>.h5`, copies raws into `raw_images/`. Closes the gap between the on-the-fly annotator and the on-disk Stage 2/3 pipeline.
- `sickling/engineering/ddp_benchmark.py` — DDP throughput benchmark + bar-chart renderer. CLI: `python -m sickling.engineering.ddp_benchmark --devices 1 --devices 2 --devices 4`.
- `configs/sweeps/bakeoff.yaml` and `configs/sweeps/multimodal_hp.yaml` — Bayesian wandb sweeps over LRs, weight decay, LLRD, batch size, fusion hidden dim, dropout.
- `RESULTS.md` template — ablation table layout + bake-off + DDP scaling + bias-check + reproducibility sections.
- 4 new tests in `tests/test_ablation.py`; total non-network suite **75/75 passing**.

Verification: end-to-end smoke ablation on 37 labeled samples (1 FOV, the only one with cached U-Net predictions on disk), 1 seed × 1 fold × 3 rows, 2 epochs each:
```
Full multimodal (A+morph)    PR-AUC = 0.948 ± 0.000   MCC = 0.655 ± 0.000
Image only (DINOv2 frozen)   PR-AUC = 0.717 ± 0.000   MCC = 0.333 ± 0.000
Morphology only              PR-AUC = 0.717 ± 0.000   MCC = 0.333 ± 0.000
```
Multimodal beats either tower alone — as the spec predicted. Numbers are not statistically meaningful (n_val ≈ 8 after the row-level 80/20 fallback) but the orchestration + tables + CIs + figures all work. Once `sickling predict` is run on the remaining FOVs in `to_be_labeled/`, the full 200-sickle / 500-non-sickle corpus becomes available for proper k-fold ablation.

## [0.6.2] — 2026-05-07 — Annotation notebook improvements

- **Priority sort by boundary polymer.** New `PRIORITY_SORT_BY_POLYMER` (default `True`) + `MIN_POLYMER_SCORE` knobs in `notebooks/02_annotate.ipynb`. The polymer score is the count of class-0 pixels inside a 4-pixel erosion ring at each instance's boundary — interior polymer is less specific to sickling than polymer at the cell's outer edge. The queue is sorted descending so likely-sickle candidates come first. On the reference FOV, 54 / 526 cells have a non-zero score — labelling those first hits the sickle population without scanning every cell.
- **Circular display outline instead of foreground mask.** The display now draws a circle of radius `1.5 × max(centroid → instance pixel distance)` around the cell centroid instead of tracing the U-Net foreground mask. Robust to U-Net segmentation noise. The yellow centroid crosshair stays. New config knob `CIRCLE_RADIUS_FACTOR = 1.5`.
- GUI status line now surfaces `polymer_score` per cell so the priority-sort behavior is visible at a glance.

## [0.6.1] — 2026-05-07 — Cell-crop labeler

- New `sickling/stage1_unet/` package — frozen U-Net inference (architecture mirrors `training 2.ipynb`, plus `load_unet` + sliding-window `predict_label_map`). Reusable from anywhere, not just notebooks.
- Created root folders `to_be_labeled/` (raw images dropped here for labeling) and `models/` (user placed `unet_fold_2_best.pth` here).
- Rewrote `notebooks/02_annotate.ipynb`: reads raw images from `to_be_labeled/`, runs the U-Net + watershed pipeline, then walks every kept instance with a Tk GUI. Hotkeys `8`=sickle, `9`=non_sickle, `0`=ambiguous, `z`=undo, `s`=skip, `Esc`=quit. Display shows a 192×192 context window with the cell outlined in red and a yellow centroid crosshair (3× upscaled).
- CSV writes happen on every keypress so a crashed session loses at most the unlabeled tail. `(source_image, x, y)` dedup against `labels.csv` makes sessions resumable.
- Sanity-checked: U-Net loads onto CUDA, predicts the 1992² FOV in a few seconds, output classes are exactly `{0, 1, 2, 3}` matching the project convention.

## [0.6.0] — 2026-05-07 — Milestone 6: validation harness + metrics + figures

- New modules under `sickling/eval/`:
  - `metrics.py` — `BinaryMetrics` dataclass; `compute_binary_metrics(y_true, y_score, threshold_strategy='max_mcc'|'max_f1'|'fixed')`; `pick_threshold_max_mcc`, `pick_threshold_max_f1`, `recall_at_precision`.
  - `bootstrap.py` — vectorized `bootstrap_metric` (handles NaN resamples) + `bootstrap_pr_curve` (pointwise band on a fixed recall grid).
  - `report.py` — `EvaluationReport` with full snapshot (metrics + CIs + PR band + raw y/score). JSON round-trippable.
  - `figures.py` — PR curve with bootstrap CI band, confusion-matrix heatmap, calibration plot. Module sets `svg.fonttype='none'` and `pdf.fonttype=42` so fonts stay as text in SVG/PDF (editable in Illustrator/Inkscape per user request).
  - `cli.py` — `run_evaluate` reloads any Stage 4/5 checkpoint, recomputes val scores, runs metrics + bootstrap, writes `report.json` + 3 SVG figures. `run_figures` re-renders from saved JSON without retraining.
- `sickling evaluate CHECKPOINT --variant ...` and `sickling figures` Typer commands are live.
- 17 new tests added (8 metrics + 4 bootstrap + 5 figures); total suite **67 tests**, all green.

Verification on the multimodal smoke checkpoint with synthetic labels:
```
multimodal_dinov2_frozen_image+morphology_fold0  fold=0  n_val=100
  PR-AUC = 0.506  [0.367, 0.653]
  MCC    = 0.188  [-0.009, 0.359]
  recall@p=0.9 = 0.022  threshold=0.533
  figures + report -> figures/eval/multimodal_dinov2_frozen_image+morphology_fold0/
```
Wide CIs are correct given the synthetic labels and 100-sample val set. SVG figures verified to contain `<text>` elements with `font-family` strings (not paths) — text remains editable in vector tools. `make figures` re-renders all reports without retraining.

## [0.5.0] — 2026-05-07 — Milestone 5: multimodal classifier

- New `sickling/stage5_multimodal/` package:
  - `morphology_features.py` — pure functions for hand-crafted shape descriptors. `FEATURE_NAMES` / `N_FEATURES = 30` (5 basic + 8 Fourier harmonics + 16 Zernike (degree 6) + 1 polymer ratio). `compute_features(crop)` takes a `(3, H, W)` tensor.
  - `Tower` ABC — `forward(x) -> Tensor[B, D]`, `D: int`. Default `trainable_param_groups`.
  - `ImageTower` — wraps any `ImageEncoder` from Stage 4; proxies LLRD param groups.
  - `MorphologyTower` — small MLP with feature-mean / feature-std buffers (set per-fold from train-set stats; travel with checkpoint).
  - `MultimodalClassifier({name: Tower}, num_classes, hidden, dropout)` — concat fusion, per-tower LR groups via `trainable_param_groups(base_lrs, llrd, head_lr)`.
  - `MultimodalCropDataset` — wraps `CropDataset`, caches morphology features in memory at construction. Train and val subsets share the cache. Yields `({'image': Tensor, 'morphology': Tensor}, label)`.
  - `MultimodalFinetuneModule` — AdamW on per-tower groups (LLRD on image, flat on morphology, separate head LR), cosine-with-warmup schedule.
  - `run_multimodal_finetune(cfg, image_variant, ..., use_image, use_morphology)` — single-fold driver. `use_image` / `use_morphology` toggles for ablation rows.
- `sickling finetune multimodal` Typer command is live with `--image-variant`, `--no-image`, `--no-morphology`, `--mae-init`, `--synth-labels`.
- New configs: `configs/finetune_multimodal.yaml`, `configs/smoke_multimodal.yaml`.
- 16 new tests added (9 morphology + 6 towers + 1 extension contract); total suite **58 tests**, all green.

Smoke result: multimodal classifier (DINOv2 frozen image tower + MorphologyTower) on 497 crops with synthetic 50/50 labels, 2 epochs, val_pr_auc=0.506 (≈ random as expected with random labels). Morphology cache built in 2.5 s at ~200 cells/s.

The 5-line modality-extension contract is enforced by `tests/test_tower_extension.py`: a `TimeTower` is added in 4 lines (subclass `Tower`, set `D`, define `forward`) and integrates with `MultimodalClassifier` without any other code change.

## [0.4.0] — 2026-05-07 — Milestone 4: representation learning bake-off

- New `sickling/data/` package: `CropDataset` (lazy per-FOV `.pt` cache, channel-aware resize: ch0 bilinear, ch1/ch2 nearest), augmentations (`train_transform`, `eval_transform`, `ssl_transform` — flips/rot90 across all channels, brightness/contrast jitter on ch0 only), `make_weighted_sampler` (50% minority per batch).
- `sickling/eval/splits.py` — `group_stratified_kfold` (groups = source_image, stratifies on FOV-level dominant label, unlabeled rows always go to train).
- New `sickling/stage4_repr/` package:
  - `ImageEncoder` ABC with `forward / standardize / freeze_backbone / trainable_param_groups`.
  - `DinoV2Encoder` (Model A — frozen ViT-S/14 via torch.hub).
  - `TimmViTEncoder` (Model B — `vit_small_patch16_224.augreg_in21k_ft_in1k`) with layer-wise LR decay (patch_embed deepest, top norm shallowest).
  - `MAEViTEncoder` (Model C — same arch as B; loads MAE checkpoint, falls back to supervised init with a warning if name unavailable). `load_mae_checkpoint` reads weights from `MAEPretrainModule` Lightning checkpoints.
  - `MAEReconstructor` — encoder + small ViT decoder + `_random_masking` + linear pred head. Reconstructs `(3, 224, 224)` patches with optional `norm_pix_loss`.
  - `LinearProbeModule`, `FinetuneModule`, `MAEPretrainModule` — Lightning modules with cosine-with-warmup LR schedules. All log `val_pr_auc` for `ModelCheckpoint` (MAE module logs `-val_recon_loss` aliased as `val_pr_auc`).
- `sickling pretrain` and `sickling finetune {variant}` Typer commands are live. Flags: `--synth-labels` (deterministic 50/50 for module smoke-testing), `--mae-init <ckpt>`, `--resume <ckpt>`, `--devices`, `--strategy`.
- Per-stage YAMLs: `configs/finetune_modelA.yaml`, `finetune_modelB.yaml`, `finetune_modelC.yaml`, `pretrain_mae.yaml`. Plus `smoke_finetune.yaml` and `smoke_pretrain.yaml` for quick validation.
- Config additions: `AugmentConfig`, `FinetuneConfig`, `MAEPretrainConfig`. `TrainingConfig.num_workers`.
- 19 new tests (6 dataset + 5 splits + 8 encoders); total suite **42 tests**, all green.

Verification:
- Linear-probe smoke (DINOv2 frozen, 2 epochs, synthetic 50/50 labels on 497 crops): `val_pr_auc=0.573` (≈ random, as expected with synthetic labels). Best checkpoint saved.
- MAE pretrain smoke (2 epochs on 497 crops, decoder=128/2/4, mask_ratio=0.75): train recon loss 1.34 → 0.70, val recon loss 0.79 → 0.70. Decoder is learning to reconstruct masked patches. Best checkpoint saved at `checkpoints/pretrain_mae/001-32.ckpt`.

## [0.3.0] — 2026-05-07 — Milestone 3: crop extraction

- `sickling/io/images.py` — `normalize_image` (single source of truth, mirrors `training 2.ipynb`), `load_raw_greyscale`, `find_raw_image`.
- `sickling/io/labels.py` — `LabelRow`, `load_labels`, `load_conditions`, `resolve_coordinate_to_instance` (point-in-mask join with explicit failure reasons: `coordinate_outside_cell`, `instance_dropped:<reason>`).
- `sickling/io/parquet.py` — `CELLS_COLUMNS` schema and `read_cells` / `write_cells` helpers.
- `sickling/stage3_crops/extract.py` — `extract_one` (3-channel: ch0=normalized raw, ch1=instance cell_body mask, ch2=instance polymer mask) + `extract_for_fov` bulk driver; `drop_if_clipped=True` is the default per user spec.
- `sickling/stage3_crops/metadata.py` — `make_cells_dataframe` enforces canonical schema with proper nullable dtypes; `write_failed_jsonl`.
- `sickling/stage3_crops/cli.py` — `run_stage3` walks `*_instances.h5`, reloads U-Net + raw, extracts crops, joins labels + conditions, writes `crops/<stem>.pt` + `cells.parquet` + `failed.jsonl`.
- `sickling crops` Typer command is now live.
- 15 new tests (8 in `test_labels.py`, 7 in `test_crops.py`); total suite 23 tests, all green.
- Fixed pydantic 2.11 deprecation warning in `PathsConfig.resolved` (`type(self).model_fields` instead of `self.model_fields`).

Verification: `make crops` on the real FOV produced 497 cells (25 clipped near edges) in `crops/D16_03_1_1_Bright Field_001.pt` (497×3×96×96 float32, 55 MB), with `cells.parquet` (497 rows × 14 cols) and `failed.jsonl` (25 entries) at the repo root. ch0 ∈ [0.009, 1.0]; ch1, ch2 ∈ {0, 1}.

## [0.2.2] — 2026-05-07 — min_area retune

- Default `instances.min_area` lowered from 800 → 550 after empirical QA via the notebook on the reference FOV.
- Real-FOV yield: `n_kept` 473 → 522 (+49 cells previously dropped as fragments are now kept; visual QA confirms they're real cells in the bimodal histogram's lower tail).
- Updated `notebooks/02_instance_qa.ipynb` sys.path-fallback cell to point users at `pip install -e .` as the proper fix.

## [0.2.1] — 2026-05-07 — Stage 2 QA visualization

- Refactored `mask_to_instances` into composable internals (`_run_watershed`, `_classify_drops`, `_apply_filters`); existing tests still pass.
- Added `mask_to_instances_with_reasons` — returns the pre-filter watershed image plus a `{pre_filter_id: reason}` mapping. Drop reasons exported as `DROP_KEPT`, `DROP_EDGE`, `DROP_MIN`, `DROP_MAX`, `DROP_EMPTY`.
- New `sickling/stage2_instances/qa.py` — 4-panel QA figure (raw, kept instances colorized, drop-reason overlay, log-x area histogram with min/max cutoffs).
- `sickling instances --qa` flag renders `stage2_qa_<stem>.png` per FOV to `figures/`. Auto-locates matching raw image (strips the `PRED_` prefix that `export_for_ilastik_correction` adds).
- `notebooks/02_instance_qa.ipynb` — interactive QA + `min_area` sweep.
- Copied `D16_03_1_1_Bright Field_001.jpg` from legacy `trainingImages/` into `raw_images/` for the QA overlay.

QA findings on the reference FOV: pre-filter area histogram is cleanly bimodal (noise mode <100 px, real-cell mode 1000–3000 px) — `min_area=800` falls in the trough between modes, so the 108 dropped fragments are legitimately U-Net background false-positives, not real cells. No retune needed.

## [0.2.0] — 2026-05-07 — Milestone 2: instance segmentation

- `sickling/io/h5.py` — `load_robust_h5`, `load_label_map` (1-based → 0-based, unannotated → 255), `write_label_map_h5` (gzip-compressed uint16), `write_ilastik_h5` (5-D round-trip).
- `sickling/stage2_instances/watershed.py` — pure function `mask_to_instances(label_map, cfg, classes) -> (uint16 image, InstanceStats)`. Foreground = polymer ∪ cell_body; morphological closing → distance transform → marker-seeded watershed; drops edge-touching, below-`min_area`, above-`max_area` instances; sequentially relabels survivors.
- `sickling/stage2_instances/cli.py` — `run_stage2(cfg, input_dir, output_dir, limit)` walks `*.h5`, writes `<stem>_instances.h5` and `_stats.parquet`.
- `sickling instances` Typer command is now live (was a stub in 0.1.0).
- Added `paths.instances` to `Config` / `configs/base.yaml`.
- `tests/conftest.py` — synthetic 5-cell fixture (isolated, polymer-ringed, touching pair, edge-touching) + variant with tiny/oversized blobs.
- `tests/test_instances.py` — 8 tests, all passing. Includes a real-h5 smoke test against the user's reference FOV.
- Re-pinned `requirements.txt` and `pyproject.toml` to the working `pytorch-cuda` conda env (torch 2.5.1, lightning 2.5.5, pydantic 2.12.3, etc.).
- Moved `PRED_D16_03_1_1_Bright Field_001.h5` from repo root into `unet_predictions/`.

Verification: `make instances` on the real FOV produced 473 cells (46 dropped at edge, 108 below `min_area=800`, 0 above `max_area=6000`) in 1.2 s.

## [0.1.0] — 2026-05-07 — Milestone 1: project skeleton

- Created `sickling/` package skeleton (one sub-package per pipeline stage + cross-cutting modules: `io`, `data`, `eval`, `ablation`, `engineering`, `cli`).
- Pinned dependencies in `requirements.txt` and `pyproject.toml` (PyTorch 2.11, CUDA 12.6, Lightning 2.5, timm 1.0, pydantic-settings 2.7).
- `sickling.config` — `pydantic-settings` root config with nested sub-models for project / paths / classes / crop / instances / representation / multimodal / validation / training / smoke. YAML deep-merge override chain via `load_config(*overrides)`.
- `configs/base.yaml` — project-wide defaults; `configs/smoke.yaml` — smoke-test override.
- `sickling.engineering` — `seed_everything`, `build_wandb_logger`, `build_checkpoint_callback`, `build_trainer` factories. `run_smoke` — one-linear-layer regressor on random tensors, exercises full Lightning + wandb (offline) + checkpoint stack on CPU.
- `sickling.cli.main` — Typer CLI with subcommands `smoke` (live), `instances`, `crops`, `pretrain`, `finetune`, `evaluate`, `ablate`, `figures` (stubbed with milestone hint until implemented).
- Root data directories created: `unet_predictions/`, `raw_images/`, `labels/`, `conditions/`, `figures/`.
- `labels/labels.csv` (coordinate-based schema) and `labels/README.md`.
- `conditions/conditions.csv` and `conditions/README.md` (no `patient_id` per user request).
- `Makefile` with one target per CLI subcommand + `install` / `dev-install` / `test` / `lint` / `clean`.
- `README.md` quickstart.
- `ARCHITECTURE.md` file-by-file index, with planned files flagged by milestone.
