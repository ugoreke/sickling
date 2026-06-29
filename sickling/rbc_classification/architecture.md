# Architecture

File-by-file index of the `sickling/` package, grouped by pipeline stage.
Entries marked **(planned)** are scaffolding only — to be filled in at the
indicated milestone. This document is updated alongside every code change.

## Pipeline overview

```
raw_images/                     unet_predictions/
   │                                │
   └─► Stage 1 (frozen U-Net) ──────┘   ← external; see `training 2.ipynb`
                │
                ▼
    Stage 2  sickling/stage2_instances/   semantic 4-class -> int instance label image
                │
                ▼
    Stage 3  sickling/stage3_crops/       per-cell 96×96×3 crops + cells.parquet
                │
                ├──► Stage 4  sickling/stage4_repr/        representation bake-off (A/B/C)
                │
                └──► Stage 5  sickling/stage5_multimodal/  modular tower classifier

                                           cross-cutting:
                                             sickling/io/         IO conventions
                                             sickling/data/       datasets/samplers
                                             sickling/eval/       metrics + bootstrap CIs
                                             sickling/ablation/   ablation runner
                                             sickling/engineering/ Lightning + wandb factories
                                             sickling/cli/        Typer entrypoints
```

## Top-level package — `sickling/`

### `sickling/__init__.py`
Package metadata. Exports `__version__`.

### `sickling/config.py`
Single source of truth. `pydantic-settings`-based `Config` with nested
sub-models for project, paths, classes, crop, instances, representation,
multimodal, validation, training, and smoke-test settings.
- `Config` — root settings model (env-var overrides via `SICKLING_…`).
- `ProjectConfig`, `PathsConfig`, `ClassesConfig`, `CropConfig`, `InstancesConfig`,
  `RepresentationConfig` (+ `DinoV2Config`, `TimmViTConfig`, `MAEConfig`),
  `MultimodalConfig`, `ValidationConfig`, `TrainingConfig`, `SmokeConfig`.
- `PathsConfig.resolved()` — returns absolute-path copy with `root` resolved against the repo root.
- `load_config(*overrides)` — loads `configs/base.yaml`, deep-merges any number of override YAMLs.

## Stage 1 — frozen U-Net inference

Training lives in `training 2.ipynb`; this package only provides the
architecture + sliding-window predictor needed to use the trained weights.

### `sickling/stage1_unet/__init__.py`
Re-exports `UNet`, `load_unet`, `predict_label_map`, `bulk_predict`.

### `sickling/stage1_unet/inference.py`
- `UNet(n_channels=1, n_classes=4)` — same architecture as `training 2.ipynb`. Loads `unet_fold_*_best.pth` checkpoints with no key remapping.
- `load_unet(model_path, n_classes=4, device=None) -> UNet` — eval-mode loader.
- `predict_label_map(model, raw_norm, tile_size=256, overlap=0.5, n_classes=4)` — sliding-window argmax. Returns `int16` 0-indexed label map matching the project's class convention (0=polymer, 1=bg, 2=cell_body, 3=cell_border).

### `sickling/stage1_unet/bulk.py`
- `bulk_predict(cfg, input_dir, model_path, copy_raws=True, overwrite=False, n_classes=4)` — runs the frozen U-Net over every image in `input_dir`, writes `unet_predictions/PRED_<stem>.h5` (Ilastik 1-based for compatibility with `training 2.ipynb`), optionally copies raws into `raw_images/`. Persists everything Stage 2 needs so the on-the-fly annotator path doesn't leave Stages 2-5 starving for data.

## Stage 2 — instance segmentation

### `sickling/stage2_instances/__init__.py`
Re-exports `InstanceStats` and `mask_to_instances`.

### `sickling/stage2_instances/watershed.py`
- `InstanceStats` (dataclass) — `n_total`, `n_kept`, `n_dropped_edge`, `n_dropped_min_area`, `n_dropped_max_area`. Has `to_dict()`.
- `mask_to_instances(label_map, cfg: InstancesConfig, classes: ClassesConfig) -> (uint16 array, InstanceStats)` — pure function. Foreground = `polymer ∪ cell_body`; morphological closing (radius `cfg.closing_radius`) → Euclidean distance transform → `peak_local_max(min_distance, threshold_rel)` → marker-seeded watershed masked to closed foreground. Drops edge-touching, < `min_area`, > `max_area` instances; relabels survivors sequentially.

### `sickling/stage2_instances/watershed.py` (additions)
- `mask_to_instances_with_reasons(label_map, cfg, classes) -> (instance_image, stats, pre_instance_image, reasons)` — same pipeline as `mask_to_instances` but also returns the unfiltered watershed image and a `{pre_filter_id: reason}` mapping. Used by Stage 2 QA and by Stage 3 when joining label coordinates that fall inside a dropped instance.
- `DROP_KEPT`, `DROP_EDGE`, `DROP_MIN`, `DROP_MAX`, `DROP_EMPTY` — string constants used as drop-reason keys.

### `sickling/stage2_instances/qa.py`
- `make_qa_figure(label_map, instance_image, pre_instance_image, reasons, cfg, raw_image=None, title=None) -> Figure` — 4-panel (A: raw, B: kept instances colorized, C: drop-reason overlay, D: log-x area histogram with min/max cutoffs).
- `render_qa_for_h5(h5_path, cfg, classes, raw_image=None)` — convenience wrapper that loads the U-Net 4-class h5 and runs the full pipeline.
- `load_raw_image(stem, raw_dir)` — find a matching `.jpg/.jpeg/.png/.tif/.tiff`, return greyscale `np.ndarray` or `None`.
- `save_qa_figure(fig, path, dpi=150)`.

### `sickling/stage2_instances/cli.py`
- `run_stage2(cfg, input_dir=None, output_dir=None, limit=None, qa=False) -> pd.DataFrame` — walks `*.h5` in `input_dir` (defaults to `cfg.paths.unet_predictions`), writes `<stem>_instances.h5` per FOV (drops `_segmentation` suffix if present) and `_stats.parquet` at the output root. With `qa=True` also renders `stage2_qa_<stem>.png` to `cfg.paths.figures`, looking up the matching raw image (PRED_-prefix-stripped) in `cfg.paths.raw_images`. Returns the stats DataFrame.

## Stage 3 — crop extraction

### `sickling/stage3_crops/__init__.py`
Re-exports `extract_one`, `extract_for_fov`, `make_cells_dataframe`, `write_failed_jsonl`.

### `sickling/stage3_crops/extract.py`
- `extract_one(raw_norm, label_map, instance_image, instance_id, cfg, classes) -> (Tensor[3,H,W] | None, meta)` — builds ch0=normalized raw, ch1=this instance's cell_body mask, ch2=this instance's polymer mask. Drops if window would clip and `cfg.drop_if_clipped=True`.
- `extract_for_fov(raw_norm, label_map, instance_image, cfg, classes) -> (Tensor[N,3,H,W], list[int], list[dict], list[dict])` — returns aligned `(tensors, instance_ids, kept_meta, failed_meta)` for one FOV.

### `sickling/stage3_crops/metadata.py`
- `make_cells_dataframe(records) -> pd.DataFrame` — coerces per-cell records into the `CELLS_COLUMNS` schema with proper nullable dtypes.
- `write_failed_jsonl(rows, path)` — one JSON per line.

### `sickling/stage3_crops/cli.py`
- `run_stage3(cfg, instances_dir=None, raw_dir=None, unet_dir=None, crops_dir=None, limit=None) -> pd.DataFrame` — for each `*_instances.h5`: matches raw + U-Net prediction by stem (strips `PRED_` and `_instances` suffix), reloads label map + drop reasons, calls `extract_for_fov`, `torch.save`s to `crops/<stem>.pt`, joins labels + conditions, writes `cells.parquet` + `failed.jsonl`.

## Stage 4 — representation learning

### `sickling/stage4_repr/__init__.py`
Re-exports the encoder classes and `build_encoder(variant, **kwargs)` factory (variants: `dinov2_frozen`, `timm_vit`, `mae`).

### `sickling/stage4_repr/encoder.py`
- `ImageEncoder` ABC — `forward(x: Tensor[B,3,H,W]) -> Tensor[B, embed_dim]`, `embed_dim`, `freeze_backbone()`, `standardize(x)`, `trainable_param_groups(base_lr, llrd)`.
- `imagenet_standardize(x)` — shared default standardization.

### `sickling/stage4_repr/dinov2_encoder.py`
- `DinoV2Encoder` — torch.hub `dinov2_vits14`, frozen-by-default, returns CLS token (384-d). `train()` overridden to keep BN/dropout in eval mode.

### `sickling/stage4_repr/timm_vit_encoder.py`
- `TimmViTEncoder` — wraps a timm ViT (default `vit_small_patch16_224.augreg_in21k_ft_in1k`), exposes per-block param groups for layer-wise LR decay (patch_embed deepest, top norm shallowest).
- `MAEViTEncoder` — same architecture, MAE-checkpoint init. Falls back to supervised init with a warning if the requested MAE name isn't in the timm registry. `load_mae_checkpoint(path)` loads encoder weights from a `MAEPretrainModule` Lightning checkpoint.

### `sickling/stage4_repr/mae_encoder.py`
- `_patchify(x, patch_size)`, `_random_masking(x, mask_ratio)` — pure helpers (tested).
- `MAEReconstructor(encoder, decoder_embed_dim, decoder_depth, decoder_num_heads, norm_pix_loss)` — encoder + small ViT decoder + linear pred head; `forward(x, mask_ratio) -> (loss, pred, target, mask)`.

### `sickling/stage4_repr/_metrics.py`
- `pr_auc(y_true, y_score)`, `mcc(y_true, y_pred)` — thin sklearn wrappers used inline by the LightningModules until milestone 6 lands the full metrics suite.

### `sickling/stage4_repr/linear_probe.py`
- `LinearProbeModule(encoder, ...)` — frozen encoder + linear head, AdamW + cosine-with-warmup. Used for **Model A**.

### `sickling/stage4_repr/finetune.py`
- `FinetuneModule(encoder, ...)` — full fine-tune with LLRD. Used for **Models B and C**.

### `sickling/stage4_repr/ssl_pretrain.py`
- `MAEPretrainModule(reconstructor, mask_ratio, ...)` — MAE continuation. Logs `train_recon_loss`, `val_recon_loss`, plus `val_pr_auc = -val_recon_loss` so the existing `ModelCheckpoint(monitor="val_pr_auc", mode="max")` keeps the best-loss epoch.

### `sickling/stage4_repr/cli.py`
- `run_finetune(cfg, variant, fold, ckpt_path, mae_init_ckpt, synth_labels, devices)` — single-fold linear-probe / fine-tune driver. Falls back to row-level 80/20 split when fewer FOVs than CV folds (smoke).
- `run_pretrain_mae(cfg, ckpt_path, devices, strategy)` — MAE continuation over the full crop corpus. 90/10 train/val split by source_image.
- `_add_synthetic_labels(df, seed)` — used only by `--synth-labels` for module smoke-testing.

## Stage 5 — multimodal classifier

### `sickling/stage5_multimodal/__init__.py`
Re-exports `Tower`, `ImageTower`, `MorphologyTower`, `MultimodalClassifier`, `MultimodalCropDataset`, `compute_features`, `FEATURE_NAMES`, `N_FEATURES`.

### `sickling/stage5_multimodal/tower.py`
- `Tower` ABC — `forward(x) -> Tensor[B, D]`, `D: int`, `trainable_param_groups(base_lr, llrd)` default impl.

### `sickling/stage5_multimodal/image_tower.py`
- `ImageTower(encoder)` — wraps any `ImageEncoder` from Stage 4. Proxies `trainable_param_groups` so the multimodal optimizer can apply LLRD on the image branch only.

### `sickling/stage5_multimodal/morphology_features.py`
- `FEATURE_NAMES` (30 names), `N_FEATURES = 30`.
- `_basic_shape(mask)` (5: area, perimeter, compactness, eccentricity, solidity), `_fourier_descriptors(mask, n_harmonics=8)` (scale-normalized boundary FFT magnitudes), `_zernike(mask, degree=6)` (16 mahotas Zernike moments), `_polymer_ratio(body, polymer)`.
- `compute_features(crop) -> np.ndarray[N_FEATURES]` — main entry; takes `(3, H, W)` tensor or array, binarizes ch1/ch2 at 0.5.

### `sickling/stage5_multimodal/morphology_tower.py`
- `MorphologyTower(in_features, hidden=64, out_features=64, dropout=0.2)` — `Linear → GELU → Dropout → Linear → GELU → Linear`.
- `set_feature_stats(mean, std)` — registers train-set standardization as buffers (travel with checkpoint).

### `sickling/stage5_multimodal/classifier.py`
- `MultimodalClassifier(towers, num_classes, hidden, dropout)` — concat tower outputs, fusion MLP. Errors clearly on missing modalities at `forward`.
- `trainable_param_groups(base_lrs, llrd, head_lr)` — per-tower LRs + a head group; LLRD applies wherever a tower overrides the default.

### `sickling/stage5_multimodal/dataset.py`
- `MultimodalCropDataset(cells_df, crops_dir, target_size, return_label, transform, morphology_cache, zero_mask_channels, zero_image_masks_only)` — wraps `CropDataset`, computes/caches morphology features in memory at construction. Yields `({'image': Tensor, 'morphology': Tensor}, label)`. The `morphology_cache` argument allows train/val splits to share precompute. `zero_image_masks_only=True` zeros ch1/ch2 on the image tensor sent to the image tower but leaves the morphology cache untouched (the cache is built from `_load_fov`'s raw `.pt` tensors). `zero_mask_channels=True` is the legacy global zero. The two flags are mutually exclusive.

### `sickling/stage5_multimodal/lightning_module.py`
- `MultimodalFinetuneModule(classifier, ...)` — AdamW on per-tower param groups (LLRD on image, flat on morphology, separate `head_lr`); cosine-with-warmup schedule.

### `sickling/stage5_multimodal/cli.py`
- `run_multimodal_finetune(cfg, image_variant, fold, ckpt_path, mae_init_ckpt, synth_labels, devices, use_image, use_morphology, zero_mask_channels, zero_image_masks_only) -> dict` — single-fold trainer. Per-fold morphology standardization on train subset (no val leak). Tower toggles via `use_image` / `use_morphology` for ablation rows. Label-prevalence gate (`cfg.validation.target_sickle_frac`) applied before fold construction so different prevalence targets can be tried without re-extracting crops. Splits are built via `make_kfold_splits(strategy=cfg.validation.fold_strategy)`.

## Cross-cutting modules

### `sickling/io/__init__.py`
### `sickling/io/h5.py`
- `load_robust_h5(path) -> np.ndarray` — squeeze-aware 2-D loader; mirrors `training 2.ipynb`.
- `load_label_map(path, n_classes=4) -> np.ndarray` — load + convert raw 1-based Ilastik labels to 0-based; sets unannotated pixels to 255.
- `write_label_map_h5(path, arr)` — write a 2-D `uint16` instance label image (gzip-compressed, key `exported_data`).
- `write_ilastik_h5(path, arr)` — write a 2-D class label map back into Ilastik 5-D `uint8` format with axistags (round-trip for human correction).
### `sickling/io/images.py`
- `normalize_image(img, percentile=99.0) -> np.ndarray` — percentile-clip + scale to [0, 1]. Single source of truth, matches `training 2.ipynb`.
- `load_raw_greyscale(path) -> np.ndarray` — load `.jpg/.png/.tif` as 2-D float32 (no normalization).
- `find_raw_image(stem, raw_dir) -> Path | None` — case-aware lookup across `RAW_EXTS`.
### `sickling/io/labels.py`
- `LabelRow` dataclass; `VALID_LABELS = ('sickle', 'non_sickle', 'ambiguous')`.
- `load_labels(path) -> list[LabelRow]` — empty template returns `[]`; bad labels raise.
- `load_conditions(path) -> dict[stem, dict]` — per-FOV metadata keyed by stem.
- `resolve_coordinate_to_instance(label_row, instance_image, pre_instance_image, drop_reasons) -> (instance_id | None, fail_reason | None)` — point-in-mask join. Failure reasons: `coordinate_outside_cell` or `instance_dropped:<reason>`.
- `gate_labels_to_prevalence(cells_df, target_sickle_frac, *, seed, policy) -> (gated_df, stats)` — down-samples whichever modelled class is in excess so the labeled subset hits the target sickle fraction (e.g. `0.10` to mimic natural prevalence on the existing sickle-enriched label corpus). Unlabeled and ambiguous rows pass through. `policy="drop_excess_majority"` is the production setting; the inverse is exposed for symmetric tests.
### `sickling/io/parquet.py`
- `CELLS_COLUMNS` — canonical column order for `cells.parquet`.
- `write_cells(df, path)`, `read_cells(path)`.
### `sickling/io/parquet.py`  *(planned, milestone 3)*
### `sickling/io/labels.py`  *(planned, milestone 3)* — load `labels.csv` and `conditions.csv`; resolve `(source_image, x, y) -> instance_id` by point-in-mask.

### `sickling/data/__init__.py`
Re-exports `CropDataset`, `build_dataset`, `labeled_subset`, `make_weighted_sampler`, augmentations.

### `sickling/data/crop_dataset.py`
- `LABEL_TO_INT = {'non_sickle': 0, 'sickle': 1}`.
- `_resize_3channel(t, target_size)` — channel-aware resize: ch0 bilinear, ch1/ch2 nearest. Tested.
- `CropDataset(cells_df, crops_dir, target_size, return_label, transform, label_to_int)` — lazy per-FOV `.pt` cache.
- `labeled_subset(cells_df, exclude_ambiguous=True)` — filter helper.
- `build_dataset(cfg, *, only_labeled, ...)` — convenience factory.

### `sickling/data/augment.py`
- `train_transform(cfg)` / `eval_transform(cfg)` / `ssl_transform(cfg)` — channel-aware: spatial flips/rot90 across all 3 channels; brightness/contrast jitter on ch0 only (ch1/ch2 stay binary).

### `sickling/data/sampler.py`
- `make_weighted_sampler(labels, minority_frac=0.5)` — `WeightedRandomSampler` factory; targets a configurable minority fraction per batch.

### `sickling/eval/__init__.py`
Re-exports `group_stratified_kfold`, `balanced_group_kfold`, `make_kfold_splits`, `fold_diagnostics`, `BinaryMetrics`, `EvaluationReport`, `compute_binary_metrics`, `bootstrap_metric`, `bootstrap_pr_curve`, `pick_threshold_max_mcc`, `recall_at_precision`, figure builders, `read_report`, `write_report`, `render_all_figures`.

### `sickling/eval/splits.py`
- `group_stratified_kfold(cells_df, n_splits, seed) -> list[(train_idx, val_idx)]` — sklearn-backed FOV-grouped k-fold, stratifies on the FOV-level dominant label. Kept for back-compatibility with the `ablation_20260516_003426` numbers.
- `balanced_group_kfold(cells_df, n_splits, seed) -> list[(train_idx, val_idx)]` — greedy multi-key bin-packer (Karmarkar–Karp-style). Same FOV-leakage-free contract, but assigns FOVs to folds so per-class cell counts are near-equal across folds. Addresses the fold-4 outlier from the discussion section.
- `make_kfold_splits(cells_df, *, n_splits, seed, strategy)` — dispatch over `'stratified' | 'balanced'`. Single hook the CLI layer uses.
- `fold_diagnostics(cells_df, splits) -> pd.DataFrame` — per-fold `[n_val, n_train, n_sickle_val, n_non_sickle_val, n_fovs_val]` table used in tests and the discussion.

### `sickling/eval/metrics.py`
- `BinaryMetrics` dataclass — `pr_auc`, `roc_auc`, `mcc`, `recall_at_p90`, `threshold_at_p90`, `f1_sickle`, `f1_non_sickle`, `threshold`, `confusion (2,2)`.
- `compute_binary_metrics(y_true, y_score, threshold_strategy='max_mcc'|'max_f1'|'fixed', threshold=None, target_precision=0.9) -> BinaryMetrics`.
- `pick_threshold_max_mcc(y_true, y_score)`, `pick_threshold_max_f1(y_true, y_score)`, `recall_at_precision(y_true, y_score, target_precision)`.

### `sickling/eval/bootstrap.py`
- `bootstrap_metric(y_true, y_score, metric_fn, n_resamples=1000, alpha=0.05, seed=42) -> (point, low, high)` — vectorized resampling, NaN-tolerant.
- `bootstrap_pr_curve(...)` — pointwise PR-curve band on a fixed recall grid, returns `{recall_grid, precision_point, precision_mean, precision_low, precision_high}`.

### `sickling/eval/report.py`
- `EvaluationReport` — full snapshot (metrics + CIs + PR band + raw `y_true`/`y_score` for re-rendering without retraining).
- `write_report(report, path)` / `read_report(path)` — JSON serialization (numpy arrays → lists; round-trippable).

### `sickling/eval/figures.py`
- `pr_curve_with_band(report)`, `confusion_matrix_heatmap(report)`, `calibration_plot(report, n_bins=10)` — return `Figure`. Module sets `mpl.rcParams['svg.fonttype'] = 'none'` and `pdf.fonttype = 42` so fonts stay as text in SVG/PDF (editable in Illustrator/Inkscape).
- `render_all_figures(report, output_dir) -> {name: path}` — writes `pr_curve.svg`, `confusion_matrix.svg`, `calibration.svg`.

### `sickling/eval/cli.py`
- `run_evaluate(cfg, checkpoint, variant, *, image_variant, fold, synth_labels, output_dir, bootstrap_resamples, mae_init_ckpt, use_image, use_morphology, zero_mask_channels, zero_image_masks_only) -> EvaluationReport` — reloads any Stage-4/5 checkpoint, scores the val fold, computes metrics + CIs + PR band, writes `report.json` + 3 SVGs. The two zero-mask flags forward to the same dataset constructors used at training time so eval mirrors the ablation row.
- `run_figures(cfg, reports_glob)` — re-render figures from existing `report.json` files; no retraining.

### `sickling/ablation/__init__.py`
Re-exports `AblationRow`, `AblationResult`, `DEFAULT_ABLATION`, `run_ablation_table`, `aggregate_results`, `load_results`, `render_markdown_table`, `render_latex_table`, `write_tables`.

### `sickling/ablation/runner.py`
- `AblationRow` dataclass — one row of the table. Fields: `name`, `variant`, `image_variant`, `use_image`, `use_morphology`, `zero_mask_channels`, `zero_image_masks_only` (per-tower variant: image tower sees zeroed ch1/ch2 but morphology features still computed on the original masks), `overrides` (dotted-key config patches), `notes`.
- `AblationResult` dataclass — one (row, seed, fold) cell with metrics + CIs + checkpoint + duration.
- `DEFAULT_ABLATION` — the PIPELINE_PLAN §4 table (full multimodal, − morphology, − image, − mask channels [both global and per-tower], − weighted sampler, A vs B vs C).
- `run_ablation_table(cfg, rows, seeds, folds, output_dir, synth_labels, skip_existing)` — orchestrates training + evaluation per (row, seed, fold). Persists `raw_results.json` after every run for crash recovery.
- `aggregate_results(results) -> pd.DataFrame` — one row per ablation row, mean / std across (seed × fold).

### `sickling/ablation/render.py`
- `render_markdown_table(agg_df, title)` / `render_latex_table(agg_df, caption, label)` — paper-ready table renderers (booktabs LaTeX with proper underscore escaping).
- `write_tables(agg_df, output_dir, title) -> {markdown, latex}` — writes both alongside the raw results.

### `sickling/engineering/__init__.py`
### `sickling/engineering/seed.py` — `seed_everything(seed, deterministic_cudnn=False)`.
### `sickling/engineering/lightning_utils.py` — `build_wandb_logger`, `build_checkpoint_callback`, `build_trainer` factories. Project/entity/precision sourced from `Config`. `build_trainer` now installs a `DurationCallback` by default so every run produces a `duration.json` next to its checkpoints.
### `sickling/engineering/duration.py`
- `DurationCallback(output_dir, run_name)` — Lightning callback that records per-epoch wall-clock, total fit time, device name, n_devices, strategy, precision, samples seen, and mean images/sec. Writes `duration.json` on `on_fit_end` so single-GPU vs DDP comparisons can be done after-the-fact by diffing two JSONs.
- `_batch_size_of(batch)` — best-effort batch-size extraction across the bare-tensor, `(x, y)`, and `({modality: tensor}, label)` batch shapes used in the project; returns 0 when unrecognised so the timer never crashes training.
### `sickling/engineering/smoke.py` — `_DummyModule` + `run_smoke`. Trains a one-linear-layer regressor on random tensors for 1 epoch on CPU, exercising the entire Lightning + wandb + checkpoint stack.
### `sickling/engineering/ddp_benchmark.py`
- `_ThroughputTimer` Lightning callback — warmup-then-measure throughput in images/sec.
- `_single_run(devices, batch_size, steps, warmup)` — runs MAE pretraining for a fixed step budget with the given GPU count.
- `run_benchmark(devices_list, batch_size, steps, warmup, output_dir)` — sweeps device counts, writes `throughput.csv` + `throughput.svg` (matched bar chart, ideal vs measured, with scaling-efficiency labels).
- `__main__` entrypoint: `python -m sickling.engineering.ddp_benchmark --devices 1 --devices 2 --devices 4`.

### `sickling/cli/__init__.py`
### `sickling/cli/main.py` — Typer app. All subcommands live: `smoke`, `predict`, `instances`, `crops` (with `--labels-csv`), `pretrain`, `finetune` (with `--target-sickle-frac` and `--fold-strategy`), `evaluate`, `figures`, `ablate` (with `--target-sickle-frac` and `--fold-strategy`).

## `Makefile` targets

| target | command |
|---|---|
| `install` / `dev-install` | `pip install -r requirements.txt` + editable install |
| `smoke` / `predict` / `instances` / `crops` / `pretrain` / `finetune` / `evaluate` / `figures` / `ablate` | one-liners over the CLI subcommands |
| `bench-ddp` | `python -m sickling.engineering.ddp_benchmark --devices 1` |
| `reproduce` | end-to-end `predict → instances → crops → ablate → figures`. Default `LABELS_CSV=labels/labels_trimmed.csv` `SEEDS=42` `FOLDS=0`. |
| `reproduce-full` | re-invokes `reproduce` with `LABELS_CSV=labels/labels.csv SEEDS=42,43,44 FOLDS=0,1,2,3,4` |
| `test` / `lint` / `clean` | pytest / ruff / workspace cleanup |

## Configs

- `configs/base.yaml` — paths, channel definitions, all hyperparameter defaults. Now includes `validation.fold_strategy` (`balanced` default), `validation.target_sickle_frac` (null default), and `validation.gate_seed`.
- `configs/pretrain_mae.yaml` — original 300-epoch single-GPU MAE continuation schedule (preserved for reproducibility).
- `configs/pretrain_mae_long.yaml` — extended 800-epoch single-GPU MAE schedule with `batch_size=192`, `grad_accum=2` (effective batch 384), warmup 40 epochs. Targets the discussion-section limitation 6 that the previous MAE run was likely under-trained.
- `configs/smoke.yaml` — tiny override for `make smoke`.
- `configs/sweeps/` — wandb sweep YAMLs (planned).

## Tests

- `tests/conftest.py` — synthetic fixtures: `synth_label_map` (256² FOV with 5 simulated cells: isolated, polymer-ringed, touching pair, edge-touching), `synth_label_map_with_blobs` (adds tiny + oversized blobs to exercise area filters).
- `tests/test_instances.py` — eight tests covering basic separation, polymer-ring containment, min/max-area filtering, edge handling, 2-D shape requirement, empty input, and a real-h5 smoke test gated on `unet_predictions/PRED_D16_03_1_1_Bright Field_001.h5`.
- `tests/test_labels.py` — eight tests for `load_labels` / `load_conditions` / `resolve_coordinate_to_instance` (kept-cell, dropped-edge, background, out-of-bounds, invalid-label) plus six new tests for `gate_labels_to_prevalence` (drops excess sickle to a target rate, drops excess non-sickle when corpus is sickle-poor, determinism, passthrough rows untouched, invalid-target rejection, requires both classes present).
- `tests/test_duration.py` — two tests for `DurationCallback` (writes `duration.json` with the expected schema after a CPU smoke fit; `_batch_size_of` handles the three batch shapes used in the project).
- `tests/test_crops.py` — seven tests for crop extraction (channel shape, drop-when-clipped, pad-when-not, polymer-only-inside-instance, no-other-instance-leakage, aligned outputs) plus a real-FOV end-to-end smoke through `run_stage3`.
- `tests/test_dataset.py` — six tests for `CropDataset` + augmentations (channel-aware resize, labeled vs unlabeled return, binary-mask preservation through random flips, lazy FOV caching) plus three new tests for `MultimodalCropDataset` mask-zeroing semantics (image-tower-only zero leaves morphology cache intact, global zero leaves morphology cache intact for legacy reasons, mutually-exclusive flag check).
- `tests/test_splits.py` — five tests for `group_stratified_kfold` (no FOV overlap, every FOV eventually in val, unlabeled rows train-only, no-labels degenerate case, balanced sickle representation per fold) plus five new tests for `balanced_group_kfold` (no FOV leakage, per-class count range $\leq$ stratified, unlabeled rows train-only, rejects $n\_{fovs}<n\_{splits}$, `make_kfold_splits` dispatch).
- `tests/test_encoders.py` — eight tests, weights downloaded once via torch.hub / huggingface (fall back to skip if offline). Covers forward shape for A/B/C, frozen-DINOv2 has no trainable params, LLRD produces decreasing LRs, MAE patchify + random-masking + reconstructor end-to-end.
- `tests/test_morphology_features.py` — nine tests: known-shape regressions (circle compactness ≈ 4π, ellipse high eccentricity, square compactness, polymer-ratio bounds, blank-input zero-handling, 2-D rejection, full feature-vector shape).
- `tests/test_towers.py` — six tests for `Tower` contract, `MorphologyTower` MLP shape + buffer round-trip, `MultimodalClassifier` concat / forward / missing-modality error / per-tower LRs.
- `tests/test_tower_extension.py` — single test demonstrating the 5-line modality-extension contract (PIPELINE_PLAN §2 Stage 5).
- `tests/test_metrics.py` — eight tests: perfect / inverted / random classifier behavior, recall@p=0.9 reachable + unreachable cases, threshold pickers, confusion matrix layout, single-class NaN handling.
- `tests/test_bootstrap.py` — four tests: CI contains point estimate, narrower for separable problems, PR-band shape + bound ordering, NaN tolerance under heavy imbalance.
- `tests/test_figures.py` — five tests: each builder returns a valid Figure, `render_all_figures` writes 3 SVGs, `EvaluationReport` JSON round-trip preserves arrays + metrics.
- `tests/test_ablation.py` — four tests: `aggregate_results` groups by row name, markdown renderer contains every row + canonical columns, LaTeX renderer is booktabs-valid with underscore escaping, single-run aggregates have zero std (not NaN).

## Notebooks

- `notebooks/02_instance_qa.ipynb` — interactive Stage 2 QA: renders the 4-panel figure for any FOV, sweeps `min_area` to inspect the bimodal area distribution, saves the chosen panel to `figures/`.
- `notebooks/02_annotate.ipynb` — Tk-based cell-crop labeler. Reads from `to_be_labeled/`, runs the frozen U-Net + watershed, iterates kept instances, hotkeys `8`/`9`/`0`/`z`/`s`. Appends to `labels/labels.csv` and dedups against existing rows so sessions are resumable.

## Inputs (filled by user)

- `unet_predictions/` — full-FOV 4-class `.h5`.
- `raw_images/` — matching raw images.
- `labels/labels.csv` + `labels/README.md`.
- `conditions/conditions.csv` + `conditions/README.md`.
