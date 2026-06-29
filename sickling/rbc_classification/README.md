# Sickle Cell Classification Pipeline

End-to-end pipeline from microscopy field-of-view (FOV) images to per-cell
sickle / non-sickle labels. Stage 1 (semantic U-Net, `training 2.ipynb`) is
frozen; this package implements stages 2–5 plus evaluation, ablation, and
reproducibility scaffolding.

See `PIPELINE_PLAN.md` for design and `ARCHITECTURE.md` for a file-by-file
index. Per-milestone changes live in `changelog.md`.

## One-line reproduce

After dropping raw images into `to_be_labeled/` and a U-Net checkpoint into
`models/unet_fold_2_best.pth`:

```bash
make dev-install
make reproduce                                            # uses labels/labels_trimmed.csv
make reproduce-full                                       # full labels, 3 seeds × 5 folds
```

`make reproduce` runs `predict → instances → crops → ablate → figures` end-to-end.
Stages are idempotent — re-running skips already-cached outputs and resumes a
partial ablation table from `figures/ablation/<ts>/raw_results.json`.

## CLI

Every command is also reachable as `python -m sickling.cli.main <cmd>` or via
the `sickling` console script (after `pip install -e .`):

| command | what it does |
|---|---|
| `sickling smoke` | dummy LightningModule sanity run (1 epoch CPU) |
| `sickling predict` | bulk U-Net inference over `to_be_labeled/` → `unet_predictions/` + `raw_images/` |
| `sickling instances` | Stage 2 watershed (add `--qa` for QA panels) |
| `sickling crops` | Stage 3 — extract per-cell tensors + `cells.parquet`. `--labels-csv` to swap source |
| `sickling pretrain` | Stage 4 — MAE continuation pretraining (Model C) |
| `sickling finetune VARIANT` | fine-tune `dinov2_frozen` / `timm_vit` / `mae` / `multimodal` |
| `sickling evaluate CKPT --variant ...` | metrics + bootstrap CIs + SVG figures |
| `sickling figures` | re-render SVGs from saved `report.json` files |
| `sickling ablate --seeds ... --folds ...` | PIPELINE_PLAN §4 table → Markdown + LaTeX |

## Notebooks

- `orchestrate.ipynb` *(at repo root)* — drives every stage with Python-level
  config overrides (no YAML strings). Flag-gated cells (`RUN_PREDICT`, `RUN_INSTANCES`,
  …). Inline Markdown render of the latest ablation table.
- `notebooks/02_annotate.ipynb` — Tk-based cell-crop labeler. Hotkeys `8`/`9`/`0`/`z`/`s`.
  Priority-sort by polymer-at-boundary score. Set `REDO_MODE = True` to edit existing labels.
- `notebooks/02_instance_qa.ipynb` — Stage 2 QA panel + `min_area` sweep.

## Where to put input data

| directory | contents |
|---|---|
| `to_be_labeled/` | raw `.jpg`/`.png` FOVs you want predicted + labeled (`sickling predict` populates the rest) |
| `unet_predictions/` | 4-class label maps (`.h5`, Ilastik 1-based) — produced by `sickling predict` |
| `raw_images/` | matching raws — copied here by `sickling predict` |
| `labels/labels.csv` | per-cell sickle labels (coordinate-based; schema in `labels/README.md`) |
| `labels/labels_trimmed.csv` | optional fast-iteration subset |
| `conditions/conditions.csv` | per-FOV metadata (oxygen, treatment); schema in `conditions/README.md` |
| `models/unet_fold_*_best.pth` | frozen U-Net checkpoints from `training 2.ipynb` |

## Configuration

- `configs/base.yaml` is the single source of truth — see `sickling/config.py`
  for the typed schema.
- Per-stage YAMLs (`configs/finetune_modelA.yaml`, `configs/smoke_*.yaml`)
  deep-merge on top.
- Environment variables prefixed `SICKLING_` override either
  (e.g. `SICKLING_PROJECT__WANDB_ENTITY=my-team`, `SICKLING_TRAINING__LR=5e-5`).
- The `orchestrate.ipynb` notebook mutates `cfg.<...>` directly in Python
  instead of writing YAML.

## Weights & Biases

Project: `sickling-classifier`. Set your entity once:

```bash
wandb login
export SICKLING_PROJECT__WANDB_ENTITY=<your_username_or_team>
```

Smoke tests run offline; production stages run online when an entity is set.
Bayesian sweep configs at `configs/sweeps/bakeoff.yaml` and `multimodal_hp.yaml`.

## DDP / multi-GPU

The MAE pretraining run is the only stage that benefits from multi-GPU. Run
the benchmark with whatever GPUs you have:

```bash
python -m sickling.engineering.ddp_benchmark --devices 1 --devices 2 --devices 4
```

Output: `figures/ddp_benchmark/throughput.{csv,svg}`. CSV column
`scaling_efficiency` is the ratio of measured to linear-ideal throughput.

## Tests

```bash
make test          # full suite (75 tests; encoder tests download weights — skip if offline)
make lint          # ruff
```

## Hardware

Default local target: A4000 16 GB. The MAE continuation pretraining is meant
for a rented multi-GPU box (RunPod); everything else fits on the A4000.
