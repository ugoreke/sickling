# ARCHITECTURE.md

> Top-level design map of the `sickling` repository.
> Arm-specific design docs live at
> [`sickling/protrusion_detection/ARCHITECTURE.md`](sickling/protrusion_detection/ARCHITECTURE.md)
> and [`sickling/rbc_classification/architecture.md`](sickling/rbc_classification/architecture.md);
> this file links into them.
> Last updated: 2026-06-29 (paper-aligned naming).

## 1. Goal

End-to-end characterization of HbS protrusion formation in sickled
gene-edited HSPC-derived RBCs from bright-field microscopy. The
pipeline below feeds the paper's main biology figure (per-condition
protrusion µm per sickle cell) and the U-Net + classifier supplementary
figures.

```
                raw bright-field images
                         │
                         ▼
              ┌──────────────────────────┐
              │  protrusion_detection    │  HITL-trained 4-class U-Net
              │   (sickling.protrusion_  │  → protrusion / bg / cell body /
              │    detection)            │     cell boundary
              └────────────┬─────────────┘
                           │  per-pixel mask
                           ▼
              ┌──────────────────────────┐
              │  rbc_classification      │  watershed → 96×96×3 cell crops
              │   (sickling.rbc_         │  → DINOv2 frozen + 30-d morphology
              │    classification)       │     multimodal classifier
              └────────────┬─────────────┘  → sickle vs non-sickle per cell
                           │
                           ▼
            per-FOV / per-condition biology metrics
            (notebooks/analysis_protrusion_per_condition.ipynb)
            - pool protrusion µm per sickle cell, by condition
            - pool sickle fraction, by condition
            - bootstrap 95% CIs over FOV resampling
```

**No train / eval leakage.** The U-Net and classifier are trained on
labelled image sets that are disjoint from the per-condition images
used for the downstream biology figures.

---

## 2. Protrusion-detection arm — HITL U-Net

See [`sickling/protrusion_detection/ARCHITECTURE.md`](sickling/protrusion_detection/ARCHITECTURE.md)
for full design.

**Classes** (class index → meaning):

| idx | meaning | comment |
|---|---|---|
| 0 | HbS protrusion | rare, thin |
| 1 | background | abundant |
| 2 | cell body | abundant |
| 3 | cell boundary | thin, between RBCs |

**Training pool**: `BootstrappedLabels/` (Ilastik-densified
pseudo-labels from sparse manual annotations) plus HITL-corrected
sparse 512-px tiles (`CorrectedTiles/`) and target-class-only mini-crops
(`MiniTilesCorrected/`). `InitialLabels/` (2 dense hand-labelled FOVs)
is held out as test; 3 additional dense FOVs are used for per-epoch
checkpoint selection.

**Model**: from-scratch U-Net (4 down / up stages, BatchNorm + ReLU,
1-channel input, 4-class logits). Backbone is swappable via
`cfg.MODEL_BACKBONE` (vanilla U-Net default, `smp_unet_resnet34/101`
and `smp_unet_efficientnet-b0/b7` available for the ablation reported
in Supp. Fig. 4).

**Loss**: composite of weighted Dice + cross-entropy, a directed
confusion penalty on the protrusion-to-background error mode (weight
0.3), a false-negative penalty on the rare classes (weight 0.1), and a
Tversky term on the protrusion class (α=0.4, β=0.6, weight 0.3).

**Tile sampling**: class-0-aware — 50% probability of seeding a crop on
a protrusion pixel — to compensate for the rarity of the protrusion
class.

**Training**: 256-px tiles, batch 16, 100 optimizer steps per epoch, 50
epochs, AdamW lr=1e-4, 5-fold CV. 8-way flip / rotation TTA at eval.

**Loop-versioned checkpoints**: every retrain writes
`<backbone>_fold_<f>_best_loop_<N>.pth` (no overwrite) so prior
generations stay available for rollback / A-B comparisons.

**Protrusion-length quantification**: connected components on the
class-0 mask → fit 2-D inertia ellipse per component → drop components
with major axis < 10 px → sum major axes in µm using the 40× Leica
calibration (500 µm per 3100 px).

---

## 3. RBC-classification arm — sickle classifier

See [`sickling/rbc_classification/architecture.md`](sickling/rbc_classification/architecture.md)
for full design.

Five staged sub-packages:

- **`stage1_unet`** — frozen U-Net 4-class inference per FOV, writes
  `unet_predictions/PRED_<stem>.h5`.
- **`stage2_instances`** — watershed instance segmentation. Foreground =
  `protrusion ∪ cell_body`; morphological closing → EDT → marker-seeded
  watershed. Per-instance area / edge filtering with drop reasons
  captured for QA.
- **`stage3_crops`** — each kept instance becomes a 96×96×3 tensor:
  channel 0 = normalised raw, channel 1 = cell-body mask, channel 2 =
  protrusion mask. Aligned with `cells.parquet`.
- **`stage4_repr`** — image representation. The published model uses
  the **frozen DINOv2 ViT-S/14** (384-d CLS token, no fine-tuning); the
  package also exposes finetune variants (timm-ViT, MAE-pretrained ViT)
  used only during pre-publication exploration.
- **`stage5_multimodal`** — image tower + 30-d morphology MLP, fused
  via a 2-layer MLP into a 2-class softmax head. Per-fold
  standardization of morphology features on the train subset.

**Training**: AdamW + cosine LR + 10% warmup; weighted sampler
targeting 50% sickle per batch. 5-fold FOV-grouped CV (no FOV
leakage). Label-prevalence gate at fold-build time preserves the
natural ~10% sickle prevalence so reported metrics reflect a realistic
operating regime. Decision threshold picked to maximise MCC on the OOF
predictions. 95% CIs are non-parametric bootstrap percentiles over 1000
resamples.

**Downstream data** (already generated, lives in
`sickling/rbc_classification/experiment_data/`):

- `per_cell.parquet` — one row per kept cell.
- `per_fov.parquet` — one row per image (counts + protrusion-length
  aggregates).
- `per_fov_dist10.parquet` — same as above, recomputed with the
  `POLYMER_MAX_DIST_FROM_CELL_PX = 10` filter (cleaner per-FOV
  measurements; used in the paper's main biology figure).
- `polymer_blobs.parquet` / `polymer_blobs_dist10.parquet` — per-blob
  geometry + drop reasons.
- `per_condition.parquet`, `pairwise_stats.parquet` — sums / medians
  per condition; pairwise Mann-Whitney + BH-FDR.

The schema for every column is documented in
[`sickling/rbc_classification/experiment_data/batch_classify_output_schema.md`](sickling/rbc_classification/experiment_data/batch_classify_output_schema.md).

---

## 4. Cross-arm notebooks

Live at `notebooks/` (top level).

### 4.1 `notebooks/colab_demo.ipynb`

End-to-end demo on the bundled `sample.jpg`. Loads both checkpoints,
runs U-Net → watershed → classifier, displays the colored overlay and
prints per-FOV summary numbers (total protrusion µm, sickle fraction,
µm protrusion per sickle cell). Designed to run unmodified in Google
Colab with a T4 GPU after pasting the OSF checkpoint URLs into the
*Setup* cell.

### 4.2 `notebooks/pixel_confusion_matrix.ipynb`

U-Net pixel-level performance on the held-out `InitialLabels` test
split. Runs the current best fold/loop checkpoint with TTA, builds the
4×4 pixel confusion matrix, renders **row-normalised (recall)** and
**column-normalised (precision)** heatmap views, writes them as
editable-text SVGs for the paper. Also dumps per-class
recall / precision / F1 + raw counts.

### 4.3 `notebooks/protrusion_length_grid.ipynb`

U-Net protrusion-length accuracy vs human ground truth. Workflow:

1. `tools/build_grid.py` tiles the 100 v4 eval crops into a single
   2560×2560 `grid_10x10.png`. Each cell shows one eval region centered
   in a 256-px tile, with light-gray padding and a dashed dotted frame
   marking what's graded.
2. Operator places Photoshop Count-tool markers at protrusion endpoints
   inside each dashed frame, exports via `count2csv.jsx`.
3. The notebook pairs consecutive marker rows, clips to the
   per-crop eval region (Liang-Barsky), and computes manual length as
   Euclidean distance between paired endpoints. Model length is
   computed two ways (skeleton pixel count and `regionprops` major
   axis) over CCs in the same eval region.
4. Output: manual-vs-model scatter + Bland–Altman per metric.

### 4.4 `notebooks/analysis_protrusion_per_condition.ipynb`

**Main biology figure.** Reads
`rbc_classification/experiment_data/per_fov_dist10.parquet` and
produces the per-condition pool stats (sickle fraction, µm protrusion
per sickle cell) with bootstrap 95% CIs over FOV resampling. No
per-FOV minimum-sickle filter is applied to the pool metric — by
construction, pool aggregation isn't sensitive to per-FOV denominator
volatility. The crowding upper bound `MAX_CELLS ≤ 500` is still
applied as a biological cleanliness filter.

Per-condition Mann-Whitney vs the negative control with BH-FDR
correction over the comparison set; output table goes into the paper as
the per-condition stats panel.

### 4.5 `notebooks/sickle_classifier_confusion_matrix.ipynb`

**Classifier supplementary figure.** Reads the 5 fold OOF
`report.json` dumps in `sickling/rbc_classification/eval_reports/`,
concatenates into a single OOF set, computes the MCC-maximising
threshold, and renders the three-panel composite: (A) PR curve with
1000-bootstrap 95% CI band, (B) confusion matrix at the OOF threshold,
(C) reliability / calibration plot. Saves SVG/PNG/CSV.

---

## 5. Repository layout

```
sickling/
├── pyproject.toml
├── README.md
├── ARCHITECTURE.md                            ← you are here
├── sample.jpg                                 demo image
├── .gitignore                                 excludes large data; see OSF below
│
├── notebooks/                                 cross-arm notebooks
│   ├── colab_demo.ipynb                       (§4.1)
│   ├── pixel_confusion_matrix.ipynb           (§4.2)
│   ├── protrusion_length_grid.ipynb           (§4.3)
│   ├── analysis_protrusion_per_condition.ipynb  (§4.4)
│   ├── sickle_classifier_confusion_matrix.ipynb (§4.5)
│   └── figures/                               generated paper figure SVG/PNG/CSV
│
├── tools/
│   └── build_grid.py
│
└── sickling/                                  installable Python package
    ├── __init__.py                            umbrella; re-exports both arms
    ├── protrusion_detection/                  ARM 1
    │   ├── __init__.py                        aliases py_modules → arm namespace
    │   ├── HITL_pipeline.ipynb
    │   ├── ARCHITECTURE.md / CHANGELOG.md / GUIDE.md
    │   ├── count2crops.jsx / count2csv.jsx    Photoshop scripts
    │   └── py_modules/                        actual implementation
    │
    └── rbc_classification/                    ARM 2
        ├── __init__.py                        aliases py_modules → arm namespace
        ├── orchestrate.ipynb
        ├── architecture.md / README.md / RESULTS.md / changelog.md
        ├── configs/                           training configs
        ├── eval_reports/                      cached OOF predictions (5 folds)
        ├── experiment_data/                   parquet outputs (paper inputs)
        ├── labels/                            sickle / non_sickle ground truth
        ├── notebooks/                         arm-specific notebooks
        ├── tests/                             pytest test suite
        └── py_modules/                        actual implementation
```

---

## 6. Install + import patterns

```bash
git clone https://github.com/ugoreke/sickling.git
cd sickling
pip install -e ".[classification]"
```

Both arms are reachable through the umbrella package:

```python
# Arm 1 — U-Net inference helpers
from sickling.protrusion_detection.config import cfg
from sickling.protrusion_detection.model import UNet
from sickling.protrusion_detection.inference import predict_probs, predict_mask
from sickling.protrusion_detection.masks import normalize_image, load_dense_mask

# Arm 2 — classifier inference + eval helpers
from sickling.rbc_classification.stage1_unet.inference import load_unet, predict_label_map
from sickling.rbc_classification.stage2_instances.watershed import mask_to_instances
from sickling.rbc_classification.stage5_multimodal.classifier import MultimodalClassifier
from sickling.rbc_classification.eval.report import read_report
```

Each arm's top-level `__init__.py` aliases the modules under its
private `py_modules/` folder back to the arm namespace, so users never
need to type `py_modules` explicitly.

---

## 7. Reproducibility

Code + small parquet outputs + cached OOF dumps + sample.jpg are in
this repo. Raw images, U-Net training data, U-Net checkpoints,
classifier checkpoints, and per-condition image folders live on OSF:

| Path in repo (gitignored) | OSF deposit (https://osf.io/gnec4/) |
|---|---|
| `sickling/protrusion_detection/InitialLabels/` | held-out U-Net test masks |
| `sickling/protrusion_detection/BootstrappedLabels/` | dense pseudo-label training pool |
| `sickling/protrusion_detection/CorrectedTiles/` | HITL-painted training tiles |
| `sickling/protrusion_detection/CorrectionPool/` | raw correction-pool images + PRED files |
| `sickling/protrusion_detection/models/` | versioned U-Net checkpoints |
| `sickling/rbc_classification/checkpoints/` | multimodal classifier checkpoints |
| `sickling/rbc_classification/raw_images/` | raw FOV bright-field stacks |
| `sickling/rbc_classification/experiment_data/{condition}/` | per-condition processed images |
| `sickling/rbc_classification/experiment_data/per_cell_morphology.pt` | 30-d morphology vectors |

After cloning the repo, download the matching OSF subset and place the
files at the indicated paths to reproduce the full pipeline. The demo
notebook (`notebooks/colab_demo.ipynb`) and the supplementary-figure
regen notebooks (`pixel_confusion_matrix.ipynb`,
`sickle_classifier_confusion_matrix.ipynb`) work without the OSF data —
they consume only the small artifacts committed to git.
