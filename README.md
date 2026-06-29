# sickling

Machine-learning pipelines for the bright-field functional assessment of
gene-edited sickle red blood cells described in:

> **Quantifying the Functional Phenotype of Sickled Red Cells Derived
> from Gene-Edited Hematopoietic Stem Cells Using Machine Learning**
> Goreke et al., *Molecular Therapy Advances* (in press).

Two coupled arms:

| Arm | What it does | Backbone |
|---|---|---|
| **`sickling.protrusion_detection`** | 4-class semantic segmentation of HbS protrusion, background, cell body, cell boundary. Trained with a human-in-the-loop (HITL) correction process from sparse manual annotations expanded with Ilastik. | From-scratch U-Net |
| **`sickling.rbc_classification`** | Per-cell sickle / non-sickle classification on 96×96 crops harvested via watershed instance segmentation from the U-Net output. | Frozen DINOv2 ViT-S/14 + MLP morphology tower (30 hand-crafted descriptors), 2-layer fusion head |

The two arms share no training data with the held-out
control / treatment images used for the downstream biological figures,
so the published per-condition numbers are computed on never-seen FOVs.

---

## Quick start

### Install

```bash
git clone https://github.com/ugoreke/sickling.git
cd sickling
pip install -e ".[classification]"
```

### Run the demo

`notebooks/colab_demo.ipynb` runs both arms end-to-end on the bundled
`sample.jpg`. Open it in Google Colab; edit the OSF checkpoint URLs in
the *Setup* cell (raw data + weights live at
[https://osf.io/gnec4/](https://osf.io/gnec4/)); execute top-to-bottom.

Local equivalent:

```python
from sickling.rbc_classification.stage1_unet.inference import load_unet, predict_label_map
from sickling.rbc_classification.io.images import normalize_image
import numpy as np
from PIL import Image

unet = load_unet("path/to/unet.pth", n_classes=4)
raw  = np.array(Image.open("sample.jpg").convert("L"))
mask = predict_label_map(unet, normalize_image(raw, 99.0))   # 0=protrusion, 1=bg, 2=cell, 3=boundary
print("protrusion fraction:", (mask == 0).mean())
```

---

## Repository layout

```
sickling/
├── pyproject.toml
├── README.md                                       ← you are here
├── ARCHITECTURE.md                                 top-level design
├── sample.jpg                                      bundled demo image
│
├── notebooks/                                      cross-arm / paper figures
│   ├── colab_demo.ipynb                            end-to-end demo, Colab-ready
│   ├── pixel_confusion_matrix.ipynb                pixel-level supp. figure (U-Net)
│   ├── protrusion_length_grid.ipynb                model vs manual protrusion length
│   ├── analysis_protrusion_per_condition.ipynb     main biology figure
│   ├── sickle_classifier_confusion_matrix.ipynb    classifier supp. figure
│   └── figures/                                    generated paper figures
│
├── tools/                                          one-shot helpers
│   └── build_grid.py                               assembles the 10×10 eval grid
│
└── sickling/                                       installable Python package
    ├── __init__.py                                 re-exports both arms
    ├── protrusion_detection/                       Arm 1: HITL U-Net
    │   ├── HITL_pipeline.ipynb                     operator notebook
    │   ├── ARCHITECTURE.md / CHANGELOG.md / GUIDE.md
    │   └── py_modules/                             actual Python source
    └── rbc_classification/                         Arm 2: sickle classifier
        ├── orchestrate.ipynb                       full classifier pipeline driver
        ├── notebooks/batch_classify.ipynb          batch inference over all FOVs
        ├── notebooks/make_publication_figures_v4.ipynb
        ├── architecture.md / RESULTS.md / README.md
        ├── configs/                                base.yaml + variants
        ├── eval_reports/                           cached OOF fold predictions
        ├── experiment_data/                        per_fov, per_cell, polymer_blobs parquets
        ├── labels/                                 sickle / non_sickle ground truth
        └── py_modules/                             actual Python source
```

Data, models, and raw image stacks **are not committed to git** — they
live on OSF: [https://osf.io/gnec4/](https://osf.io/gnec4/). The
`.gitignore` reflects this. See *Reproducibility* below for which OSF
file maps to which local path.

---

## Reproducing the paper figures

| Figure / table | Notebook | Inputs |
|---|---|---|
| Main biology (per-condition protrusion µm / sickle cell, pool stats) | `notebooks/analysis_protrusion_per_condition.ipynb` | `experiment_data/per_fov_dist10.parquet`, `polymer_blobs_dist10.parquet` |
| Supp. fig — pixel-level U-Net confusion + per-class F1 | `notebooks/pixel_confusion_matrix.ipynb` | held-out U-Net test FOVs + best checkpoint |
| Supp. fig — model vs manual protrusion length | `notebooks/protrusion_length_grid.ipynb` | `MiniTilesForEval/grid_10x10.png` + Photoshop CSV |
| Supp. fig — classifier PR + confusion + calibration | `notebooks/sickle_classifier_confusion_matrix.ipynb` | `eval_reports/*/report.json` (5 folds, committed) |
| Supp. fig — segmentation loss-and-architecture ablation | `sickling/protrusion_detection/...` (ablation runner) | 7 model variants × 5 folds |

All saved figures use `mpl.rcParams['svg.fonttype'] = 'none'` so SVG
text stays editable in Illustrator / Inkscape.

---

## HITL training pipeline

Operator opens `sickling/protrusion_detection/HITL_pipeline.ipynb` and
walks through one HITL loop:

1. **Predict** on the correction pool with the current best fold.
2. **Mine** false-negative-aware tiles (soft protrusion-probability +
   cross-fold disagreement).
3. **Paint** corrections in Ilastik (sparse manual annotations expanded
   to dense pseudo-labels).
4. **Retrain** from scratch on the updated pool (5-fold or single
   carryover after a k-fold round).
5. **Measure** — per-class Dice on val, TP / FP plot, polymer-only
   monitor on `BootstrappedLabels`. Trajectory in
   `metrics/iteration_log.csv`.

See [`sickling/protrusion_detection/GUIDE.md`](sickling/protrusion_detection/GUIDE.md)
for the operator-level walkthrough and
[`sickling/protrusion_detection/ARCHITECTURE.md`](sickling/protrusion_detection/ARCHITECTURE.md)
for the design rationale (loss composition, class-aware tile sampling,
loop-versioned checkpoints, etc).

---

## Sickle classification pipeline

Five stages (`sickling.rbc_classification.{stage1_unet,
stage2_instances, stage3_crops, stage4_repr, stage5_multimodal}`):

1. **stage1_unet** — frozen U-Net 4-class inference per FOV.
2. **stage2_instances** — watershed instance segmentation from
   `cell_body ∪ protrusion`.
3. **stage3_crops** — 96×96×3 per-cell tensors (raw / cell-body mask /
   protrusion mask).
4. **stage4_repr** — DINOv2 ViT-S/14 frozen image tower (384-d CLS).
5. **stage5_multimodal** — fusion of image tower + 30-d morphology MLP,
   2-class softmax. 5-fold FOV-grouped CV, MCC-maximising decision
   threshold, 1000-bootstrap CIs.

Driver notebook: `sickling/rbc_classification/orchestrate.ipynb`. Batch
inference across the full per-condition dataset:
`sickling/rbc_classification/notebooks/batch_classify.ipynb`.

---

## Terminology

The paper uses **HbS protrusion** for the rigid HbS-dependent structure
that pokes beyond the deoxygenated RBC membrane. The codebase uses
*protrusion* in module / class / function names. A few parquet column
names retain the legacy `polymer_*` prefix (e.g. `polymer_length_um`)
because they were generated before the manuscript landed on the
"protrusion" wording; their values *are* the HbS protrusion length /
mask / area / etc. of the paper.

---

## License

MIT. See [`LICENSE`](LICENSE) (if present) or the SPDX identifier in
`pyproject.toml`.

---

## Citation

Bibtex once the manuscript is in print:

```bibtex
@article{goreke2026hbs,
  author  = {Goreke, Utku and Chen, Julia and Ansong-Ansongton, Yaw Ofosu Nyansa and Perez, Nathan M. and Kamath, Dipti and Gurkan, Umut and Giannikopoulos, Petros and Nguyen, David},
  title   = {Quantifying the Functional Phenotype of Sickled Red Cells Derived from Gene-Edited Hematopoietic Stem Cells Using Machine Learning},
  journal = {Molecular Therapy Advances},
  year    = {2026},
}
```
