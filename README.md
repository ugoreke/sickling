# sickling

Two machine-learning models for the bright-field functional assessment of
gene-edited sickle red blood cells published in:

> **Quantifying the Functional Phenotype of Sickled Red Cells Derived
> from Gene-Edited Hematopoietic Stem Cells Using Machine Learning**
> Goreke et al., *Molecular Therapy Advances* (in press).

This repository is a **model-development showcase**: it walks you through
how the two models were designed, trained, and validated, and lets you
run them on a bundled bright-field sample. It is **not** a paper-figure
reproduction package — the raw per-condition image dataset is large and
lives off-repo (see [*Data & weights*](#data--weights) below).

---

## The two models

| Arm | What it does | Backbone | Where it lives |
|---|---|---|---|
| **`sickling.protrusion_detection`** | 4-class semantic segmentation of HbS protrusion / background / cell body / cell boundary. Trained with a **human-in-the-loop** correction process from sparse manual annotations expanded with Ilastik. | From-scratch U-Net (256-px tiles, class-0-aware sampling, composite Dice + CE + Tversky + directed FN penalty loss) | `sickling/protrusion_detection/` |
| **`sickling.rbc_classification`** | Per-cell sickle / non-sickle classification on 96×96 crops harvested via watershed instance segmentation from the U-Net output. | **Frozen DINOv2 ViT-S/14** image tower + MLP morphology tower over 30 hand-crafted shape descriptors, 2-layer fusion head | `sickling/rbc_classification/` |

Chain them and you get, per FOV: total HbS protrusion length (µm),
sickle fraction, and µm of protrusion per sickle cell — the biological
readout the paper reports.

---

## Try it in Colab (5 minutes)

Open [`notebooks/colab_demo.ipynb`](notebooks/colab_demo.ipynb) in Google
Colab. Paste the checkpoint URLs into the *Setup* cell (see
[*Data & weights*](#data--weights)) and run top-to-bottom. The notebook
loads the bundled `sample.jpg`, runs the U-Net to produce the 4-class
mask, watershed-segments individual RBCs, classifies each one as
sickle / non-sickle, and prints the per-FOV summary numbers.

Local one-liner:

```python
from sickling.rbc_classification.stage1_unet.inference import load_unet, predict_label_map
from sickling.rbc_classification.io.images import normalize_image
import numpy as np
from PIL import Image

unet = load_unet("path/to/unet.pth", n_classes=4)
raw  = np.array(Image.open("sample.jpg").convert("L"))
mask = predict_label_map(unet, normalize_image(raw, 99.0))    # 0=protrusion, 1=bg, 2=cell body, 3=cell boundary
print("protrusion fraction:", (mask == 0).mean())
```

---

## What each notebook shows

All under `notebooks/`:

| Notebook | Shows | Runnable? |
|---|---|---|
| `colab_demo.ipynb` | Both models end-to-end on the bundled sample | Yes (needs checkpoint URLs from Google Drive) |
| `sickle_classifier_confusion_matrix.ipynb` | Classifier eval on the 5-fold OOF predictions committed under `eval_reports/` | Yes (self-contained) |
| `pixel_confusion_matrix.ipynb` | U-Net pixel-level performance on the held-out `InitialLabels` test split | Needs the U-Net checkpoint + `InitialLabels/*.h5` (Google Drive) |
| `protrusion_length_grid.ipynb` | U-Net protrusion-length accuracy vs manual Photoshop-Count measurement on a 10×10 evaluation grid | Needs `MiniTilesForEval/` + `grid_10x10_counts.csv` (Google Drive) |
| `analysis_protrusion_per_condition.ipynb` | Reproduces the paper's Figure 2e metric — pool per-condition protrusion µm per sickle cell + sickle fraction + bootstrap CIs (see "Idea O" cell at the bottom). Earlier cells contain the methodology exploration (fiber decomposition, threshold sweeps, per-FOV distribution views, dist-10 persistence step) as a record of how the paper metric was arrived at. | Yes for Idea O + the exploration cells (committed parquets); full re-sweeps need per-condition images (Google Drive) |

Plus the model-arm notebooks:

- `sickling/protrusion_detection/HITL_pipeline.ipynb` — the operator's
  HITL correction loop (predict → mine → paint in Ilastik → retrain →
  measure). See the arm's `ARCHITECTURE.md` and `GUIDE.md` for the
  operator walkthrough.
- `sickling/rbc_classification/orchestrate.ipynb` — full classifier
  pipeline driver.
- `sickling/rbc_classification/notebooks/batch_classify.ipynb` — batch
  inference over an entire condition folder.

---

## Install

```bash
git clone https://github.com/ugoreke/sickling.git
cd sickling
pip install -e ".[classification]"
```

Both arms are importable through the umbrella package:

```python
# Protrusion arm
from sickling.protrusion_detection.config import cfg
from sickling.protrusion_detection.model import UNet
from sickling.protrusion_detection.inference import predict_probs

# Classifier arm
from sickling.rbc_classification.stage1_unet.inference import load_unet, predict_label_map
from sickling.rbc_classification.stage2_instances.watershed import mask_to_instances
from sickling.rbc_classification.stage5_multimodal.classifier import MultimodalClassifier
from sickling.rbc_classification.eval.report import read_report
```

---

## Data & weights

**In this repo:** code, docs, `sample.jpg`, the 5 fold OOF prediction
dumps (`sickling/rbc_classification/eval_reports/`), the small parquet
outputs of the classifier pipeline (`per_fov*.parquet`, `per_cell.parquet`,
`polymer_blobs*.parquet`, `per_condition.parquet`, `pairwise_stats.parquet`)
and the 30-d morphology feature tensor (`per_cell_morphology.pt`).

**Off-repo** (too big / raw data hosting):

| Asset | Where | Used by |
|---|---|---|
| U-Net checkpoint (best fold / loop) | Google Drive — paste URL into `notebooks/colab_demo.ipynb` | Colab demo + `pixel_confusion_matrix.ipynb` |
| DINOv2 + morphology classifier checkpoint | Google Drive — paste URL into `notebooks/colab_demo.ipynb` | Colab demo + `orchestrate.ipynb` |
| Raw sickle / non-sickle cell crops used to train the classifier | Google Drive | `orchestrate.ipynb` re-training |
| HITL protrusion training labels (`InitialLabels/`, `BootstrappedLabels/`, `CorrectedTiles/`, `MiniTilesCorrected/`) | Google Drive | `HITL_pipeline.ipynb` re-training |
| Per-condition FOV images (`experiment_data/{A-UNT, ALHi, ALLo, S-UNT, SE1, SE2, SLHi, SLN1}/`) | Not currently hosted | `batch_classify.ipynb` full-dataset sweep |

`.gitignore` reflects this split — cloning the repo gets you everything
you need to explore the classifier eval and run the Colab demo end-to-end
once you plug in the two checkpoint URLs.

---

## Terminology

The paper uses **HbS protrusion** for the rigid HbS-dependent structures
that poke beyond the deoxygenated RBC membrane. This repository uses
*protrusion* in module names, class names, notebook titles, and
user-facing text. A few Python identifiers and parquet column names
retain the legacy `polymer_*` prefix (the models were built before the
paper's wording settled) — their values *are* the HbS protrusion
length / mask / area of the paper.

---

## License

MIT.

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
