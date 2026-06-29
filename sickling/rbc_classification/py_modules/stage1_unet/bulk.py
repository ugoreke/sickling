"""Bulk U-Net prediction over a directory of raw images.

Mirrors the labeling notebook's on-the-fly pipeline but persists outputs so
the downstream Stage 2 / 3 / 4 / 5 stages have everything they need on disk.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from sickling.rbc_classification.py_modules.config import Config
from sickling.rbc_classification.py_modules.io.h5 import write_ilastik_h5
from sickling.rbc_classification.py_modules.io.images import RAW_EXTS, load_raw_greyscale, normalize_image
from sickling.rbc_classification.py_modules.stage1_unet.inference import load_unet, predict_label_map


def bulk_predict(
    cfg: Config,
    *,
    input_dir: Path,
    model_path: Path,
    copy_raws: bool = True,
    overwrite: bool = False,
    n_classes: int = 4,
) -> dict[str, int]:
    """Run the frozen U-Net over every image in ``input_dir``.

    For each image:
      * write a 4-class label map h5 to ``cfg.paths.unet_predictions`` as
        ``PRED_<stem>.h5`` (Ilastik 1-based for compatibility).
      * if ``copy_raws=True``, copy the raw file into ``cfg.paths.raw_images``.

    Returns a counts dict.
    """
    paths = cfg.paths.resolved()
    paths.unet_predictions.mkdir(parents=True, exist_ok=True)
    if copy_raws:
        paths.raw_images.mkdir(parents=True, exist_ok=True)

    raw_files: list[Path] = []
    for ext in RAW_EXTS:
        raw_files.extend(sorted(input_dir.glob(f"*.{ext}")))
    if not raw_files:
        raise FileNotFoundError(f"No images in {input_dir}.")

    model = load_unet(model_path, n_classes=n_classes)
    counts = {"n_predicted": 0, "n_skipped_existing": 0, "n_raws_copied": 0}

    for raw_path in tqdm(raw_files, desc="bulk U-Net predict"):
        stem = raw_path.stem
        out_pred = paths.unet_predictions / f"PRED_{stem}.h5"

        if copy_raws:
            dest = paths.raw_images / raw_path.name
            if not dest.exists() or overwrite:
                shutil.copy2(raw_path, dest)
                counts["n_raws_copied"] += 1

        if out_pred.exists() and not overwrite:
            counts["n_skipped_existing"] += 1
            continue

        raw = load_raw_greyscale(raw_path)
        raw_norm = normalize_image(raw, cfg.crop.norm_percentile)
        label_map = predict_label_map(model, raw_norm, n_classes=n_classes)
        # Convert 0-based -> 1-based for the Ilastik convention (consistent with
        # `training 2.ipynb`'s `export_for_ilastik_correction`).
        write_ilastik_h5(out_pred, (label_map.astype(np.int32) + 1).astype(np.uint8))
        counts["n_predicted"] += 1

    print(
        f"{counts['n_predicted']} predicted, "
        f"{counts['n_skipped_existing']} existing-skipped, "
        f"{counts['n_raws_copied']} raws copied to {paths.raw_images}."
    )
    return counts
