"""End-to-end Stage 3 driver.

For each FOV with a Stage-2 instance label image:

    1. Load the raw greyscale (matched by stem under ``raw_images/``) and
       percentile-normalize it with the same single-source-of-truth as
       ``training 2.ipynb``.
    2. Reload the U-Net 4-class label map from ``unet_predictions/`` and
       re-run ``mask_to_instances_with_reasons`` so we have the *unfiltered*
       watershed image — needed to explain *why* a labeled coordinate fell
       inside a dropped instance.
    3. Build per-cell 3-channel crops via :func:`extract_for_fov`.
    4. ``torch.save`` the tensors to ``crops/<source_stem>.pt``.
    5. Resolve any labeled coordinates to instance IDs.
    6. Append rows to a per-cell list; failed coords / clipped crops go to
       ``failed.jsonl``.

Final output: ``cells.parquet`` (one row per kept crop, joined with labels +
conditions) and ``failed.jsonl``.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from sickling.rbc_classification.py_modules.config import Config
from sickling.rbc_classification.py_modules.io.h5 import load_label_map
from sickling.rbc_classification.py_modules.io.images import find_raw_image, load_raw_greyscale, normalize_image
from sickling.rbc_classification.py_modules.io.labels import (
    LabelRow,
    load_conditions,
    load_labels,
    resolve_coordinate_to_instance,
)
from sickling.rbc_classification.py_modules.io.parquet import write_cells
from sickling.rbc_classification.py_modules.stage2_instances.watershed import mask_to_instances_with_reasons
from sickling.rbc_classification.py_modules.stage3_crops.extract import extract_for_fov
from sickling.rbc_classification.py_modules.stage3_crops.metadata import make_cells_dataframe, write_failed_jsonl


def _instance_files(instances_dir: Path) -> list[Path]:
    return sorted(p for p in instances_dir.glob("*_instances.h5") if p.is_file())


def _raw_stem_for(instances_path: Path) -> str:
    """``PRED_<x>_instances.h5`` → ``<x>``. Falls back to dropping the
    ``_instances`` suffix if the ``PRED_`` prefix is absent."""
    stem = instances_path.stem
    if stem.endswith("_instances"):
        stem = stem[: -len("_instances")]
    if stem.startswith("PRED_"):
        stem = stem[len("PRED_"):]
    return stem


def _unet_pred_for(raw_stem: str, unet_dir: Path) -> Path | None:
    """Map the raw stem back to the U-Net prediction h5 (``PRED_<stem>.h5``)."""
    candidate = unet_dir / f"PRED_{raw_stem}.h5"
    if candidate.exists():
        return candidate
    candidate = unet_dir / f"{raw_stem}.h5"
    if candidate.exists():
        return candidate
    return None


def run_stage3(
    cfg: Config,
    instances_dir: Path | None = None,
    raw_dir: Path | None = None,
    unet_dir: Path | None = None,
    crops_dir: Path | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Execute Stage 3 end-to-end. Returns the cells DataFrame.

    Side effects (relative to ``cfg.paths.resolved()``):
        * writes ``crops/<stem>.pt`` per FOV.
        * writes ``cells.parquet`` at the repo root.
        * writes ``failed.jsonl`` at the repo root.
    """
    paths = cfg.paths.resolved()
    instances_dir = instances_dir or paths.instances
    raw_dir = raw_dir or paths.raw_images
    unet_dir = unet_dir or paths.unet_predictions
    crops_dir = crops_dir or paths.crops
    crops_dir.mkdir(parents=True, exist_ok=True)

    files = _instance_files(instances_dir)
    if limit is not None:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(
            f"No *_instances.h5 in {instances_dir}. Run `sickling instances` first."
        )

    label_rows = load_labels(paths.labels_csv)
    labels_by_stem: dict[str, list[LabelRow]] = defaultdict(list)
    for row in label_rows:
        labels_by_stem[Path(row.source_image).stem].append(row)

    conditions = load_conditions(paths.conditions / "conditions.csv")

    cell_records: list[dict] = []
    failed_records: list[dict] = []
    n_fovs_skipped_no_raw = 0

    for instances_path in tqdm(files, desc="Stage 3 crop extraction"):
        raw_stem = _raw_stem_for(instances_path)

        raw_path = find_raw_image(raw_stem, raw_dir)
        if raw_path is None:
            n_fovs_skipped_no_raw += 1
            failed_records.append({
                "source_image": instances_path.name,
                "reason": "raw_image_missing",
                "context": {"expected_stem": raw_stem, "raw_dir": str(raw_dir)},
            })
            continue

        unet_path = _unet_pred_for(raw_stem, unet_dir)
        if unet_path is None:
            failed_records.append({
                "source_image": instances_path.name,
                "reason": "unet_prediction_missing",
                "context": {"expected_stem": raw_stem, "unet_dir": str(unet_dir)},
            })
            continue

        # 1. raw greyscale + normalization.
        raw = load_raw_greyscale(raw_path)
        raw_norm = normalize_image(raw, cfg.crop.norm_percentile)

        # 2. label map + filtered + pre-filter instance images + drop reasons.
        label_map = load_label_map(unet_path, n_classes=4)
        instance_image_filtered, _stats, pre_inst, drop_reasons = (
            mask_to_instances_with_reasons(label_map, cfg.instances, cfg.classes)
        )

        if instance_image_filtered.shape != raw_norm.shape:
            failed_records.append({
                "source_image": instances_path.name,
                "reason": "shape_mismatch",
                "context": {
                    "raw_shape": list(raw_norm.shape),
                    "instance_shape": list(instance_image_filtered.shape),
                },
            })
            continue

        # 3. extract crops.
        tensors, instance_ids, kept_meta, failed_meta = extract_for_fov(
            raw_norm=raw_norm,
            label_map=label_map,
            instance_image=instance_image_filtered,
            cfg=cfg.crop,
            classes=cfg.classes,
        )

        # 4. save .pt for the FOV.
        out_pt = crops_dir / f"{raw_stem}.pt"
        torch.save(
            {
                "tensors": tensors,
                "instance_ids": torch.tensor(instance_ids, dtype=torch.int32),
            },
            out_pt,
        )

        # ---- 5. resolve labels (if any) and build per-cell records ----
        cond = conditions.get(raw_stem, {})
        # Map kept instance_id -> (position, meta) for label join.
        position_by_id = {iid: pos for pos, iid in enumerate(instance_ids)}

        # First, register the labeled-cell rows. We mark them by mutating the
        # base meta when the label resolves; unresolved labels go to failed.
        labels_for_fov = labels_by_stem.get(raw_stem, [])
        labels_resolved: dict[int, str] = {}    # filtered instance_id -> label
        seen_resolved: set[int] = set()
        for lbl in labels_for_fov:
            iid, fail = resolve_coordinate_to_instance(
                lbl, instance_image_filtered, pre_inst, drop_reasons
            )
            if iid is None:
                failed_records.append({
                    "source_image": raw_path.name,
                    "reason": fail,
                    "context": {"x": lbl.x, "y": lbl.y, "label": lbl.label},
                })
                continue
            if iid in seen_resolved:
                failed_records.append({
                    "source_image": raw_path.name,
                    "reason": "duplicate_instance",
                    "context": {"x": lbl.x, "y": lbl.y, "instance_id": iid},
                })
                continue
            seen_resolved.add(iid)
            labels_resolved[iid] = lbl.label

        # Emit one record per kept crop.
        for pos, (iid, meta) in enumerate(zip(instance_ids, kept_meta, strict=True)):
            assert pos == position_by_id[iid]
            label = labels_resolved.get(iid)
            cell_records.append({
                "source_image": raw_path.name,
                "instance_id": int(iid),
                "position": int(pos),
                "centroid_x": meta["centroid_x"],
                "centroid_y": meta["centroid_y"],
                "area": int(meta["area"]),
                "bbox_x0": int(meta["bbox_x0"]),
                "bbox_y0": int(meta["bbox_y0"]),
                "bbox_x1": int(meta["bbox_x1"]),
                "bbox_y1": int(meta["bbox_y1"]),
                "has_label": label is not None,
                "label": label,
                "oxygen_pct": cond.get("oxygen_pct"),
                "treatment": cond.get("treatment"),
            })

        # Clipped or otherwise failed crops.
        for fm in failed_meta:
            failed_records.append({
                "source_image": raw_path.name,
                "reason": fm.pop("reason", "clipped"),
                "context": fm,
            })

    cells_df = make_cells_dataframe(cell_records)
    write_cells(cells_df, paths.root / cfg.paths.cells_parquet)
    write_failed_jsonl(failed_records, paths.root / cfg.paths.failed_jsonl)

    n_kept = len(cells_df)
    n_labeled = int(cells_df["has_label"].sum()) if n_kept else 0
    n_failed = len(failed_records)
    print(
        f"{len(files)} FOVs · cells_kept={n_kept} (labeled={n_labeled}) · "
        f"failed={n_failed} · "
        f"skipped_no_raw={n_fovs_skipped_no_raw} · "
        f"crops: {crops_dir} · cells.parquet at root"
    )
    return cells_df
