"""Driver for Stage 2 — walk a directory of U-Net 4-class h5 predictions,
run ``mask_to_instances`` on each, and write integer instance label images
plus a per-FOV stats parquet for QA.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from sickling.rbc_classification.py_modules.config import Config
from sickling.rbc_classification.py_modules.io.h5 import load_label_map, write_label_map_h5
from sickling.rbc_classification.py_modules.stage2_instances.qa import (
    load_raw_image,
    make_qa_figure,
    save_qa_figure,
)
from sickling.rbc_classification.py_modules.stage2_instances.watershed import (
    mask_to_instances,
    mask_to_instances_with_reasons,
)


def _stem_no_ilastik_segmentation_suffix(path: Path) -> str:
    """``foo_segmentation.h5`` -> ``foo`` to round-trip with `training 2.ipynb`'s
    naming. Plain ``foo.h5`` is unchanged."""
    stem = path.stem
    return stem[: -len("_segmentation")] if stem.endswith("_segmentation") else stem


def _stage2_input_files(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.glob("*.h5") if p.is_file())


def run_stage2(
    cfg: Config,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    limit: int | None = None,
    qa: bool = False,
) -> pd.DataFrame:
    """Run Stage 2 over every ``*.h5`` in ``input_dir``.

    Args:
        cfg: project config.
        input_dir: defaults to ``cfg.paths.unet_predictions`` (resolved).
        output_dir: defaults to ``cfg.paths.instances`` (resolved).
        limit: process only the first N files (smoke / debugging).
        qa: if True, also render a 4-panel QA PNG per FOV to ``cfg.paths.figures``
            (drop reasons, area histogram, raw overlay if a matching raw exists).

    Writes:
        * ``<output_dir>/<stem>_instances.h5`` per FOV (uint16 label image).
        * ``<output_dir>/_stats.parquet`` with columns
          ``[source_image, n_total, n_kept, n_dropped_edge,
              n_dropped_min_area, n_dropped_max_area]``.

    Returns the stats DataFrame so callers (tests, notebooks) can inspect it.
    """
    paths = cfg.paths.resolved()
    input_dir = input_dir or paths.unet_predictions
    output_dir = output_dir or paths.instances
    output_dir.mkdir(parents=True, exist_ok=True)

    files = _stage2_input_files(input_dir)
    if limit is not None:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No *.h5 found in {input_dir}.")

    figures_dir = paths.figures if qa else None
    raw_dir = paths.raw_images if qa else None

    rows: list[dict[str, object]] = []
    for fpath in tqdm(files, desc="Stage 2 instance segmentation"):
        label_map = load_label_map(fpath, n_classes=4)

        if qa:
            instance_img, stats, pre_inst, reasons = mask_to_instances_with_reasons(
                label_map, cfg.instances, cfg.classes
            )
        else:
            instance_img, stats = mask_to_instances(label_map, cfg.instances, cfg.classes)

        out_stem = _stem_no_ilastik_segmentation_suffix(fpath)
        out_path = output_dir / f"{out_stem}_instances.h5"
        write_label_map_h5(out_path, instance_img)

        if qa:
            assert figures_dir is not None and raw_dir is not None
            # Strip the "PRED_" prefix that `export_for_ilastik_correction` adds
            # so the raw-image lookup matches the original training stem.
            raw_stem = out_stem[len("PRED_"):] if out_stem.startswith("PRED_") else out_stem
            raw = load_raw_image(raw_stem, raw_dir)
            fig = make_qa_figure(
                label_map=label_map,
                instance_image=instance_img,
                pre_instance_image=pre_inst,
                reasons=reasons,
                cfg=cfg.instances,
                raw_image=raw,
                title=fpath.name,
            )
            save_qa_figure(fig, figures_dir / f"stage2_qa_{out_stem}.png")

        rows.append({"source_image": fpath.name, **stats.to_dict()})

    stats_df = pd.DataFrame(rows)
    stats_df.to_parquet(output_dir / "_stats.parquet", index=False)

    totals = stats_df[
        ["n_total", "n_kept", "n_dropped_edge", "n_dropped_min_area", "n_dropped_max_area"]
    ].sum()
    print(
        f"{len(files)}/{len(files)} FOVs · "
        f"n_kept={int(totals['n_kept'])} · "
        f"dropped: edge={int(totals['n_dropped_edge'])}, "
        f"min={int(totals['n_dropped_min_area'])}, "
        f"max={int(totals['n_dropped_max_area'])} · "
        f"out: {output_dir}"
    )
    return stats_df
