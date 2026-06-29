"""Tests for Stage 3 crop extraction — synthetic FOVs + real-h5 smoke."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from sickling.rbc_classification.py_modules.config import ClassesConfig, CropConfig, InstancesConfig, load_config
from sickling.rbc_classification.py_modules.stage2_instances.watershed import mask_to_instances
from sickling.rbc_classification.py_modules.stage3_crops.extract import extract_for_fov, extract_one

CLASSES = ClassesConfig()


def _crop_cfg(**kw) -> CropConfig:
    return CropConfig(**{**CropConfig().model_dump(), **kw})


def test_extract_one_3channel_shape(synth_label_map):
    """Crop tensor is (3, size, size) float32; channels follow the project order."""
    inst, _ = mask_to_instances(synth_label_map, InstancesConfig(), CLASSES)
    raw = synth_label_map.astype(np.float32) / synth_label_map.max()  # dummy raw
    iid = int(inst.max())  # any kept id

    cfg = _crop_cfg(size=64)  # smaller than default to fit within 256² fixture
    tensor, meta = extract_one(raw, synth_label_map, inst, iid, cfg, CLASSES)

    assert tensor is not None
    assert tensor.shape == (3, 64, 64)
    assert tensor.dtype == torch.float32
    assert "centroid_x" in meta and "area" in meta


def test_extract_drops_clipped_when_flagged(synth_label_map):
    """Cell A (centroid 64, 64) — a 200-px crop centered there clips top/left."""
    inst, _ = mask_to_instances(
        synth_label_map, InstancesConfig(drop_edge_touching=False), CLASSES
    )
    raw = synth_label_map.astype(np.float32)
    iid = int(inst[64, 64])
    assert iid != 0

    tensor, meta = extract_one(
        raw, synth_label_map, inst, iid, _crop_cfg(size=200, drop_if_clipped=True), CLASSES
    )
    assert tensor is None
    assert meta["area"] > 0


def test_clipped_pad_when_drop_disabled(synth_label_map):
    inst, _ = mask_to_instances(
        synth_label_map, InstancesConfig(drop_edge_touching=False), CLASSES
    )
    raw = synth_label_map.astype(np.float32)
    iid = int(inst[64, 64])
    tensor, _meta = extract_one(
        raw, synth_label_map, inst, iid, _crop_cfg(size=200, drop_if_clipped=False), CLASSES
    )
    assert tensor is not None
    assert tensor.shape == (3, 200, 200)


def test_polymer_channel_only_inside_instance(synth_label_map):
    """Cell B (centroid 64, 160) has a polymer ring → ch2 must be non-empty.
    Cell A (centroid 64, 64) has no polymer → ch2 must be zero."""
    inst, _ = mask_to_instances(synth_label_map, InstancesConfig(), CLASSES)
    raw = synth_label_map.astype(np.float32)

    iid_a = int(inst[64, 64])
    iid_b = int(inst[64, 160])
    assert iid_a != 0 and iid_b != 0 and iid_a != iid_b

    cfg = _crop_cfg(size=80)
    t_a, _ = extract_one(raw, synth_label_map, inst, iid_a, cfg, CLASSES)
    t_b, _ = extract_one(raw, synth_label_map, inst, iid_b, cfg, CLASSES)

    assert t_a[2].sum() == 0, "Cell A should have no polymer."
    assert t_b[2].sum() > 0, "Cell B should have polymer in ch2."
    assert t_b[1].sum() > 0, "Cell B should still have a body in ch1."


def test_other_instances_not_in_channels(synth_label_map):
    """ch1/ch2 must be masked to *this* instance — touching cells C and D
    overlap in pixel space; one's crop should not include the other's body."""
    inst, _ = mask_to_instances(synth_label_map, InstancesConfig(), CLASSES)
    raw = synth_label_map.astype(np.float32)

    iid_c = int(inst[192, 80])
    iid_d = int(inst[192, 128])
    assert iid_c != 0 and iid_d != 0 and iid_c != iid_d

    t_c, _ = extract_one(raw, synth_label_map, inst, iid_c, _crop_cfg(size=80), CLASSES)
    # Restrict to cell C's crop window: row 152..232, col 40..120. Cell D's
    # body pixels (row 192, col 128) might appear in ch0 (raw) but never in
    # ch1, since ch1 is gated on instance_image == iid_c.
    body_pixels_c = int(t_c[1].sum())
    assert body_pixels_c > 0
    # Sanity: the body channel pixel count equals the # of pixels in inst==iid_c
    # falling inside the window.
    rows, cols = np.where(inst == iid_c)
    cy, cx = int(round(rows.mean())), int(round(cols.mean()))
    half = 40
    in_window = ((rows >= cy - half) & (rows < cy + half)
                 & (cols >= cx - half) & (cols < cx + half))
    expected_body = int(((synth_label_map[rows[in_window], cols[in_window]] == CLASSES.cell_body)
                        & (inst[rows[in_window], cols[in_window]] == iid_c)).sum())
    assert body_pixels_c == expected_body


def test_extract_for_fov_returns_aligned_lists(synth_label_map):
    inst, stats = mask_to_instances(synth_label_map, InstancesConfig(), CLASSES)
    raw = synth_label_map.astype(np.float32)
    cfg = _crop_cfg(size=80)
    tensors, instance_ids, kept_meta, failed_meta = extract_for_fov(
        raw, synth_label_map, inst, cfg, CLASSES
    )
    assert tensors.shape[0] == len(instance_ids) == len(kept_meta)
    # Some kept cells may still clip on the smallish 256² fixture; total accounted for.
    assert tensors.shape[0] + len(failed_meta) == stats.n_kept


# ---------- Real-h5 smoke ----------

REAL_INSTANCES = Path("instances/PRED_D16_03_1_1_Bright Field_001_instances.h5")
REAL_RAW = Path("raw_images/D16_03_1_1_Bright Field_001.jpg")


@pytest.mark.skipif(
    not REAL_INSTANCES.exists() or not REAL_RAW.exists(),
    reason="Real Stage-2 outputs or raw image not present.",
)
def test_run_stage3_on_real_fov(tmp_path):
    """End-to-end smoke: run the Stage 3 pipeline on the real FOV. Asserts a
    cells.parquet with > 400 rows and a 3-channel .pt on disk."""
    cfg = load_config()
    paths = cfg.paths.resolved()

    # Redirect outputs into tmp_path so we don't pollute the repo.
    cfg.paths.root = tmp_path
    cfg.paths.crops = Path("crops")
    cfg.paths.cells_parquet = Path("cells.parquet")
    cfg.paths.failed_jsonl = Path("failed.jsonl")
    # Inputs stay at the repo root.
    from sickling.rbc_classification.py_modules.stage3_crops.cli import run_stage3

    df = run_stage3(
        cfg,
        instances_dir=paths.instances,
        raw_dir=paths.raw_images,
        unet_dir=paths.unet_predictions,
        crops_dir=tmp_path / "crops",
    )
    assert len(df) > 400, f"Expected >400 cells, got {len(df)}."
    pt_files = list((tmp_path / "crops").glob("*.pt"))
    assert pt_files, "Expected at least one .pt crops file."

    obj = torch.load(pt_files[0], weights_only=True)
    assert obj["tensors"].shape[1:] == (3, cfg.crop.size, cfg.crop.size)
    assert obj["tensors"].shape[0] == len(obj["instance_ids"])
