"""Tests for ``sickling.io.labels`` — CSV loading + coordinate resolution."""
from __future__ import annotations

import pytest

from sickling.rbc_classification.py_modules.config import ClassesConfig, InstancesConfig
from sickling.rbc_classification.py_modules.io.labels import (
    LabelRow,
    load_conditions,
    load_labels,
    resolve_coordinate_to_instance,
)
from sickling.rbc_classification.py_modules.stage2_instances.watershed import mask_to_instances_with_reasons

CLASSES = ClassesConfig()


def test_load_labels_empty_template(tmp_path):
    """The shipped template has only a header — should return an empty list."""
    p = tmp_path / "labels.csv"
    p.write_text("source_image,x,y,label,annotator,notes\n")
    assert load_labels(p) == []


def test_load_labels_parses_rows(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text(
        "source_image,x,y,label,annotator,notes\n"
        "roi_1.jpg,10,20,sickle,UG,\n"
        "roi_2.jpg,5,5,non_sickle,,partial\n"
    )
    rows = load_labels(p)
    assert len(rows) == 2
    assert rows[0] == LabelRow(
        source_image="roi_1.jpg", x=10, y=20, label="sickle", annotator="UG", notes=None
    )
    assert rows[1].label == "non_sickle"
    assert rows[1].notes == "partial"


def test_load_labels_rejects_invalid_label(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("source_image,x,y,label\nroi_1.jpg,10,20,bogus\n")
    with pytest.raises(ValueError, match="invalid label"):
        load_labels(p)


def test_load_conditions(tmp_path):
    p = tmp_path / "conditions.csv"
    p.write_text(
        "source_image,oxygen_pct,treatment,date,notes\n"
        "roi_1.jpg,2,DMSO,2026-04-12,control\n"
        "roi_2,21,GBT440,,\n"
    )
    out = load_conditions(p)
    assert set(out.keys()) == {"roi_1", "roi_2"}
    assert out["roi_1"]["oxygen_pct"] == 2.0
    assert out["roi_1"]["treatment"] == "DMSO"
    assert out["roi_2"]["oxygen_pct"] == 21.0
    assert out["roi_2"]["date"] is None


def test_resolve_coordinate_inside_kept_cell(synth_label_map):
    """Coordinate falls inside an instance that survived all filters → returns its id."""
    inst, _stats, pre, reasons = mask_to_instances_with_reasons(
        synth_label_map, InstancesConfig(), CLASSES
    )
    # Pick the centroid of cell A (which is at row 64, col 64).
    iid_at = int(inst[64, 64])
    assert iid_at != 0
    row = LabelRow(source_image="roi.jpg", x=64, y=64, label="sickle")
    iid, reason = resolve_coordinate_to_instance(row, inst, pre, reasons)
    assert reason is None
    assert iid == iid_at


def test_resolve_coordinate_in_dropped_edge_instance(synth_label_map):
    """Cell E touches the right edge → dropped. Coord inside it should resolve to None
    with reason ``instance_dropped:edge``."""
    inst, _stats, pre, reasons = mask_to_instances_with_reasons(
        synth_label_map, InstancesConfig(), CLASSES
    )
    # Cell E is at (32, 232) — pick its center.
    row = LabelRow(source_image="roi.jpg", x=232, y=32, label="sickle")
    iid, reason = resolve_coordinate_to_instance(row, inst, pre, reasons)
    assert iid is None
    assert reason == "instance_dropped:edge"


def test_resolve_coordinate_in_background(synth_label_map):
    """Coord in plain background → ``coordinate_outside_cell``."""
    inst, _stats, pre, reasons = mask_to_instances_with_reasons(
        synth_label_map, InstancesConfig(), CLASSES
    )
    row = LabelRow(source_image="roi.jpg", x=10, y=10, label="sickle")
    iid, reason = resolve_coordinate_to_instance(row, inst, pre, reasons)
    assert iid is None
    assert reason == "coordinate_outside_cell"


def test_resolve_coordinate_out_of_bounds(synth_label_map):
    inst, _stats, pre, reasons = mask_to_instances_with_reasons(
        synth_label_map, InstancesConfig(), CLASSES
    )
    row = LabelRow(source_image="roi.jpg", x=99999, y=99999, label="sickle")
    iid, reason = resolve_coordinate_to_instance(row, inst, pre, reasons)
    assert iid is None
    assert reason == "coordinate_outside_cell"


# ---------------------------------------------------------------------------
# gate_labels_to_prevalence
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd

from sickling.rbc_classification.py_modules.io.labels import gate_labels_to_prevalence


def _synthetic_label_df(n_sickle: int, n_non_sickle: int, n_unlabeled: int = 0, n_ambiguous: int = 0) -> pd.DataFrame:
    rows = []
    for i in range(n_sickle):
        rows.append({"source_image": f"fov_{i % 50}.jpg", "has_label": True, "label": "sickle"})
    for i in range(n_non_sickle):
        rows.append({"source_image": f"fov_{i % 50}.jpg", "has_label": True, "label": "non_sickle"})
    for i in range(n_ambiguous):
        rows.append({"source_image": f"fov_{i % 50}.jpg", "has_label": True, "label": "ambiguous"})
    for i in range(n_unlabeled):
        rows.append({"source_image": f"fov_{i % 50}.jpg", "has_label": False, "label": None})
    return pd.DataFrame(rows)


def test_gate_to_natural_prevalence_drops_excess_sickle():
    """The real-world case: corpus is sickle-enriched (39%); gate to 10%."""
    df = _synthetic_label_df(n_sickle=713, n_non_sickle=1108, n_unlabeled=83000, n_ambiguous=30)
    gated, stats = gate_labels_to_prevalence(df, target_sickle_frac=0.10, seed=0)
    assert abs(stats["achieved_frac"] - 0.10) < 0.01
    # All 1108 non_sickle survive; sickle is downsampled.
    assert stats["n_non_sickle_kept"] == 1108
    assert stats["n_sickle_kept"] < 713
    # Pass-through rows stay.
    assert (gated["has_label"] == False).sum() == 83000
    assert (gated["label"] == "ambiguous").sum() == 30


def test_gate_drops_excess_non_sickle_when_already_below_target():
    """When current sickle frac < target, the helper drops the non-sickle excess
    to push the frac UP to the target. Symmetric inverse of the natural-prevalence
    case."""
    df = _synthetic_label_df(n_sickle=50, n_non_sickle=950)  # 5% sickle
    gated, stats = gate_labels_to_prevalence(df, target_sickle_frac=0.20, seed=0)
    assert abs(stats["achieved_frac"] - 0.20) < 0.02
    # All 50 sickle survive; non-sickle is downsampled to ~200.
    assert stats["n_sickle_kept"] == 50
    # n_non_keep = round(50 * 0.8 / 0.2) = 200
    assert stats["n_non_sickle_kept"] == 200


def test_gate_is_deterministic():
    df = _synthetic_label_df(n_sickle=400, n_non_sickle=600)
    a, _ = gate_labels_to_prevalence(df, target_sickle_frac=0.10, seed=42)
    b, _ = gate_labels_to_prevalence(df, target_sickle_frac=0.10, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_gate_preserves_passthrough_rows():
    df = _synthetic_label_df(n_sickle=300, n_non_sickle=600, n_unlabeled=200, n_ambiguous=15)
    gated, stats = gate_labels_to_prevalence(df, target_sickle_frac=0.10, seed=1)
    # Unlabeled and ambiguous rows pass through untouched in count.
    assert (gated["has_label"] == False).sum() == 200
    assert (gated["label"] == "ambiguous").sum() == 15
    # Modelled rows obey target.
    n_sickle_kept = (gated["label"] == "sickle").sum()
    n_non_kept = (gated["label"] == "non_sickle").sum()
    achieved = n_sickle_kept / (n_sickle_kept + n_non_kept)
    assert abs(achieved - 0.10) < 0.01


def test_gate_rejects_invalid_target():
    df = _synthetic_label_df(n_sickle=10, n_non_sickle=10)
    with pytest.raises(ValueError):
        gate_labels_to_prevalence(df, target_sickle_frac=0.0)
    with pytest.raises(ValueError):
        gate_labels_to_prevalence(df, target_sickle_frac=1.5)


def test_gate_requires_both_classes_present():
    df = _synthetic_label_df(n_sickle=0, n_non_sickle=100)
    with pytest.raises(ValueError):
        gate_labels_to_prevalence(df, target_sickle_frac=0.10)
