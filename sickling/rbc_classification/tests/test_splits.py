"""Tests for ``sickling.eval.splits`` — both the original stratified splitter
and the new ``balanced_group_kfold`` greedy bin-packer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sickling.rbc_classification.py_modules.eval.splits import (
    balanced_group_kfold,
    fold_diagnostics,
    group_stratified_kfold,
    make_kfold_splits,
)


def _synth_cells(n_fovs: int, cells_per_fov: int, sickle_per_fov: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_fovs):
        for c in range(cells_per_fov):
            is_sickle = c < sickle_per_fov
            rows.append({
                "source_image": f"fov_{f:03d}.jpg",
                "instance_id": c + 1,
                "position": c,
                "centroid_x": 0.0, "centroid_y": 0.0,
                "area": 1000,
                "bbox_x0": 0, "bbox_y0": 0, "bbox_x1": 96, "bbox_y1": 96,
                "has_label": True,
                "label": "sickle" if is_sickle else "non_sickle",
                "oxygen_pct": None, "treatment": None,
            })
    df = pd.DataFrame(rows)
    return df.iloc[rng.permutation(len(df))].reset_index(drop=True)


def test_no_source_image_overlap_train_val():
    df = _synth_cells(n_fovs=10, cells_per_fov=20, sickle_per_fov=2)
    splits = group_stratified_kfold(df, n_splits=5, seed=0)
    for train_idx, val_idx in splits:
        train_fovs = set(df.iloc[train_idx]["source_image"])
        val_fovs = set(df.iloc[val_idx]["source_image"])
        assert train_fovs.isdisjoint(val_fovs), "FOV leakage between train and val."


def test_every_fov_eventually_in_val():
    df = _synth_cells(n_fovs=10, cells_per_fov=10, sickle_per_fov=1)
    splits = group_stratified_kfold(df, n_splits=5, seed=0)
    val_fovs_total = set()
    for _train_idx, val_idx in splits:
        val_fovs_total.update(df.iloc[val_idx]["source_image"])
    assert val_fovs_total == set(df["source_image"])


def test_unlabeled_rows_only_in_train():
    df = _synth_cells(n_fovs=6, cells_per_fov=10, sickle_per_fov=1)
    df.loc[df.index[:20], "has_label"] = False
    df.loc[df.index[:20], "label"] = None

    splits = group_stratified_kfold(df, n_splits=3, seed=0)
    for train_idx, val_idx in splits:
        train_unlabeled = (~df.iloc[train_idx]["has_label"]).sum()
        val_unlabeled = (~df.iloc[val_idx]["has_label"]).sum()
        assert val_unlabeled == 0
        assert train_unlabeled == 20  # all 20 unlabeled rows present in every train


def test_no_labels_returns_train_only():
    df = _synth_cells(n_fovs=4, cells_per_fov=5, sickle_per_fov=0)
    df["has_label"] = False
    df["label"] = None
    splits = group_stratified_kfold(df, n_splits=3, seed=0)
    for train_idx, val_idx in splits:
        assert val_idx.size == 0
        assert train_idx.size == len(df)


def test_stratification_balances_across_folds():
    """Each fold's val set should contain at least one sickle FOV."""
    df = _synth_cells(n_fovs=10, cells_per_fov=10, sickle_per_fov=1)
    splits = group_stratified_kfold(df, n_splits=5, seed=0)
    for _train_idx, val_idx in splits:
        n_sickle_in_val = (df.iloc[val_idx]["label"] == "sickle").sum()
        assert n_sickle_in_val > 0


# ---------------------------------------------------------------------------
# balanced_group_kfold
# ---------------------------------------------------------------------------


def _heavily_skewed_cells(n_fovs: int = 30, seed: int = 0) -> pd.DataFrame:
    """A 30-FOV synthetic corpus where one FOV carries 10x as many labels as
    the rest. This is the synthetic analogue of the real-data fold-4
    pathology documented in ablation_20260516_003426/discussion.tex."""
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_fovs):
        # FOV 0 is the heavyweight; everyone else gets ~5 cells.
        if f == 0:
            n_cells = 50
            sickle = 30
        else:
            n_cells = 5
            sickle = rng.integers(0, n_cells + 1)
        labels = ["sickle"] * int(sickle) + ["non_sickle"] * int(n_cells - sickle)
        rng.shuffle(labels)
        for c, lbl in enumerate(labels):
            rows.append({
                "source_image": f"fov_{f:03d}.jpg",
                "instance_id": c + 1,
                "position": c,
                "centroid_x": 0.0, "centroid_y": 0.0,
                "area": 1000,
                "bbox_x0": 0, "bbox_y0": 0, "bbox_x1": 96, "bbox_y1": 96,
                "has_label": True,
                "label": lbl,
                "oxygen_pct": None, "treatment": None,
            })
    return pd.DataFrame(rows)


def test_balanced_kfold_no_fov_leakage():
    df = _heavily_skewed_cells(n_fovs=20, seed=0)
    splits = balanced_group_kfold(df, n_splits=5, seed=0)
    assert len(splits) == 5
    for train_idx, val_idx in splits:
        train_fovs = set(df.iloc[train_idx]["source_image"])
        val_fovs = set(df.iloc[val_idx]["source_image"])
        assert train_fovs.isdisjoint(val_fovs), "FOV leakage between train and val."


def test_balanced_kfold_equalises_per_class_counts():
    """The bin-packer should make per-class val counts noticeably more even
    than the stratified splitter on a heavily-skewed FOV corpus."""
    df = _heavily_skewed_cells(n_fovs=30, seed=1)

    bal = balanced_group_kfold(df, n_splits=5, seed=1)
    strat = group_stratified_kfold(df, n_splits=5, seed=1)

    bal_diag = fold_diagnostics(df, bal)
    strat_diag = fold_diagnostics(df, strat)

    # Range of per-fold validation sickle counts should be smaller (more
    # balanced) under the new splitter.
    bal_range = bal_diag["n_sickle_val"].max() - bal_diag["n_sickle_val"].min()
    strat_range = strat_diag["n_sickle_val"].max() - strat_diag["n_sickle_val"].min()
    assert bal_range <= strat_range, (
        f"balanced range={bal_range}, stratified range={strat_range}; "
        f"balanced splitter should be no worse"
    )
    # And per-fold totals should not be dominated by one fold either.
    bal_total_range = bal_diag["n_val"].max() - bal_diag["n_val"].min()
    strat_total_range = strat_diag["n_val"].max() - strat_diag["n_val"].min()
    assert bal_total_range <= strat_total_range, (
        f"balanced total range={bal_total_range}, stratified={strat_total_range}"
    )


def test_balanced_kfold_unlabeled_only_in_train():
    df = _heavily_skewed_cells(n_fovs=15, seed=2)
    df.loc[df.index[:30], "has_label"] = False
    df.loc[df.index[:30], "label"] = None
    splits = balanced_group_kfold(df, n_splits=3, seed=2)
    for train_idx, val_idx in splits:
        train_unlabeled = (~df.iloc[train_idx]["has_label"]).sum()
        val_unlabeled = (~df.iloc[val_idx]["has_label"]).sum()
        assert val_unlabeled == 0
        # All 30 unlabeled rows are appended to every train fold.
        assert train_unlabeled == 30


def test_balanced_kfold_rejects_too_few_fovs():
    df = _heavily_skewed_cells(n_fovs=2, seed=3)
    with pytest.raises(ValueError):
        balanced_group_kfold(df, n_splits=5, seed=3)


def test_make_kfold_splits_dispatch():
    df = _heavily_skewed_cells(n_fovs=10, seed=4)
    bal = make_kfold_splits(df, n_splits=5, seed=4, strategy="balanced")
    strat = make_kfold_splits(df, n_splits=5, seed=4, strategy="stratified")
    assert len(bal) == 5 and len(strat) == 5
    with pytest.raises(ValueError):
        make_kfold_splits(df, strategy="not-a-real-strategy")  # type: ignore[arg-type]


def _rare_minority_cells(n_sickle: int, n_non_sickle: int, seed: int) -> pd.DataFrame:
    """Mirror the real-world post-gate layout: minority cells are spread
    across many small FOVs, majority cells are concentrated in a handful of
    heavy FOVs. This is the synthetic analogue of the 10%-prevalence corpus
    from ``figures/ablation/ablation_20260517_154137/``."""
    rng = np.random.default_rng(seed)
    rows = []
    # 23 heavy non-sickle FOVs (mirror the real data); spread 1108 cells.
    heavy_fovs = [f"heavy_{i:02d}.jpg" for i in range(23)]
    cells_per_heavy = n_non_sickle // len(heavy_fovs)
    for fov in heavy_fovs:
        for c in range(cells_per_heavy):
            rows.append({
                "source_image": fov,
                "instance_id": c + 1, "position": c,
                "centroid_x": 0.0, "centroid_y": 0.0,
                "area": 1000, "bbox_x0": 0, "bbox_y0": 0,
                "bbox_x1": 96, "bbox_y1": 96,
                "has_label": True, "label": "non_sickle",
                "oxygen_pct": None, "treatment": None,
            })
    # 96 light sickle FOVs (1–3 sickle cells each).
    light_fovs = [f"light_{i:03d}.jpg" for i in range(96)]
    remaining = n_sickle
    for fov in light_fovs:
        if remaining <= 0:
            break
        k = int(min(remaining, rng.integers(1, 4)))
        for c in range(k):
            rows.append({
                "source_image": fov,
                "instance_id": c + 1, "position": c,
                "centroid_x": 0.0, "centroid_y": 0.0,
                "area": 800, "bbox_x0": 0, "bbox_y0": 0,
                "bbox_x1": 96, "bbox_y1": 96,
                "has_label": True, "label": "sickle",
                "oxygen_pct": None, "treatment": None,
            })
        remaining -= k
    return pd.DataFrame(rows)


def test_balanced_kfold_keeps_rare_class_spread_at_10pct_prevalence():
    """Regression for the v0.9.1 fix: under 10% prevalence with sickle
    spread across many small FOVs, every fold must end up with at least
    one sickle cell, AND no single fold may hoard more than ~40% of the
    global sickle budget. The pre-fix raw-count cost function failed both
    invariants — folds 1/3/4 ended up with 0–2 sickles each while folds
    0/2 hoarded 56–62 sickles."""
    df = _rare_minority_cells(n_sickle=123, n_non_sickle=1108, seed=7)
    splits = balanced_group_kfold(df, n_splits=5, seed=7)
    diag = fold_diagnostics(df, splits)
    sickle_counts = diag["n_sickle_val"].to_numpy()
    total_sickle = int(sickle_counts.sum())

    # Every fold has at least one sickle val cell.
    assert (sickle_counts > 0).all(), f"some folds got zero sickle: {sickle_counts.tolist()}"
    # No fold owns more than 40% of the rare class. (Even allocation would
    # be 20% per fold; we leave a healthy margin for the integer-FOV
    # granularity.)
    assert sickle_counts.max() / total_sickle < 0.40, (
        f"one fold hoards too much sickle: {sickle_counts.tolist()}, "
        f"max share = {sickle_counts.max() / total_sickle:.2%}"
    )
    # Non-sickle stays balanced too (the original property we shipped in v0.9.0).
    nons = diag["n_non_sickle_val"].to_numpy()
    assert nons.max() / max(nons.min(), 1) < 1.5, f"non-sickle imbalance: {nons.tolist()}"
