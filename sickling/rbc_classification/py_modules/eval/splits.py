"""Group-stratified k-fold splitters for ``cells.parquet``.

Two strategies are exposed:

* :func:`group_stratified_kfold` — the original sklearn-backed splitter.
  Groups = ``source_image``; stratifies on the FOV-level dominant label.
  Tends to produce uneven per-fold validation sizes when one FOV carries
  far more labels than the rest (the situation we hit in the
  ablation_20260516_003426 discussion, fold 4 limitation).

* :func:`balanced_group_kfold` — greedy multi-key bin-packing splitter.
  Groups = ``source_image`` (same leakage-free invariant), but the
  assignment is chosen to keep both the per-fold sickle count and the
  per-fold non-sickle count as close to equal as possible. Drop-in
  replacement for :func:`group_stratified_kfold`.

The downstream chooser is :func:`make_kfold_splits`, which reads the
``cfg.validation.fold_strategy`` field and dispatches to whichever
splitter the user asked for.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from sickling.rbc_classification.py_modules.data.crop_dataset import LABEL_TO_INT

FoldStrategy = Literal["stratified", "balanced"]


def _label_int_or_nan(row: pd.Series) -> float:
    if not bool(row.get("has_label", False)):
        return np.nan
    label = row.get("label")
    if label is None or label == "ambiguous":
        return np.nan
    return float(LABEL_TO_INT[label])


def _prep_labeled(cells_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Reset index, append ``label_int``, return ``(cells, labeled_mask, unlabeled_idx)``."""
    if "has_label" not in cells_df.columns or "source_image" not in cells_df.columns:
        raise ValueError("cells_df missing required columns: has_label, source_image.")
    cells = cells_df.reset_index(drop=True).copy()
    cells["label_int"] = cells.apply(_label_int_or_nan, axis=1)
    labeled_mask = cells["label_int"].notna()
    unlabeled_idx = np.where(~labeled_mask.to_numpy())[0]
    return cells, labeled_mask, unlabeled_idx


def group_stratified_kfold(
    cells_df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return ``[(train_idx, val_idx), ...]`` over ``cells_df`` row indices.

    The fold assignment is computed on the labeled subset, where each FOV
    contributes one (sickle_count, non_sickle_count) tuple. We assign FOVs to
    folds by stratifying on the FOV-level dominant label, then expand back to
    cell indices. Unlabeled cells from all FOVs are appended to every train
    split.
    """
    cells, labeled_mask, unlabeled_idx = _prep_labeled(cells_df)
    labeled = cells[labeled_mask]
    if labeled.empty:
        all_idx = np.arange(len(cells))
        return [(all_idx, np.array([], dtype=np.int64)) for _ in range(n_splits)]

    by_fov = (
        labeled.groupby("source_image")["label_int"].apply(
            lambda s: 1 if (s == 1).any() else 0
        )
    )
    fov_names = by_fov.index.to_numpy()
    fov_labels = by_fov.to_numpy().astype(np.int64)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for train_fovs_idx, val_fovs_idx in skf.split(fov_names, fov_labels):
        train_fovs = set(fov_names[train_fovs_idx].tolist())
        val_fovs = set(fov_names[val_fovs_idx].tolist())

        train_idx = np.where(
            cells["source_image"].isin(train_fovs).to_numpy() & labeled_mask.to_numpy()
        )[0]
        val_idx = np.where(
            cells["source_image"].isin(val_fovs).to_numpy() & labeled_mask.to_numpy()
        )[0]

        train_idx = np.concatenate([train_idx, unlabeled_idx])
        train_idx.sort()
        splits.append((train_idx, val_idx))

    return splits


def _fov_class_counts(labeled: pd.DataFrame) -> pd.DataFrame:
    """One row per FOV with columns ``[sickle, non_sickle, total]``."""
    grp = labeled.groupby("source_image")["label_int"]
    sickle = grp.apply(lambda s: int((s == 1).sum()))
    nonsickle = grp.apply(lambda s: int((s == 0).sum()))
    out = pd.DataFrame({"sickle": sickle, "non_sickle": nonsickle})
    out["total"] = out["sickle"] + out["non_sickle"]
    return out


def _greedy_balanced_assignment(
    fov_counts: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> dict[str, int]:
    """Greedy multi-key bin-packing of FOVs into ``n_splits`` folds.

    Heuristic (Karmarkar–Karp-style):
        1. Order FOVs by ``-total`` (heaviest first) for deterministic
           dominance, breaking ties via ``rng.permutation`` so the
           ordering depends on ``seed``.
        2. For each FOV, drop it into the fold that minimises a *normalised*
           cost over per-class loads. The normalisation is the fraction of
           the global per-class budget already allocated; this makes the
           rare class drive the decision whenever it is much smaller than
           the majority. (The pre-v0.9.1 raw-count formulation was
           dominated by whichever class was numerically larger, so a 10%
           sickle gate produced fold-0/2 = 60+ sickle, fold-1/3/4 ≈ 1
           sickle each — see ``ablation_20260517_154137``.)
        3. Final ties broken by smallest total load (also normalised),
           then by RNG.

    Returns ``{source_image: fold_index}``.
    """
    rng = np.random.default_rng(seed)
    fovs = fov_counts.index.to_numpy()
    perm = rng.permutation(len(fovs))
    order = sorted(perm, key=lambda i: -int(fov_counts.iloc[i]["total"]))
    fov_order = [fovs[i] for i in order]

    fold_sickle = np.zeros(n_splits, dtype=np.int64)
    fold_nonsickle = np.zeros(n_splits, dtype=np.int64)
    assignment: dict[str, int] = {}

    # Global per-class budgets — guard against zero so the divisions below
    # never blow up when one class is empty (the public entry rejects that
    # case anyway, but keep the helper robust).
    total_sickle = max(int(fov_counts["sickle"].sum()), 1)
    total_nonsickle = max(int(fov_counts["non_sickle"].sum()), 1)

    for name in fov_order:
        s = int(fov_counts.loc[name, "sickle"])
        n = int(fov_counts.loc[name, "non_sickle"])
        # Cost of placing this FOV in each candidate fold: maximum *share*
        # of either class's global budget already in that fold after the
        # hypothetical assignment. Quantise to int via ``round`` so the
        # ``lexsort`` keys stay stable across NumPy versions.
        cand_s_frac = (fold_sickle + s) / total_sickle
        cand_n_frac = (fold_nonsickle + n) / total_nonsickle
        cost = np.maximum(cand_s_frac, cand_n_frac)
        # Tie-break on total fraction allocated; then RNG.
        tie_break = (cand_s_frac + cand_n_frac)
        # ``lexsort`` is stable and minimises the LAST key — pass
        # ``(rand, tie_break, cost)`` so cost is primary.
        rand = rng.integers(0, 10_000, size=n_splits)
        # Scale floats to int keys for lexsort to behave deterministically
        # under float comparisons. 1e9 gives ~9-digit precision, well above
        # any per-FOV granularity.
        cost_key = np.round(cost * 1_000_000_000).astype(np.int64)
        tie_key = np.round(tie_break * 1_000_000_000).astype(np.int64)
        key = np.lexsort((rand, tie_key, cost_key))
        choice = int(key[0])
        assignment[name] = choice
        fold_sickle[choice] += s
        fold_nonsickle[choice] += n

    return assignment


def balanced_group_kfold(
    cells_df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """FOV-group k-fold that balances per-class cell counts across folds.

    Same return contract as :func:`group_stratified_kfold`:
        ``[(train_idx, val_idx), ..., (train_idx, val_idx)]`` (length
        ``n_splits``). Unlabeled cells (if any) are appended to every
        training fold and never appear in any validation fold.

    The intent is to fix the fold 4 instability documented in
    ``figures/ablation/ablation_20260516_003426/discussion.tex`` —
    where one heavy FOV dominated the smallest fold and crashed the
    minority class.
    """
    cells, labeled_mask, unlabeled_idx = _prep_labeled(cells_df)
    labeled = cells[labeled_mask]
    if labeled.empty:
        all_idx = np.arange(len(cells))
        return [(all_idx, np.array([], dtype=np.int64)) for _ in range(n_splits)]

    fov_counts = _fov_class_counts(labeled)
    if len(fov_counts) < n_splits:
        raise ValueError(
            f"balanced_group_kfold needs >= n_splits ({n_splits}) FOVs, "
            f"got {len(fov_counts)}."
        )

    assignment = _greedy_balanced_assignment(fov_counts, n_splits, seed)
    fov_to_fold = pd.Series(assignment, name="fold")
    cell_fold = cells["source_image"].map(fov_to_fold)
    cell_fold_arr = cell_fold.to_numpy()

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    labeled_arr = labeled_mask.to_numpy()
    for fold_idx in range(n_splits):
        val_idx = np.where((cell_fold_arr == fold_idx) & labeled_arr)[0]
        train_idx = np.where((cell_fold_arr != fold_idx) & labeled_arr)[0]
        train_idx = np.concatenate([train_idx, unlabeled_idx])
        train_idx.sort()
        splits.append((train_idx, val_idx))

    return splits


def fold_diagnostics(
    cells_df: pd.DataFrame,
    splits: Iterable[tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Per-fold sanity table: ``[fold, n_val, n_train, n_sickle_val, n_non_sickle_val, n_fovs_val]``.

    Useful for asserting that a fold strategy actually produced what it
    promised; the test suite and the ablation discussion both call this.
    """
    cells, _, _ = _prep_labeled(cells_df)
    rows = []
    for i, (train_idx, val_idx) in enumerate(splits):
        v = cells.iloc[val_idx]
        rows.append({
            "fold": i,
            "n_val": int(len(val_idx)),
            "n_train": int(len(train_idx)),
            "n_sickle_val": int((v["label_int"] == 1).sum()),
            "n_non_sickle_val": int((v["label_int"] == 0).sum()),
            "n_fovs_val": int(v["source_image"].nunique()),
        })
    return pd.DataFrame(rows)


def make_kfold_splits(
    cells_df: pd.DataFrame,
    *,
    n_splits: int = 5,
    seed: int = 42,
    strategy: FoldStrategy = "balanced",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Dispatch to the requested splitter. Single hook the CLI layer uses
    so the choice lives in one place."""
    if strategy == "stratified":
        return group_stratified_kfold(cells_df, n_splits=n_splits, seed=seed)
    if strategy == "balanced":
        return balanced_group_kfold(cells_df, n_splits=n_splits, seed=seed)
    raise ValueError(f"Unknown fold_strategy={strategy!r}; expected 'stratified' or 'balanced'.")
