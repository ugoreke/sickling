"""Per-cell label and per-FOV condition CSV readers + the coordinate→instance
resolver used by Stage 3.

The labels CSV is keyed by ``(source_image, x, y)`` rather than ``instance_id``
because instance IDs only exist *after* Stage 2 runs — annotators work in
pixel space. Stage 3 joins coordinates to instances by point-in-mask.

This module also exposes :func:`gate_labels_to_prevalence` — a label-balancing
helper that down-samples whichever class is in excess so the labeled subset of
``cells.parquet`` hits a target sickle fraction. We use it to mimic the natural
~10% sickle prevalence on the existing label corpus before the user finishes
collecting more non-sickle labels.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

VALID_LABELS = ("sickle", "non_sickle", "ambiguous")


@dataclass(frozen=True)
class LabelRow:
    """One row from ``labels/labels.csv``. ``source_image`` may include the
    file extension; the join layer strips it before matching."""
    source_image: str
    x: int
    y: int
    label: str
    annotator: str | None = None
    notes: str | None = None


def _stem(name: str) -> str:
    return Path(name).stem


def load_labels(path: str | Path) -> list[LabelRow]:
    """Read ``labels.csv``. Returns ``[]`` if the file has only the header
    (the project ships an empty template by default).

    Raises ``ValueError`` if a row's ``label`` is not in :data:`VALID_LABELS`.
    """
    path = Path(path)
    rows: list[LabelRow] = []
    if not path.exists():
        return rows

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for ln, row in enumerate(reader, start=2):
            if not row.get("source_image"):
                continue  # blank line
            label = (row.get("label") or "").strip()
            if label not in VALID_LABELS:
                raise ValueError(
                    f"{path}:{ln}: invalid label {label!r}, expected one of {VALID_LABELS}."
                )
            rows.append(
                LabelRow(
                    source_image=row["source_image"].strip(),
                    x=int(row["x"]),
                    y=int(row["y"]),
                    label=label,
                    annotator=(row.get("annotator") or "").strip() or None,
                    notes=(row.get("notes") or "").strip() or None,
                )
            )
    return rows


def load_conditions(path: str | Path) -> dict[str, dict[str, object]]:
    """Read ``conditions.csv``. Returns ``{source_stem: {oxygen_pct, treatment,
    date, notes}}``. ``oxygen_pct`` is parsed as float; missing optional fields
    are ``None``.
    """
    path = Path(path)
    out: dict[str, dict[str, object]] = {}
    if not path.exists():
        return out
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for ln, row in enumerate(reader, start=2):
            if not row.get("source_image"):
                continue
            stem = _stem(row["source_image"].strip())
            oxygen_raw = (row.get("oxygen_pct") or "").strip()
            try:
                oxygen = float(oxygen_raw) if oxygen_raw else None
            except ValueError as e:
                raise ValueError(f"{path}:{ln}: oxygen_pct {oxygen_raw!r} not numeric.") from e
            out[stem] = {
                "oxygen_pct": oxygen,
                "treatment": (row.get("treatment") or "").strip() or None,
                "date": (row.get("date") or "").strip() or None,
                "notes": (row.get("notes") or "").strip() or None,
            }
    return out


def resolve_coordinate_to_instance(
    label_row: LabelRow,
    instance_image: np.ndarray,
    pre_instance_image: np.ndarray,
    drop_reasons: dict[int, str],
) -> tuple[int | None, str | None]:
    """Resolve ``(x, y)`` → kept instance id, or fail with a reason.

    Args:
        label_row: the labeled coordinate.
        instance_image: filtered uint16 label image (0 = bg, 1..N kept).
        pre_instance_image: unfiltered watershed output before any drops
            (so we can tell *why* a coordinate fell into a dropped instance).
        drop_reasons: ``{pre_filter_id: 'kept'|'edge'|'min_area'|...}`` from
            ``mask_to_instances_with_reasons``.

    Returns:
        ``(instance_id, None)`` on success — ``instance_id`` is the value in
        ``instance_image`` (1..N).
        ``(None, fail_reason)`` on failure — where ``fail_reason`` is one of
        ``'coordinate_outside_cell'`` or ``'instance_dropped:<reason>'``.
    """
    h, w = instance_image.shape
    x, y = label_row.x, label_row.y
    if not (0 <= x < w and 0 <= y < h):
        return None, "coordinate_outside_cell"

    kept_id = int(instance_image[y, x])
    if kept_id > 0:
        return kept_id, None

    # The kept image is 0 here. Two sub-cases:
    #   - coord falls inside background (no instance ever existed here)
    #   - coord falls inside a dropped instance (pre_instance_image > 0)
    pre_id = int(pre_instance_image[y, x])
    if pre_id == 0:
        return None, "coordinate_outside_cell"
    return None, f"instance_dropped:{drop_reasons.get(pre_id, 'unknown')}"


# ---------------------------------------------------------------------------
# Label-prevalence gating
# ---------------------------------------------------------------------------

GatePolicy = Literal["drop_excess_majority", "drop_excess_minority"]


def gate_labels_to_prevalence(
    cells_df: pd.DataFrame,
    target_sickle_frac: float,
    *,
    seed: int = 42,
    policy: GatePolicy = "drop_excess_majority",
) -> tuple[pd.DataFrame, dict]:
    """Down-sample the over-represented class so the labeled subset hits
    ``target_sickle_frac``.

    The labeled subset is defined as ``has_label == True`` AND ``label`` in
    ``{"sickle", "non_sickle"}`` (ambiguous rows are passed through as
    non-modelled and unlabeled rows are passed through untouched — they are
    SSL fodder, not classifier inputs).

    Down-sampling preserves the FOV distribution of the surviving class as
    much as possible (we sample without replacement uniformly across
    candidate row indices). The returned DataFrame is a view-safe copy with
    the original index reset.

    Args:
        cells_df: ``cells.parquet`` DataFrame (has ``has_label`` and ``label``).
        target_sickle_frac: desired ``n_sickle / (n_sickle + n_non_sickle)``
            in the returned frame (e.g. ``0.10``).
        seed: RNG seed for the down-sample draw.
        policy: ``"drop_excess_majority"`` (default) drops from whichever
            class is currently over the target ratio; ``"drop_excess_minority"``
            is the inverse, useful only for synthetic tests.

    Returns:
        Tuple ``(gated_df, stats)``. ``stats`` is a dict with keys
        ``n_sickle_in`` / ``n_non_sickle_in`` / ``n_sickle_kept`` /
        ``n_non_sickle_kept`` / ``n_dropped`` / ``achieved_frac`` /
        ``policy``. The dropped rows are physically removed from
        ``gated_df`` — they are not flipped to ``has_label=False``, because
        the downstream FOV-grouped k-fold splitter operates on the labeled
        subset directly. Unlabeled and ambiguous rows are returned as-is.
    """
    if not 0.0 < target_sickle_frac < 1.0:
        raise ValueError(
            f"target_sickle_frac must be in (0, 1), got {target_sickle_frac}."
        )
    if "has_label" not in cells_df.columns or "label" not in cells_df.columns:
        raise ValueError("cells_df missing required columns: has_label, label.")

    df = cells_df.reset_index(drop=True).copy()
    labeled_mask = df["has_label"].fillna(False).astype(bool)
    in_modelled = labeled_mask & df["label"].isin(["sickle", "non_sickle"])

    sickle_idx = df.index[in_modelled & (df["label"] == "sickle")].to_numpy()
    nonsickle_idx = df.index[in_modelled & (df["label"] == "non_sickle")].to_numpy()

    n_s, n_n = len(sickle_idx), len(nonsickle_idx)
    if n_s == 0 or n_n == 0:
        raise ValueError(
            f"Cannot gate: need both classes present (got n_sickle={n_s}, n_non_sickle={n_n})."
        )

    current_frac = n_s / (n_s + n_n)

    rng = np.random.default_rng(seed)
    keep_sickle = sickle_idx
    keep_nonsickle = nonsickle_idx

    # Solve for how many of each class to keep so that
    # n_s_keep / (n_s_keep + n_n_keep) == target.
    if policy == "drop_excess_majority":
        if current_frac > target_sickle_frac:
            # Sickles in excess → drop sickle to match target, keep all non_sickle.
            target_n_sickle = int(round(target_sickle_frac * n_n / (1 - target_sickle_frac)))
            target_n_sickle = max(target_n_sickle, 1)
            target_n_sickle = min(target_n_sickle, n_s)
            keep_sickle = rng.choice(sickle_idx, size=target_n_sickle, replace=False)
        elif current_frac < target_sickle_frac:
            # Non-sickles in excess → drop non_sickle to match target, keep all sickle.
            target_n_non = int(round(n_s * (1 - target_sickle_frac) / target_sickle_frac))
            target_n_non = max(target_n_non, 1)
            target_n_non = min(target_n_non, n_n)
            keep_nonsickle = rng.choice(nonsickle_idx, size=target_n_non, replace=False)
        # else: already at target ± rounding — no-op.
    else:  # "drop_excess_minority" — symmetric inverse policy used by tests.
        if current_frac < target_sickle_frac:
            target_n_sickle = int(round(target_sickle_frac * n_n / (1 - target_sickle_frac)))
            target_n_sickle = max(target_n_sickle, 1)
            target_n_sickle = min(target_n_sickle, n_s)
            keep_sickle = rng.choice(sickle_idx, size=target_n_sickle, replace=False)
        elif current_frac > target_sickle_frac:
            target_n_non = int(round(n_s * (1 - target_sickle_frac) / target_sickle_frac))
            target_n_non = max(target_n_non, 1)
            target_n_non = min(target_n_non, n_n)
            keep_nonsickle = rng.choice(nonsickle_idx, size=target_n_non, replace=False)

    kept_modelled = np.concatenate([keep_sickle, keep_nonsickle])
    kept_mask = np.zeros(len(df), dtype=bool)
    kept_mask[kept_modelled] = True

    # Pass-through for non-modelled rows (ambiguous, unlabeled).
    passthrough_mask = (~in_modelled.to_numpy())
    final_mask = kept_mask | passthrough_mask

    gated = df.loc[final_mask].reset_index(drop=True)

    n_s_kept = len(keep_sickle)
    n_n_kept = len(keep_nonsickle)
    achieved = n_s_kept / (n_s_kept + n_n_kept) if (n_s_kept + n_n_kept) > 0 else 0.0
    stats = {
        "n_sickle_in": n_s,
        "n_non_sickle_in": n_n,
        "n_sickle_kept": n_s_kept,
        "n_non_sickle_kept": n_n_kept,
        "n_dropped": (n_s - n_s_kept) + (n_n - n_n_kept),
        "achieved_frac": achieved,
        "target_frac": target_sickle_frac,
        "policy": policy,
        "seed": seed,
    }
    return gated, stats
