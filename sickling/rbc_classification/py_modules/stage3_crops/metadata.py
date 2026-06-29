"""Build ``cells.parquet`` and ``failed.jsonl`` from per-FOV extraction outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_failed_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    """One JSON object per line. Append-friendly format for incremental runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def make_cells_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Coerce a list of per-cell records into the canonical column order
    expected by :data:`sickling.io.parquet.CELLS_COLUMNS`. Missing optional
    fields are filled with ``None`` / ``False`` as appropriate."""
    from sickling.rbc_classification.py_modules.io.parquet import CELLS_COLUMNS

    df = pd.DataFrame(records)
    for col in CELLS_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[list(CELLS_COLUMNS)]

    # Type tightening — parquet handles None as null.
    df["instance_id"] = df["instance_id"].astype("Int32")
    df["position"] = df["position"].astype("Int32")
    df["area"] = df["area"].astype("Int64")
    for col in ("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"):
        df[col] = df[col].astype("Int32")
    df["has_label"] = df["has_label"].fillna(False).astype(bool)
    return df
