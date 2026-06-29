"""Schema for ``cells.parquet`` and tiny read/write helpers.

Each row is one cell. ``position`` is the index into the per-FOV
``crops/<source_stem>.pt`` tensor — this lets the Dataset map
``(source_image, instance_id) -> tensor`` in O(1) without scanning.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CELLS_COLUMNS: tuple[str, ...] = (
    "source_image",
    "instance_id",
    "position",
    "centroid_x",
    "centroid_y",
    "area",
    "bbox_x0",
    "bbox_y0",
    "bbox_x1",
    "bbox_y1",
    "has_label",
    "label",
    "oxygen_pct",
    "treatment",
)


def write_cells(df: pd.DataFrame, path: str | Path) -> None:
    """Write ``cells.parquet`` after asserting the schema is exactly
    :data:`CELLS_COLUMNS` (in order)."""
    if list(df.columns) != list(CELLS_COLUMNS):
        missing = set(CELLS_COLUMNS) - set(df.columns)
        extra = set(df.columns) - set(CELLS_COLUMNS)
        raise ValueError(
            f"cells.parquet schema mismatch. missing={sorted(missing)}, extra={sorted(extra)}"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_cells(path: str | Path) -> pd.DataFrame:
    """Read ``cells.parquet`` and reorder to the canonical column order."""
    df = pd.read_parquet(path)
    return df[list(CELLS_COLUMNS)]
