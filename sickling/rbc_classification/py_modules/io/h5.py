"""HDF5 IO conventions, mirroring ``training 2.ipynb`` so the same files are
readable by both the Stage 1 U-Net pipeline and downstream code.

Files written by Ilastik / `export_for_ilastik_correction` are 5-D
``(t, z, y, x, c)`` ``uint8`` arrays with 1-based class labels (0 = unannotated).
The instance-segmentation output is a 2-D ``uint16`` integer label image with
0 = background, 1..N = instance IDs.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

# Match `training 2.ipynb`'s axistags string for round-tripping into Ilastik.
_AXISTAGS_5D = """{
      "axes": [
        {"key": "t", "typeFlags": 2, "resolution": 0, "description": ""},
        {"key": "z", "typeFlags": 2, "resolution": 0, "description": ""},
        {"key": "y", "typeFlags": 2, "resolution": 0, "description": ""},
        {"key": "x", "typeFlags": 2, "resolution": 0, "description": ""},
        {"key": "c", "typeFlags": 1, "resolution": 0, "description": ""}
      ]
    }"""


def load_robust_h5(path: str | Path) -> np.ndarray:
    """Load a 2-D array from an Ilastik-style h5 file.

    Squeezes singleton dimensions, then if the result is still > 2-D applies the
    same channel heuristic as ``training 2.ipynb``: drop the smallest axis first.
    """
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        key = "exported_data" if "exported_data" in keys else ("data" if "data" in keys else keys[0])
        data = f[key][()]

    data = np.squeeze(data)
    if data.ndim > 2:
        if data.shape[-1] < 5:
            data = data[..., 0]
        elif data.shape[0] < 5:
            data = data[0, ...]
    if data.ndim != 2:
        raise ValueError(f"Expected 2-D array after squeeze, got shape {data.shape} from {path}")
    return data


def load_label_map(path: str | Path, n_classes: int = 4) -> np.ndarray:
    """Load a U-Net 4-class prediction and convert to 0-based labels.

    Raw Ilastik 1-based labels (1..N) are mapped to (0..N-1). Pixels labeled
    0 (unannotated) are kept as 255 (`IGNORE_INDEX`) so downstream code can
    treat them uniformly with the U-Net training convention. Asserts that no
    label exceeds ``n_classes``.
    """
    raw = load_robust_h5(path).astype(np.int64)

    if raw.min() < 0:
        raise ValueError(f"Negative label found in {path}: min={raw.min()}")

    out = np.full_like(raw, fill_value=255, dtype=np.int16)
    valid = raw > 0
    shifted = raw[valid] - 1
    if shifted.max(initial=-1) >= n_classes:
        raise ValueError(
            f"{path} contains label {shifted.max() + 1} but n_classes={n_classes}."
        )
    out[valid] = shifted
    return out


def write_label_map_h5(path: str | Path, arr: np.ndarray) -> None:
    """Write a 2-D integer instance label image as a plain h5 dataset.

    No 5-D wrapping or axistags — this output is consumed by downstream
    sickling code, not Ilastik. Use ``write_ilastik_h5`` for that round-trip.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(arr)
    with h5py.File(path, "w") as f:
        f.create_dataset("exported_data", data=arr, compression="gzip", compression_opts=4)


def write_ilastik_h5(path: str | Path, arr: np.ndarray) -> None:
    """Write a 2-D class label map in the 5-D ``uint8`` Ilastik convention.

    Used only when round-tripping predictions back into Ilastik for human
    correction (mirrors ``training 2.ipynb``'s ``export_for_ilastik_correction``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D label map, got shape {arr.shape}")
    out = np.ascontiguousarray(arr[None, None, :, :, None].astype(np.uint8))
    with h5py.File(path, "w") as f:
        dset = f.create_dataset("exported_data", data=out, dtype="uint8")
        dset.attrs["axistags"] = _AXISTAGS_5D
