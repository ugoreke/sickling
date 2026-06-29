"""H5 mask I/O and the single normalization path used at train + inference.

Label conventions in this project (two flavours on disk):

1. **Training-ready files** (``InitialLabels/*.h5``, ``BootstrappedLabels/*.h5``):
   already in the training convention — ``0..N-1`` valid, ``255`` = ignore.
   Load with ``load_dense_mask`` (alias for ``load_robust_h5``).

2. **Ilastik round-trip files** (``CorrectedTiles/*.h5`` straight from ilastik,
   and ``CorrectionPool/PRED_*.h5`` we write for ilastik to import):
   1-based, ``0`` = unannotated. Load with ``load_ilastik_mask`` which
   subtracts one and marks raw-zero pixels as ``cfg.IGNORE_INDEX``.

Either way the in-memory representation matches ``cfg.IGNORE_INDEX`` (255)
for ignore, so the rest of the pipeline does not need to know which folder
the mask came from.
"""

from __future__ import annotations

import h5py
import numpy as np

from .config import cfg


def load_robust_h5(filepath: str) -> np.ndarray:
    """Read a numpy array from an H5 file written by ilastik or this project.

    Squeezes singleton t/z/c axes; copes with channel-first or channel-last
    storage when a small channel axis sneaks through.
    """
    with h5py.File(filepath, "r") as f:
        keys = list(f.keys())
        key = "exported_data" if "exported_data" in keys else ("data" if "data" in keys else keys[0])
        data = f[key][()]

    data = np.squeeze(data)
    if data.ndim > 2:
        if data.shape[-1] < 5:
            data = data[..., 0]
        elif data.shape[0] < 5:
            data = data[0, ...]
    return data


def load_ilastik_mask(filepath: str) -> np.ndarray:
    """Convert a raw 1-based ilastik mask to training convention (0..N-1 / 255).

    Unannotated pixels (raw 0) become ``cfg.IGNORE_INDEX``. Use this for any
    file that came straight out of the ilastik label exporter — i.e. freshly
    painted ``CorrectedTiles`` and the ``PRED_*.h5`` files we write into
    ``CorrectionPool`` for ilastik to re-import.
    """
    raw = load_robust_h5(filepath).astype(np.int64)
    out = np.full_like(raw, cfg.IGNORE_INDEX)
    valid = raw > 0
    out[valid] = raw[valid] - 1
    return out


# ``InitialLabels`` and ``BootstrappedLabels`` are stored in the training
# convention already (0..N-1 valid, 255 = ignore). No conversion needed.
load_dense_mask = load_robust_h5


def load_bootstrap_label(filepath: str) -> np.ndarray:
    """Loader for ``BootstrappedLabels`` and ``InitialLabels`` that
    auto-detects whether the file is stored in the 0-based training
    convention (legacy, hand-curated and hand-converted files) or the
    1-based ilastik convention (newly written by ``generate_bootstrap_preds``
    so ilastik can render the starting PRED in the painting UI).

    Discriminator:
      - any pixel == ``cfg.IGNORE_INDEX``                       → 0-based.
      - else max non-ignore value < ``cfg.N_CLASSES``           → 0-based.
      - else (max value == ``cfg.N_CLASSES``, no ignore)        → 1-based,
        subtract one and treat raw zero as ``cfg.IGNORE_INDEX`` (ilastik's
        "unannotated").

    This makes the file produced by ``generate_bootstrap_preds`` paint-able
    in ilastik without breaking the read path; freshly painted ilastik
    output drops in the same way (operator paints onto a fully-filled
    PRED, the resulting file is still 1-based with no zeros and no 255s,
    so the heuristic correctly subtracts one).
    """
    raw = load_robust_h5(filepath).astype(np.int64)
    if (raw == cfg.IGNORE_INDEX).any():
        return raw   # 0-based, legacy convention
    non_ignore = raw[raw != cfg.IGNORE_INDEX]
    if non_ignore.size == 0 or non_ignore.max() < cfg.N_CLASSES:
        return raw   # 0-based, just no ignore pixels in this file
    # 1-based ilastik output. Subtract one; raw 0 becomes IGNORE_INDEX.
    out = np.full_like(raw, cfg.IGNORE_INDEX)
    valid = raw > 0
    out[valid] = raw[valid] - 1
    return out


def normalize_image(img_np: np.ndarray) -> np.ndarray:
    """Percentile-clip + scale to [0, 1]. Identical at train and inference."""
    p = np.percentile(img_np, cfg.NORM_PERCENTILE)
    denom = p if p > 0 else (img_np.max() if img_np.max() > 0 else 1.0)
    return np.clip(img_np / denom, 0, 1).astype(np.float32)


AXISTAGS_5D_JSON = """{
  "axes": [
    {"key": "t", "typeFlags": 2, "resolution": 0, "description": ""},
    {"key": "z", "typeFlags": 2, "resolution": 0, "description": ""},
    {"key": "y", "typeFlags": 2, "resolution": 0, "description": ""},
    {"key": "x", "typeFlags": 2, "resolution": 0, "description": ""},
    {"key": "c", "typeFlags": 1, "resolution": 0, "description": ""}
  ]
}"""


def save_ilastik_mask(out_path: str, mask_0based: np.ndarray) -> None:
    """Write a 2D 0-based mask as ilastik-importable Labels (5D, 1-based)."""
    arr = (mask_0based.astype(np.uint8) + 1)
    arr = np.ascontiguousarray(arr[None, None, :, :, None])
    with h5py.File(out_path, "w") as f:
        dset = f.create_dataset("exported_data", data=arr, dtype="uint8")
        dset.attrs["axistags"] = AXISTAGS_5D_JSON
