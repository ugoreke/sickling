"""Hand-crafted shape descriptors for the morphology tower.

Per-cell features computed from the crop's instance-mask channels (ch1 = body,
ch2 = polymer). Per PIPELINE_PLAN §2 Stage 5:

    Basic shape    : area, perimeter, compactness (perim^2 / area), eccentricity,
                     solidity (area / convex_hull_area)
    Fourier        : magnitude of the first 8 boundary descriptors (rotation-invariant)
    Zernike        : moments up to degree 6 (16 numbers via mahotas)
    Polymer ratio  : polymer_area / cell_area  ∈ [0, 1]

Total feature dim with these defaults = 5 + 8 + 16 + 1 = 30.

All features are computed in pixel units; the morphology tower standardizes
them via a buffer of train-set means/stds at training start.
"""
from __future__ import annotations

import mahotas
import numpy as np
import torch
from skimage.measure import find_contours, regionprops

# Default feature-set knobs.
N_FOURIER = 8
ZERNIKE_DEGREE = 6
N_ZERNIKE = 16  # determined empirically for degree=6 via mahotas

_BASIC_NAMES = ("area", "perimeter", "compactness", "eccentricity", "solidity")
FEATURE_NAMES: tuple[str, ...] = (
    *_BASIC_NAMES,
    *(f"fourier_{i}" for i in range(N_FOURIER)),
    *(f"zernike_{i}" for i in range(N_ZERNIKE)),
    "polymer_ratio",
)
N_FEATURES = len(FEATURE_NAMES)


def _basic_shape(mask: np.ndarray) -> np.ndarray:
    """area / perimeter / compactness / eccentricity / solidity from a binary mask."""
    if mask.sum() == 0:
        return np.zeros(5, dtype=np.float32)
    props_list = regionprops(mask.astype(np.int32))
    if not props_list:
        return np.zeros(5, dtype=np.float32)
    props = max(props_list, key=lambda p: p.area)  # largest connected component
    area = float(props.area)
    perimeter = float(props.perimeter) if props.perimeter > 0 else 1.0
    compactness = perimeter * perimeter / max(area, 1.0)
    eccentricity = float(props.eccentricity)
    convex_area = float(props.convex_area) if props.convex_area > 0 else max(area, 1.0)
    solidity = area / convex_area
    return np.array(
        [area, perimeter, compactness, eccentricity, solidity], dtype=np.float32
    )


def _fourier_descriptors(mask: np.ndarray, n_harmonics: int = N_FOURIER) -> np.ndarray:
    """Magnitudes of the first ``n_harmonics`` boundary-FFT coefficients,
    normalized by |F1| so the descriptor is scale-invariant.

    Returns zeros if no contour is found (defensive: blank mask)."""
    if mask.sum() == 0:
        return np.zeros(n_harmonics, dtype=np.float32)
    contours = find_contours(mask.astype(np.float32), level=0.5)
    if not contours:
        return np.zeros(n_harmonics, dtype=np.float32)
    contour = max(contours, key=len)
    # Treat boundary as a complex sequence z_t = x_t + i y_t, FFT, take |.|.
    z = contour[:, 1] + 1j * contour[:, 0]
    coeffs = np.fft.fft(z)
    # Skip DC (k=0). Normalize by |coeff_1| for scale invariance.
    mags = np.abs(coeffs)
    norm = mags[1] if mags.shape[0] > 1 and mags[1] > 0 else 1.0
    out = np.zeros(n_harmonics, dtype=np.float32)
    available = min(n_harmonics, max(mags.shape[0] - 1, 0))
    out[:available] = mags[1 : 1 + available] / norm
    return out


def _zernike(mask: np.ndarray, degree: int = ZERNIKE_DEGREE) -> np.ndarray:
    """Mahotas Zernike moments up to ``degree``. Pads to ``N_ZERNIKE`` if a
    smaller-degree call returns fewer."""
    if mask.sum() == 0:
        return np.zeros(N_ZERNIKE, dtype=np.float32)
    # Radius covers the cell — use bbox half-diagonal of mask.
    rows, cols = np.where(mask)
    cy, cx = rows.mean(), cols.mean()
    rmax = max(np.sqrt((rows - cy) ** 2 + (cols - cx) ** 2).max(), 1.0)
    z = mahotas.features.zernike_moments(mask.astype(np.uint8), radius=float(rmax), degree=degree)
    out = np.zeros(N_ZERNIKE, dtype=np.float32)
    out[: len(z)] = z[:N_ZERNIKE]
    return out


def _polymer_ratio(body_mask: np.ndarray, polymer_mask: np.ndarray) -> float:
    body_area = float(body_mask.sum())
    polymer_area = float(polymer_mask.sum())
    if body_area + polymer_area == 0:
        return 0.0
    # Definition: polymer extent relative to total cell footprint (body ∪ polymer).
    return polymer_area / (body_area + polymer_area)


def compute_features(crop: torch.Tensor | np.ndarray) -> np.ndarray:
    """Extract the full morphology feature vector for one crop.

    Args:
        crop: ``(3, H, W)`` tensor or array. ch0 raw (unused), ch1 cell-body
            mask, ch2 polymer mask. Masks are binarized via ``> 0.5``.

    Returns:
        ``np.ndarray[float32, N_FEATURES]``.
    """
    c = crop.detach().cpu().numpy() if torch.is_tensor(crop) else np.asarray(crop)
    if c.ndim != 3 or c.shape[0] < 3:
        raise ValueError(f"Expected (3, H, W) crop, got shape {c.shape}.")

    body = (c[1] > 0.5).astype(np.uint8)
    polymer = (c[2] > 0.5).astype(np.uint8)
    full_mask = ((body + polymer) > 0).astype(np.uint8)

    basic = _basic_shape(full_mask)
    fourier = _fourier_descriptors(full_mask)
    zern = _zernike(full_mask)
    pratio = np.array([_polymer_ratio(body, polymer)], dtype=np.float32)
    return np.concatenate([basic, fourier, zern, pratio]).astype(np.float32)
