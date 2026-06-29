"""Tests for ``sickling.stage5_multimodal.morphology_features``.

Known shape regressions: a circle should be near-perfectly compact
(compactness ≈ 4π ≈ 12.57) and have low eccentricity; an ellipse should have
high eccentricity; a square should have compactness 16; a ring should
produce a non-trivial polymer ratio when polymer pixels are added separately.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sickling.rbc_classification.py_modules.stage5_multimodal.morphology_features import (
    FEATURE_NAMES,
    N_FEATURES,
    _basic_shape,
    _fourier_descriptors,
    _polymer_ratio,
    _zernike,
    compute_features,
)


def _circle_mask(size: int = 96, radius: int = 30) -> np.ndarray:
    yy, xx = np.ogrid[:size, :size]
    cy, cx = size // 2, size // 2
    return ((yy - cy) ** 2 + (xx - cx) ** 2 <= radius * radius).astype(np.uint8)


def _square_mask(size: int = 96, side: int = 40) -> np.ndarray:
    arr = np.zeros((size, size), dtype=np.uint8)
    cy, cx = size // 2, size // 2
    h = side // 2
    arr[cy - h : cy + h, cx - h : cx + h] = 1
    return arr


def _ellipse_mask(size: int = 96, a: int = 30, b: int = 12) -> np.ndarray:
    yy, xx = np.ogrid[:size, :size]
    cy, cx = size // 2, size // 2
    return (((yy - cy) ** 2) / (b * b) + ((xx - cx) ** 2) / (a * a) <= 1.0).astype(np.uint8)


def test_basic_circle_compactness():
    mask = _circle_mask(radius=30)
    feats = _basic_shape(mask)
    area, perim, compactness, ecc, sol = feats
    # Circle: area ≈ π r², perim ≈ 2π r, compactness = perim^2 / area ≈ 4π ≈ 12.57.
    assert 11.0 < compactness < 14.0, f"circle compactness off: {compactness}"
    assert ecc < 0.4, f"circle eccentricity too high: {ecc}"
    assert sol > 0.97, f"circle solidity should be ~1: {sol}"


def test_basic_ellipse_eccentricity():
    feats = _basic_shape(_ellipse_mask(a=30, b=10))
    _, _, _, ecc, _ = feats
    assert ecc > 0.85, f"flat ellipse eccentricity should be high: {ecc}"


def test_basic_square_compactness_above_circle():
    """Square's perim^2/area = (4s)^2 / s^2 = 16. Higher than the circle's ≈12.57."""
    feats = _basic_shape(_square_mask(side=40))
    _, _, compactness, _, sol = feats
    assert compactness > 12.5
    assert sol > 0.95


def test_polymer_ratio_zero_when_no_polymer():
    body = _circle_mask(radius=20)
    polymer = np.zeros_like(body)
    assert _polymer_ratio(body, polymer) == 0.0


def test_polymer_ratio_positive_when_polymer_present():
    body = _circle_mask(radius=20)
    polymer = _circle_mask(radius=30) ^ body  # ring around the body
    r = _polymer_ratio(body, polymer)
    assert 0.0 < r < 1.0


def test_fourier_zero_for_blank():
    out = _fourier_descriptors(np.zeros((96, 96), dtype=np.uint8))
    assert out.shape == (8,)
    assert out.sum() == 0.0


def test_zernike_zero_for_blank():
    out = _zernike(np.zeros((96, 96), dtype=np.uint8))
    assert out.shape == (16,)
    assert np.all(out == 0)


def test_compute_features_full_shape():
    crop = torch.zeros(3, 96, 96)
    body = _circle_mask(radius=20)
    polymer = _circle_mask(radius=30) ^ body
    crop[1] = torch.from_numpy(body).float()
    crop[2] = torch.from_numpy(polymer).float()

    feats = compute_features(crop)
    assert feats.shape == (N_FEATURES,)
    assert feats.dtype == np.float32
    assert len(FEATURE_NAMES) == N_FEATURES
    # area > 0
    assert feats[0] > 0


def test_compute_features_rejects_2d_input():
    with pytest.raises(ValueError):
        compute_features(np.zeros((96, 96), dtype=np.float32))
