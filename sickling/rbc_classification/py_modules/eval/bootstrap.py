"""Bootstrap confidence intervals for scalar metrics + the PR curve band.

Vectorized in NumPy: build a ``(n_resamples, n_samples)`` index matrix once
and stride over it.

Per PIPELINE_PLAN §3: 1000 resamples, 95% CI by default (alpha=0.05).
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import precision_recall_curve


def _resample_indices(n: int, n_resamples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_resamples, n))


def bootstrap_metric(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return ``(point_estimate, ci_low, ci_high)``.

    Resamples ``(y_true, y_score)`` pairs with replacement. Resamples that
    end up with only one class get NaN — those are excluded from the
    percentile computation.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = y_true.shape[0]
    point = float(metric_fn(y_true, y_score))

    idx = _resample_indices(n, n_resamples, seed)
    values = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        ii = idx[i]
        try:
            values[i] = metric_fn(y_true[ii], y_score[ii])
        except (ValueError, RuntimeError):
            values[i] = np.nan

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return point, float("nan"), float("nan")
    lo = float(np.percentile(finite, 100 * alpha / 2))
    hi = float(np.percentile(finite, 100 * (1 - alpha / 2)))
    return point, lo, hi


def bootstrap_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    n_thresholds: int = 200,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """PR-curve precision band as a function of recall.

    For each resample, compute its (precision, recall) curve and interpolate
    precision onto a fixed monotone-decreasing recall grid. Return the
    pointwise mean / lower / upper percentiles plus the point-estimate curve.

    Returns dict with keys:
        ``recall_grid``       — (G,) descending from 1.0 to 0.0.
        ``precision_point``   — (G,) precision on the unbootstrapped sample.
        ``precision_mean``    — (G,) mean precision across resamples.
        ``precision_low``     — (G,) lower CI bound at each recall.
        ``precision_high``    — (G,) upper CI bound at each recall.
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)
    n = y_true.shape[0]

    grid = np.linspace(1.0, 0.0, n_thresholds)

    def _interp_pr(yt: np.ndarray, ys: np.ndarray) -> np.ndarray:
        if np.unique(yt).size < 2:
            return np.full(n_thresholds, np.nan)
        p, r, _ = precision_recall_curve(yt, ys)
        # precision_recall_curve returns r descending from 1 to 0.
        # np.interp wants ascending xp, so flip both.
        order = np.argsort(r)
        return np.interp(grid, r[order], p[order])

    point_curve = _interp_pr(y_true, y_score)

    idx = _resample_indices(n, n_resamples, seed)
    resampled = np.empty((n_resamples, n_thresholds), dtype=np.float64)
    for i in range(n_resamples):
        ii = idx[i]
        resampled[i] = _interp_pr(y_true[ii], y_score[ii])

    finite_mask = np.isfinite(resampled)
    with np.errstate(invalid="ignore"):
        # Per-recall percentiles, ignoring NaN rows.
        precision_mean = np.nanmean(resampled, axis=0)
        precision_low = np.nanpercentile(resampled, 100 * alpha / 2, axis=0)
        precision_high = np.nanpercentile(resampled, 100 * (1 - alpha / 2), axis=0)

    return {
        "recall_grid": grid,
        "precision_point": point_curve,
        "precision_mean": precision_mean,
        "precision_low": precision_low,
        "precision_high": precision_high,
        "n_resamples_finite": np.array(finite_mask.any(axis=1).sum()),
    }
