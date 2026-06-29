"""Tests for ``sickling.eval.bootstrap``."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, matthews_corrcoef

from sickling.rbc_classification.py_modules.eval.bootstrap import bootstrap_metric, bootstrap_pr_curve


def _safe_ap(yt, ys):
    return float(average_precision_score(yt, ys)) if np.unique(yt).size == 2 else float("nan")


def test_ci_contains_point_for_clean_problem():
    rng = np.random.default_rng(0)
    n = 400
    y_true = (rng.random(n) < 0.3).astype(np.int64)
    y_score = y_true + 0.05 * rng.normal(size=n)
    point, lo, hi = bootstrap_metric(y_true, y_score, _safe_ap, n_resamples=200, seed=0)
    assert lo <= point <= hi
    assert (hi - lo) < 0.1, f"separable problem should have a tight CI, got [{lo}, {hi}]"


def test_ci_wider_for_random_problem():
    rng = np.random.default_rng(0)
    n = 400
    y_true = (rng.random(n) < 0.3).astype(np.int64)
    y_score = rng.random(n)
    point, lo, hi = bootstrap_metric(y_true, y_score, _safe_ap, n_resamples=200, seed=0)
    assert lo <= point <= hi
    assert (hi - lo) > 0.02


def test_pr_band_shape_and_monotonicity():
    rng = np.random.default_rng(1)
    n = 300
    y_true = (rng.random(n) < 0.2).astype(np.int64)
    y_score = y_true + 0.2 * rng.normal(size=n)

    band = bootstrap_pr_curve(y_true, y_score, n_resamples=100, n_thresholds=64, seed=0)
    grid = band["recall_grid"]
    assert grid.shape == (64,)
    # Recall grid is descending.
    assert grid[0] > grid[-1]
    # Bound ordering at every recall.
    finite = np.isfinite(band["precision_low"]) & np.isfinite(band["precision_high"])
    assert np.all(band["precision_low"][finite] <= band["precision_high"][finite])
    assert band["precision_point"].shape == (64,)


def test_bootstrap_metric_handles_nan_resamples():
    """Pure-noise small set → some resamples may have only one class. Coverage
    should still produce finite percentiles by skipping those."""
    rng = np.random.default_rng(0)
    n = 30
    y_true = (rng.random(n) < 0.05).astype(np.int64)  # heavily imbalanced
    y_score = rng.random(n)

    def _mcc(yt, ys):
        if np.unique(yt).size < 2:
            return float("nan")
        return float(matthews_corrcoef(yt, (ys >= 0.5).astype(np.int64)))

    point, lo, hi = bootstrap_metric(y_true, y_score, _mcc, n_resamples=200, seed=0)
    # Even with NaNs we expect finite percentile bounds.
    assert np.isfinite(lo)
    assert np.isfinite(hi)
