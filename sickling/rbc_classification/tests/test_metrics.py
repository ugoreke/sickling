"""Tests for ``sickling.eval.metrics``."""
from __future__ import annotations

import numpy as np

from sickling.rbc_classification.py_modules.eval.metrics import (
    compute_binary_metrics,
    pick_threshold_max_mcc,
    recall_at_precision,
)


def test_perfect_classifier_metrics():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    m = compute_binary_metrics(y_true, y_score)
    assert m.pr_auc == 1.0
    assert m.roc_auc == 1.0
    assert m.mcc == 1.0
    assert m.f1_sickle == 1.0
    assert m.f1_non_sickle == 1.0
    assert (m.confusion == np.array([[3, 0], [0, 3]])).all()


def test_inverted_classifier_metrics():
    """Scores anti-correlated with labels → MCC at the picked threshold should
    flip to a positive value because the threshold-picker uses MCC = max."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    m = compute_binary_metrics(y_true, y_score)
    # PR-AUC of an anti-correlated classifier ≈ prevalence (= 0.5 here, but
    # the AP scoring rule is more punishing — just assert it's below random).
    assert m.pr_auc < 0.5
    assert m.roc_auc == 0.0


def test_random_classifier_mcc_near_zero():
    rng = np.random.default_rng(0)
    n = 4000
    y_true = (rng.random(n) > 0.95).astype(np.int64)  # ~5% sickle
    y_score = rng.random(n)
    m = compute_binary_metrics(y_true, y_score)
    assert abs(m.mcc) < 0.1


def test_recall_at_p90_separable():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    rec, _t = recall_at_precision(y_true, y_score, target_precision=0.9)
    assert rec == 1.0


def test_recall_at_p90_unreachable():
    """If no operating point achieves precision=0.9, return 0.0 / +inf."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_score = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    rec, t = recall_at_precision(y_true, y_score, target_precision=0.9)
    assert rec == 0.0
    assert np.isposinf(t)


def test_pick_threshold_max_mcc_picks_separator():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    t = pick_threshold_max_mcc(y_true, y_score)
    # Any threshold in (0.3, 0.7] is optimal.
    assert 0.3 < t <= 0.7


def test_confusion_matrix_layout():
    """Sklearn's confusion_matrix uses [[TN, FP], [FN, TP]] with labels=[0,1]."""
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.9, 0.1, 0.9])  # one TN, one FP, one FN, one TP
    m = compute_binary_metrics(y_true, y_score, threshold_strategy="fixed", threshold=0.5)
    assert (m.confusion == np.array([[1, 1], [1, 1]])).all()


def test_single_class_metrics_are_nan():
    y_true = np.zeros(10, dtype=np.int64)
    y_score = np.linspace(0, 1, 10)
    m = compute_binary_metrics(y_true, y_score)
    assert np.isnan(m.pr_auc)
    assert np.isnan(m.roc_auc)
