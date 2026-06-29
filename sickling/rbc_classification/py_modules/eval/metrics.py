"""Binary classification metrics for the sickle-vs-non-sickle task.

Headline metric (PIPELINE_PLAN §3): PR-AUC. Reported alongside ROC-AUC, MCC at
the picked threshold, recall@precision=0.9, per-class F1, and the 2×2
confusion matrix.

Threshold selection: by default pick the score threshold that maximizes MCC
on the val set (``"max_mcc"``); ``"fixed"`` and ``"max_f1"`` are also supported.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
)


@dataclass(frozen=True)
class BinaryMetrics:
    pr_auc: float
    roc_auc: float
    mcc: float
    recall_at_p90: float
    threshold_at_p90: float
    f1_sickle: float
    f1_non_sickle: float
    threshold: float
    confusion: np.ndarray  # (2, 2) [[TN, FP], [FN, TP]]


def _safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def pick_threshold_max_mcc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Sweep candidate thresholds (= unique scores) and return the one with the
    highest MCC. Falls back to 0.5 if no class diversity is present."""
    if np.unique(y_true).size < 2:
        return 0.5
    candidates = np.unique(y_score)
    if candidates.size > 200:  # cap the sweep cost
        candidates = np.linspace(candidates.min(), candidates.max(), 200)
    best_t, best_mcc = 0.5, -np.inf
    for t in candidates:
        preds = (y_score >= t).astype(np.int64)
        if np.unique(preds).size < 2:
            continue
        m = matthews_corrcoef(y_true, preds)
        if m > best_mcc:
            best_mcc, best_t = m, float(t)
    return best_t


def pick_threshold_max_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return 0.5
    p, r, t = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns N+1 p/r and N thresholds.
    f1 = 2 * p * r / np.where(p + r > 0, p + r, 1.0)
    if t.size == 0:
        return 0.5
    return float(t[max(int(np.argmax(f1[:-1])), 0)])


def recall_at_precision(
    y_true: np.ndarray, y_score: np.ndarray, target_precision: float = 0.9
) -> tuple[float, float]:
    """Return ``(recall, threshold)`` at the highest threshold whose precision
    meets or exceeds ``target_precision``. If no operating point achieves
    that precision, returns ``(0.0, +inf)``.
    """
    if np.unique(y_true).size < 2:
        return float("nan"), float("nan")
    p, r, t = precision_recall_curve(y_true, y_score)
    # p, r have length N+1; t has length N. Drop the last (recall=0) point.
    p, r = p[:-1], r[:-1]
    mask = p >= target_precision
    if not mask.any():
        return 0.0, float("inf")
    # Among valid thresholds, pick the one with highest recall.
    valid_r = np.where(mask, r, -np.inf)
    idx = int(np.argmax(valid_r))
    return float(r[idx]), float(t[idx])


def compute_binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold_strategy: Literal["max_mcc", "max_f1", "fixed"] = "max_mcc",
    threshold: float | None = None,
    target_precision: float = 0.9,
) -> BinaryMetrics:
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)

    pr_auc = _safe_pr_auc(y_true, y_score)
    roc_auc = _safe_roc_auc(y_true, y_score)
    rec_p90, t_p90 = recall_at_precision(y_true, y_score, target_precision)

    if threshold_strategy == "fixed":
        if threshold is None:
            raise ValueError("threshold_strategy='fixed' requires `threshold` argument.")
        t = float(threshold)
    elif threshold_strategy == "max_f1":
        t = pick_threshold_max_f1(y_true, y_score)
    else:
        t = pick_threshold_max_mcc(y_true, y_score)

    preds = (y_score >= t).astype(np.int64)

    if np.unique(y_true).size < 2 or np.unique(preds).size < 2:
        mcc = float("nan")
    else:
        mcc = float(matthews_corrcoef(y_true, preds))

    f1_sickle = float(f1_score(y_true, preds, pos_label=1, zero_division=0))
    f1_non = float(f1_score(y_true, preds, pos_label=0, zero_division=0))

    cm = sk_confusion_matrix(y_true, preds, labels=[0, 1])
    return BinaryMetrics(
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        mcc=mcc,
        recall_at_p90=rec_p90,
        threshold_at_p90=t_p90,
        f1_sickle=f1_sickle,
        f1_non_sickle=f1_non,
        threshold=float(t),
        confusion=cm,
    )
