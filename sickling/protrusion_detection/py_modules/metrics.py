"""Evaluation metrics and the per-iteration HITL trajectory log.

- ``per_class_dice`` / ``confusion_matrix`` / ``target_recall_precision``
  are used both for per-epoch checkpoint selection and for the trajectory.
- ``polymer_monitor`` is the secondary signal on BootstrappedLabels
  (polymer-positive binary; the other classes are imperfect there so we
  collapse them all into "not polymer"). Per-image, then averaged.
- ``append_iteration_row`` writes one row to ``metrics/iteration_log.csv``
  per retrain. ``plot_trajectory`` renders the trajectory PNG.
"""

from __future__ import annotations

import csv
import datetime as _dt
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import cfg


# --- Per-image dice ----------------------------------------------------------

def per_class_dice(pred: torch.Tensor, gt: torch.Tensor,
                   n_classes: int = None, ignore_index: int = None) -> np.ndarray:
    """Dice per class on a single image. Returns float array of shape (n_classes,)."""
    n_classes = n_classes or cfg.N_CLASSES
    ignore_index = ignore_index if ignore_index is not None else cfg.IGNORE_INDEX
    valid = (gt != ignore_index).float()
    out = np.zeros(n_classes, dtype=np.float64)
    for c in range(n_classes):
        p = (pred == c).float() * valid
        t = (gt == c).float() * valid
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        if union == 0:
            out[c] = 1.0 if t.sum() == 0 else 0.0
        else:
            out[c] = (2.0 * inter / union).item()
    return out


def confusion_matrix(pred: torch.Tensor, gt: torch.Tensor,
                     n_classes: int = None, ignore_index: int = None) -> np.ndarray:
    """Counts confusion matrix; rows = true, cols = pred."""
    n_classes = n_classes or cfg.N_CLASSES
    ignore_index = ignore_index if ignore_index is not None else cfg.IGNORE_INDEX
    p = pred.cpu().numpy().ravel() if torch.is_tensor(pred) else np.asarray(pred).ravel()
    g = gt.cpu().numpy().ravel() if torch.is_tensor(gt) else np.asarray(gt).ravel()
    valid = (g >= 0) & (g < n_classes) & (g != ignore_index)
    p, g = p[valid], g[valid]
    idx = g * n_classes + p
    counts = np.bincount(idx, minlength=n_classes * n_classes)
    return counts.reshape(n_classes, n_classes)


def target_recall_precision(cm: np.ndarray, target_classes: Sequence[int]) -> Dict[int, Dict[str, float]]:
    """From a confusion matrix, per-target-class recall and precision."""
    out: Dict[int, Dict[str, float]] = {}
    for c in target_classes:
        tp = float(cm[c, c])
        fn = float(cm[c, :].sum() - tp)
        fp = float(cm[:, c].sum() - tp)
        out[c] = {
            "recall": tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
            "precision": tp / (tp + fp) if (tp + fp) > 0 else float("nan"),
        }
    return out


# --- Polymer-only monitor on BootstrappedLabels -----------------------------

POLYMER_CLASS = 0


def polymer_monitor(pred: torch.Tensor, gt: torch.Tensor,
                    ignore_index: int = None) -> Tuple[float, float]:
    """Binary polymer vs not-polymer recall, precision on one image.

    Returns ``(recall, precision)`` as floats, NaN where the denominator
    is zero. ``ignore_index`` pixels are excluded.
    """
    ignore_index = ignore_index if ignore_index is not None else cfg.IGNORE_INDEX
    valid = (gt != ignore_index)
    p = (pred == POLYMER_CLASS) & valid
    t = (gt == POLYMER_CLASS) & valid
    tp = float((p & t).sum().item())
    fp = float((p & ~t).sum().item())
    fn = float((~p & t).sum().item())
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    return recall, precision


# --- Trajectory log ----------------------------------------------------------

_TRAJECTORY_CSV = "iteration_log.csv"
_TRAJECTORY_PNG = "trajectory.png"
_TP_FP_PNG = "tp_fp_trajectory.png"


def trajectory_csv_path() -> str:
    return os.path.join(cfg.METRICS_DIR, _TRAJECTORY_CSV)


def latest_best_fold(default: Optional[int] = None) -> int:
    """Fold index the next training round should focus on.

    Reads the most recent row of ``metrics/iteration_log.csv`` and returns its
    ``best_fold``. After a kfold loop this is the fold that won — switching
    ``FOLD_MODE`` to ``'single'`` then carries that winner forward as the
    fold the next single retrain trains. After a single loop it's just the
    same fold that loop trained. Falls back to ``default`` (or
    ``cfg.BEST_FOLD``) when the log is missing or unparseable.
    """
    fallback = default if default is not None else cfg.BEST_FOLD
    path = trajectory_csv_path()
    if not os.path.exists(path):
        return fallback
    try:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return fallback
    if not rows:
        return fallback
    bf = rows[-1].get("best_fold", "")
    try:
        return int(float(bf))
    except (TypeError, ValueError):
        return fallback


def append_iteration_row(row: Dict[str, float | str | int]) -> str:
    """Append one row to ``metrics/iteration_log.csv``, creating it if absent.

    Caller fills the columns it wants tracked; the CSV is written in the
    order of the union of all rows' keys (so adding new columns later is OK).
    """
    os.makedirs(cfg.METRICS_DIR, exist_ok=True)
    path = trajectory_csv_path()
    row = dict(row)
    row.setdefault("timestamp", _dt.datetime.now().isoformat(timespec="seconds"))

    if os.path.exists(path):
        with open(path, newline="") as f:
            existing_fields = next(csv.reader(f), [])
        all_fields = list(existing_fields) + [k for k in row.keys() if k not in existing_fields]
        rows = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        rows.append({k: row.get(k, "") for k in all_fields})
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in all_fields})
    else:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
    return path


def plot_trajectory(out_path: Optional[str] = None) -> Optional[str]:
    """Render ``metrics/trajectory.png`` from the iteration log.

    Plots, per training run on the x-axis (``n_corrected_tiles``):
    - per-class dice on the val split (one line per class)
    - target-class recall (dashed)
    """
    import matplotlib.pyplot as plt

    path = trajectory_csv_path()
    if not os.path.exists(path):
        return None
    rows: List[Dict[str, str]] = []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    def _f(k: str, r: Dict[str, str]) -> float:
        v = r.get(k, "")
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    n_tiles = [int(float(r.get("n_corrected_tiles", 0))) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e63946", "#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#8e44ad"]
    for c in range(cfg.N_CLASSES):
        ys = [_f(f"val_dice_class_{c}", r) for r in rows]
        ax.plot(n_tiles, ys, marker="o", color=colors[c % len(colors)],
                label=f"val dice class {c}")
    for c in cfg.TARGET_CLASSES:
        ys = [_f(f"val_recall_class_{c}", r) for r in rows]
        ax.plot(n_tiles, ys, marker="x", linestyle="--",
                color=colors[c % len(colors)],
                label=f"val recall class {c}")

    ax.set_xlabel("# corrected tiles at training time")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("HITL trajectory")
    fig.tight_layout()
    out_path = out_path or os.path.join(cfg.METRICS_DIR, _TRAJECTORY_PNG)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_tp_fp_trajectory(
    out_path: Optional[str] = None,
    target_classes: Optional[Sequence[int]] = None,
) -> Optional[str]:
    """Render ``metrics/tp_fp_trajectory.png`` — TP rate and FP fraction per class.

    For each class in ``target_classes`` (default ``cfg.TARGET_CLASSES`` — the
    classes you're currently trying to fix), plots on the same axes against
    ``n_corrected_tiles``:

    - **TP rate** = ``val_recall_class_<c>``       (solid, circle marker)
      Fraction of true-class pixels the model finds. Climbing = closing FNs.
    - **FP fraction** = ``1 - val_precision_class_<c>``   (dashed, x marker)
      Fraction of predicted-class pixels that are wrong. Dropping = closing
      over-firing. Computed from precision in the log; the FP rate against
      true negatives is not used because for rare classes (polymer ~0.8 %)
      it is dominated by the negative count and barely moves.

    Rows that don't carry a column for a class plot as NaN and are skipped
    by matplotlib — historical rows from earlier ``TARGET_CLASSES`` settings
    don't break the figure.
    """
    import matplotlib.pyplot as plt

    path = trajectory_csv_path()
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        rows: List[Dict[str, str]] = list(csv.DictReader(f))
    if not rows:
        return None

    target_classes = list(target_classes) if target_classes is not None else list(cfg.TARGET_CLASSES)
    if not target_classes:
        return None

    def _f(k: str, r: Dict[str, str]) -> float:
        v = r.get(k, "")
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    n_tiles = [int(float(r.get("n_corrected_tiles", 0) or 0)) for r in rows]
    colors = ["#e63946", "#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#8e44ad"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for c in target_classes:
        col = colors[c % len(colors)]
        recall = [_f(f"val_recall_class_{c}", r) for r in rows]
        prec = [_f(f"val_precision_class_{c}", r) for r in rows]
        # FP fraction of predictions = 1 - precision (NaN where precision is NaN).
        fp_frac = [float("nan") if p != p else (1.0 - p) for p in prec]
        ax.plot(n_tiles, recall, marker="o", linestyle="-", color=col,
                label=f"class {c}: TP rate (recall)")
        ax.plot(n_tiles, fp_frac, marker="x", linestyle="--", color=col,
                label=f"class {c}: FP frac (1 − precision)")

    ax.set_xlabel("# corrected tiles at training time")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Per-class TP rate vs FP fraction on val (lower FP, higher TP = better)")
    fig.tight_layout()
    out_path = out_path or os.path.join(cfg.METRICS_DIR, _TP_FP_PNG)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
