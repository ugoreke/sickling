"""Lightweight metric helpers used inside Stage 4 LightningModules.

Full metrics + bootstrap CIs land in milestone 6 (``sickling/eval/metrics.py``).
For now we only need scalar values for ``ModelCheckpoint`` to monitor.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, matthews_corrcoef


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    # NumPy can't convert bf16/fp16 tensors directly — cast to float32 first.
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def pr_auc(y_true: torch.Tensor | np.ndarray, y_score: torch.Tensor | np.ndarray) -> float:
    yt = _to_numpy(y_true)
    ys = _to_numpy(y_score)
    if np.unique(yt).size < 2:
        return float("nan")
    return float(average_precision_score(yt, ys))


def mcc(y_true: torch.Tensor | np.ndarray, y_pred: torch.Tensor | np.ndarray) -> float:
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    if np.unique(yt).size < 2:
        return float("nan")
    return float(matthews_corrcoef(yt, yp))
