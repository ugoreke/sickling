"""Reproducibility helpers — seed every RNG that any pipeline stage touches."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic_cudnn: bool = False) -> None:
    """Seed Python, NumPy, PyTorch (CPU + CUDA), and the env ``PYTHONHASHSEED``.

    Lightning's own ``seed_everything`` does this too; this wrapper exists so
    standalone scripts (e.g. data preprocessing CLIs) seed identically without
    needing to import Lightning.

    Set ``deterministic_cudnn=True`` for fully reproducible runs at the cost of
    a slowdown — recommended for ablation rows, not for pretraining.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
