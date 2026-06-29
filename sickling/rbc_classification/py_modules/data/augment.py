"""Channel-aware augmentations.

Spatial augmentations (flip / rot90) apply to all three channels uniformly.
Photometric augmentations (brightness / contrast jitter) apply **only to ch0**
because ch1 and ch2 are binary instance masks — perturbing them would corrupt
the morphology signal.

Transforms are implemented as module-level callable *classes* (not closures)
so they pickle cleanly when ``DataLoader(num_workers > 0)`` spawns worker
processes on Windows / macOS. Closures returned from factory functions cannot
be pickled, which crashes the worker pool.
"""
from __future__ import annotations

import random

import torch
import torchvision.transforms.functional as TF

from sickling.rbc_classification.py_modules.config import AugmentConfig


def _random_flip_rot(t: torch.Tensor, cfg: AugmentConfig) -> torch.Tensor:
    if random.random() < cfg.hflip_p:
        t = TF.hflip(t)
    if random.random() < cfg.vflip_p:
        t = TF.vflip(t)
    if random.random() < cfg.rot90_p:
        k = random.choice((1, 2, 3))
        t = torch.rot90(t, k=k, dims=(-2, -1))
    return t


def _photometric_ch0(t: torch.Tensor, cfg: AugmentConfig) -> torch.Tensor:
    """Brightness + contrast jitter on ch0 only. ch1/ch2 unchanged."""
    if cfg.brightness_jitter > 0:
        delta = (random.random() * 2 - 1) * cfg.brightness_jitter
        t[0] = (t[0] + delta).clamp(0.0, 1.0)
    if cfg.contrast_jitter > 0:
        scale = 1.0 + (random.random() * 2 - 1) * cfg.contrast_jitter
        mean = t[0].mean()
        t[0] = ((t[0] - mean) * scale + mean).clamp(0.0, 1.0)
    return t


class _TrainTransform:
    """Spatial flips/rot90 + photometric jitter on ch0. Picklable."""

    def __init__(self, cfg: AugmentConfig) -> None:
        self.cfg = cfg

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        x = _random_flip_rot(x, self.cfg)
        x = _photometric_ch0(x, self.cfg)
        return x


class _EvalTransform:
    """Identity (resize already done in the Dataset). Picklable."""

    def __init__(self, cfg: AugmentConfig) -> None:
        self.cfg = cfg

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x


def train_transform(cfg: AugmentConfig) -> _TrainTransform:
    return _TrainTransform(cfg)


def eval_transform(cfg: AugmentConfig) -> _EvalTransform:
    return _EvalTransform(cfg)


def ssl_transform(cfg: AugmentConfig) -> _TrainTransform:
    """Currently same as train_transform. MAE doesn't use multi-crop; can swap
    in DINO-style multi-crop here if we ever want to compare SSL methods."""
    return _TrainTransform(cfg)
