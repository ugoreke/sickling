"""Frozen U-Net inference helpers.

Architecture mirrors the model defined in ``training 2.ipynb`` so the
``unet_fold_*_best.pth`` checkpoints load with no key remapping. Use
:func:`load_unet` and :func:`predict_label_map` to score new raw images.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class _DoubleConv(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """Same architecture as ``training 2.ipynb``: 1-channel input, configurable
    output classes. Default 4 classes matches the project convention
    (0=polymer, 1=background, 2=cell_body, 3=cell_border)."""

    def __init__(self, n_channels: int = 1, n_classes: int = 4) -> None:
        super().__init__()
        self.inc = _DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(512, 1024))
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv_up1 = _DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv_up2 = _DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up3 = _DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up4 = _DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.conv_up1(torch.cat([x4, self.up1(self.down4(x4))], dim=1))
        x = self.conv_up2(torch.cat([x3, self.up2(x)], dim=1))
        x = self.conv_up3(torch.cat([x2, self.up3(x)], dim=1))
        x = self.conv_up4(torch.cat([x1, self.up4(x)], dim=1))
        return self.outc(x)


def load_unet(
    model_path: str | Path,
    n_classes: int = 4,
    device: str | torch.device | None = None,
) -> UNet:
    """Load a saved ``unet_fold_*_best.pth`` checkpoint into a fresh ``UNet``."""
    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = UNet(n_channels=1, n_classes=n_classes).to(device)
    state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_label_map(
    model: UNet,
    raw_norm: np.ndarray,
    tile_size: int = 256,
    overlap: float = 0.5,
    n_classes: int = 4,
) -> np.ndarray:
    """Sliding-window argmax prediction.

    Args:
        model: a U-Net in eval mode.
        raw_norm: 2-D float32 array, percentile-normalized to [0, 1] using the
            project's ``normalize_image``.
        tile_size: window size (must match training).
        overlap: 0.5 = 50% stride overlap.
        n_classes: number of output classes.

    Returns:
        ``np.ndarray[int16]`` 0-indexed label map of the same H×W as ``raw_norm``.
    """
    if raw_norm.ndim != 2:
        raise ValueError(f"Expected 2-D raw image, got shape {raw_norm.shape}")

    device = next(model.parameters()).device
    h, w = raw_norm.shape
    if h < tile_size or w < tile_size:
        raise ValueError(
            f"Image {h}x{w} smaller than tile_size={tile_size}; pad upstream first."
        )

    img_t = torch.from_numpy(raw_norm).float().unsqueeze(0).unsqueeze(0).to(device)

    stride = max(int(tile_size * (1.0 - overlap)), 1)
    prob_map = torch.zeros((n_classes, h, w), device=device, dtype=torch.float32)
    count_map = torch.zeros((n_classes, h, w), device=device, dtype=torch.float32)

    ys = list(range(0, h, stride))
    xs = list(range(0, w, stride))
    if ys[-1] + tile_size < h:
        ys.append(h - tile_size)
    if xs[-1] + tile_size < w:
        xs.append(w - tile_size)

    for y in ys:
        for x in xs:
            y0 = max(0, min(y, h - tile_size))
            x0 = max(0, min(x, w - tile_size))
            crop = img_t[:, :, y0 : y0 + tile_size, x0 : x0 + tile_size]
            probs = torch.softmax(model(crop), dim=1).squeeze(0)
            prob_map[:, y0 : y0 + tile_size, x0 : x0 + tile_size] += probs
            count_map[:, y0 : y0 + tile_size, x0 : x0 + tile_size] += 1

    avg = prob_map / count_map.clamp(min=1.0)
    return torch.argmax(avg, dim=0).cpu().numpy().astype(np.int16)
