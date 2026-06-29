"""Vanilla U-Net (1 grayscale input channel, configurable output classes),
plus a backbone factory and filename parser so heavier SMP encoders can
slot in via ``cfg.MODEL_BACKBONE``.

Preserved verbatim from ``training_2.ipynb`` — see ARCHITECTURE.md §10. The
default ``n_classes`` matches ``cfg.N_CLASSES`` so the notebook never
instantiates the wrong head by accident.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import cfg


class DoubleConv(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels: int = 1, n_classes: int | None = None) -> None:
        super().__init__()
        if n_classes is None:
            n_classes = cfg.N_CLASSES
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)
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


# ----------------------------------------------------------------------------
# Backbone factory
# ----------------------------------------------------------------------------
#
# Filename convention: ``<backbone-tag>_fold_<f>_best_loop_<N>.pth``. So the
# existing ``unet_fold_2_best_loop_3.pth`` files map to backbone="unet" out
# of the box, and SMP runs write e.g. ``smp_unet_efficientnet-b0_fold_2_best_loop_4.pth``.
# Multiple backbones coexist across loops without colliding.
#
# Supported backbones:
#   "unet"                            — the vanilla UNet above (default).
#   "smp_unet_<encoder>"              — segmentation_models_pytorch UNet with
#                                       the given encoder. ImageNet-pretrained,
#                                       1-channel input (SMP averages the
#                                       first conv weights across channels).
#                                       e.g. "smp_unet_efficientnet-b0",
#                                            "smp_unet_efficientnet-b7".

_HEAVY_BACKBONES = {"smp_unet_efficientnet-b7"}


def build_model(backbone: Optional[str] = None) -> nn.Module:
    """Construct the model per ``backbone`` (default ``cfg.MODEL_BACKBONE``).

    Raises ``ImportError`` if SMP is requested but not installed, and
    ``ValueError`` for an unknown backbone string.
    """
    bk = backbone if backbone is not None else cfg.MODEL_BACKBONE
    if bk == "unet":
        return UNet(1, cfg.N_CLASSES)
    if bk.startswith("smp_unet_"):
        encoder = bk[len("smp_unet_"):]
        try:
            import segmentation_models_pytorch as smp
        except ImportError as e:
            raise ImportError(
                f"cfg.MODEL_BACKBONE={bk!r} needs segmentation_models_pytorch. "
                f"Install with: pip install segmentation-models-pytorch"
            ) from e
        if bk in _HEAVY_BACKBONES:
            print(f"ℹ️  {bk}: heavy encoder — consider reducing cfg.BATCH_SIZE if you OOM.")
        return smp.Unet(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=1,
            classes=cfg.N_CLASSES,
        )
    raise ValueError(
        f"Unknown MODEL_BACKBONE={bk!r}. Supported: 'unet', 'smp_unet_<encoder>'."
    )


# Order matters for the prefix scan in parse_backbone_from_ckpt_path —
# longer prefixes first so "smp_unet_efficientnet-b0" wins over "smp_unet_".
_KNOWN_BACKBONE_TAGS: Tuple[str, ...] = (
    "smp_unet_efficientnet-b7",
    "smp_unet_efficientnet-b0",
    "unet",
)


def parse_backbone_from_ckpt_path(path: str) -> str:
    """Recover the backbone tag from a ``<backbone>_fold_<f>_best_loop_<N>.pth``
    filename. Falls back to ``'unet'`` for legacy filenames so existing
    pre-tag checkpoints keep loading."""
    base = os.path.basename(path)
    for tag in _KNOWN_BACKBONE_TAGS:
        if base.startswith(tag + "_fold_"):
            return tag
    return "unet"
