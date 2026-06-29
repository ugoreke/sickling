"""Sickle cell classification pipeline.

Five stages:
    1. Frozen U-Net 4-class semantic segmentation (external — see ``training 2.ipynb``).
    2. Instance segmentation (this package, ``stage2_instances``).
    3. Per-cell crop extraction (``stage3_crops``).
    4. Representation learning bake-off (``stage4_repr``).
    5. Multimodal classifier (``stage5_multimodal``).
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sickling")
except PackageNotFoundError:  # not installed in editable mode yet
    __version__ = "0.1.0"

__all__ = ["__version__"]
