"""sickling.rbc_classification — sickle / non-sickle cell classifier arm.

DINOv2 ViT-S/14 image tower (frozen) + MLP morphology tower over 30
hand-crafted shape descriptors, fused by a 2-layer MLP into a 2-class
softmax. See ``Sickle cell classification`` in the *Methods* section of
Goreke et al. for full training detail.

The implementation modules live under
``sickling.rbc_classification.py_modules``. This ``__init__`` aliases
every submodule at the ``sickling.rbc_classification.X`` level via
``sys.modules`` so user-facing imports such as
``from sickling.rbc_classification.eval.report import read_report``
work directly.
"""

from __future__ import annotations

import sys as _sys

_SUBMODULES = (
    "config",
    "cli",
    "io",
    "data",
    "engineering",
    "ablation",
    "eval",
    "stage1_unet",
    "stage2_instances",
    "stage3_crops",
    "stage4_repr",
    "stage5_multimodal",
)
for _name in _SUBMODULES:
    try:
        __import__(f"sickling.rbc_classification.py_modules.{_name}")
        _sys.modules[f"sickling.rbc_classification.{_name}"] = (
            _sys.modules[f"sickling.rbc_classification.py_modules.{_name}"]
        )
    except Exception:
        # A submodule may have optional heavy dependencies; alias what we can.
        pass

__all__ = ("py_modules",)
