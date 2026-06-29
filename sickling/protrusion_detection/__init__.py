"""sickling.protrusion_detection — HITL U-Net pixel segmentation arm.

Four-class semantic segmentation (0 = HbS protrusion, 1 = background,
2 = cell body, 3 = cell boundary) trained with a human-in-the-loop
correction process. The "protrusion" class corresponds to what's
discussed throughout Goreke et al. as the rigid HbS-dependent
structure that protrudes beyond the deoxygenated RBC membrane.

The implementation modules live under
``sickling.protrusion_detection.py_modules`` for neatness (so the arm
folder cleanly separates code from data folders). This ``__init__``
re-exposes every submodule at the ``sickling.protrusion_detection.X``
level via ``sys.modules`` aliasing so user-facing imports such as
``from sickling.protrusion_detection.config import cfg`` work directly.
"""

from __future__ import annotations

import sys as _sys

# Eagerly import each py_modules submodule and alias it at the
# sickling.protrusion_detection.X level. Dependency order doesn't
# strictly matter because the submodules use relative imports
# (`from .X import Y`) so Python's import machinery resolves siblings
# within py_modules.
_SUBMODULES = (
    "config",
    "masks",
    "paths",
    "model",
    "losses",
    "sampler",
    "datasets",
    "metrics",
    "inference",
    "mining",
    "splits",
    "minicrops",
    "densify",
    "viz",
    "train",
    "bootstrap",
    "correction",
)
for _name in _SUBMODULES:
    __import__(f"sickling.protrusion_detection.py_modules.{_name}")
    _sys.modules[f"sickling.protrusion_detection.{_name}"] = (
        _sys.modules[f"sickling.protrusion_detection.py_modules.{_name}"]
    )

# Convenience top-level re-export.
from .py_modules.config import Config, cfg

__all__ = ("Config", "cfg", "py_modules")
