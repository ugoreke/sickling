"""sickling — top-level Python package for the polished HbS-protrusion +
sickle-cell-classification pipeline accompanying Goreke et al., *Mol
Ther Adv* (in press).

The two scientific arms live as sub-packages:

- ``sickling.protrusion_detection`` — HITL-trained U-Net pixel
  segmentation arm. Produces the polymer / cell / background / cell-
  boundary mask used in ``Figure 2`` and the polymer-length test in
  ``Supplementary Figure X``.
- ``sickling.rbc_classification`` — DINOv2 + morphology multimodal
  classifier that turns the per-cell crops from the U-Net mask into
  sickle / non-sickle predictions.

Each sub-package's actual implementation modules live in a private
``py_modules`` folder; this package's ``__init__`` re-exports them at
the arm level so user-facing imports look like::

    from sickling.protrusion_detection.config import cfg
    from sickling.rbc_classification.eval.report import read_report

regardless of the underlying ``py_modules`` layout.
"""
from __future__ import annotations

import importlib as _importlib

# Eagerly import each arm. Their own __init__.py performs the
# ``py_modules.X`` → ``<arm>.X`` aliasing.
protrusion_detection = _importlib.import_module(
    "sickling.protrusion_detection"
)
rbc_classification = _importlib.import_module(
    "sickling.rbc_classification"
)

__all__ = ("protrusion_detection", "rbc_classification")
