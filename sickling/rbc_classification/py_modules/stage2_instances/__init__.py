"""Stage 2 — semantic 4-class label map -> integer instance label image."""
from sickling.rbc_classification.py_modules.stage2_instances.watershed import (
    DROP_EDGE,
    DROP_EMPTY,
    DROP_KEPT,
    DROP_MAX,
    DROP_MIN,
    InstanceStats,
    mask_to_instances,
    mask_to_instances_with_reasons,
)

__all__ = [
    "DROP_EDGE",
    "DROP_EMPTY",
    "DROP_KEPT",
    "DROP_MAX",
    "DROP_MIN",
    "InstanceStats",
    "mask_to_instances",
    "mask_to_instances_with_reasons",
]
