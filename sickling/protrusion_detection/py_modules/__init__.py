"""HITL polymer-detection segmentation pipeline.

Package layout:
    config        Single Config dataclass; pruned of legacy keys.
    paths         Folder discovery, filename builders, provenance parsing.
    masks         h5 mask I/O + 1-based/255-ignore conversion.
    datasets      MicroscopyDataset (whole-image) + TileDataset (sparse tiles).
    sampler       Target-class-aware crop sampler (inverse-frequency).
    model         UNet.
    losses        WeightedDice, Tversky, directed-confusion / FN penalties.
    inference     Sliding-window probability map, TTA, ensemble, predict_mask.
    mining        FN-aware tile candidate scoring + dup guard.
    metrics       Per-class dice, polymer monitor, confusion, trajectory log.
    train         kfold / single training loop, from-scratch every run.
    bootstrap     Throwaway generator + final bootstrap-mode training.
    correction    Correction round orchestrator (predict -> mine -> stage).

Public entry points live on `sickling.protrusion_detection.api`.
"""

from .config import Config, cfg

__all__ = ["Config", "cfg"]
