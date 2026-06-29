"""Single source of truth for project configuration.

Edit the `Config` class to switch modes, point at folders, or tune training.
All other modules read from `cfg` (the module-level instance) and never
hard-code paths or hyperparameters.

See ARCHITECTURE.md §11 for the field list and rationale; §12 for the design
decisions baked into the defaults.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch


def _default_base_dir() -> str:
    """Arm folder root — contains the data folders (``InitialLabels``,
    ``BootstrappedLabels``, …) and the ``py_modules`` Python sub-package.

    Derived from ``__file__`` so renaming the outer repo folder (or
    moving the project between machines) doesn't need a code edit.
    Layout: this file is ``sickling/protrusion_detection/py_modules/config.py``, so
    the arm folder is two ``parent`` hops up.
    """
    from pathlib import Path
    return str(Path(__file__).resolve().parent.parent)


@dataclass
class Config:
    # ---- Mode switches -----------------------------------------------------
    RUN_MODE: str = "correction"   # "bootstrap" | "correction"
    FOLD_MODE: str = "single"      # "kfold" | "single"
    BEST_FOLD: int = 2             # 1-based fold index used in single mode

    # ---- Model backbone ----------------------------------------------------
    # Default is the hand-rolled UNet (1-channel in, N_CLASSES out). Swap to
    # an SMP backbone to test whether model size is the val ceiling — e.g.
    # ``"smp_unet_efficientnet-b0"`` or ``"smp_unet_efficientnet-b7"``.
    # Checkpoints are tagged by backbone in the filename so backbones can
    # coexist across loops (``<backbone>_fold_<f>_best_loop_<N>.pth``).
    # Heavy encoders may need a smaller BATCH_SIZE to fit on your GPU.
    MODEL_BACKBONE: str = "unet"

    # ---- Densify fill model (decoupled from training) ---------------------
    # The clean model used to fill non-target pixels when densifying
    # positive-only labels. Decoupled from MODEL_BACKBONE so you can run the
    # SMP experiment with the original clean UNet for fills (the user's
    # backup loop_0 model).
    DENSIFY_BACKBONE: str = "unet"
    DENSIFY_FOLD: int = 2
    DENSIFY_LOOP: int = 0

    # ---- Folder roots ------------------------------------------------------
    BASE_DIR: str = field(default_factory=_default_base_dir)

    # Filled in __post_init__ once BASE_DIR is known.
    INITIAL_LABELS_DIR: str = ""
    BOOTSTRAP_LABELS_DIR: str = ""
    CORRECTION_POOL_DIR: str = ""
    TILES_TODO_DIR: str = ""
    CORRECTED_TILES_DIR: str = ""
    BOOTSTRAP_RAW_DIR: str = ""    # defaults to CORRECTION_POOL_DIR (canonical raw store)
    MODELS_DIR: str = ""           # checkpoints
    METRICS_DIR: str = ""          # iteration_log.csv + trajectory.png
    VIZ_DIR: str = ""              # colored-jpg renders
    EVAL_DIR: str = ""             # Panel B/C figures

    # ---- Data conventions --------------------------------------------------
    RAW_EXTS: Tuple[str, ...] = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
    IGNORE_INDEX: int = 255
    N_CLASSES: int = 4

    # ---- Training hyperparameters -----------------------------------------
    TILE_SIZE: int = 256
    BATCH_SIZE: int = 16
    EPOCHS: int = 50
    STEPS_PER_EPOCH: int = 100
    LR: float = 1e-4
    N_FOLDS: int = 5
    NORM_PERCENTILE: float = 99.0

    # ---- Target-class sampling (generalized from CLASS0_*) -----------------
    TARGET_CLASSES: List[int] = field(default_factory=lambda: [0])
    TARGET_CROP_PROB: float = 0.5

    # ---- Loss configuration (preserved from training_2.ipynb) --------------
    BOOSTED_CLASSES: dict = field(default_factory=lambda: {0: 5.0, 3: 3.0})
    DIRECTED_CONFUSION_PENALTY: List[Tuple[int, int, float]] = field(
        default_factory=lambda: [(0, 1, 2.0), (0, 2, 1.0), (0, 3, 1.0)]
    )
    DIRECTED_FN_PENALTY: List[Tuple[int, int]] = field(
        default_factory=lambda: [(1, 0), (2, 0), (3, 0)]
    )
    CONFUSION_WEIGHT: float = 0.3
    FN_WEIGHT: float = 0.1
    TVERSKY_CLASSES: List[int] = field(default_factory=lambda: [0])
    TVERSKY_WEIGHT: float = 0.3
    TVERSKY_ALPHA: float = 0.4
    TVERSKY_BETA: float = 0.6

    # ---- Val/test split inside InitialLabels -------------------------------
    # When VAL_STEMS is empty (default), the first TRUTH_VAL_COUNT (sorted)
    # files are val and the remainder is held-out test for Panel B/C.
    # When VAL_STEMS is set, those exact stems form val (everything else in
    # InitialLabels = test) — useful for balancing polymer density across
    # the val set, since sorted-first can land you on the polymer-light end.
    TRUTH_VAL_COUNT: int = 2
    VAL_STEMS: List[str] = field(default_factory=list)

    # ---- Mining (correction mode) -----------------------------------------
    CORRECTION_TILE_SIZE: int = 512
    MINING_BATCH_SIZE: int = 30
    MINING_SCORE: str = "softprob+disagreement"  # auto-fallback to soft-prob in single mode
    MINING_LAMBDA: float = 1.0                   # disagreement weight
    MINING_STRIDE: int = 256                     # 50% overlap on 512-px windows
    MINING_DUP_IOU: float = 0.25                 # skip candidates overlapping existing tiles
    MINING_EXCLUDE_LABELED_STEMS: bool = True    # never mine from InitialLabels/BootstrappedLabels stems

    # ---- PRED rotation -----------------------------------------------------
    PRED_BATCH_SIZE: int = 150    # pool images refreshed per correction round

    # ---- Mini-crops (single-class, fast-label workflow) -------------------
    # When dense 512-px tiles plateau, switch to mining many small adaptive
    # crops centered on predicted target-class blobs. You paint *only* the
    # target class in each crop; the densify-on-retrain hook fills in
    # non-target pixels from the clean (loop_0) model so we never reintroduce
    # the positive-only-tile bug. Mini-crops live in their own folders so
    # they don't tangle with the 512-px parsing.
    MINI_TILES_TODO_DIR: str = ""        # default: <BASE_DIR>/MiniTilesToBeCorrected
    MINI_TILES_CORRECTED_DIR: str = ""   # default: <BASE_DIR>/MiniTilesCorrected
    MINI_CROP_TARGET_CLASS: int = 0      # single class to focus on (polymer)
    MINI_CROP_PADDING: int = 32          # context margin around the predicted blob
    MINI_CROP_MIN: int = 64              # min crop side
    MINI_CROP_MAX: int = 192             # max crop side
    MINI_CROP_BATCH_SIZE: int = 200      # how many to stage per mining call
    MINI_CROP_MIN_CC_AREA: int = 3       # discard predicted blobs smaller than this (noise)
    MINI_CROP_DUP_IOU: float = 0.25      # skip a crop overlapping an already-staged one
    # (Mini-crop fill model is configured via DENSIFY_{BACKBONE,FOLD,LOOP}.)

    # ---- Eval set (mini-crop held-out test) -------------------------------
    # A separate folder for hand-labeled mini-crops used **only** as a
    # held-out test surface — never enters the training pool. Crops are
    # centered on predicted polymer connected components (same convention
    # as the §3.1 training mini-crops) so every painted crop tests the
    # actual question — "is the model right about this polymer?" — instead
    # of being burnt on background pixels.
    MINI_TILES_FOR_EVAL_DIR: str = ""    # default: <BASE_DIR>/MiniTilesForEval
    MINI_EVAL_N_CROPS: int = 100         # default n_crops for stage_eval_minicrops
    # Adaptive sizing (bbox + padding, clamped). Decoupled from
    # MINI_CROP_* so the eval crops can be smaller / faster to paint than
    # the training crops. Defaults are deliberately smaller than the §3.1
    # training defaults: typical eval crop ~80–128 px, paint in 30–60 s.
    MINI_EVAL_CROP_MIN: int = 64
    MINI_EVAL_CROP_MAX: int = 128
    MINI_EVAL_CROP_PADDING: int = 24

    # ---- Tile promotion to val/test ---------------------------------------
    # If True, the strict whole-image leakage barrier fires (ARCHITECTURE.md §8).
    PROMOTE_TILES_TO_VAL: bool = False

    # ---- Device ------------------------------------------------------------
    DEVICE: torch.device = field(default_factory=lambda: (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    ))

    # ------------------------------------------------------------------------
    def __post_init__(self) -> None:
        b = self.BASE_DIR
        if not self.INITIAL_LABELS_DIR:    self.INITIAL_LABELS_DIR = os.path.join(b, "InitialLabels")
        if not self.BOOTSTRAP_LABELS_DIR:  self.BOOTSTRAP_LABELS_DIR = os.path.join(b, "BootstrappedLabels")
        if not self.CORRECTION_POOL_DIR:   self.CORRECTION_POOL_DIR = os.path.join(b, "CorrectionPool")
        if not self.TILES_TODO_DIR:        self.TILES_TODO_DIR = os.path.join(b, "TilesToBeCorrected")
        if not self.CORRECTED_TILES_DIR:   self.CORRECTED_TILES_DIR = os.path.join(b, "CorrectedTiles")
        if not self.MINI_TILES_TODO_DIR:      self.MINI_TILES_TODO_DIR = os.path.join(b, "MiniTilesToBeCorrected")
        if not self.MINI_TILES_CORRECTED_DIR: self.MINI_TILES_CORRECTED_DIR = os.path.join(b, "MiniTilesCorrected")
        if not self.MINI_TILES_FOR_EVAL_DIR:  self.MINI_TILES_FOR_EVAL_DIR = os.path.join(b, "MiniTilesForEval")
        if not self.BOOTSTRAP_RAW_DIR:     self.BOOTSTRAP_RAW_DIR = self.CORRECTION_POOL_DIR
        if not self.MODELS_DIR:            self.MODELS_DIR = os.path.join(b, "models")
        if not self.METRICS_DIR:           self.METRICS_DIR = os.path.join(b, "metrics")
        if not self.VIZ_DIR:               self.VIZ_DIR = os.path.join(b, "viz")
        if not self.EVAL_DIR:              self.EVAL_DIR = os.path.join(b, "evaluation_truth")

        if self.RUN_MODE not in {"bootstrap", "correction"}:
            raise ValueError(f"RUN_MODE must be 'bootstrap' or 'correction', got {self.RUN_MODE!r}")
        if self.FOLD_MODE not in {"kfold", "single"}:
            raise ValueError(f"FOLD_MODE must be 'kfold' or 'single', got {self.FOLD_MODE!r}")
        if not (1 <= self.BEST_FOLD <= self.N_FOLDS):
            raise ValueError(f"BEST_FOLD={self.BEST_FOLD} out of range 1..{self.N_FOLDS}")

    # ---- Checkpoint loop versioning ---------------------------------------
    # Every retrain writes a *new* loop (``..._best_loop_<N>.pth``) instead of
    # overwriting the previous model. Training writes ``next_loop_index()``
    # (latest + 1); inference reads ``latest_loop_index()`` (newest trained).
    # Pass an explicit ``loop`` to target a specific generation (e.g. loop 0
    # for the clean pre-correction model used to densify labels).

    def fold_ckpt_path(self, fold_1based: int, loop: int,
                       backbone: Optional[str] = None) -> str:
        """Checkpoint path for a (fold, loop) pair under the chosen backbone.

        Filenames embed the backbone tag (``<backbone>_fold_<f>_best_loop_<N>.pth``)
        so multiple backbones coexist across loops without colliding.
        Default backbone = ``self.MODEL_BACKBONE``.
        """
        bk = backbone if backbone is not None else self.MODEL_BACKBONE
        return os.path.join(self.MODELS_DIR, f"{bk}_fold_{fold_1based}_best_loop_{loop}.pth")

    def latest_loop_index(self, backbone: Optional[str] = None) -> int:
        """Highest loop index present under MODELS_DIR for ``backbone``, or -1."""
        import glob
        import re
        bk = backbone if backbone is not None else self.MODEL_BACKBONE
        rx = re.compile(rf"^{re.escape(bk)}_fold_\d+_best_loop_(\d+)\.pth$")
        loops = [
            int(m.group(1))
            for p in glob.glob(os.path.join(self.MODELS_DIR, f"{bk}_fold_*_best_loop_*.pth"))
            if (m := rx.search(os.path.basename(p)))
        ]
        return max(loops) if loops else -1

    def next_loop_index(self, backbone: Optional[str] = None) -> int:
        """Loop index a fresh retrain should write (never overwrites)."""
        return self.latest_loop_index(backbone=backbone) + 1

    def ensure_dirs(self) -> None:
        for d in (self.MODELS_DIR, self.METRICS_DIR, self.VIZ_DIR, self.EVAL_DIR,
                  self.TILES_TODO_DIR, self.CORRECTED_TILES_DIR,
                  self.MINI_TILES_TODO_DIR, self.MINI_TILES_CORRECTED_DIR,
                  self.MINI_TILES_FOR_EVAL_DIR):
            os.makedirs(d, exist_ok=True)


cfg = Config()
