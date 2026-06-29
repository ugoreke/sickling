"""Stage 5 driver — single-fold multimodal classifier training."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from sickling.rbc_classification.py_modules.config import Config
from sickling.rbc_classification.py_modules.data import (
    LABEL_TO_INT,
    eval_transform,
    labeled_subset,
    make_weighted_sampler,
    train_transform,
)
from sickling.rbc_classification.py_modules.engineering.lightning_utils import build_trainer
from sickling.rbc_classification.py_modules.eval.splits import make_kfold_splits
from sickling.rbc_classification.py_modules.io.labels import gate_labels_to_prevalence
from sickling.rbc_classification.py_modules.io.parquet import read_cells
from sickling.rbc_classification.py_modules.stage4_repr import build_encoder
from sickling.rbc_classification.py_modules.stage4_repr.cli import _add_synthetic_labels
from sickling.rbc_classification.py_modules.stage5_multimodal.classifier import MultimodalClassifier
from sickling.rbc_classification.py_modules.stage5_multimodal.dataset import MultimodalCropDataset
from sickling.rbc_classification.py_modules.stage5_multimodal.image_tower import ImageTower
from sickling.rbc_classification.py_modules.stage5_multimodal.lightning_module import MultimodalFinetuneModule
from sickling.rbc_classification.py_modules.stage5_multimodal.morphology_features import N_FEATURES
from sickling.rbc_classification.py_modules.stage5_multimodal.morphology_tower import MorphologyTower


def _resolve_split(cfg: Config, labeled_df: pd.DataFrame, fold: int) -> tuple[np.ndarray, np.ndarray]:
    n_fovs = labeled_df["source_image"].nunique()
    if n_fovs < cfg.validation.cv_folds:
        rng = np.random.default_rng(cfg.project.seed)
        all_idx = np.arange(len(labeled_df))
        rng.shuffle(all_idx)
        cut = max(int(len(all_idx) * 0.8), 1)
        return all_idx[:cut], all_idx[cut:]
    splits = make_kfold_splits(
        labeled_df,
        n_splits=cfg.validation.cv_folds,
        seed=cfg.project.seed,
        strategy=cfg.validation.fold_strategy,
    )
    return splits[fold]


def run_multimodal_finetune(
    cfg: Config,
    *,
    image_variant: str = "dinov2_frozen",
    fold: int | None = None,
    ckpt_path: Path | None = None,
    mae_init_ckpt: Path | None = None,
    synth_labels: bool = False,
    devices: int | str = "auto",
    use_morphology: bool = True,
    use_image: bool = True,
    zero_mask_channels: bool = False,
    zero_image_masks_only: bool = False,
    zero_cell_body_only: bool = False,
    zero_polymer_only: bool = False,
    dilate_cell_body_px: int = 0,
) -> dict:
    """Train ``MultimodalFinetuneModule`` on a single fold.

    Towers default to ``{image, morphology}`` per spec. Disable either with
    ``use_image=False`` / ``use_morphology=False`` for ablation rows.

    ``zero_mask_channels`` zeros ch1/ch2 globally (image tower input *and*
    the morphology cache). ``zero_image_masks_only`` zeros ch1/ch2 only on
    the image tower input — the morphology cache continues to read the
    original masks. The latter is the per-tower test from the
    ``ablation_20260516_003426`` discussion (Limitation 5).
    """
    if not (use_image or use_morphology):
        raise ValueError("Need at least one tower; got use_image=False, use_morphology=False.")

    paths = cfg.paths.resolved()
    cells_df = read_cells(paths.root / cfg.paths.cells_parquet)
    if synth_labels:
        cells_df = _add_synthetic_labels(cells_df, seed=cfg.project.seed)

    labeled_df = labeled_subset(cells_df)
    if labeled_df.empty:
        raise RuntimeError(
            "No labeled cells in cells.parquet. Fill in labels/labels.csv or use --synth-labels."
        )

    if cfg.validation.target_sickle_frac is not None:
        labeled_df, gate_stats = gate_labels_to_prevalence(
            labeled_df,
            target_sickle_frac=cfg.validation.target_sickle_frac,
            seed=cfg.validation.gate_seed,
        )
        print(
            f"[label gate] target={gate_stats['target_frac']:.3f} "
            f"achieved={gate_stats['achieved_frac']:.3f} "
            f"kept sickle={gate_stats['n_sickle_kept']}/{gate_stats['n_sickle_in']} "
            f"non_sickle={gate_stats['n_non_sickle_kept']}/{gate_stats['n_non_sickle_in']}"
        )

    fold = cfg.finetune.fold if fold is None else fold
    train_idx, val_idx = _resolve_split(cfg, labeled_df, fold)

    train_tf = train_transform(cfg.augment)
    val_tf = eval_transform(cfg.augment)

    base_train = MultimodalCropDataset(
        cells_df=labeled_df, crops_dir=paths.crops,
        target_size=cfg.crop.resize_to_vit,
        return_label=True, transform=train_tf,
        zero_mask_channels=zero_mask_channels,
        zero_image_masks_only=zero_image_masks_only,
        zero_cell_body_only=zero_cell_body_only,
        zero_polymer_only=zero_polymer_only,
        dilate_cell_body_px=dilate_cell_body_px,
    )
    base_val = MultimodalCropDataset(
        cells_df=labeled_df, crops_dir=paths.crops,
        target_size=cfg.crop.resize_to_vit,
        return_label=True, transform=val_tf,
        morphology_cache=base_train.morphology,  # share cache across train/val datasets
        zero_mask_channels=zero_mask_channels,
        zero_image_masks_only=zero_image_masks_only,
        zero_cell_body_only=zero_cell_body_only,
        zero_polymer_only=zero_polymer_only,
        dilate_cell_body_px=dilate_cell_body_px,
    )
    train_ds = Subset(base_train, train_idx.tolist())
    val_ds = Subset(base_val, val_idx.tolist())

    # Per-fold morphology standardization on the train subset only (no val leak).
    train_feats = base_train.morphology[train_idx]
    feat_mean = train_feats.mean(dim=0)
    feat_std = train_feats.std(dim=0)

    # Sampler — minority-balanced.
    train_labels = np.array([LABEL_TO_INT[labeled_df.iloc[i]["label"]] for i in train_idx])
    sampler = None
    if cfg.finetune.minority_frac > 0 and np.unique(train_labels).size == 2:
        sampler = make_weighted_sampler(train_labels, cfg.finetune.minority_frac)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=cfg.training.num_workers,
        persistent_workers=cfg.training.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        persistent_workers=cfg.training.num_workers > 0,
    )

    towers: dict = {}
    if use_image:
        encoder = build_encoder(image_variant)
        if image_variant == "mae" and mae_init_ckpt is not None:
            from sickling.rbc_classification.py_modules.stage4_repr.timm_vit_encoder import MAEViTEncoder
            assert isinstance(encoder, MAEViTEncoder)
            encoder.load_mae_checkpoint(mae_init_ckpt)
        towers["image"] = ImageTower(encoder)
    if use_morphology:
        morph_tower = MorphologyTower(in_features=N_FEATURES)
        morph_tower.set_feature_stats(feat_mean, feat_std)
        towers["morphology"] = morph_tower

    classifier = MultimodalClassifier(
        towers,
        num_classes=cfg.multimodal.num_classes,
        hidden=cfg.multimodal.fusion_hidden,
        dropout=cfg.multimodal.dropout,
    )
    module = MultimodalFinetuneModule(
        classifier=classifier,
        backbone_lr=cfg.finetune.backbone_lr,
        morphology_lr=cfg.finetune.head_lr,
        head_lr=cfg.finetune.head_lr,
        llrd=cfg.finetune.llrd,
        weight_decay=cfg.training.weight_decay,
        warmup_epochs=cfg.training.warmup_epochs,
        max_epochs=cfg.training.max_epochs,
        sickle_class_index=cfg.finetune.sickle_class_index,
        label_smoothing=cfg.finetune.label_smoothing,
    )

    tower_tag = "+".join(towers.keys())
    # Encode mask-zeroing + dilation variants in the run_name so checkpoints +
    # eval reports for different ablation rows don't collide on disk.
    if zero_mask_channels:
        mask_tag = "_zeromasks"
    elif zero_image_masks_only:
        mask_tag = "_zeroimgmasks"
    elif zero_cell_body_only:
        mask_tag = "_zerocellbody"
    elif zero_polymer_only:
        mask_tag = "_zeropolymer"
    else:
        mask_tag = ""
    dilate_tag = f"_dilcb{dilate_cell_body_px}" if dilate_cell_body_px > 0 else ""
    run_name = f"multimodal_{image_variant}_{tower_tag}{mask_tag}{dilate_tag}_fold{fold}"
    trainer = build_trainer(
        cfg, run_name=run_name, devices=devices, offline_wandb=True,
        tags=["stage5", "multimodal", tower_tag, image_variant, f"fold{fold}"],
    )
    trainer.fit(module, train_loader, val_loader, ckpt_path=ckpt_path)

    metrics = {k: float(v) for k, v in trainer.callback_metrics.items() if torch.is_tensor(v)}
    best_ckpt = trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else ""
    return {"best_checkpoint": best_ckpt, **metrics}
