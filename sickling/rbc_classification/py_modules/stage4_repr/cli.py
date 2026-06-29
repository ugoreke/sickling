"""Stage 4 CLI drivers — fine-tune (any of A/B/C) and MAE continuation."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from sickling.rbc_classification.py_modules.config import Config
from sickling.rbc_classification.py_modules.data import (
    LABEL_TO_INT,
    CropDataset,
    eval_transform,
    labeled_subset,
    make_weighted_sampler,
    ssl_transform,
    train_transform,
)
from sickling.rbc_classification.py_modules.engineering.lightning_utils import build_trainer
from sickling.rbc_classification.py_modules.eval.splits import make_kfold_splits
from sickling.rbc_classification.py_modules.io.labels import gate_labels_to_prevalence
from sickling.rbc_classification.py_modules.io.parquet import read_cells
from sickling.rbc_classification.py_modules.stage4_repr import build_encoder
from sickling.rbc_classification.py_modules.stage4_repr.finetune import FinetuneModule
from sickling.rbc_classification.py_modules.stage4_repr.linear_probe import LinearProbeModule
from sickling.rbc_classification.py_modules.stage4_repr.mae_encoder import MAEReconstructor
from sickling.rbc_classification.py_modules.stage4_repr.ssl_pretrain import MAEPretrainModule

VARIANTS = ("dinov2_frozen", "timm_vit", "mae")


def _add_synthetic_labels(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Mark every unlabeled row with a deterministic synthetic 50/50 label.

    Used only by ``--synth-labels`` for smoke-testing fine-tune modules before
    real labels exist. NEVER call this in production runs.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    n = len(df)
    synth = rng.integers(0, 2, size=n)
    df["label"] = np.where(synth == 1, "sickle", "non_sickle")
    df["has_label"] = True
    return df


def _build_dataloaders(
    cfg: Config,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cells_df: pd.DataFrame,
    crops_dir: Path,
    *,
    return_label: bool,
    zero_mask_channels: bool = False,
) -> tuple[DataLoader, DataLoader]:
    train_tf = train_transform(cfg.augment) if return_label else ssl_transform(cfg.augment)
    val_tf = eval_transform(cfg.augment)

    base_train = CropDataset(
        cells_df=cells_df,
        crops_dir=crops_dir,
        target_size=cfg.crop.resize_to_vit,
        return_label=return_label,
        transform=train_tf,
        zero_mask_channels=zero_mask_channels,
    )
    base_val = CropDataset(
        cells_df=cells_df,
        crops_dir=crops_dir,
        target_size=cfg.crop.resize_to_vit,
        return_label=return_label,
        transform=val_tf,
        zero_mask_channels=zero_mask_channels,
    )
    train_ds = Subset(base_train, train_idx.tolist())
    val_ds = Subset(base_val, val_idx.tolist())

    sampler = None
    if return_label and cfg.finetune.minority_frac > 0:
        train_labels = np.array([LABEL_TO_INT[cells_df.iloc[i]["label"]] for i in train_idx])
        if np.unique(train_labels).size == 2:
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
    return train_loader, val_loader


def run_finetune(
    cfg: Config,
    variant: Literal["dinov2_frozen", "timm_vit", "mae"],
    *,
    fold: int | None = None,
    ckpt_path: Path | None = None,
    mae_init_ckpt: Path | None = None,
    synth_labels: bool = False,
    devices: int | str = "auto",
    zero_mask_channels: bool = False,
) -> dict:
    """Fine-tune (or linear-probe) one of A/B/C on a single fold.

    Returns a dict with the best checkpoint path and val_pr_auc / val_mcc.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")

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
    n_fovs = labeled_df["source_image"].nunique()
    if n_fovs < cfg.validation.cv_folds:
        # Too few FOVs for proper k-fold (e.g. smoke runs with synth labels on
        # one FOV). Fall back to a deterministic row-level 80/20 split.
        rng = np.random.default_rng(cfg.project.seed)
        all_idx = np.arange(len(labeled_df))
        rng.shuffle(all_idx)
        cut = max(int(len(all_idx) * 0.8), 1)
        train_idx, val_idx = all_idx[:cut], all_idx[cut:]
    else:
        splits = make_kfold_splits(
            labeled_df,
            n_splits=cfg.validation.cv_folds,
            seed=cfg.project.seed,
            strategy=cfg.validation.fold_strategy,
        )
        train_idx, val_idx = splits[fold]

    train_loader, val_loader = _build_dataloaders(
        cfg, train_idx, val_idx, labeled_df, paths.crops,
        return_label=True, zero_mask_channels=zero_mask_channels,
    )

    encoder = build_encoder(variant)
    if variant == "mae" and mae_init_ckpt is not None:
        from sickling.rbc_classification.py_modules.stage4_repr.timm_vit_encoder import MAEViTEncoder
        assert isinstance(encoder, MAEViTEncoder)
        encoder.load_mae_checkpoint(mae_init_ckpt)

    if variant == "dinov2_frozen":
        module = LinearProbeModule(
            encoder=encoder,
            head_lr=cfg.finetune.head_lr,
            weight_decay=cfg.training.weight_decay,
            warmup_epochs=cfg.training.warmup_epochs,
            max_epochs=cfg.training.max_epochs,
            sickle_class_index=cfg.finetune.sickle_class_index,
            label_smoothing=cfg.finetune.label_smoothing,
        )
    else:
        module = FinetuneModule(
            encoder=encoder,
            head_lr=cfg.finetune.head_lr,
            backbone_lr=cfg.finetune.backbone_lr,
            llrd=cfg.finetune.llrd,
            weight_decay=cfg.training.weight_decay,
            warmup_epochs=cfg.training.warmup_epochs,
            max_epochs=cfg.training.max_epochs,
            sickle_class_index=cfg.finetune.sickle_class_index,
            label_smoothing=cfg.finetune.label_smoothing,
        )

    run_name = f"finetune_{variant}_fold{fold}"
    trainer = build_trainer(
        cfg,
        run_name=run_name,
        devices=devices,
        offline_wandb=True,
        tags=["stage4", variant, f"fold{fold}"],
    )
    trainer.fit(module, train_loader, val_loader, ckpt_path=ckpt_path)

    metrics = {k: float(v) for k, v in trainer.callback_metrics.items() if torch.is_tensor(v)}
    best_ckpt = trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else ""
    return {"best_checkpoint": best_ckpt, **metrics}


def run_pretrain_mae(
    cfg: Config,
    *,
    ckpt_path: Path | None = None,
    devices: int | str = "auto",
    strategy: str = "auto",
) -> dict:
    """Run MAE continuation pretraining over the entire cell-crop corpus.

    Splits 90/10 train/val by ``source_image`` for a held-out reconstruction
    loss curve. No labels needed.
    """
    paths = cfg.paths.resolved()
    cells_df = read_cells(paths.root / cfg.paths.cells_parquet)

    fovs = cells_df["source_image"].unique().tolist()
    rng = np.random.default_rng(cfg.project.seed)
    rng.shuffle(fovs)
    cut = max(int(len(fovs) * 0.9), 1)
    train_fovs = set(fovs[:cut])
    val_fovs = set(fovs[cut:]) if len(fovs) > 1 else set(fovs)

    train_idx = np.where(cells_df["source_image"].isin(train_fovs).to_numpy())[0]
    val_idx = np.where(cells_df["source_image"].isin(val_fovs).to_numpy())[0]
    if val_idx.size == 0:
        # Single FOV — overlap train/val for smoke runs.
        val_idx = train_idx[: max(len(train_idx) // 10, 1)]

    train_loader, val_loader = _build_dataloaders(
        cfg, train_idx, val_idx, cells_df, paths.crops, return_label=False
    )

    encoder = build_encoder("mae")
    reconstructor = MAEReconstructor(
        encoder=encoder,
        decoder_embed_dim=cfg.mae_pretrain.decoder_embed_dim,
        decoder_depth=cfg.mae_pretrain.decoder_depth,
        decoder_num_heads=cfg.mae_pretrain.decoder_num_heads,
        norm_pix_loss=cfg.mae_pretrain.norm_pix_loss,
    )
    module = MAEPretrainModule(
        reconstructor=reconstructor,
        mask_ratio=cfg.mae_pretrain.mask_ratio,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        warmup_epochs=cfg.training.warmup_epochs,
        max_epochs=cfg.training.max_epochs,
    )

    run_name = "pretrain_mae"
    trainer = build_trainer(
        cfg,
        run_name=run_name,
        devices=devices,
        strategy=strategy,
        offline_wandb=True,
        tags=["stage4", "mae", "pretrain"],
    )
    trainer.fit(module, train_loader, val_loader, ckpt_path=ckpt_path)

    metrics = {k: float(v) for k, v in trainer.callback_metrics.items() if torch.is_tensor(v)}
    best_ckpt = trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else ""
    return {"best_checkpoint": best_ckpt, **metrics}
