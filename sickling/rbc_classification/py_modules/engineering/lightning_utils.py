"""Factories for ``WandbLogger`` and ``ModelCheckpoint`` configured against
project-wide defaults so individual training scripts stay terse."""
from __future__ import annotations

import warnings
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from sickling.rbc_classification.py_modules.config import Config
from sickling.rbc_classification.py_modules.engineering.duration import DurationCallback

# Silence two harmless-by-design Lightning warnings that fire on every run:
#   1. "Found N module(s) in eval mode at the start of training" — DINOv2 is
#      intentionally frozen and `train()` is overridden to keep it in eval mode.
#   2. "The number of training batches (X) is smaller than the logging interval"
#      — fires for tiny fast-mode / smoke runs; not actionable.
warnings.filterwarnings(
    "ignore",
    message=r"Found \d+ module\(s\) in eval mode at the start of training.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"The number of training batches \(\d+\) is smaller than the logging interval.*",
    category=UserWarning,
)


def build_wandb_logger(
    cfg: Config,
    run_name: str,
    tags: list[str] | None = None,
    offline: bool = False,
) -> WandbLogger:
    """Construct a ``WandbLogger`` against the project-wide entity / project."""
    paths = cfg.paths.resolved()
    paths.wandb_dir.mkdir(parents=True, exist_ok=True)
    return WandbLogger(
        project=cfg.project.name,
        entity=cfg.project.wandb_entity,
        name=run_name,
        tags=tags or [],
        save_dir=str(paths.wandb_dir),
        offline=offline,
    )


def build_checkpoint_callback(
    cfg: Config,
    run_name: str,
    monitor: str | None = None,
    mode: str | None = None,
) -> ModelCheckpoint:
    """Save best-by-monitor and last; resumes are exact (incl. RNG state)."""
    paths = cfg.paths.resolved()
    ckpt_dir: Path = paths.checkpoints / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="{epoch:03d}-{step}",
        monitor=monitor or cfg.training.monitor_metric,
        mode=mode or cfg.training.monitor_mode,
        save_top_k=cfg.training.save_top_k,
        save_last=True,
        auto_insert_metric_name=False,
    )


def build_trainer(
    cfg: Config,
    run_name: str,
    *,
    max_epochs: int | None = None,
    devices: int | str = "auto",
    strategy: str = "auto",
    precision: str | None = None,
    extra_callbacks: list[pl.Callback] | None = None,
    offline_wandb: bool = False,
    tags: list[str] | None = None,
) -> pl.Trainer:
    """Default Lightning ``Trainer`` with W&B logger, checkpointing, LR monitor.

    Stage-specific scripts override only what differs (e.g. ``strategy="ddp"``
    for the MAE pretraining run).
    """
    paths = cfg.paths.resolved()
    ckpt_dir = paths.checkpoints / run_name
    callbacks: list[pl.Callback] = [
        build_checkpoint_callback(cfg, run_name),
        LearningRateMonitor(logging_interval="epoch"),
        DurationCallback(output_dir=ckpt_dir, run_name=run_name),
    ]
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    return pl.Trainer(
        max_epochs=max_epochs or cfg.training.max_epochs,
        precision=precision or cfg.training.precision,
        accumulate_grad_batches=cfg.training.grad_accum,
        devices=devices,
        strategy=strategy,
        logger=build_wandb_logger(cfg, run_name, tags=tags, offline=offline_wandb),
        callbacks=callbacks,
        log_every_n_steps=10,
        deterministic=False,
    )
