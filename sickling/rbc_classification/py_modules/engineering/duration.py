"""Per-run wall-clock + throughput accounting for Lightning training jobs.

The :class:`DurationCallback` is installed by default via
:func:`sickling.engineering.lightning_utils.build_trainer`. After every fit
it writes ``duration.json`` next to the checkpoint directory so that:

* the ablation runner's per-cell ``duration_seconds`` is corroborated by an
  independent timer that survives Python exceptions in the runner,
* multi-run DDP comparisons (e.g. ``make pretrain DEVICES=1`` vs
  ``--devices 4 --strategy ddp``) can be done after-the-fact without any
  rerun, by diffing the two ``duration.json`` files.

JSON schema (stable, additive only)::

    {
        "run_name": str,
        "device_name": str,           # e.g. "NVIDIA RTX A4000" / "cpu"
        "n_devices": int,
        "strategy": str,              # "auto" | "ddp" | ...
        "precision": str,             # "bf16-mixed" | "32-true" | ...
        "max_epochs": int,
        "completed_epochs": int,
        "fit_seconds": float,
        "epoch_seconds": [float, ...],
        "imgs_per_second_mean": float,
        "samples_seen_total": int,
    }
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import pytorch_lightning as pl

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


def _device_name() -> str:
    if torch is not None and torch.cuda.is_available():
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            return "cuda"
    return f"cpu ({platform.processor() or platform.machine()})"


class DurationCallback(pl.Callback):
    """Record per-epoch and total wall-clock + a rough images/sec for a fit."""

    def __init__(self, output_dir: str | Path, run_name: str) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.run_name = run_name
        self._fit_start: float = 0.0
        self._epoch_start: float = 0.0
        self._epoch_times: list[float] = []
        self._samples_seen: int = 0
        # Fields filled from the trainer at on_fit_start.
        self._strategy_name: str = "unknown"
        self._n_devices: int = 1
        self._precision: str = "unknown"
        self._max_epochs: int = 0

    # ------------------------------------------------------------------ hooks
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._fit_start = time.perf_counter()
        self._epoch_times.clear()
        self._samples_seen = 0
        # Best-effort capture of trainer config; never block fit on it.
        try:
            self._strategy_name = type(trainer.strategy).__name__
        except Exception:
            self._strategy_name = "unknown"
        try:
            self._n_devices = int(getattr(trainer, "num_devices", 1))
        except Exception:
            self._n_devices = 1
        self._precision = str(getattr(trainer, "precision", "unknown"))
        self._max_epochs = int(getattr(trainer, "max_epochs", 0) or 0)

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._epoch_start = time.perf_counter()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._epoch_times.append(time.perf_counter() - self._epoch_start)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        # Track samples seen for an approximate throughput. ``batch`` can be a
        # dict (multimodal), a tuple (image, label), or a bare tensor (SSL).
        n = _batch_size_of(batch)
        if n > 0:
            self._samples_seen += n

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        total = time.perf_counter() - self._fit_start
        n_epochs = len(self._epoch_times)
        imgs_per_sec = (self._samples_seen / total) if total > 0 else 0.0
        payload = {
            "run_name": self.run_name,
            "device_name": _device_name(),
            "n_devices": self._n_devices,
            "strategy": self._strategy_name,
            "precision": self._precision,
            "max_epochs": self._max_epochs,
            "completed_epochs": n_epochs,
            "fit_seconds": float(total),
            "epoch_seconds": [float(t) for t in self._epoch_times],
            "imgs_per_second_mean": float(imgs_per_sec),
            "samples_seen_total": int(self._samples_seen),
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "duration.json", "w") as f:
            json.dump(payload, f, indent=2)


def _batch_size_of(batch) -> int:
    """Best-effort batch-size extraction across the three batch shapes we use.

    Returns 0 if the batch can't be measured (the timer is informational, so
    we never want it to crash training).
    """
    try:
        if torch is not None and torch.is_tensor(batch):
            return int(batch.shape[0])
        if isinstance(batch, (tuple, list)) and batch:
            head = batch[0]
            if isinstance(head, dict) and head:
                first = next(iter(head.values()))
                if torch is not None and torch.is_tensor(first):
                    return int(first.shape[0])
            elif torch is not None and torch.is_tensor(head):
                return int(head.shape[0])
        if isinstance(batch, dict) and batch:
            first = next(iter(batch.values()))
            if torch is not None and torch.is_tensor(first):
                return int(first.shape[0])
    except Exception:
        return 0
    return 0
