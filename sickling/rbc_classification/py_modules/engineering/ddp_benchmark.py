"""DDP throughput benchmark.

Trains the MAE pretraining module for a fixed number of steps at 1, 2, 4 GPUs
(configurable) and records images/sec + scaling efficiency. Designed for the
rented multi-GPU box per PIPELINE_PLAN §5.

Output: CSV + matplotlib bar chart in ``figures/ddp_benchmark/``.

Usage::

    # Single-node, run the benchmark with whatever GPU counts you have available:
    python -m sickling.engineering.ddp_benchmark --devices 1 --devices 2 --devices 4

    # On a single A4000 box (sanity-check the harness):
    python -m sickling.engineering.ddp_benchmark --devices 1
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch

from sickling.rbc_classification.py_modules.config import load_config
from sickling.rbc_classification.py_modules.data import CropDataset, ssl_transform
from sickling.rbc_classification.py_modules.engineering.lightning_utils import build_trainer
from sickling.rbc_classification.py_modules.io.parquet import read_cells
from sickling.rbc_classification.py_modules.stage4_repr import build_encoder
from sickling.rbc_classification.py_modules.stage4_repr.mae_encoder import MAEReconstructor
from sickling.rbc_classification.py_modules.stage4_repr.ssl_pretrain import MAEPretrainModule


class _ThroughputTimer(pl.Callback):
    """Measure images/sec across ``warmup_steps`` then ``measure_steps``."""

    def __init__(self, warmup_steps: int, measure_steps: int) -> None:
        self.warmup_steps = warmup_steps
        self.measure_steps = measure_steps
        self.steps_seen = 0
        self._start_time: float | None = None
        self._batch_sizes: list[int] = []
        self.images_per_sec: float | None = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        bsz = batch.shape[0] if torch.is_tensor(batch) else batch[0].shape[0]
        self.steps_seen += 1
        if self.steps_seen == self.warmup_steps:
            self._start_time = time.time()
        elif self._start_time is not None:
            self._batch_sizes.append(bsz)
            if self.steps_seen >= self.warmup_steps + self.measure_steps:
                elapsed = time.time() - self._start_time
                total_images = sum(self._batch_sizes) * trainer.world_size
                self.images_per_sec = total_images / max(elapsed, 1e-6)
                trainer.should_stop = True


def _single_run(devices: int, batch_size: int, steps: int, warmup: int) -> dict:
    """Run inside the current process with the given device count."""
    cfg = load_config()
    paths = cfg.paths.resolved()
    cells_df = read_cells(paths.root / cfg.paths.cells_parquet)

    ds = CropDataset(
        cells_df=cells_df,
        crops_dir=paths.crops,
        target_size=cfg.crop.resize_to_vit,
        return_label=False,
        transform=ssl_transform(cfg.augment),
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=cfg.training.num_workers,
        persistent_workers=cfg.training.num_workers > 0,
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
        max_epochs=1,
        warmup_epochs=0,
    )

    timer = _ThroughputTimer(warmup_steps=warmup, measure_steps=steps)
    trainer = build_trainer(
        cfg,
        run_name=f"ddp_bench_d{devices}",
        max_epochs=1,
        devices=devices,
        strategy="ddp" if devices > 1 else "auto",
        offline_wandb=True,
        tags=["ddp_benchmark"],
        extra_callbacks=[timer],
    )
    # Set a hard step cap as a fallback in case the timer doesn't fire.
    trainer.limit_train_batches = warmup + steps + 5
    trainer.fit(module, train_dataloaders=loader)
    return {
        "devices": devices,
        "batch_size_per_gpu": batch_size,
        "global_batch_size": batch_size * devices,
        "images_per_sec": float(timer.images_per_sec or 0.0),
    }


def run_benchmark(
    devices_list: list[int],
    batch_size: int = 64,
    steps: int = 30,
    warmup: int = 5,
    output_dir: Path | None = None,
) -> list[dict]:
    """Run the benchmark across ``devices_list`` device counts in-process."""
    output_dir = Path(output_dir) if output_dir else (
        Path("figures") / "ddp_benchmark"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for d in devices_list:
        print(f"\n=== Benchmarking devices={d} ===")
        res = _single_run(d, batch_size=batch_size, steps=steps, warmup=warmup)
        results.append(res)
        print(f"  images/sec = {res['images_per_sec']:.1f}")

    csv_path = output_dir / "throughput.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()) + ["scaling_efficiency"])
        writer.writeheader()
        base_thru = results[0]["images_per_sec"]
        for r in results:
            ideal = base_thru * (r["devices"] / results[0]["devices"])
            r["scaling_efficiency"] = float(r["images_per_sec"] / ideal) if ideal > 0 else 0.0
            writer.writerow(r)
    print(f"throughput.csv -> {csv_path}")

    _render_bar_chart(results, output_dir / "throughput.svg")
    return results


def _render_bar_chart(results: list[dict], path: Path) -> None:
    devices = [r["devices"] for r in results]
    thru = [r["images_per_sec"] for r in results]
    base = thru[0]
    ideal = [base * (d / devices[0]) for d in devices]

    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    x = np.arange(len(devices))
    width = 0.4
    ax.bar(x - width / 2, ideal, width, color="#cccccc", label="Ideal (linear)")
    ax.bar(x + width / 2, thru, width, color="#3a7ca5", label="Measured")
    for i, (t, ideal_v) in enumerate(zip(thru, ideal, strict=True)):
        eff = (t / ideal_v) if ideal_v > 0 else 0.0
        ax.text(i + width / 2, t + max(thru) * 0.01, f"{eff * 100:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d} GPU{'s' if d > 1 else ''}" for d in devices])
    ax.set_ylabel("Images / sec")
    ax.set_title("DDP scaling — MAE pretraining throughput")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    fig.savefig(path, format="svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="DDP throughput benchmark")
    parser.add_argument("--devices", action="append", type=int, default=None,
                        help="GPU count; repeat for each config (e.g. --devices 1 --devices 2).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    devices_list = args.devices or [1]
    run_benchmark(
        devices_list=devices_list,
        batch_size=args.batch_size,
        steps=args.steps,
        warmup=args.warmup,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
