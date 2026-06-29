"""End-to-end evaluation driver.

Reloads a Lightning checkpoint produced by Stage 4 / Stage 5 training,
recomputes the validation-set scores, runs metrics + bootstrap, writes a
JSON report, and renders SVG figures.

Variant -> module reconstruction map (mirrors the finetune CLI):
    dinov2_frozen -> LinearProbeModule
    timm_vit      -> FinetuneModule
    mae           -> FinetuneModule (with optional --mae-init seed)
    multimodal    -> MultimodalFinetuneModule
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, matthews_corrcoef
from torch.utils.data import DataLoader, Subset

from sickling.rbc_classification.py_modules.config import Config
from sickling.rbc_classification.py_modules.data import (
    CropDataset,
    eval_transform,
    labeled_subset,
)
from sickling.rbc_classification.py_modules.eval.bootstrap import bootstrap_metric, bootstrap_pr_curve
from sickling.rbc_classification.py_modules.eval.figures import render_all_figures
from sickling.rbc_classification.py_modules.eval.metrics import compute_binary_metrics
from sickling.rbc_classification.py_modules.eval.report import EvaluationReport, read_report, write_report
from sickling.rbc_classification.py_modules.eval.splits import make_kfold_splits
from sickling.rbc_classification.py_modules.io.labels import gate_labels_to_prevalence
from sickling.rbc_classification.py_modules.io.parquet import read_cells
from sickling.rbc_classification.py_modules.stage4_repr import build_encoder
from sickling.rbc_classification.py_modules.stage4_repr.cli import _add_synthetic_labels
from sickling.rbc_classification.py_modules.stage4_repr.finetune import FinetuneModule
from sickling.rbc_classification.py_modules.stage4_repr.linear_probe import LinearProbeModule
from sickling.rbc_classification.py_modules.stage5_multimodal.classifier import MultimodalClassifier
from sickling.rbc_classification.py_modules.stage5_multimodal.dataset import MultimodalCropDataset
from sickling.rbc_classification.py_modules.stage5_multimodal.image_tower import ImageTower
from sickling.rbc_classification.py_modules.stage5_multimodal.lightning_module import MultimodalFinetuneModule
from sickling.rbc_classification.py_modules.stage5_multimodal.morphology_tower import MorphologyTower

VARIANTS = ("dinov2_frozen", "timm_vit", "mae", "multimodal")


def _resolve_split(cfg: Config, labeled_df: pd.DataFrame, fold: int) -> tuple[np.ndarray, np.ndarray]:
    n_fovs = labeled_df["source_image"].nunique()
    if n_fovs < cfg.validation.cv_folds:
        rng = np.random.default_rng(cfg.project.seed)
        all_idx = np.arange(len(labeled_df))
        rng.shuffle(all_idx)
        cut = max(int(len(all_idx) * 0.8), 1)
        return all_idx[:cut], all_idx[cut:]
    return make_kfold_splits(
        labeled_df,
        n_splits=cfg.validation.cv_folds,
        seed=cfg.project.seed,
        strategy=cfg.validation.fold_strategy,
    )[fold]


def _build_image_module(
    cfg: Config, variant: str, mae_init_ckpt: Path | None
) -> tuple[torch.nn.Module, str]:
    """Construct the (untrained) module skeleton for the given variant — to
    be filled in by ``load_from_checkpoint``."""
    encoder = build_encoder(variant)
    if variant == "mae" and mae_init_ckpt is not None:
        from sickling.rbc_classification.py_modules.stage4_repr.timm_vit_encoder import MAEViTEncoder
        assert isinstance(encoder, MAEViTEncoder)
        encoder.load_mae_checkpoint(mae_init_ckpt)
    if variant == "dinov2_frozen":
        return LinearProbeModule(encoder=encoder), "image"
    return FinetuneModule(encoder=encoder), "image"


def _build_multimodal_module(
    cfg: Config,
    image_variant: str,
    mae_init_ckpt: Path | None,
    n_morph_features: int,
    use_image: bool,
    use_morphology: bool,
) -> MultimodalFinetuneModule:
    towers: dict = {}
    if use_image:
        encoder = build_encoder(image_variant)
        if image_variant == "mae" and mae_init_ckpt is not None:
            from sickling.rbc_classification.py_modules.stage4_repr.timm_vit_encoder import MAEViTEncoder
            assert isinstance(encoder, MAEViTEncoder)
            encoder.load_mae_checkpoint(mae_init_ckpt)
        towers["image"] = ImageTower(encoder)
    if use_morphology:
        towers["morphology"] = MorphologyTower(in_features=n_morph_features)
    classifier = MultimodalClassifier(
        towers,
        num_classes=cfg.multimodal.num_classes,
        hidden=cfg.multimodal.fusion_hidden,
        dropout=cfg.multimodal.dropout,
    )
    return MultimodalFinetuneModule(classifier=classifier)


@torch.no_grad()
def _score_loader(module: torch.nn.Module, loader: DataLoader, sickle_idx: int) -> tuple[np.ndarray, np.ndarray]:
    device = next(module.parameters()).device
    module.eval()
    scores: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for batch in loader:
        if isinstance(batch[0], dict):
            inputs, y = batch
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            logits = module(inputs)
        else:
            x, y = batch
            x = x.to(device, non_blocking=True)
            logits = module(x)
        probs = F.softmax(logits, dim=-1)
        scores.append(probs[:, sickle_idx].detach().float().cpu().numpy())
        targets.append(y.numpy() if not torch.is_tensor(y) else y.cpu().numpy())
    return np.concatenate(scores), np.concatenate(targets)


def run_evaluate(
    cfg: Config,
    checkpoint: Path,
    variant: str,
    *,
    image_variant: str = "dinov2_frozen",
    fold: int | None = None,
    synth_labels: bool = False,
    output_dir: Path | None = None,
    bootstrap_resamples: int | None = None,
    mae_init_ckpt: Path | None = None,
    use_image: bool = True,
    use_morphology: bool = True,
    zero_mask_channels: bool = False,
    zero_image_masks_only: bool = False,
    zero_cell_body_only: bool = False,
    zero_polymer_only: bool = False,
    dilate_cell_body_px: int = 0,
) -> EvaluationReport:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")

    paths = cfg.paths.resolved()
    cells_df = read_cells(paths.root / cfg.paths.cells_parquet)
    if synth_labels:
        cells_df = _add_synthetic_labels(cells_df, seed=cfg.project.seed)
    labeled_df = labeled_subset(cells_df)
    if labeled_df.empty:
        raise RuntimeError("No labeled cells. Fill labels.csv or use --synth-labels.")

    if cfg.validation.target_sickle_frac is not None:
        labeled_df, _gate_stats = gate_labels_to_prevalence(
            labeled_df,
            target_sickle_frac=cfg.validation.target_sickle_frac,
            seed=cfg.validation.gate_seed,
        )

    fold = cfg.finetune.fold if fold is None else fold
    _train_idx, val_idx = _resolve_split(cfg, labeled_df, fold)

    val_tf = eval_transform(cfg.augment)
    if variant == "multimodal":
        base_val = MultimodalCropDataset(
            cells_df=labeled_df, crops_dir=paths.crops,
            target_size=cfg.crop.resize_to_vit,
            return_label=True, transform=val_tf,
            zero_mask_channels=zero_mask_channels,
            zero_image_masks_only=zero_image_masks_only,
            zero_cell_body_only=zero_cell_body_only,
            zero_polymer_only=zero_polymer_only,
            dilate_cell_body_px=dilate_cell_body_px,
        )
        n_morph = base_val.n_morphology_features
        module = _build_multimodal_module(
            cfg, image_variant=image_variant, mae_init_ckpt=mae_init_ckpt,
            n_morph_features=n_morph, use_image=use_image, use_morphology=use_morphology,
        )
        # Lightning's load_from_checkpoint requires the same constructor args
        # used at training time. We use load_state_dict on a freshly built module
        # to side-step hparam reconstruction. Buffers (incl. morphology stats)
        # come from the checkpoint state_dict.
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        module.load_state_dict(state["state_dict"], strict=False)
    else:
        base_val = CropDataset(
            cells_df=labeled_df, crops_dir=paths.crops,
            target_size=cfg.crop.resize_to_vit,
            return_label=True, transform=val_tf,
            zero_mask_channels=zero_mask_channels or zero_image_masks_only,
            zero_cell_body_only=zero_cell_body_only,
            zero_polymer_only=zero_polymer_only,
            dilate_cell_body_px=dilate_cell_body_px,
        )
        module, _ = _build_image_module(cfg, variant, mae_init_ckpt)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        module.load_state_dict(state["state_dict"], strict=False)

    val_ds = Subset(base_val, val_idx.tolist())
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=cfg.training.num_workers,
        persistent_workers=cfg.training.num_workers > 0,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = module.to(device)
    sickle_idx = cfg.finetune.sickle_class_index
    y_score, y_true_int = _score_loader(module, val_loader, sickle_idx)
    y_true = (y_true_int == sickle_idx).astype(np.int64)

    metrics = compute_binary_metrics(y_true, y_score)

    n_resamples = bootstrap_resamples or cfg.validation.bootstrap_resamples
    alpha = cfg.validation.bootstrap_alpha
    seed = cfg.project.seed

    metrics_ci: dict[str, tuple[float, float, float]] = {}
    metrics_ci["pr_auc"] = bootstrap_metric(
        y_true, y_score,
        lambda yt, ys: float(average_precision_score(yt, ys)) if np.unique(yt).size == 2 else float("nan"),
        n_resamples=n_resamples, alpha=alpha, seed=seed,
    )
    metrics_ci["mcc"] = bootstrap_metric(
        y_true, y_score,
        lambda yt, ys: (
            float(matthews_corrcoef(yt, (ys >= metrics.threshold).astype(np.int64)))
            if np.unique(yt).size == 2 and np.unique((ys >= metrics.threshold).astype(np.int64)).size == 2
            else float("nan")
        ),
        n_resamples=n_resamples, alpha=alpha, seed=seed,
    )
    pr_band = bootstrap_pr_curve(
        y_true, y_score, n_resamples=n_resamples, alpha=alpha, seed=seed,
    )

    run_name = checkpoint.parent.name
    output_dir = Path(output_dir) if output_dir is not None else (paths.figures / "eval" / run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = EvaluationReport(
        run_name=run_name,
        variant=variant,
        fold=fold,
        checkpoint=str(checkpoint),
        n_val=int(y_true.size),
        metrics=metrics,
        metrics_ci=metrics_ci,
        pr_band=pr_band,
        y_true=y_true,
        y_score=y_score,
        timestamp=datetime.now(UTC).isoformat(),
    )
    write_report(report, output_dir / "report.json")
    render_all_figures(report, output_dir)

    pa_p, pa_lo, pa_hi = metrics_ci["pr_auc"]
    mc_p, mc_lo, mc_hi = metrics_ci["mcc"]
    print(
        f"{run_name}  fold={fold}  n_val={report.n_val}\n"
        f"  PR-AUC = {pa_p:.3f}  [{pa_lo:.3f}, {pa_hi:.3f}]\n"
        f"  MCC    = {mc_p:.3f}  [{mc_lo:.3f}, {mc_hi:.3f}]\n"
        f"  recall@p=0.9 = {metrics.recall_at_p90:.3f}  threshold={metrics.threshold:.3f}\n"
        f"  figures + report -> {output_dir}"
    )
    return report


def run_figures(cfg: Config, reports_glob: Path | None = None) -> list[Path]:
    """Re-render figures from any saved ``report.json`` files. Returns the
    list of report paths processed."""
    paths_cfg = cfg.paths.resolved()
    base = paths_cfg.figures / "eval"
    candidates = (
        sorted(base.rglob("report.json")) if reports_glob is None
        else sorted(Path().glob(str(reports_glob)))
    )
    if not candidates:
        print(f"No report.json found under {base}.")
        return []
    for rp in candidates:
        report = read_report(rp)
        render_all_figures(report, rp.parent)
        print(f"re-rendered: {rp.parent}")
    return candidates
