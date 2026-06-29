"""Ablation runner — PIPELINE_PLAN §4 table.

Every row is a `(variant, optional config overrides, optional tower flags)`
spec. The runner crosses each row with ``seeds × folds`` and produces an
``AblationResult`` per cell. Results are written to disk after every run so
a crash mid-table resumes from where it left off.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from sickling.rbc_classification.py_modules.config import Config
from sickling.rbc_classification.py_modules.engineering.seed import seed_everything
from sickling.rbc_classification.py_modules.eval.cli import run_evaluate
from sickling.rbc_classification.py_modules.eval.metrics import BinaryMetrics
from sickling.rbc_classification.py_modules.stage4_repr.cli import run_finetune
from sickling.rbc_classification.py_modules.stage5_multimodal.cli import run_multimodal_finetune


@dataclass(frozen=True)
class AblationRow:
    """One row of the ablation table.

    ``name`` is the human-readable identifier shown in the rendered table.
    ``variant`` is one of ``dinov2_frozen``, ``timm_vit``, ``mae``, ``multimodal``.
    ``image_variant`` only matters for the multimodal case (= which image
    encoder the image tower wraps).
    ``use_image`` / ``use_morphology`` toggle multimodal towers.
    ``zero_mask_channels`` zeros out ch1/ch2 in every crop tensor at Dataset
    time — only meaningful when the image tower is used (Models B/C/multimodal).
    ``minority_frac=0`` disables the weighted sampler (natural-prevalence row).
    ``overrides`` is a dict of dotted ``key.path`` -> value to deep-merge into
    the Config before each run (e.g. ``{'finetune.minority_frac': 0.0}``).
    """
    name: str
    variant: str
    image_variant: str = "dinov2_frozen"
    use_image: bool = True
    use_morphology: bool = True
    zero_mask_channels: bool = False
    zero_image_masks_only: bool = False
    zero_cell_body_only: bool = False
    zero_polymer_only: bool = False
    dilate_cell_body_px: int = 0
    overrides: dict = field(default_factory=dict)
    notes: str = ""


@dataclass
class AblationResult:
    row_name: str
    variant: str
    seed: int
    fold: int
    pr_auc: float
    pr_auc_ci: tuple[float, float]
    mcc: float
    mcc_ci: tuple[float, float]
    recall_at_p90: float
    f1_sickle: float
    f1_non_sickle: float
    threshold: float
    n_val: int
    checkpoint: str
    eval_dir: str
    duration_seconds: float
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _apply_overrides(cfg: Config, overrides: dict) -> Config:
    """Apply ``{'a.b.c': value}`` overrides onto a config copy."""
    if not overrides:
        return cfg
    merged: dict = cfg.model_dump(mode="python")
    for key, value in overrides.items():
        node = merged
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return Config(**merged)


# Default table mirrors PIPELINE_PLAN §4.
DEFAULT_ABLATION: list[AblationRow] = [
    AblationRow(
        name="Full multimodal (DINOv2 frozen + morph)",
        variant="multimodal", image_variant="dinov2_frozen",
        use_image=True, use_morphology=True,
    ),
    AblationRow(
        name="- morphology tower",
        variant="multimodal", image_variant="dinov2_frozen",
        use_image=True, use_morphology=False,
        notes="Does shape add over learned features?",
    ),
    AblationRow(
        name="- image tower (morphology only)",
        variant="multimodal", image_variant="dinov2_frozen",
        use_image=False, use_morphology=True,
        notes="Plan B baseline: morphology-only ceiling.",
    ),
    AblationRow(
        name="- mask channels (ch1=ch2=0)",
        variant="multimodal", image_variant="dinov2_frozen",
        use_image=True, use_morphology=True, zero_mask_channels=True,
        notes="Does explicit mask channel help the ViT?",
    ),
    AblationRow(
        name="- mask channels (image tower only, morphology kept)",
        variant="multimodal", image_variant="dinov2_frozen",
        use_image=True, use_morphology=True, zero_image_masks_only=True,
        notes="Per-tower mask test from discussion limitation 5: zero ch1/ch2 "
              "for the image tower while morphology features stay computed on "
              "the un-zeroed masks.",
    ),
    AblationRow(
        name="- weighted sampler (natural prevalence)",
        variant="multimodal", image_variant="dinov2_frozen",
        overrides={"finetune.minority_frac": 0.0},
        notes="How much does balancing matter at train time?",
    ),
    AblationRow(
        name="Image = DINOv2 frozen (linear probe)",
        variant="dinov2_frozen",
    ),
    AblationRow(
        name="Image = ViT-S supervised (full FT)",
        variant="timm_vit",
    ),
    AblationRow(
        name="Image = MAE init (full FT)",
        variant="mae",
    ),
]


def _flatten_metrics(report) -> dict:
    m: BinaryMetrics = report.metrics
    ci = report.metrics_ci
    return {
        "pr_auc": float(m.pr_auc),
        "pr_auc_ci": (float(ci.get("pr_auc", (m.pr_auc, np.nan, np.nan))[1]),
                       float(ci.get("pr_auc", (m.pr_auc, np.nan, np.nan))[2])),
        "mcc": float(m.mcc),
        "mcc_ci": (float(ci.get("mcc", (m.mcc, np.nan, np.nan))[1]),
                    float(ci.get("mcc", (m.mcc, np.nan, np.nan))[2])),
        "recall_at_p90": float(m.recall_at_p90),
        "f1_sickle": float(m.f1_sickle),
        "f1_non_sickle": float(m.f1_non_sickle),
        "threshold": float(m.threshold),
    }


def _finish_wandb_run() -> None:
    """Close any open wandb run so the next ``WandbLogger`` starts a fresh one
    instead of warning about reuse. Safe to call when no run is active."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass


def _train_one(cfg: Config, row: AblationRow, seed: int, fold: int, synth_labels: bool) -> dict:
    cfg = _apply_overrides(cfg, row.overrides)
    seed_everything(seed)
    try:
        if row.variant == "multimodal":
            return run_multimodal_finetune(
                cfg,
                image_variant=row.image_variant,
                fold=fold,
                synth_labels=synth_labels,
                use_image=row.use_image,
                use_morphology=row.use_morphology,
                zero_mask_channels=row.zero_mask_channels,
                zero_image_masks_only=row.zero_image_masks_only,
                zero_cell_body_only=row.zero_cell_body_only,
                zero_polymer_only=row.zero_polymer_only,
                dilate_cell_body_px=row.dilate_cell_body_px,
            )
        return run_finetune(
            cfg,
            variant=row.variant,
            fold=fold,
            synth_labels=synth_labels,
            zero_mask_channels=row.zero_mask_channels or row.zero_image_masks_only,
        )
    finally:
        _finish_wandb_run()


def _eval_one(
    cfg: Config, row: AblationRow, fold: int, ckpt: str, synth_labels: bool
) -> dict:
    cfg = _apply_overrides(cfg, row.overrides)
    try:
        report = run_evaluate(
            cfg,
            checkpoint=Path(ckpt),
            variant=row.variant,
            image_variant=row.image_variant,
            fold=fold,
            synth_labels=synth_labels,
            use_image=row.use_image,
            use_morphology=row.use_morphology,
            zero_mask_channels=row.zero_mask_channels,
            zero_image_masks_only=row.zero_image_masks_only,
            zero_cell_body_only=row.zero_cell_body_only,
            zero_polymer_only=row.zero_polymer_only,
            dilate_cell_body_px=row.dilate_cell_body_px,
        )
        return {
            "report": report,
            "metrics": _flatten_metrics(report),
        }
    finally:
        _finish_wandb_run()


def run_ablation_table(
    cfg: Config,
    rows: list[AblationRow] | None = None,
    seeds: tuple[int, ...] = (42,),
    folds: tuple[int, ...] = (0,),
    output_dir: Path | None = None,
    synth_labels: bool = False,
    skip_existing: bool = True,
) -> list[AblationResult]:
    """Run every (row, seed, fold) combination. Persists ``raw_results.json``
    after every run so the table is crash-resumable.

    Returns the flat list of results in run order.
    """
    rows = rows or DEFAULT_ABLATION
    paths = cfg.paths.resolved()
    output_dir = output_dir or (paths.figures / "ablation" / time.strftime("ablation_%Y%m%d_%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "raw_results.json"
    results: list[AblationResult] = []
    if raw_path.exists() and skip_existing:
        with open(raw_path) as f:
            for d in json.load(f):
                d["pr_auc_ci"] = tuple(d["pr_auc_ci"])
                d["mcc_ci"] = tuple(d["mcc_ci"])
                results.append(AblationResult(**d))
    done_keys = {(r.row_name, r.seed, r.fold) for r in results}

    n_cells = len(rows) * len(seeds) * len(folds)
    cell_idx = 0
    for row in rows:
        for seed in seeds:
            for fold in folds:
                cell_idx += 1
                key = (row.name, seed, fold)
                if key in done_keys:
                    print(f"[{cell_idx}/{n_cells}] skip (cached) — {row.name} seed={seed} fold={fold}")
                    continue
                t0 = time.time()
                print(f"[{cell_idx}/{n_cells}] TRAIN — {row.name} seed={seed} fold={fold}")
                try:
                    train_out = _train_one(cfg, row, seed, fold, synth_labels)
                    ckpt = train_out.get("best_checkpoint", "")
                    if not ckpt:
                        raise RuntimeError("No best_checkpoint returned by trainer.")
                    eval_out = _eval_one(cfg, row, fold, ckpt, synth_labels)
                    duration = time.time() - t0
                    m = eval_out["metrics"]
                    result = AblationResult(
                        row_name=row.name,
                        variant=row.variant,
                        seed=seed,
                        fold=fold,
                        pr_auc=m["pr_auc"],
                        pr_auc_ci=m["pr_auc_ci"],
                        mcc=m["mcc"],
                        mcc_ci=m["mcc_ci"],
                        recall_at_p90=m["recall_at_p90"],
                        f1_sickle=m["f1_sickle"],
                        f1_non_sickle=m["f1_non_sickle"],
                        threshold=m["threshold"],
                        n_val=int(eval_out["report"].n_val),
                        checkpoint=str(ckpt),
                        eval_dir=str(Path(ckpt).parent),
                        duration_seconds=duration,
                        notes=row.notes,
                    )
                    results.append(result)
                    print(
                        f"   PR-AUC={result.pr_auc:.3f} {tuple(round(v, 3) for v in result.pr_auc_ci)}"
                        f"   MCC={result.mcc:.3f}   ({duration:.1f}s)"
                    )
                except Exception as e:
                    print(f"   FAILED: {e!r}")
                    continue
                # Persist after each run.
                with open(raw_path, "w") as f:
                    json.dump([r.to_dict() for r in results], f, indent=2)

    return results


def load_results(path: Path) -> list[AblationResult]:
    with open(path) as f:
        data = json.load(f)
    out: list[AblationResult] = []
    for d in data:
        d["pr_auc_ci"] = tuple(d["pr_auc_ci"])
        d["mcc_ci"] = tuple(d["mcc_ci"])
        out.append(AblationResult(**d))
    return out


def aggregate_results(results: list[AblationResult]) -> pd.DataFrame:
    """One row per ablation row, with mean / std across all (seed, fold) cells."""
    rows = []
    df = pd.DataFrame([r.to_dict() for r in results])
    if df.empty:
        return df
    for row_name, sub in df.groupby("row_name", sort=False):
        rows.append({
            "row_name": row_name,
            "variant": sub["variant"].iloc[0],
            "n_runs": len(sub),
            "pr_auc_mean": float(sub["pr_auc"].mean()),
            "pr_auc_std": float(sub["pr_auc"].std(ddof=0)) if len(sub) > 1 else 0.0,
            "pr_auc_ci_lo": float(sub["pr_auc"].mean() - sub["pr_auc"].std(ddof=0)),  # placeholder for renderer
            "pr_auc_ci_hi": float(sub["pr_auc"].mean() + sub["pr_auc"].std(ddof=0)),
            "mcc_mean": float(sub["mcc"].mean()),
            "mcc_std": float(sub["mcc"].std(ddof=0)) if len(sub) > 1 else 0.0,
            "recall_at_p90_mean": float(sub["recall_at_p90"].mean()),
            "recall_at_p90_std": float(sub["recall_at_p90"].std(ddof=0)) if len(sub) > 1 else 0.0,
            "f1_sickle_mean": float(sub["f1_sickle"].mean()),
            "f1_non_sickle_mean": float(sub["f1_non_sickle"].mean()),
            "notes": sub.iloc[0].get("notes", ""),
        })
    return pd.DataFrame(rows)
