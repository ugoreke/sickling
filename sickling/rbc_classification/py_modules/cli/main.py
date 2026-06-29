"""Top-level CLI. Subcommands map 1:1 to the Makefile targets.

    sickling smoke               Run dummy LightningModule end-to-end (CPU, 1 epoch).
    sickling instances           Stage 2 — semantic mask -> integer instance label image.
    sickling crops               Stage 3 — extract per-cell 96x96 crops + cells.parquet.
    sickling pretrain            Stage 4 — MAE continuation pretraining (Model C).
    sickling finetune            Stage 4/5 — fine-tune one of {A,B,C} or the multimodal head.
    sickling evaluate            Compute metrics + figures from a trained checkpoint.
    sickling ablate              Run the ablation table (PIPELINE_PLAN §4).
    sickling figures             Re-render figures from saved metric JSON files.
"""
from __future__ import annotations

from pathlib import Path

import typer

from sickling.rbc_classification.py_modules.config import load_config
from sickling.rbc_classification.py_modules.engineering.seed import seed_everything

app = typer.Typer(
    name="sickling",
    add_completion=False,
    help="Sickle cell classification pipeline.",
    no_args_is_help=True,
)


@app.command()
def smoke(
    config: list[Path] = typer.Option(
        None, "--config", "-c", help="Override YAML on top of base.yaml. Repeatable."
    ),
) -> None:
    """End-to-end scaffolding sanity check: dummy LightningModule, 1 epoch, CPU."""
    from sickling.rbc_classification.py_modules.engineering.smoke import run_smoke

    cfg = load_config(*(config or [Path("configs/smoke.yaml")]))
    seed_everything(cfg.project.seed)
    run_smoke(cfg)


@app.command()
def predict(
    config: list[Path] = typer.Option(None, "--config", "-c"),
    input_dir: Path = typer.Option(
        Path("to_be_labeled"), "--input", "-i",
        help="Folder of raw images to predict.",
    ),
    model_path: Path = typer.Option(
        Path("models/unet_fold_2_best.pth"), "--model",
        help="Frozen U-Net checkpoint.",
    ),
    no_copy_raws: bool = typer.Option(
        False, "--no-copy-raws",
        help="Skip copying raws into raw_images/.",
    ),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Bulk-run the frozen U-Net over a directory of raw images. Writes
    ``unet_predictions/PRED_<stem>.h5`` (Ilastik-1-based) and optionally
    copies the raw images into ``raw_images/`` so the downstream stages
    have everything on disk."""
    from sickling.rbc_classification.py_modules.stage1_unet import bulk_predict

    cfg = load_config(*(config or []))
    seed_everything(cfg.project.seed)
    bulk_predict(
        cfg,
        input_dir=input_dir,
        model_path=model_path,
        copy_raws=not no_copy_raws,
        overwrite=overwrite,
    )


@app.command()
def instances(
    config: list[Path] = typer.Option(None, "--config", "-c"),
    input_dir: Path | None = typer.Option(
        None, "--input", help="Override paths.unet_predictions for this run."
    ),
    output_dir: Path | None = typer.Option(
        None, "--output", help="Where to write integer instance label images."
    ),
    limit: int | None = typer.Option(None, "--limit", help="Process only first N FOVs."),
    qa: bool = typer.Option(
        False, "--qa", help="Also render 4-panel QA PNGs to figures/ per FOV."
    ),
) -> None:
    """Stage 2 — convert 4-class label maps to integer instance label images."""
    from sickling.rbc_classification.py_modules.stage2_instances.cli import run_stage2

    cfg = load_config(*(config or []))
    seed_everything(cfg.project.seed)
    run_stage2(cfg, input_dir=input_dir, output_dir=output_dir, limit=limit, qa=qa)


@app.command()
def crops(
    config: list[Path] = typer.Option(None, "--config", "-c"),
    limit: int | None = typer.Option(None, "--limit", help="Process only first N FOVs."),
    labels_csv: Path | None = typer.Option(
        None, "--labels-csv",
        help="Override cfg.paths.labels_csv (e.g. labels/labels_trimmed.csv).",
    ),
) -> None:
    """Stage 3 — extract per-cell 3-channel crops and build cells.parquet."""
    from sickling.rbc_classification.py_modules.stage3_crops.cli import run_stage3

    cfg = load_config(*(config or []))
    if labels_csv is not None:
        cfg.paths.labels_csv = labels_csv
    seed_everything(cfg.project.seed)
    run_stage3(cfg, limit=limit)


@app.command()
def pretrain(
    config: list[Path] = typer.Option(None, "--config", "-c"),
    devices: int = typer.Option(1, help="Number of GPUs for DDP MAE pretraining."),
    strategy: str = typer.Option("auto", help="Lightning strategy ('ddp' for multi-GPU)."),
    resume: Path | None = typer.Option(None, help="Resume from a Lightning checkpoint."),
) -> None:
    """Stage 4 — MAE continuation pretraining (Model C)."""
    from sickling.rbc_classification.py_modules.stage4_repr.cli import run_pretrain_mae

    cfg = load_config(*(config or []))
    seed_everything(cfg.project.seed)
    out = run_pretrain_mae(cfg, ckpt_path=resume, devices=devices, strategy=strategy)
    typer.echo(f"MAE pretrain done. best_ckpt={out.get('best_checkpoint')!r}")


@app.command()
def finetune(
    variant: str = typer.Argument(
        ..., help="One of: dinov2_frozen | timm_vit | mae | multimodal"
    ),
    config: list[Path] = typer.Option(None, "--config", "-c"),
    fold: int | None = typer.Option(None, help="CV fold index (defaults to cfg.finetune.fold)."),
    resume: Path | None = typer.Option(None, help="Resume from a Lightning checkpoint."),
    mae_init: Path | None = typer.Option(
        None, help="MAE pretrain checkpoint to seed the image encoder."
    ),
    image_variant: str = typer.Option(
        "dinov2_frozen",
        help="(multimodal only) Which image encoder to use inside the image tower.",
    ),
    no_image: bool = typer.Option(False, "--no-image", help="(multimodal) drop the image tower."),
    no_morphology: bool = typer.Option(
        False, "--no-morphology", help="(multimodal) drop the morphology tower."
    ),
    synth_labels: bool = typer.Option(
        False, "--synth-labels",
        help="Assign deterministic 50/50 synthetic labels for module smoke-testing. "
             "Never use for real evaluation.",
    ),
    devices: int = typer.Option(1, help="Number of GPUs."),
    target_sickle_frac: float | None = typer.Option(
        None, "--target-sickle-frac",
        help="If set, down-sample the labeled subset so this is the sickle "
             "fraction (e.g. 0.10 mimics natural prevalence).",
    ),
    fold_strategy: str | None = typer.Option(
        None, "--fold-strategy",
        help="Override cfg.validation.fold_strategy: 'balanced' or 'stratified'.",
    ),
) -> None:
    """Fine-tune Model A/B/C or the multimodal classifier on a single fold."""
    cfg = load_config(*(config or []))
    if target_sickle_frac is not None:
        cfg.validation.target_sickle_frac = target_sickle_frac
    if fold_strategy is not None:
        cfg.validation.fold_strategy = fold_strategy
    seed_everything(cfg.project.seed)

    if variant == "multimodal":
        from sickling.rbc_classification.py_modules.stage5_multimodal.cli import run_multimodal_finetune
        out = run_multimodal_finetune(
            cfg,
            image_variant=image_variant,
            fold=fold,
            ckpt_path=resume,
            mae_init_ckpt=mae_init,
            synth_labels=synth_labels,
            devices=devices,
            use_image=not no_image,
            use_morphology=not no_morphology,
        )
    else:
        from sickling.rbc_classification.py_modules.stage4_repr.cli import run_finetune
        out = run_finetune(
            cfg,
            variant=variant,
            fold=fold,
            ckpt_path=resume,
            mae_init_ckpt=mae_init,
            synth_labels=synth_labels,
            devices=devices,
        )

    typer.echo(
        f"Finetune {variant} done. best_ckpt={out.get('best_checkpoint')!r} "
        f"val_pr_auc={out.get('val_pr_auc')} val_mcc={out.get('val_mcc')}"
    )


@app.command()
def evaluate(
    checkpoint: Path = typer.Argument(..., exists=True, help="Lightning checkpoint to evaluate."),
    variant: str = typer.Option(
        ..., "--variant", help="One of: dinov2_frozen | timm_vit | mae | multimodal"
    ),
    config: list[Path] = typer.Option(None, "--config", "-c"),
    fold: int | None = typer.Option(None, help="CV fold (defaults to cfg.finetune.fold)."),
    image_variant: str = typer.Option("dinov2_frozen", help="(multimodal) image encoder variant."),
    no_image: bool = typer.Option(False, "--no-image"),
    no_morphology: bool = typer.Option(False, "--no-morphology"),
    mae_init: Path | None = typer.Option(None, help="MAE pretrain checkpoint (variant=mae)."),
    synth_labels: bool = typer.Option(False, "--synth-labels"),
    bootstrap_resamples: int | None = typer.Option(None, help="Override cfg.validation.bootstrap_resamples."),
    output_dir: Path | None = typer.Option(None, help="Override default figures/eval/<run_name>/."),
) -> None:
    """Compute headline metrics + bootstrap CIs, write JSON report + SVG figures."""
    from sickling.rbc_classification.py_modules.eval.cli import run_evaluate

    cfg = load_config(*(config or []))
    seed_everything(cfg.project.seed)
    run_evaluate(
        cfg,
        checkpoint=checkpoint,
        variant=variant,
        image_variant=image_variant,
        fold=fold,
        synth_labels=synth_labels,
        output_dir=output_dir,
        bootstrap_resamples=bootstrap_resamples,
        mae_init_ckpt=mae_init,
        use_image=not no_image,
        use_morphology=not no_morphology,
    )


@app.command()
def ablate(
    config: list[Path] = typer.Option(None, "--config", "-c"),
    seeds: str = typer.Option("42", help="Comma-separated seeds, e.g. '42,43,44'."),
    folds: str = typer.Option("0", help="Comma-separated fold indices, e.g. '0,1,2,3,4'."),
    output_dir: Path | None = typer.Option(None, help="Output dir; default figures/ablation/<ts>."),
    synth_labels: bool = typer.Option(False, "--synth-labels", help="Use synthetic labels (smoke)."),
    no_resume: bool = typer.Option(False, "--no-resume", help="Ignore cached raw_results.json."),
    target_sickle_frac: float | None = typer.Option(
        None, "--target-sickle-frac",
        help="If set, gate labels.csv to this sickle fraction before fold construction.",
    ),
    fold_strategy: str | None = typer.Option(
        None, "--fold-strategy",
        help="Override cfg.validation.fold_strategy ('balanced' or 'stratified').",
    ),
) -> None:
    """Run the full ablation table from PIPELINE_PLAN §4 and render Markdown + LaTeX tables."""
    from sickling.rbc_classification.py_modules.ablation import (
        DEFAULT_ABLATION,
        aggregate_results,
        run_ablation_table,
        write_tables,
    )

    cfg = load_config(*(config or []))
    if target_sickle_frac is not None:
        cfg.validation.target_sickle_frac = target_sickle_frac
    if fold_strategy is not None:
        cfg.validation.fold_strategy = fold_strategy
    seed_everything(cfg.project.seed)
    seed_list = tuple(int(s) for s in seeds.split(",") if s.strip())
    fold_list = tuple(int(s) for s in folds.split(",") if s.strip())

    results = run_ablation_table(
        cfg,
        rows=DEFAULT_ABLATION,
        seeds=seed_list,
        folds=fold_list,
        output_dir=output_dir,
        synth_labels=synth_labels,
        skip_existing=not no_resume,
    )
    if not results:
        typer.echo("No results produced — check the run logs.")
        raise typer.Exit(code=1)

    agg = aggregate_results(results)
    out_dir = Path(output_dir) if output_dir else None
    if out_dir is None:
        # Pull the latest output dir created inside run_ablation_table.
        ablation_root = cfg.paths.resolved().figures / "ablation"
        out_dir = max(ablation_root.iterdir(), key=lambda p: p.stat().st_mtime)
    paths = write_tables(agg, out_dir)
    typer.echo(f"\nRendered tables:\n  markdown: {paths['markdown']}\n  latex:    {paths['latex']}")


@app.command()
def figures(
    config: list[Path] = typer.Option(None, "--config", "-c"),
    reports_glob: Path | None = typer.Option(
        None, "--reports", help="Glob for report.json files (default: figures/eval/**/report.json)."
    ),
) -> None:
    """Re-render SVG figures from every ``report.json`` under ``figures/eval/``."""
    from sickling.rbc_classification.py_modules.eval.cli import run_figures

    cfg = load_config(*(config or []))
    run_figures(cfg, reports_glob=reports_glob)


if __name__ == "__main__":
    app()
