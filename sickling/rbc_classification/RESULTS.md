# Sickle cell classification — results

> Auto-fill the numerical cells from `figures/ablation/<run>/table_markdown.md`
> after running `sickling ablate`. Keep the structure stable; this doc is what
> the writeup pulls from.

## Headline

- **Backbone winner (Stage 4 bake-off):** _TBD — fill after running `sickling ablate`._
- **Multimodal vs image-only:** _TBD._
- **Best operating point:** PR-AUC `_._` (95% CI [_, _]), MCC `_._`, recall@p=0.9 `_._`.

## Ablation table (PIPELINE_PLAN §4)

<!-- Pasted from figures/ablation/<run>/table_markdown.md -->

| Row | PR-AUC | MCC | recall@p=0.9 | F1 (sickle) | F1 (non-sickle) | runs |
|-----|--------|-----|--------------|-------------|------------------|------|
| _placeholder_ | _._ ± _._ | _._ ± _._ | _._ ± _._ | _._ | _._ | _ |

## Bake-off (Models A vs B vs C)

| Backbone | PR-AUC | MCC | Train cost | Notes |
|----------|--------|-----|-----------|-------|
| A — DINOv2 frozen + linear probe | _ | _ | cheapest (only head trained) | day-one baseline |
| B — ViT-S/16 ImageNet-supervised + full FT | _ | _ | moderate | LLRD 0.65 |
| C — MAE init + full FT | _ | _ | most expensive (continuation MAE run) | uses unlabeled corpus |

## DDP scaling (resume artifact)

<!-- Fill from figures/ddp_benchmark/throughput.csv -->

| GPUs | Batch / GPU | Global batch | Images / sec | Scaling efficiency |
|-----:|------------:|------------:|------------:|-------------------:|
| 1 | _ | _ | _ | 100% (baseline) |
| 2 | _ | _ | _ | _% |
| 4 | _ | _ | _ | _% |

Throughput bar chart: [`figures/ddp_benchmark/throughput.svg`](figures/ddp_benchmark/throughput.svg).

## Bias / sanity checks (PIPELINE_PLAN §3)

- By **source ROI**: stratified PR-AUC across FOVs. _TBD._
- By **oxygen condition**: stratified PR-AUC at 21% vs 2%. _TBD._
- By **treatment**: stratified PR-AUC at DMSO vs drug arms. _TBD._
- By **mask quality**: high vs low Dice on the U-Net per-FOV — does classifier
  performance depend on segmentation quality? _TBD._

## Reproducibility

- Every run loads from `configs/base.yaml` + optional override YAML.
- All experiments seed every RNG via `seed_everything(cfg.project.seed)`.
- Dependencies pinned in `requirements.txt`.
- End-to-end: `make instances && make crops && sickling ablate`.
