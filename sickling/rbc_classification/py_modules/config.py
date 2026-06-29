"""Single source of truth for paths, channel definitions, and hyperparameters.

The base configuration lives in ``configs/base.yaml``. Per-stage YAMLs override
individual fields; environment variables (prefix ``SICKLING_``) override
everything (useful for ``WANDB_ENTITY`` etc.).

Loading::

    from sickling.rbc_classification.py_modules.config import load_config
    cfg = load_config()                                   # base.yaml only
    cfg = load_config("configs/finetune_modelA.yaml")     # base + override
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_YAML = REPO_ROOT / "configs" / "base.yaml"


class ProjectConfig(BaseModel):
    name: str = "sickling-classifier"
    wandb_entity: str | None = None
    seed: int = 42


class PathsConfig(BaseModel):
    """All paths are resolved relative to ``root`` (which is itself resolved
    relative to the repo root if it is itself relative)."""
    root: Path = Path(".")
    raw_images: Path = Path("raw_images")
    unet_predictions: Path = Path("unet_predictions")
    instances: Path = Path("instances")
    labels: Path = Path("labels")
    labels_csv: Path = Path("labels/labels.csv")  # relative to root by default
    conditions: Path = Path("conditions")
    crops: Path = Path("crops")
    cells_parquet: Path = Path("cells.parquet")
    failed_jsonl: Path = Path("failed.jsonl")
    checkpoints: Path = Path("checkpoints")
    figures: Path = Path("figures")
    wandb_dir: Path = Path("wandb_logs")

    def resolved(self) -> PathsConfig:
        """Return a copy with every path made absolute."""
        root = self.root if self.root.is_absolute() else (REPO_ROOT / self.root).resolve()
        out = self.model_copy()
        out.root = root
        for field in type(self).model_fields:
            if field == "root":
                continue
            val: Path = getattr(self, field)
            setattr(out, field, val if val.is_absolute() else (root / val))
        return out


class ClassesConfig(BaseModel):
    polymer: int = 0
    background: int = 1
    cell_body: int = 2
    cell_border: int = 3


class CropConfig(BaseModel):
    size: int = 96
    resize_to_vit: int = 224
    norm_percentile: float = 99.0
    drop_if_clipped: bool = True


class InstancesConfig(BaseModel):
    closing_radius: int = 2
    peak_min_distance: int = 12
    peak_threshold_rel: float = 0.1
    min_area: int = 550
    max_area: int = 6000
    drop_edge_touching: bool = True


class DinoV2Config(BaseModel):
    repo: str = "facebookresearch/dinov2"
    model: str = "dinov2_vits14"
    embed_dim: int = 384


class TimmViTConfig(BaseModel):
    model: str = "vit_small_patch16_224.augreg_in21k_ft_in1k"
    embed_dim: int = 384
    layer_wise_lr_decay: float = 0.65


class MAEConfig(BaseModel):
    model: str = "vit_small_patch16_224.mae"
    embed_dim: int = 384
    mask_ratio: float = 0.75
    # Documentation field only — the trainer reads ``cfg.training.max_epochs``.
    # 800 matches ``configs/pretrain_mae_long.yaml``; older 300-epoch runs
    # remain reproducible via ``configs/pretrain_mae.yaml``.
    pretrain_epochs: int = 800


class RepresentationConfig(BaseModel):
    variants: list[str] = Field(
        default_factory=lambda: ["dinov2_frozen", "timm_vit_supervised", "mae_continued"]
    )
    dinov2: DinoV2Config = DinoV2Config()
    timm_vit: TimmViTConfig = TimmViTConfig()
    mae: MAEConfig = MAEConfig()


class MultimodalConfig(BaseModel):
    fusion_hidden: int = 256
    dropout: float = 0.3
    num_classes: int = 2


class ValidationConfig(BaseModel):
    cv_folds: int = 5
    group_by: str = "source_image"
    bootstrap_resamples: int = 1000
    bootstrap_alpha: float = 0.05
    natural_prevalence_eval: bool = True
    weighted_sampler_minority_frac: float = 0.5
    # Fold construction strategy. ``balanced`` is the FOV-grouped greedy
    # bin-packer that equalises per-class cell counts across folds; see
    # ``sickling.eval.splits.balanced_group_kfold``. ``stratified`` is the
    # original sklearn-backed splitter (kept for reproducibility of the
    # ablation_20260516_003426 results).
    fold_strategy: str = "balanced"
    # Optional label-prevalence gate applied to the labeled subset before
    # fold construction. ``None`` disables the gate. ``0.10`` mimics the
    # natural ~10% sickle prevalence on the existing label corpus before
    # the user finishes collecting more non-sickle labels.
    target_sickle_frac: float | None = None
    gate_seed: int = 42


class TrainingConfig(BaseModel):
    precision: str = "bf16-mixed"
    batch_size: int = 64
    grad_accum: int = 1
    max_epochs: int = 50
    lr: float = 1.0e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    monitor_metric: str = "val_pr_auc"
    monitor_mode: str = "max"
    save_top_k: int = 3
    num_workers: int = 4


class AugmentConfig(BaseModel):
    """Train-time augmentation knobs. All flips / rot90 are channel-aware."""
    hflip_p: float = 0.5
    vflip_p: float = 0.5
    rot90_p: float = 0.75
    brightness_jitter: float = 0.10   # applied to ch0 only; ch1/ch2 kept binary
    contrast_jitter: float = 0.10


class FinetuneConfig(BaseModel):
    """Hyperparameters for the linear-probe and full-finetune Lightning modules."""
    head_lr: float = 1.0e-3
    backbone_lr: float = 1.0e-4
    llrd: float = 0.65
    label_smoothing: float = 0.0
    minority_frac: float = 0.5         # WeightedRandomSampler target sickle fraction
    fold: int = 0
    sickle_class_index: int = 1        # int label for "sickle" in the binary head


class MAEPretrainConfig(BaseModel):
    """Hyperparameters for masked-autoencoder continuation on the 40k cell crops."""
    mask_ratio: float = 0.75
    decoder_embed_dim: int = 256
    decoder_depth: int = 4
    decoder_num_heads: int = 8
    norm_pix_loss: bool = True         # standard MAE-paper trick
    periodic_probe_every: int = 0      # 0 = disabled. Otherwise: linear-probe MCC every N epochs.


class SmokeConfig(BaseModel):
    """Used only by ``make smoke``."""
    num_samples: int = 64
    input_dim: int = 16


class Config(BaseSettings):
    """Root configuration. Loaded from YAML and overridable via env vars
    prefixed with ``SICKLING_`` (e.g. ``SICKLING_PROJECT__WANDB_ENTITY=foo``)."""
    model_config = SettingsConfigDict(
        env_prefix="SICKLING_",
        env_nested_delimiter="__",
        extra="allow",
    )

    project: ProjectConfig = ProjectConfig()
    paths: PathsConfig = PathsConfig()
    classes: ClassesConfig = ClassesConfig()
    crop: CropConfig = CropConfig()
    instances: InstancesConfig = InstancesConfig()
    representation: RepresentationConfig = RepresentationConfig()
    multimodal: MultimodalConfig = MultimodalConfig()
    validation: ValidationConfig = ValidationConfig()
    training: TrainingConfig = TrainingConfig()
    augment: AugmentConfig = AugmentConfig()
    finetune: FinetuneConfig = FinetuneConfig()
    mae_pretrain: MAEPretrainConfig = MAEPretrainConfig()
    smoke: SmokeConfig = SmokeConfig()


def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``b`` into ``a`` and return the result."""
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Top-level YAML in {path} must be a mapping, got {type(loaded)}.")
    return loaded


def load_config(*overrides: str | Path) -> Config:
    """Load ``base.yaml`` and apply zero or more override YAML files in order."""
    merged: dict[str, Any] = _load_yaml(DEFAULT_BASE_YAML)
    for path in overrides:
        merged = _deep_merge(merged, _load_yaml(Path(path)))
    return Config(**merged)
