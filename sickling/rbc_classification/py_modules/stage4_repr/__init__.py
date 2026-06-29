"""Stage 4 — representation learning bake-off (Models A/B/C)."""
from sickling.rbc_classification.py_modules.stage4_repr.dinov2_encoder import DinoV2Encoder
from sickling.rbc_classification.py_modules.stage4_repr.encoder import ImageEncoder
from sickling.rbc_classification.py_modules.stage4_repr.mae_encoder import MAEReconstructor
from sickling.rbc_classification.py_modules.stage4_repr.timm_vit_encoder import MAEViTEncoder, TimmViTEncoder

__all__ = [
    "DinoV2Encoder",
    "ImageEncoder",
    "MAEReconstructor",
    "MAEViTEncoder",
    "TimmViTEncoder",
]


def build_encoder(variant: str, **kwargs) -> ImageEncoder:
    """Construct one of the bake-off encoders by string name.

    ``variant`` ∈ {``dinov2_frozen``, ``timm_vit``, ``mae``, ``mae_init``}.
    """
    if variant == "dinov2_frozen":
        return DinoV2Encoder(**kwargs)
    if variant == "timm_vit":
        return TimmViTEncoder(**kwargs)
    if variant in ("mae", "mae_init"):
        return MAEViTEncoder(**kwargs)
    raise ValueError(f"Unknown encoder variant: {variant!r}")
