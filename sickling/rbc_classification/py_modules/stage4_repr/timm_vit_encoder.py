"""Model B — timm ViT-S/16, ImageNet-supervised, full fine-tune with LLRD.

Reused as the backbone for **Model C** (MAE continuation) — the only
difference between B and C is the source of the pretrained checkpoint.
"""
from __future__ import annotations

import timm
import torch

from sickling.rbc_classification.py_modules.stage4_repr.encoder import ImageEncoder


class TimmViTEncoder(ImageEncoder):
    """Wraps a timm ViT and exposes per-block parameter groups for LLRD."""

    embed_dim = 384

    def __init__(
        self,
        model_name: str = "vit_small_patch16_224.augreg_in21k_ft_in1k",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        # ``num_classes=0`` removes the classification head; pooled output is the [CLS] embedding.
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = int(self.backbone.num_features)
        # The model exposes its expected mean/std. Cache for standardize().
        self._mean = tuple(self.backbone.default_cfg.get("mean", (0.485, 0.456, 0.406)))
        self._std = tuple(self.backbone.default_cfg.get("std", (0.229, 0.224, 0.225)))

    def standardize(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self._mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self._std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x - mean) / std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.standardize(x)
        return self.backbone(x)

    def trainable_param_groups(
        self, base_lr: float, llrd: float | None = None
    ) -> list[dict]:
        """Layer-wise LR decay over patch_embed → blocks → norm.

        Block ``i`` (0-indexed, deepest = last) gets ``base_lr * llrd^(N-i)``.
        Patch_embed gets the most-decayed LR. Norm + cls_token at the top use base_lr.
        Returns one param group per layer-with-trainable-params.
        """
        if llrd is None or llrd >= 1.0:
            return super().trainable_param_groups(base_lr=base_lr)

        groups: list[dict] = []
        blocks = list(getattr(self.backbone, "blocks", []))
        n_layers = len(blocks) + 2  # +1 for patch_embed, +1 for top norm/cls
        # patch_embed (deepest) → smallest LR.
        pe_params = list(self.backbone.patch_embed.parameters())
        cls_token = getattr(self.backbone, "cls_token", None)
        pos_embed = getattr(self.backbone, "pos_embed", None)
        if cls_token is not None:
            pe_params.append(cls_token)
        if pos_embed is not None:
            pe_params.append(pos_embed)
        groups.append({"params": pe_params, "lr": base_lr * (llrd ** n_layers)})

        for i, blk in enumerate(blocks):
            # Block 0 (closest to input) is deeper than block -1 (closest to head).
            depth_from_top = (len(blocks) - 1) - i
            lr = base_lr * (llrd ** (depth_from_top + 1))
            groups.append({"params": list(blk.parameters()), "lr": lr})

        # Top norm + remaining params — full lr.
        used = set(map(id, [p for g in groups for p in g["params"]]))
        top_params = [p for p in self.backbone.parameters() if id(p) not in used]
        if top_params:
            groups.append({"params": top_params, "lr": base_lr})
        return groups


class MAEViTEncoder(TimmViTEncoder):
    """Same architecture as ``TimmViTEncoder``; loads from a MAE checkpoint
    instead of the supervised one. Falls back to the supervised init if the
    requested MAE name is not in the timm registry — emits a warning so the
    user notices."""

    def __init__(self, model_name: str = "vit_small_patch16_224.mae") -> None:
        try:
            super().__init__(model_name=model_name, pretrained=True)
        except (RuntimeError, ValueError) as e:
            import warnings
            warnings.warn(
                f"MAE checkpoint {model_name!r} not loadable ({e}); falling back to "
                "supervised vit_small_patch16_224.augreg_in21k_ft_in1k. Replace with "
                "your locally pretrained MAE checkpoint when available.",
                stacklevel=2,
            )
            super().__init__(
                model_name="vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=True
            )

    def load_mae_checkpoint(self, path) -> None:
        """Load encoder weights from a Lightning checkpoint produced by
        ``MAEPretrainModule``. Decoder weights and any non-encoder keys are
        ignored — they were only needed for SSL."""
        state = torch.load(path, weights_only=True, map_location="cpu")
        sd = state.get("state_dict", state)
        encoder_sd = {
            k.removeprefix("encoder.backbone."): v
            for k, v in sd.items()
            if k.startswith("encoder.backbone.")
        }
        if not encoder_sd:
            raise ValueError(
                f"No 'encoder.backbone.*' keys in {path}. Was this produced by "
                "MAEPretrainModule?"
            )
        missing, unexpected = self.backbone.load_state_dict(encoder_sd, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys when loading MAE checkpoint: {unexpected}")
        if missing:
            # Some keys (e.g. classifier head) won't be in the state dict — fine.
            pass
