"""Masked-autoencoder reconstructor used during MAE continuation pretraining.

The encoder is a ``TimmViTEncoder`` (initialized from MAE weights via
``MAEViTEncoder``). The decoder is a small ViT trained to reconstruct
masked image patches. After SSL we discard the decoder and keep the encoder.

Implementation follows the original MAE paper (He et al. 2022):
    1. Patchify x and apply random masking (mask_ratio of patches hidden).
    2. Encoder runs on the *visible* patches only.
    3. Decoder receives encoder output + mask tokens at masked positions, plus
       a separate decoder positional embedding, and predicts the masked pixels.
    4. Loss = MSE on masked patches only (optionally per-patch normalized).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from sickling.rbc_classification.py_modules.stage4_repr.timm_vit_encoder import MAEViTEncoder


def _patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B, C, H, W) -> (B, num_patches, patch_size*patch_size*C)."""
    b, c, h, w = x.shape
    if h % patch_size or w % patch_size:
        raise ValueError(f"H={h} W={w} not divisible by patch_size={patch_size}.")
    x = x.reshape(b, c, h // patch_size, patch_size, w // patch_size, patch_size)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    return x.reshape(b, (h // patch_size) * (w // patch_size), patch_size * patch_size * c)


def _random_masking(
    x: torch.Tensor, mask_ratio: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample random masking via argsort of uniform noise.

    Returns:
        x_kept: (B, num_kept, D) — only visible tokens, in shuffled order.
        mask:   (B, N) — 1 if masked, 0 if kept.
        ids_restore: (B, N) — indices to undo the shuffle.
    """
    b, n, d = x.shape
    n_kept = int(n * (1 - mask_ratio))
    noise = torch.rand(b, n, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :n_kept]

    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, d))
    mask = torch.ones(b, n, device=x.device)
    mask[:, :n_kept] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_kept, mask, ids_restore


class MAEReconstructor(nn.Module):
    """Wraps a ViT encoder with a small ViT decoder for masked reconstruction."""

    def __init__(
        self,
        encoder: MAEViTEncoder,
        decoder_embed_dim: int = 256,
        decoder_depth: int = 4,
        decoder_num_heads: int = 8,
        norm_pix_loss: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.norm_pix_loss = norm_pix_loss

        backbone = encoder.backbone
        self.patch_size = int(backbone.patch_embed.patch_size[0])
        self.num_patches = int(backbone.patch_embed.num_patches)
        self.encoder_dim = int(backbone.num_features)

        # Decoder: linear projection + small ViT + linear prediction head.
        self.decoder_embed = nn.Linear(self.encoder_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim)  # +1 for CLS slot
        )
        self.decoder_blocks = nn.ModuleList([
            Block(dim=decoder_embed_dim, num_heads=decoder_num_heads, mlp_ratio=4.0)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        # Predict patch_size * patch_size * 3 (3 channels: ch0 + ch1 + ch2).
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_size ** 2 * 3, bias=True)

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

    def _encode_visible(
        self, x: torch.Tensor, mask_ratio: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Patchify → embed → mask → run encoder on visible patches."""
        backbone = self.encoder.backbone
        # Standardize like the encoder does in `forward`.
        x = self.encoder.standardize(x)
        # Run patch_embed (turns image into patch tokens) before masking.
        # timm ViT exposes ``patch_embed`` returning (B, N, D).
        tokens = backbone.patch_embed(x)
        if hasattr(backbone, "_pos_embed"):
            tokens = backbone._pos_embed(tokens)  # adds pos_embed + cls token
        else:  # fallback for older timm versions
            cls = backbone.cls_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat([cls, tokens], dim=1) + backbone.pos_embed

        cls = tokens[:, :1]
        patch_tokens = tokens[:, 1:]
        x_kept, mask, ids_restore = _random_masking(patch_tokens, mask_ratio)
        x = torch.cat([cls, x_kept], dim=1)

        for blk in backbone.blocks:
            x = blk(x)
        x = backbone.norm(x)
        return x, mask, ids_restore

    def _decode(self, latent: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        x = self.decoder_embed(latent)
        cls, x_kept = x[:, :1], x[:, 1:]
        b, n_kept, d = x_kept.shape
        n_full = ids_restore.shape[1]
        n_masked = n_full - n_kept

        mask_tokens = self.mask_token.expand(b, n_masked, d)
        x_full = torch.cat([x_kept, mask_tokens], dim=1)
        x_full = torch.gather(
            x_full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, d)
        )
        x = torch.cat([cls, x_full], dim=1)
        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x[:, 1:]  # drop CLS — only patch predictions

    def forward(
        self, x: torch.Tensor, mask_ratio: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(loss, pred, target, mask)``."""
        latent, mask, ids_restore = self._encode_visible(x, mask_ratio)
        pred = self._decode(latent, ids_restore)
        target = _patchify(x, self.patch_size)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6).sqrt()

        loss = ((pred - target) ** 2).mean(dim=-1)  # per-patch
        loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
        return loss, pred, target, mask
