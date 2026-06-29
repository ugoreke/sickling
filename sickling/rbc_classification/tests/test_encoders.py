"""Smoke tests for the three Stage 4 encoders.

These tests download pretrained weights on first run (DINOv2 via torch.hub,
timm ViT via huggingface). They are network-gated: if the download fails,
the test is skipped — the smoke is *that the API works given weights*, not
network reliability.
"""
from __future__ import annotations

import pytest
import torch

from sickling.rbc_classification.py_modules.stage4_repr import build_encoder
from sickling.rbc_classification.py_modules.stage4_repr.mae_encoder import MAEReconstructor, _patchify, _random_masking


def _try_build(variant: str):
    try:
        return build_encoder(variant)
    except (RuntimeError, OSError, FileNotFoundError) as e:
        pytest.skip(f"Could not load weights for {variant}: {e}")


@pytest.mark.parametrize("variant", ["dinov2_frozen", "timm_vit", "mae"])
def test_encoder_forward_shape(variant):
    encoder = _try_build(variant)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        z = encoder(x)
    assert z.shape == (2, encoder.embed_dim)


def test_dinov2_frozen_param_groups_empty():
    encoder = _try_build("dinov2_frozen")
    groups = encoder.trainable_param_groups(base_lr=1e-3)
    assert groups == []


def test_timm_vit_llrd_param_groups_have_decreasing_lr():
    encoder = _try_build("timm_vit")
    groups = encoder.trainable_param_groups(base_lr=1e-4, llrd=0.65)
    lrs = [g["lr"] for g in groups]
    assert min(lrs) < max(lrs), "LLRD must produce a non-trivial LR range."
    # Patch_embed group (group 0) is the most-decayed.
    assert lrs[0] == min(lrs)


def test_random_masking_correct_kept_count():
    n_patches = 196
    x = torch.randn(2, n_patches, 384)
    x_kept, mask, ids_restore = _random_masking(x, mask_ratio=0.75)
    assert x_kept.shape == (2, int(n_patches * 0.25), 384)
    assert mask.sum().item() == 2 * int(n_patches * 0.75)
    assert ids_restore.shape == (2, n_patches)


def test_patchify_round_trip_shape():
    x = torch.randn(1, 3, 224, 224)
    patches = _patchify(x, patch_size=16)
    assert patches.shape == (1, 196, 16 * 16 * 3)


def test_mae_reconstructor_forward():
    encoder = _try_build("mae")
    rec = MAEReconstructor(
        encoder=encoder, decoder_embed_dim=128, decoder_depth=2, decoder_num_heads=4
    )
    x = torch.rand(2, 3, 224, 224)
    loss, pred, target, mask = rec(x, mask_ratio=0.75)
    assert loss.ndim == 0  # scalar
    assert pred.shape == target.shape
    assert mask.shape[0] == x.shape[0]
    assert torch.isfinite(loss)
