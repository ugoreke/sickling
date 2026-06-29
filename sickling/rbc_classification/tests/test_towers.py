"""Tests for the Stage 5 tower stack — Tower contract, MorphologyTower MLP,
MultimodalClassifier wiring with arbitrary modality dicts."""
from __future__ import annotations

import torch

from sickling.rbc_classification.py_modules.stage5_multimodal import (
    MorphologyTower,
    MultimodalClassifier,
    Tower,
)


class _StubTower(Tower):
    """Identity-passthrough tower used to test the contract without weights."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.D = out_dim
        self.proj = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def test_tower_contract():
    tower = _StubTower(in_dim=10, out_dim=4)
    out = tower(torch.randn(8, 10))
    assert out.shape == (8, tower.D)


def test_morphology_tower_forward_shape():
    tower = MorphologyTower(in_features=30, hidden=64, out_features=64)
    out = tower(torch.randn(7, 30))
    assert out.shape == (7, 64)


def test_morphology_tower_uses_feature_stats():
    tower = MorphologyTower(in_features=4, hidden=8, out_features=8)
    mean = torch.tensor([1.0, 2.0, 3.0, 4.0])
    std = torch.tensor([0.5, 0.5, 0.5, 0.5])
    tower.set_feature_stats(mean, std)
    # Buffers should travel with state_dict.
    sd = tower.state_dict()
    assert "feature_mean" in sd and "feature_std" in sd
    torch.testing.assert_close(sd["feature_mean"], mean)
    torch.testing.assert_close(sd["feature_std"], std)


def test_multimodal_classifier_concat_and_forward():
    towers = {
        "alpha": _StubTower(in_dim=8, out_dim=3),
        "beta": _StubTower(in_dim=4, out_dim=5),
    }
    classifier = MultimodalClassifier(towers, num_classes=2, hidden=16, dropout=0.0)
    assert classifier.total_embed_dim == 8
    inputs = {"alpha": torch.randn(2, 8), "beta": torch.randn(2, 4)}
    logits = classifier(inputs)
    assert logits.shape == (2, 2)


def test_multimodal_classifier_missing_modality_errors():
    classifier = MultimodalClassifier(
        {"alpha": _StubTower(in_dim=4, out_dim=2)}, num_classes=2, dropout=0.0
    )
    import pytest
    with pytest.raises(KeyError, match="missing modality"):
        classifier({"beta": torch.randn(1, 4)})


def test_classifier_param_groups_have_per_tower_lrs():
    towers = {
        "alpha": _StubTower(in_dim=3, out_dim=2),
        "beta": _StubTower(in_dim=3, out_dim=2),
    }
    cls = MultimodalClassifier(towers, num_classes=2, dropout=0.0)
    groups = cls.trainable_param_groups(
        base_lrs={"alpha": 1e-4, "beta": 1e-3}, head_lr=5e-3
    )
    lrs = [g["lr"] for g in groups]
    assert 1e-4 in lrs and 1e-3 in lrs and 5e-3 in lrs
