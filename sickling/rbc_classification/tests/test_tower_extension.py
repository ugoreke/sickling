"""Demonstrates that adding a new modality is a 5-line code change.

Per PIPELINE_PLAN §2 Stage 5 contract: subclassing ``Tower`` once and
registering the instance in the classifier dict is the entire integration —
no changes to ``MultimodalClassifier`` or any existing tower required.

If this test breaks, the modality contract has been compromised.
"""
from __future__ import annotations

import torch

from sickling.rbc_classification.py_modules.stage5_multimodal import MorphologyTower, MultimodalClassifier, Tower


# === The ENTIRE integration: 4 lines below this comment, line-counted. ===
class TimeTower(Tower):                                                # 1
    D = 32                                                             # 2
    def forward(self, x): return torch.zeros(x.shape[0], self.D, device=x.device)  # 3
# === End of integration. ===                                          # 4


def test_third_modality_drops_in_unchanged():
    morph = MorphologyTower(in_features=10, out_features=16)
    classifier = MultimodalClassifier(
        {"morphology": morph, "time": TimeTower()},
        num_classes=2, hidden=32, dropout=0.0,
    )
    out = classifier({
        "morphology": torch.randn(4, 10),
        "time": torch.randn(4, 7),  # arbitrary input shape — TimeTower discards it
    })
    assert out.shape == (4, 2)
    # Both modalities present in modules.
    assert set(classifier.modalities) == {"morphology", "time"}
