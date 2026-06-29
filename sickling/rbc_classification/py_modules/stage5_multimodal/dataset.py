"""Multimodal dataset wrapper.

Returns ``({'image': Tensor[3,H,W], 'morphology': Tensor[F]}, label)``.
The morphology features are computed once at construction and cached in
memory — for ~40k cells this is ~40 s and ~5 MB.

Two mask-zeroing modes:

* ``zero_mask_channels=True`` zeros ch1/ch2 at the dataset level — both
  the image tensor returned to the image tower *and* the morphology
  features (because they would be derived from zeroed masks). This is the
  semantics used by the original ``- mask channels`` ablation row in
  ``DEFAULT_ABLATION``.
* ``zero_image_masks_only=True`` zeros ch1/ch2 on the image tensor only.
  The morphology features are computed from the **un-zeroed** ``.pt``
  tensors and so retain all 30 shape descriptors. This is the per-tower
  test recommended in
  ``figures/ablation/ablation_20260516_003426/discussion.tex`` §
  Limitations item 5.

The two flags are mutually exclusive; ``zero_image_masks_only=True``
takes precedence and is the recommended setting for the new ablation row.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from sickling.rbc_classification.py_modules.data.crop_dataset import CropDataset
from sickling.rbc_classification.py_modules.stage5_multimodal.morphology_features import N_FEATURES, compute_features


class MultimodalCropDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cells_df: pd.DataFrame,
        crops_dir: Path,
        target_size: int = 224,
        return_label: bool = True,
        transform: Callable | None = None,
        morphology_cache: torch.Tensor | None = None,
        zero_mask_channels: bool = False,
        zero_image_masks_only: bool = False,
        zero_cell_body_only: bool = False,
        zero_polymer_only: bool = False,
        dilate_cell_body_px: int = 0,
    ) -> None:
        n_flags = sum([
            zero_mask_channels, zero_image_masks_only,
            zero_cell_body_only, zero_polymer_only,
        ])
        if n_flags > 1:
            raise ValueError(
                "Pass at most one of zero_mask_channels / zero_image_masks_only / "
                "zero_cell_body_only / zero_polymer_only — they have different semantics."
            )

        # The inner CropDataset zeroes ch1/ch2 (or just ch1) in tensor space so
        # the image tower input is masked while the morphology cache below
        # (which reads the raw .pt FOV cache via ``_load_fov``) keeps the
        # original un-zeroed masks.
        self._inner = CropDataset(
            cells_df=cells_df,
            crops_dir=crops_dir,
            target_size=target_size,
            return_label=return_label,
            transform=transform,
            zero_mask_channels=zero_mask_channels or zero_image_masks_only,
            zero_cell_body_only=zero_cell_body_only,
            zero_polymer_only=zero_polymer_only,
            dilate_cell_body_px=dilate_cell_body_px,
        )
        self.return_label = return_label
        self.zero_mask_channels = zero_mask_channels
        self.zero_image_masks_only = zero_image_masks_only
        self.zero_cell_body_only = zero_cell_body_only
        self.zero_polymer_only = zero_polymer_only
        self.dilate_cell_body_px = int(dilate_cell_body_px)
        self.cells_df = self._inner.df
        self.crops_dir = self._inner.crops_dir

        if morphology_cache is not None:
            if morphology_cache.shape[0] != len(self.cells_df):
                raise ValueError(
                    f"morphology_cache rows ({morphology_cache.shape[0]}) != "
                    f"cells_df rows ({len(self.cells_df)})."
                )
            self.morphology = morphology_cache
        else:
            self.morphology = self._build_morphology_cache()

    @property
    def n_morphology_features(self) -> int:
        return self.morphology.shape[1]

    def _build_morphology_cache(self) -> torch.Tensor:
        feats = np.zeros((len(self.cells_df), N_FEATURES), dtype=np.float32)
        # Iterate raw 96x96 crops (pre-resize) for the morphology computation —
        # masks are stored as binary in the source .pt file, so we read them
        # directly from the inner dataset's per-FOV cache.
        for i in tqdm(range(len(self.cells_df)), desc="Caching morphology features"):
            row = self.cells_df.iloc[i]
            stem = Path(row["source_image"]).stem
            tensor = self._inner._load_fov(stem)[int(row["position"])]  # (3, 96, 96)
            feats[i] = compute_features(tensor)
        return torch.from_numpy(feats)

    def __len__(self) -> int:
        return len(self.cells_df)

    def __getitem__(self, idx: int):
        item = self._inner[idx]
        morph = self.morphology[idx]
        if self.return_label:
            image, label = item
            return {"image": image, "morphology": morph}, label
        return {"image": item, "morphology": morph}
