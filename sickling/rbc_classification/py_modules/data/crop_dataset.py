"""Dataset that serves per-cell 3-channel tensors from per-FOV ``.pt`` files.

One ``CropDataset`` instance is reusable across milestones — it's the input
to the linear-probe, fine-tune, MAE pretraining, and (later) multimodal
fusion training.

Storage contract (set in milestone 3):
    crops/<source_stem>.pt = {'tensors': float32[N, 3, 96, 96],
                              'instance_ids': int32[N]}
    cells.parquet rows have ``source_image, instance_id, position`` so the
    Dataset can map (source_image, position) -> tensor in O(1).
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from sickling.rbc_classification.py_modules.config import Config
from sickling.rbc_classification.py_modules.io.parquet import read_cells

LABEL_TO_INT: dict[str, int] = {"non_sickle": 0, "sickle": 1}


def _dilate_binary(mask: torch.Tensor, px: int) -> torch.Tensor:
    """Binary dilation by ``px`` pixels via max-pool. Operates on a (H, W) tensor."""
    if px <= 0:
        return mask
    k = 2 * px + 1
    x = mask.float().unsqueeze(0).unsqueeze(0)
    out = F.max_pool2d(x, kernel_size=k, stride=1, padding=px)
    return out.squeeze(0).squeeze(0)


def _resize_3channel(t: torch.Tensor, target_size: int) -> torch.Tensor:
    """Resize a (3, H, W) tensor: ch0 bilinear (greyscale), ch1/ch2 nearest (binary)."""
    if t.shape[-1] == target_size and t.shape[-2] == target_size:
        return t
    ch0 = F.interpolate(
        t[0:1].unsqueeze(0), size=target_size, mode="bilinear", align_corners=False
    ).squeeze(0)
    chmasks = F.interpolate(
        t[1:].unsqueeze(0), size=target_size, mode="nearest"
    ).squeeze(0)
    return torch.cat([ch0, chmasks], dim=0)


class CropDataset(Dataset):
    """Backed by ``cells.parquet`` + per-FOV ``.pt`` files.

    Args:
        cells_df: pre-filtered cells DataFrame (caller chooses labeled-only,
            ambiguous excluded, etc.).
        crops_dir: directory containing ``<stem>.pt`` files.
        target_size: output spatial size (default 224 for ViTs).
        return_label: if True, ``__getitem__`` returns ``(tensor, label_int)``;
            else just ``tensor``. Caller must ensure labels are present when
            ``return_label=True``.
        transform: optional callable applied to the (3, target_size, target_size)
            tensor *after* resize.
        label_to_int: override the default ``{'non_sickle': 0, 'sickle': 1}`` map.
    """

    def __init__(
        self,
        cells_df: pd.DataFrame,
        crops_dir: Path,
        target_size: int = 224,
        return_label: bool = False,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        label_to_int: dict[str, int] | None = None,
        zero_mask_channels: bool = False,
        zero_cell_body_only: bool = False,
        zero_polymer_only: bool = False,
        dilate_cell_body_px: int = 0,
    ) -> None:
        if "position" not in cells_df.columns:
            raise ValueError("cells_df must contain a 'position' column.")
        n_zero_flags = sum([zero_mask_channels, zero_cell_body_only, zero_polymer_only])
        if n_zero_flags > 1:
            raise ValueError(
                "Pass at most one of zero_mask_channels / zero_cell_body_only / zero_polymer_only."
            )
        self.df = cells_df.reset_index(drop=True)
        self.crops_dir = Path(crops_dir)
        self.target_size = target_size
        self.return_label = return_label
        self.transform = transform
        self.label_to_int = label_to_int or LABEL_TO_INT
        self.zero_mask_channels = zero_mask_channels
        self.zero_cell_body_only = zero_cell_body_only
        self.zero_polymer_only = zero_polymer_only
        self.dilate_cell_body_px = int(dilate_cell_body_px)
        self._fov_cache: dict[str, torch.Tensor] = {}  # stem -> (N, 3, 96, 96) tensor

    def __len__(self) -> int:
        return len(self.df)

    def _load_fov(self, stem: str) -> torch.Tensor:
        if stem not in self._fov_cache:
            obj = torch.load(self.crops_dir / f"{stem}.pt", weights_only=True)
            self._fov_cache[stem] = obj["tensors"]
        return self._fov_cache[stem]

    def _stem(self, source_image: str) -> str:
        return Path(source_image).stem

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        stem = self._stem(row["source_image"])
        position = int(row["position"])
        tensor = self._load_fov(stem)[position]

        # Dilate ch1 (cell-body mask) in the original 96x96 source space so the
        # boundary region is included. Done before resize: resize uses nearest
        # for masks, so dilation amount stays meaningful in source pixels.
        if self.dilate_cell_body_px > 0:
            tensor = tensor.clone()
            tensor[1] = _dilate_binary(tensor[1], self.dilate_cell_body_px)

        tensor = _resize_3channel(tensor, self.target_size)

        if self.zero_mask_channels:
            tensor = tensor.clone()
            tensor[1] = 0
            tensor[2] = 0
        elif self.zero_cell_body_only:
            tensor = tensor.clone()
            tensor[1] = 0  # cell_body mask zeroed; ch2 polymer mask kept
        elif self.zero_polymer_only:
            tensor = tensor.clone()
            tensor[2] = 0  # polymer mask zeroed; ch1 cell_body mask kept

        if self.transform is not None:
            tensor = self.transform(tensor)

        if not self.return_label:
            return tensor
        label = self.label_to_int[row["label"]]
        return tensor, label


def labeled_subset(cells_df: pd.DataFrame, exclude_ambiguous: bool = True) -> pd.DataFrame:
    """Return rows where ``has_label == True`` (and optionally drop ``ambiguous``)."""
    df = cells_df[cells_df["has_label"].fillna(False).astype(bool)]
    if exclude_ambiguous:
        df = df[df["label"] != "ambiguous"]
    return df.reset_index(drop=True)


def build_dataset(
    cfg: Config,
    *,
    only_labeled: bool,
    exclude_ambiguous: bool = True,
    transform: Callable | None = None,
    cells_df: pd.DataFrame | None = None,
) -> CropDataset:
    """Convenience constructor that reads cells.parquet and applies the
    standard label / ambiguous filters."""
    paths = cfg.paths.resolved()
    df = cells_df if cells_df is not None else read_cells(paths.root / cfg.paths.cells_parquet)
    if only_labeled:
        df = labeled_subset(df, exclude_ambiguous=exclude_ambiguous)
    return CropDataset(
        cells_df=df,
        crops_dir=paths.crops,
        target_size=cfg.crop.resize_to_vit,
        return_label=only_labeled,
        transform=transform,
    )
