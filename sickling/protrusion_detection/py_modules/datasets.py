"""PyTorch datasets for whole-image and sparse tile training data.

Both datasets share preprocessing:
- Image loaded as float32 grayscale, then ``masks.normalize_image`` (single
  source of truth; identical at train and inference).
- Masks loaded via the caller-supplied ``mask_loader`` so the dataset stays
  agnostic about 1-based vs 0-based on-disk conventions.
- Augmentations: hflip / vflip / 90-deg rotations.
- Crop sampling: ``sampler.sample_crop`` with ``cfg.TARGET_CROP_PROB`` over
  ``cfg.TARGET_CLASSES`` (inverse-frequency weighting).

For ``MicroscopyDataset`` the per-epoch length is decoupled from the number
of source images (``BATCH_SIZE * STEPS_PER_EPOCH``) so a few whole images
still yield many crops.

For ``TileDataset`` each painted 512-px tile contributes one random 256-px
crop per __getitem__ call.
"""

from __future__ import annotations

import random
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .config import cfg
from .masks import load_robust_h5, normalize_image
from .sampler import CenterIndex, build_center_index, sample_crop


FilePair = Tuple[str, str]
MaskLoader = Callable[[str], np.ndarray]


def _augment(img_t: torch.Tensor, mask_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """hflip/vflip/{0,90,180,270}-deg rotation. Mask uses NEAREST."""
    if random.random() > 0.5:
        img_t, mask_t = TF.hflip(img_t), TF.hflip(mask_t)
    if random.random() > 0.5:
        img_t, mask_t = TF.vflip(img_t), TF.vflip(mask_t)
    rot = random.choice([0, 90, 180, 270])
    if rot:
        img_t = TF.rotate(img_t, rot)
        mask_t = TF.rotate(mask_t, rot, interpolation=transforms.InterpolationMode.NEAREST)
    return img_t, mask_t


class MicroscopyDataset(Dataset):
    """Whole-image dataset. Use for InitialLabels and BootstrappedLabels.

    Parameters
    ----------
    file_pairs:
        Iterable of ``(raw_image_path, mask_path)`` pairs.
    is_train:
        If True, ``__getitem__`` samples random crops (with augmentation) and
        ``__len__`` reports ``BATCH_SIZE * STEPS_PER_EPOCH``. If False, returns
        the full image + full mask one at a time for sliding-window eval.
    mask_loader:
        Function turning ``mask_path`` into a 2D int mask in the training
        convention (0..N-1 valid, 255 = ignore).
    tile_size:
        Crop size in train mode. Defaults to ``cfg.TILE_SIZE``.
    """

    def __init__(
        self,
        file_pairs: Sequence[FilePair],
        is_train: bool = True,
        mask_loader: MaskLoader = load_robust_h5,
        tile_size: Optional[int] = None,
    ) -> None:
        self.is_train = is_train
        self.tile_size = tile_size or cfg.TILE_SIZE
        self.images: List[np.ndarray] = []
        self.masks: List[np.ndarray] = []
        self.centers: List[CenterIndex] = []

        for img_path, mask_path in tqdm(list(file_pairs), desc="Loading whole-image data"):
            img = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
            img = normalize_image(img)
            mask = mask_loader(mask_path).astype(np.int64)

            self.images.append(img)
            self.masks.append(mask)
            self.centers.append(
                build_center_index(mask, cfg.TARGET_CLASSES, self.tile_size)
                if is_train else
                CenterIndex(per_class={}, inv_freq_weights={})
            )

    def __len__(self) -> int:
        return cfg.BATCH_SIZE * cfg.STEPS_PER_EPOCH if self.is_train else len(self.images)

    def __getitem__(self, idx: int):
        if self.is_train:
            i = random.randint(0, len(self.images) - 1)
            img, mask, ci = self.images[i], self.masks[i], self.centers[i]
            h, w = img.shape
            top, left = sample_crop(h, w, ci, self.tile_size, cfg.TARGET_CROP_PROB)
            img_crop = img[top:top + self.tile_size, left:left + self.tile_size]
            mask_crop = mask[top:top + self.tile_size, left:left + self.tile_size]

            img_t = torch.from_numpy(img_crop).float().unsqueeze(0)
            mask_t = torch.from_numpy(mask_crop).float().unsqueeze(0)
            img_t, mask_t = _augment(img_t, mask_t)
            return img_t, mask_t.long().squeeze(0)

        img = torch.from_numpy(self.images[idx]).float().unsqueeze(0)
        mask = torch.from_numpy(self.masks[idx]).long()
        return img, mask


class TileDataset(Dataset):
    """Sparse-tile dataset for CorrectedTiles.

    Each tile is a painted ``CORRECTION_TILE_SIZE``-px crop with partial
    labels (untouched = ``cfg.IGNORE_INDEX``). At train time we draw a single
    ``TILE_SIZE``-px sub-crop from a random tile per ``__getitem__`` call,
    biasing the sub-crop toward target-class pixels exactly as for whole
    images.

    Tiles smaller than ``TILE_SIZE`` are padded with the ignore-index so the
    sampler still produces a fixed-shape output.
    """

    def __init__(
        self,
        tile_pairs: Sequence[FilePair],
        mask_loader: MaskLoader,
        tile_size: Optional[int] = None,
    ) -> None:
        self.tile_size = tile_size or cfg.TILE_SIZE
        self.images: List[np.ndarray] = []
        self.masks: List[np.ndarray] = []
        self.centers: List[CenterIndex] = []

        for img_path, mask_path in tqdm(list(tile_pairs), desc="Loading tile data"):
            img = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
            img = normalize_image(img)
            mask = mask_loader(mask_path).astype(np.int64)

            img, mask = self._pad_to_tile(img, mask)
            self.images.append(img)
            self.masks.append(mask)
            self.centers.append(build_center_index(mask, cfg.TARGET_CLASSES, self.tile_size))

    def _pad_to_tile(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = img.shape
        pad_h = max(0, self.tile_size - h)
        pad_w = max(0, self.tile_size - w)
        if pad_h or pad_w:
            img = np.pad(img, ((0, pad_h), (0, pad_w)), mode="reflect")
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant",
                          constant_values=cfg.IGNORE_INDEX)
        return img, mask

    def __len__(self) -> int:
        # Train-time only — match the whole-image dataset's per-epoch length so
        # ConcatDataset behaves uniformly.
        return cfg.BATCH_SIZE * cfg.STEPS_PER_EPOCH

    def __getitem__(self, idx: int):
        i = random.randint(0, len(self.images) - 1)
        img, mask, ci = self.images[i], self.masks[i], self.centers[i]
        h, w = img.shape
        top, left = sample_crop(h, w, ci, self.tile_size, cfg.TARGET_CROP_PROB)
        img_crop = img[top:top + self.tile_size, left:left + self.tile_size]
        mask_crop = mask[top:top + self.tile_size, left:left + self.tile_size]

        img_t = torch.from_numpy(img_crop).float().unsqueeze(0)
        mask_t = torch.from_numpy(mask_crop).float().unsqueeze(0)
        img_t, mask_t = _augment(img_t, mask_t)
        return img_t, mask_t.long().squeeze(0)
