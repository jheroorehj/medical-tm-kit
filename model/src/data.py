"""RFMiD dataset, fundus-specific preprocessing, and transform pipelines."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .labels import ALL_LABELS

HF_DATASET_REPO = "ctmedtech/RFMID"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class SplitSpec:
    """Where one official split lives, both on HF and on disk."""

    name: str
    csv: str
    images: str

    @property
    def local_csv(self) -> str:
        return self.csv

    @property
    def local_images(self) -> str:
        return self.images


SPLITS = {
    "train": SplitSpec(
        "train",
        "Training_Set/Training_Set/RFMiD_Training_Labels.csv",
        "Training_Set/Training_Set/Training",
    ),
    "val": SplitSpec(
        "val",
        "Evaluation_Set/Evaluation_Set/RFMiD_Validation_Labels.csv",
        "Evaluation_Set/Evaluation_Set/Validation",
    ),
    "test": SplitSpec(
        "test",
        "Test_Set/Test_Set/RFMiD_Testing_Labels.csv",
        "Test_Set/Test_Set/Test",
    ),
}


# --------------------------------------------------------------------------
# Fundus preprocessing
# --------------------------------------------------------------------------

def crop_fov(img: Image.Image, tol: int = 12) -> Image.Image:
    """Crop the black surround off a fundus photo, keeping the circular FOV.

    RFMiD mixes three cameras with different aspect ratios and FOV sizes, so
    the raw letterboxing varies a lot between images. Cropping to the retina
    normalises scale across cameras before the resize.
    """
    arr = np.asarray(img.convert("L"))
    mask = arr > tol
    if not mask.any():
        return img
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1
    if bottom - top < 32 or right - left < 32:
        return img
    return img.crop((left, top, right, bottom))


def pad_to_square(img: Image.Image, fill: int = 0) -> Image.Image:
    """Letterbox to a square so the resize does not distort the retina."""
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    canvas = Image.new(img.mode, (side, side), fill)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


class FundusPrepare:
    """crop FOV -> square pad. Deterministic, shared by train and eval."""

    def __init__(self, fov_crop: bool = True, square_pad: bool = True, tol: int = 12):
        self.fov_crop = fov_crop
        self.square_pad = square_pad
        self.tol = tol

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.fov_crop:
            img = crop_fov(img, self.tol)
        if self.square_pad:
            img = pad_to_square(img)
        return img


def build_transforms(img_size: int = 224, train: bool = False, fov_crop: bool = True):
    prepare = FundusPrepare(fov_crop=fov_crop)
    if train:
        return T.Compose([
            prepare,
            T.RandomResizedCrop(img_size, scale=(0.75, 1.0), ratio=(0.9, 1.111)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomApply([T.RandomRotation(20)], p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            T.RandomErasing(p=0.25, scale=(0.02, 0.1)),
        ])
    return T.Compose([
        prepare,
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

class RFMiDDataset(Dataset):
    """Multi-label RFMiD split.

    Returns ``(image, target, image_id)`` where ``target`` is a 46-dim float
    vector ordered as :data:`src.labels.ALL_LABELS`.
    """

    def __init__(
        self,
        root: str | os.PathLike,
        split: str,
        img_size: int = 224,
        train: bool = False,
        fov_crop: bool = True,
        transform=None,
    ):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {list(SPLITS)}, got {split!r}")
        spec = SPLITS[split]
        self.root = Path(root)
        self.split = split
        self.images_dir = self.root / spec.local_images
        csv_path = self.root / spec.local_csv
        if not csv_path.exists():
            raise FileNotFoundError(
                f"{csv_path} not found. Run: python -m scripts.download_data --root {self.root}"
            )
        self.df = pd.read_csv(csv_path)
        missing = [c for c in ALL_LABELS if c not in self.df.columns]
        if missing:
            raise ValueError(f"{csv_path} is missing label columns: {missing}")
        self.targets = self.df[ALL_LABELS].to_numpy(dtype=np.float32)
        self.ids = self.df["ID"].astype(str).tolist()
        self.transform = transform or build_transforms(img_size, train=train, fov_crop=fov_crop)

    def __len__(self) -> int:
        return len(self.df)

    def _path(self, image_id: str) -> Path:
        for ext in (".png", ".PNG", ".jpg", ".jpeg"):
            p = self.images_dir / f"{image_id}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"no image for ID={image_id} under {self.images_dir}")

    def __getitem__(self, idx: int):
        image_id = self.ids[idx]
        img = Image.open(self._path(image_id)).convert("RGB")
        x = self.transform(img)
        y = torch.from_numpy(self.targets[idx])
        return x, y, image_id

    def pos_counts(self) -> np.ndarray:
        """Positives per class — used for pos_weight and for reporting."""
        return self.targets.sum(axis=0)


def download_rfmid(root: str | os.PathLike, splits=("train", "val", "test")) -> Path:
    """Pull the official RFMiD splits from the Hugging Face Hub into ``root``."""
    from huggingface_hub import snapshot_download

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    patterns: list[str] = []
    for name in splits:
        spec = SPLITS[name]
        patterns.append(spec.csv)
        patterns.append(f"{spec.images}/*")
    snapshot_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        local_dir=str(root),
        allow_patterns=patterns,
    )
    return root
