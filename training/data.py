"""Dataset and augmentation pipeline for CRC H&E tiles."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TileDataset(Dataset):
    def __init__(self, manifest, transform=None, target="multiclass"):
        self.df = pd.read_csv(manifest) if isinstance(manifest, (str, Path)) else manifest
        self.df = self.df.reset_index(drop=True)
        self.transform = transform
        self.target = target

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        y = int(row["is_tumor"] if self.target == "binary" else row["label_idx"])
        return img, y


class RandomRot90:
    """Histology tiles have no canonical orientation -- rotate freely."""

    def __call__(self, img):
        return img.rotate(90 * int(torch.randint(0, 4, (1,))), expand=False)


def build_transforms(train: bool, size: int = 224):
    if not train:
        return transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    # Heavy color jitter is the main defense against stain/scanner shift between
    # the NCT training cohort and any new lab's slides.
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.9, 1.11)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            RandomRot90(),
            transforms.ColorJitter(brightness=0.25, contrast=0.25,
                                   saturation=0.25, hue=0.08),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
        ]
    )


def make_loader(manifest, train, batch_size=128, size=224, workers=8,
                target="multiclass", shuffle=None):
    ds = TileDataset(manifest, build_transforms(train, size), target=target)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train if shuffle is None else shuffle,
        num_workers=workers,
        pin_memory=True,
        drop_last=train,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )


def class_weights(manifest, target="multiclass") -> torch.Tensor:
    df = pd.read_csv(manifest)
    col = "is_tumor" if target == "binary" else "label_idx"
    n = df[col].value_counts().sort_index().to_numpy()
    w = n.sum() / (len(n) * n)
    return torch.tensor(w, dtype=torch.float32)
