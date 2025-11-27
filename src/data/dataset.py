"""Stanford Dogs dataset loader."""

import os
from pathlib import Path
from typing import Optional, Callable, Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image
import json


class StanfordDogsDataset(Dataset):
    """
    Stanford Dogs dataset loader.

    The dataset structure is expected to be:
    dataset_path/
    ├── Images/
    │   ├── n02085620-Chihuahua/
    │   ├── n02085782-Japanese_spaniel/
    │   └── ...
    └── annotations/
    """

    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        train_ratio: float = 0.8,
    ):
        """
        Initialize Stanford Dogs dataset.

        Args:
            dataset_path: Path to the downloaded dataset
            split: 'train' or 'val'
            transform: Optional transform to apply to images
            train_ratio: Ratio of data to use for training (default: 0.8)
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.transform = transform
        self.train_ratio = train_ratio

        # Find images directory
        images_dir = self.dataset_path / "Images"
        if not images_dir.exists():
            # Try alternative structure
            images_dir = self.dataset_path / "stanford-dogs-dataset" / "Images"
            if not images_dir.exists():
                raise ValueError(
                    f"Could not find Images directory in {dataset_path}"
                )

        # Load class names and images
        self.classes = sorted(
            [d.name for d in images_dir.iterdir() if d.is_dir()]
        )
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}

        # Collect all image paths with labels
        self.samples = []
        for class_name in self.classes:
            class_dir = images_dir / class_name
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            for img_path in images:
                self.samples.append((img_path, self.class_to_idx[class_name]))

        # Split into train/val
        torch.manual_seed(42)
        indices = torch.randperm(len(self.samples)).tolist()
        split_idx = int(len(self.samples) * train_ratio)

        if split == "train":
            self.samples = [self.samples[i] for i in indices[:split_idx]]
        else:
            self.samples = [self.samples[i] for i in indices[split_idx:]]

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item by index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image, label)
        """
        img_path, label = self.samples[idx]

        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Error loading image {img_path}: {e}")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_num_classes(self) -> int:
        """Return number of classes."""
        return len(self.classes)

    def get_class_names(self) -> list[str]:
        """Return list of class names."""
        return self.classes

