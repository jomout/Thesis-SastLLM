from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Optional

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from sastllm.configs.logging_config import get_logger

logger = get_logger(__name__)


class RepositoryTensorDataset(Dataset):
    """Tensor dataset for repository-level classification."""

    def __init__(self, *, ids: torch.Tensor, X: torch.Tensor, y: torch.Tensor) -> None:
        if X.ndim not in (2, 3):
            raise ValueError("X must have shape (N, D) or (N, T, D).")
        if y.ndim != 1:
            raise ValueError("y must have shape (N,).")
        if ids.ndim != 1:
            raise ValueError("ids must have shape (N,).")
        if X.size(0) != y.size(0) or ids.size(0) != X.size(0):
            raise ValueError("ids, X, and y must share the same first dimension.")
        self.ids = ids
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return int(self.X.size(0))

    def __getitem__(self, index: int):
        return self.ids[index], self.X[index], self.y[index]


class RepositoryDataModule(LightningDataModule):
    def __init__(
        self,
        *,
        dataset: RepositoryTensorDataset,
        train_indices: Sequence[int],
        val_indices: Sequence[int],
        test_indices: Sequence[int],
        batch_size: int,
        num_workers: int = 1,
        pin_memory: Optional[bool] = None,
        use_weighted_sampler: bool = True,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.train_indices = list(train_indices)
        self.val_indices = list(val_indices)
        self.test_indices = list(test_indices)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory if pin_memory is not None else torch.cuda.is_available()
        self.use_weighted_sampler = use_weighted_sampler

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_ds = torch.utils.data.Subset(self.dataset, self.train_indices)
        self.val_ds = torch.utils.data.Subset(self.dataset, self.val_indices)
        self.test_ds = torch.utils.data.Subset(self.dataset, self.test_indices)
        logger.info("Train dataset size: %s", len(self.train_ds))
        logger.info("Validation dataset size: %s", len(self.val_ds))
        logger.info("Test dataset size: %s", len(self.test_ds))

    def train_dataloader(self) -> DataLoader:
        sampler = make_weighted_sampler(self.train_ds, num_classes=self.num_classes) if self.use_weighted_sampler and len(self.train_ds) else None
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds)

    def predict_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds)

    def dataloader_for_split(self, split: Literal["train", "val", "test"]) -> DataLoader:
        self.setup()
        if split == "train":
            return self._loader(self.train_ds)
        if split == "test":
            return self._loader(self.test_ds)
        if split == "val":
            return self._loader(self.val_ds)
        raise ValueError("split must be one of: train, val, test.")

    @property
    def num_classes(self) -> int:
        valid_labels = self.dataset.y[self.dataset.y >= 0]
        return int(valid_labels.max().item()) + 1 if len(valid_labels) else 0

    def _loader(self, dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


def make_weighted_sampler(dataset, *, num_classes: int) -> WeightedRandomSampler:
    labels = np.array([int(dataset[index][2].item()) for index in range(len(dataset))])
    class_counts = np.bincount(labels, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[labels]
    sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
    return WeightedRandomSampler(weights=sample_weights.tolist(), num_samples=len(sample_weights), replacement=True)
