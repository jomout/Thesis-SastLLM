from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from sastllm.configs.logging_config import get_logger
from sastllm.utils.repository_encoder import BINARY_LABEL_MAP

logger = get_logger(__name__)


class CodeDataset(Dataset):
    """
    A PyTorch Dataset for tabular **classification** data.

    Stores features `X` and integer class targets `y` as tensors.

    Args:
        X: Feature tensor of shape (N, D).
        y: Target tensor of shape (N,) with integer class labels, or (N, C) for one-hot.

    Raises:
        AssertionError: If `X` is not 2D; if `y` is not 1D or 2D; or if the first
                        dimensions of `X` and `y` differ.
    """

    def __init__(self, ids: torch.Tensor, X: torch.Tensor, y: torch.Tensor):
        """
        Initializes the dataset with features and targets.

        Args:
            ids (torch.Tensor): Tensor of sample identifiers of shape (N,).
            X (torch.Tensor): The feature tensor of shape (N, D).
            y (torch.Tensor): The target tensor of shape (N,) or (N, C).

        Raises:
            AssertionError: If `X` is not 2D, `y` is not 1D or 2D, or the first
                            dimensions of `X` and `y` do not match.
        """
        assert X.ndim == 2, "X must be (N, D)"
        assert y.ndim in (1, 2), "y must be (N,) or (N, C)"
        assert X.size(0) == y.size(0), "X and y must share first dim"
        assert ids.shape[0] == X.size(0)
        self.ids = ids
        self.X = X
        self.y = y

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return self.X.size(0)

    def __getitem__(self, idx: int):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the feature tensor and target tensor for the
                   specified index.
        """
        # print(f"Fetching item at index: {idx}, ID: {self.ids[idx]}, Label: {self.y[idx]}")
        return self.ids[idx], self.X[idx], self.y[idx]


class CodeDataModule(LightningDataModule):
    """
    A PyTorch Lightning DataModule that handles dataset splitting and
    DataLoader creation for training, validation, and testing.

    This module wraps around a provided dataset and automatically splits it
    into train/validation/test sets based on given ratios. It also ensures
    reproducibility by using a fixed random seed for dataset shuffling.

    Parameters
    ----------
    train_dataset : CodeDataset
        The dataset to be used for training.
    validation_dataset : CodeDataset
        The dataset to be used for validation.
    test_dataset : CodeDataset
        The dataset to be used for testing.
    batch_size : int, optional, default=128
        Number of samples per batch for all dataloaders.
    num_workers : int, optional, default=1
        Number of subprocesses to use for data loading.
    pin_memory : bool, optional
        Whether to pin memory for faster GPU transfers.
        Defaults to ``True`` if device is "cuda", otherwise ``False``.
    random_seed : int, optional, default=42
        Seed used to make dataset splitting reproducible.
    device : str, optional, default="cpu"
        Device type (e.g., "cpu" or "cuda") used for deciding default pin_memory.
    """

    def __init__(
        self,
        full_dataset: CodeDataset,
        train_indices: List[int],
        val_indices: List[int],
        test_indices: List[int],
        batch_size: int = 128,
        num_workers: int = 1,
        pin_memory: Optional[bool] = None,
        random_seed: int = 42,
        device: str = "cpu",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory if pin_memory is not None else (device == "cuda")
        self.random_seed = random_seed

        self.full_dataset = full_dataset
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Sets up the datasets for training, validation, and testing.

        This method is called by Lightning at the beginning of training,
        validation, or testing. It splits the full dataset into train,
        validation, and test sets based on provided indices.

        Parameters
        ----------
        stage : Literal["fit", "validate", "test", "predict"], optional
            Stage of setup ("fit", "validate", "test", or "predict").
            Not used in this implementation.
        """

        self.train_ds = torch.utils.data.Subset(self.full_dataset, self.train_indices)
        self.val_ds = torch.utils.data.Subset(self.full_dataset, self.val_indices)
        self.test_ds = torch.utils.data.Subset(self.full_dataset, self.test_indices)

        logger.info(f"Train dataset size: {len(self.train_ds)}")
        logger.info(f"Validation dataset size: {len(self.val_ds)}")
        logger.info(f"Test dataset size: {len(self.test_ds)}")

        #! For testing purposes, train on a balanced subset of malicious and benign repositories
        # self.train_ds = self._balance_dataset(self.train_ds)
        # self.val_ds = self._balance_dataset(self.val_ds)
        # self.test_ds = self._balance_dataset(self.test_ds)

    def train_dataloader(self) -> DataLoader:
        """
        Create the training DataLoader.

        Returns
        -------
        torch.utils.data.DataLoader
            DataLoader wrapping the training dataset.
        """
        num_classes = int(self.full_dataset.y.max().item()) + 1
        print(f"Dataset: Number of classes: {num_classes}")
        sampler = make_sampler(self.train_ds, num_classes=num_classes)

        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create the validation DataLoader.

        Returns
        -------
        torch.utils.data.DataLoader
            DataLoader wrapping the validation dataset.
        """
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Create the test DataLoader.

        Returns
        -------
        torch.utils.data.DataLoader
            DataLoader wrapping the test dataset.
        """
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=False,
            # sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


def make_sampler(dataset, num_classes: int):
    # Pull integer class labels from dataset
    labels = np.array([dataset[i][2].item() for i in range(len(dataset))])

    class_counts = np.bincount(labels, minlength=num_classes)

    # guard against zero-count classes
    class_weights = 1.0 / (class_counts + 1e-6)

    sample_weights = class_weights[labels]

    # Normalize weights
    sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)

    return WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(sample_weights),
        replacement=True,
    )


def balance_dataset(data: CodeDataset) -> CodeDataset:
    """
    For testing purposes, balance the dataset to have equal number of benign and malicious
    repositories
    """
    # Identify indices for each class
    label_to_index = {v: k for k, v in BINARY_LABEL_MAP.items()}
    benign_idx = label_to_index.get("benign")
    malicious_idx = label_to_index.get("malicious")

    if benign_idx is None or malicious_idx is None:
        logger.warning("Cannot balance dataset: 'benign' or 'malicious' label not found.")
        return data

    benign_indices = (data.y == benign_idx).nonzero(as_tuple=True)[0]
    malicious_indices = (data.y == malicious_idx).nonzero(as_tuple=True)[0]

    # Determine the smaller class size
    min_size = min(len(benign_indices), len(malicious_indices))
    if min_size == 0:
        logger.warning("Cannot balance dataset: one of the classes has zero samples.")
        return data

    # Randomly sample from each class to create a balanced dataset
    sampled_benign = benign_indices[torch.randperm(len(benign_indices))[:min_size]]
    sampled_malicious = malicious_indices[torch.randperm(len(malicious_indices))[:min_size]]

    balanced_indices = torch.cat([sampled_benign, sampled_malicious])
    balanced_X = data.X[balanced_indices]
    balanced_y = data.y[balanced_indices]
    balanced_ids = data.ids[balanced_indices.numpy()]

    logger.info(f"Balanced dataset to {min_size} samples per class.")

    return CodeDataset(ids=balanced_ids, X=balanced_X, y=balanced_y)
