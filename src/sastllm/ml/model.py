from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from lightning.pytorch.core import LightningModule


class CodeModel(LightningModule):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        l1_lambda: float = 0.0,
        class_counts: Optional[Dict[int, int]],
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_counts"])

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.2),
            nn.Linear(256, output_dim),
        )

        if class_counts is not None:
            counts = torch.tensor([class_counts[i] for i in range(output_dim)], dtype=torch.float)
            weights = 1.0 / (counts + 1e-6)
        else:
            weights = torch.ones(output_dim, dtype=torch.float)

        weights = weights / weights.sum()

        print("Using class weights:", weights)

        self.criterion = nn.CrossEntropyLoss(weight=weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor.
        Returns:
            Output tensor.
        """
        return self.layers(x)

    def _l1_on_linear_weights(self) -> torch.Tensor:
        """
        Calculate L1 regularization on linear layer weights.
        Returns:
            L1 regularization term.
        """
        if self.hparams.l1_lambda <= 0.0:  # type: ignore
            return torch.zeros(1, device=self.device).sum()  # keeps graph friendly no-op
        l1 = 0.0
        for m in self.modules():
            if isinstance(m, nn.Linear):
                l1 = l1 + m.weight.abs().sum()
        return self.hparams.l1_lambda * l1  # type: ignore

    def _shared_step(self, batch, stage: str):
        """
        Shared step for training, validation, and testing.
        Args:
            batch: Input batch.
            stage: Stage of the step ("train", "val", "test").
        Returns:
            Loss value.
        """
        _, X, y = batch
        logits = self(X)

        loss = self.criterion(logits, y.long())
        if stage == "train" and self.hparams.l1_lambda > 0:  # type: ignore
            loss = loss + self._l1_on_linear_weights()
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        """
        Training step.
        Args:
            batch: Input batch.
            batch_idx: Index of the batch.
        Returns:
            Training loss.
        """
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        Args:
            batch: Input batch.
            batch_idx: Index of the batch.
        Returns:
            Validation loss.
        """
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        """
        Test step.
        Args:
            batch: Input batch.
            batch_idx: Index of the batch.
        Returns:
            Test loss.
        """
        self._shared_step(batch, "test")

    def predict_step(self, batch, batch_idx):
        """
        Prediction step.
        Args:
            batch: Input batch.
            batch_idx: Index of the batch.
        Returns:
            Tuple of ids and predictions.
        """
        ids, X, _ = batch
        logits = self(X)
        return ids, logits

    def configure_optimizers(self):
        """
        Configure optimizers.
        Returns:
            Optimizer.
        """
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,  # type: ignore
            weight_decay=self.hparams.weight_decay,  # type: ignore
        )
        return opt
