from __future__ import annotations

from abc import abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from lightning.pytorch.core import LightningModule


class RepositoryClassifierModule(LightningModule):
    """Base Lightning module for repository classifiers."""

    def __init__(
        self,
        *,
        output_dim: int,
        lr: float,
        weight_decay: float,
        l1_lambda: float,
        class_counts: Optional[dict[int, int]],
        use_class_weights: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["class_counts"])
        self.output_dim = output_dim

        criterion_weight: torch.Tensor | None = None
        if use_class_weights and class_counts is not None:
            counts = torch.tensor([class_counts.get(i, 0) for i in range(output_dim)], dtype=torch.float)
            weights = 1.0 / (counts + 1e-6)
            weights = weights / weights.sum()
            criterion_weight = weights
        else:
            weights = torch.ones(output_dim, dtype=torch.float)
        self.register_buffer("class_weights", weights)
        self.criterion = nn.CrossEntropyLoss(weight=criterion_weight)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def predict_step(self, batch, batch_idx):
        ids, features, _ = batch
        return ids, self(features)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,  # type: ignore[attr-defined]
            weight_decay=self.hparams.weight_decay,  # type: ignore[attr-defined]
        )

    def _shared_step(self, batch, stage: str):
        _, features, labels = batch
        logits = self(features)
        loss = self.criterion(logits, labels.long())
        if stage == "train" and self.hparams.l1_lambda > 0:  # type: ignore[attr-defined]
            loss = loss + self._linear_l1_penalty()
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == labels).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", accuracy, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def _linear_l1_penalty(self) -> torch.Tensor:
        penalty = torch.zeros((), device=self.device)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                penalty = penalty + module.weight.abs().sum()
        return self.hparams.l1_lambda * penalty  # type: ignore[attr-defined]
