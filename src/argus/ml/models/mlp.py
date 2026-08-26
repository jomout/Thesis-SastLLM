from __future__ import annotations

import torch
from torch import nn

from .base import RepositoryClassifierModule


class MLPRepositoryClassifier(RepositoryClassifierModule):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.2,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        l1_lambda: float = 0.0,
        class_counts: dict[int, int] | None = None,
        use_class_weights: bool = False,
    ) -> None:
        super().__init__(
            output_dim=output_dim,
            lr=lr,
            weight_decay=weight_decay,
            l1_lambda=l1_lambda,
            class_counts=class_counts,
            use_class_weights=use_class_weights,
        )
        self.save_hyperparameters(ignore=["class_counts"])
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(p=dropout),
                ]
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
