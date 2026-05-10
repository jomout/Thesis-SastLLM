from __future__ import annotations

from typing import Optional

from .base import RepositoryClassifierModule
from .mlp import MLPRepositoryClassifier


def build_model(
    *,
    name: str,
    input_dim: int,
    output_dim: int,
    lr: float,
    weight_decay: float,
    l1_lambda: float,
    class_counts: Optional[dict[int, int]],
    hidden_dims: tuple[int, ...] = (512, 256),
    dropout: float = 0.2,
) -> RepositoryClassifierModule:
    if name == "mlp":
        return MLPRepositoryClassifier(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            l1_lambda=l1_lambda,
            class_counts=class_counts,
        )
    raise ValueError(f"Unsupported repository classifier model: {name!r}.")
