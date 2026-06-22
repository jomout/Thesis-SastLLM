from __future__ import annotations

from typing import Literal, Optional

from .base import RepositoryClassifierModule
from .lstm import LSTMRepositoryClassifier
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
    use_class_weights: bool = True,
    hidden_dims: tuple[int, ...] = (512, 256),
    embedding_dim: int = 128,
    hidden_dim: int = 128,
    num_layers: int = 1,
    dropout: float = 0.2,
    bidirectional: bool = False,
    pooling: Literal["last", "mean", "max"] = "last",
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
            use_class_weights=use_class_weights,
        )
    if name == "lstm":
        return LSTMRepositoryClassifier(
            input_dim=input_dim,
            output_dim=output_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            pooling=pooling,
            lr=lr,
            weight_decay=weight_decay,
            l1_lambda=l1_lambda,
            class_counts=class_counts,
            use_class_weights=use_class_weights,
        )
    raise ValueError(f"Unsupported repository classifier model: {name!r}.")
