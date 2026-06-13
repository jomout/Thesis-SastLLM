from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn

from .base import RepositoryClassifierModule


class LSTMRepositoryClassifier(RepositoryClassifierModule):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
        pooling: Literal["last", "mean", "max"] = "last",
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        l1_lambda: float = 0.0,
        class_counts: Optional[dict[int, int]] = None,
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
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0.")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0.")
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0.")
        if pooling not in {"last", "mean", "max"}:
            raise ValueError("pooling must be one of: 'last', 'mean', 'max'.")

        self.save_hyperparameters(ignore=["class_counts"])
        self.pooling = pooling
        self.embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        output_width = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(output_width, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            valid_steps = x != 0
            sequence = self.embedding(x.long())
        elif x.ndim == 3:
            valid_steps = x.abs().sum(dim=2) > 0
            sequence = x.float()
            if sequence.size(2) != self.lstm.input_size:
                raise ValueError(
                    f"LSTMRepositoryClassifier received vector input width {sequence.size(2)}, "
                    f"but the LSTM input width is {self.lstm.input_size}. Use token sequences or set embedding_dim accordingly."
                )
        else:
            raise ValueError(f"LSTMRepositoryClassifier expects input shape (N, T) or (N, T, D), got {tuple(x.shape)}.")
        outputs, _ = self.lstm(sequence)
        pooled = self._pool(outputs, valid_steps)
        return self.classifier(self.dropout(pooled))

    def _pool(self, outputs: torch.Tensor, valid_steps: torch.Tensor) -> torch.Tensor:
        raw_lengths = valid_steps.sum(dim=1)
        lengths = raw_lengths.clamp(min=1)
        has_valid_steps = raw_lengths > 0

        if self.pooling == "last":
            indexes = (lengths - 1).view(-1, 1, 1).expand(-1, 1, outputs.size(2))
            pooled = outputs.gather(dim=1, index=indexes).squeeze(1)
            return torch.where(has_valid_steps.unsqueeze(1), pooled, torch.zeros_like(pooled))

        mask = valid_steps.unsqueeze(-1)
        if self.pooling == "mean":
            masked = outputs * mask
            return masked.sum(dim=1) / lengths.unsqueeze(1)

        masked = outputs.masked_fill(~mask, torch.finfo(outputs.dtype).min)
        pooled = masked.max(dim=1).values
        return torch.where(has_valid_steps.unsqueeze(1), pooled, torch.zeros_like(pooled))
