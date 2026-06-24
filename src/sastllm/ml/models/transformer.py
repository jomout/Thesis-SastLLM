from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn

from .base import RepositoryClassifierModule


class TransformerRepositoryClassifier(RepositoryClassifierModule):
    """Transformer encoder over ordered functionality-cluster tokens."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        embedding_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        feedforward_dim: int = 256,
        max_sequence_length: int = 512,
        dropout: float = 0.2,
        pooling: Literal["last", "mean", "max"] = "mean",
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
        if input_dim <= 1:
            raise ValueError("input_dim must include padding plus at least one cluster token.")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0.")
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0.")
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0.")
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")
        if feedforward_dim <= 0:
            raise ValueError("feedforward_dim must be > 0.")
        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be > 0.")
        if pooling not in {"last", "mean", "max"}:
            raise ValueError("pooling must be one of: 'last', 'mean', 'max'.")

        self.save_hyperparameters(ignore=["class_counts"])
        self.max_sequence_length = max_sequence_length
        self.pooling = pooling
        self.token_embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_sequence_length, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,  # Set this to true
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"TransformerRepositoryClassifier expects token input shape (N, T), got {tuple(x.shape)}.")
        if x.size(1) == 0:
            raise ValueError("Transformer token sequences must contain at least one timestep.")
        if x.size(1) > self.max_sequence_length:
            raise ValueError(f"Sequence length {x.size(1)} exceeds configured maximum {self.max_sequence_length}.")

        tokens = x.long()
        valid_steps = tokens != 0
        has_valid_steps = valid_steps.any(dim=1)
        padding_mask = ~valid_steps

        # PyTorch attention cannot safely consume a row where every token is masked.
        safe_padding_mask = padding_mask.clone()
        safe_padding_mask[~has_valid_steps, 0] = False

        positions = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
        sequence = self.token_embedding(tokens) + self.position_embedding(positions)
        encoded = self.encoder(sequence, src_key_padding_mask=safe_padding_mask)
        encoded = self.output_norm(encoded)
        pooled = self._pool(encoded, valid_steps, has_valid_steps)
        return self.classifier(self.dropout(pooled))

    def _pool(self, encoded: torch.Tensor, valid_steps: torch.Tensor, has_valid_steps: torch.Tensor) -> torch.Tensor:
        lengths = valid_steps.sum(dim=1).clamp(min=1)
        if self.pooling == "last":
            indexes = (lengths - 1).view(-1, 1, 1).expand(-1, 1, encoded.size(2))
            pooled = encoded.gather(dim=1, index=indexes).squeeze(1)
        elif self.pooling == "mean":
            pooled = (encoded * valid_steps.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(1)
        else:
            masked = encoded.masked_fill(~valid_steps.unsqueeze(-1), torch.finfo(encoded.dtype).min)
            pooled = masked.max(dim=1).values
        return torch.where(has_valid_steps.unsqueeze(1), pooled, torch.zeros_like(pooled))
