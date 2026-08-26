from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn

from .base import RepositoryClassifierModule
from .config import TransformerInputEncoding


class TransformerRepositoryClassifier(RepositoryClassifierModule):
    """Transformer encoder over ordered cluster tokens or a sparse cluster distribution."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        input_encoding: TransformerInputEncoding = "ordered_tokens",
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
        if input_encoding not in {"ordered_tokens", "cluster_distribution"}:
            raise ValueError("input_encoding must be either 'ordered_tokens' or 'cluster_distribution'.")
        if input_encoding == "ordered_tokens" and input_dim <= 1:
            raise ValueError("input_dim must include padding plus at least one cluster token.")
        if input_encoding == "cluster_distribution" and input_dim <= 0:
            raise ValueError("input_dim must include at least one cluster feature.")
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
        if input_encoding == "cluster_distribution" and pooling == "last":
            raise ValueError("pooling='last' is not meaningful for cluster-distribution input; use 'mean' or 'max'.")

        self.save_hyperparameters(ignore=["class_counts"])
        self.input_dim = input_dim
        self.input_encoding = input_encoding
        self.max_sequence_length = max_sequence_length
        self.pooling = pooling
        if input_encoding == "ordered_tokens":
            self.token_embedding: nn.Embedding | None = nn.Embedding(input_dim, embedding_dim, padding_idx=0)
            self.position_embedding: nn.Embedding | None = nn.Embedding(max_sequence_length, embedding_dim)
            self.cluster_embedding: nn.Embedding | None = None
            self.frequency_projection: nn.Linear | None = None
        else:
            self.token_embedding = None
            self.position_embedding = None
            self.cluster_embedding = nn.Embedding(input_dim, embedding_dim)
            self.frequency_projection = nn.Linear(1, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Set this to true
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"TransformerRepositoryClassifier expects a rank-2 input, got shape {tuple(x.shape)}.")

        sequence, valid_steps = self._embed_input(x)
        has_valid_steps = valid_steps.any(dim=1)
        padding_mask = ~valid_steps

        # PyTorch attention cannot safely consume a row where every token is masked.
        safe_padding_mask = padding_mask.clone()
        safe_padding_mask[~has_valid_steps, 0] = False

        encoded = self.encoder(sequence, src_key_padding_mask=safe_padding_mask)
        encoded = self.output_norm(encoded)
        pooled = self._pool(encoded, valid_steps, has_valid_steps)
        return self.classifier(self.dropout(pooled))

    def _embed_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.input_encoding == "ordered_tokens":
            return self._embed_ordered_tokens(x)
        return self._embed_cluster_distribution(x)

    def _embed_ordered_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.size(1) == 0:
            raise ValueError("Transformer token sequences must contain at least one timestep.")
        if x.size(1) > self.max_sequence_length:
            raise ValueError(f"Sequence length {x.size(1)} exceeds configured maximum {self.max_sequence_length}.")
        if self.token_embedding is None or self.position_embedding is None:
            raise RuntimeError("Ordered-token input layers are not initialized.")

        tokens = x.long()
        valid_steps = tokens != 0
        positions = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
        sequence = self.token_embedding(tokens) + self.position_embedding(positions)
        return sequence, valid_steps

    def _embed_cluster_distribution(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.size(1) != self.input_dim:
            raise ValueError(f"Expected cluster-distribution input shape (N, {self.input_dim}), got {tuple(x.shape)}.")
        if torch.any(x < 0):
            raise ValueError("Cluster-distribution features must be non-negative.")
        if self.cluster_embedding is None or self.frequency_projection is None:
            raise RuntimeError("Cluster-distribution input layers are not initialized.")

        # Attention is quadratic, so represent only the strongest nonzero
        # clusters while retaining both cluster identity and frequency.
        token_count = min(self.max_sequence_length, self.input_dim)
        frequencies, cluster_ids = torch.topk(x.float(), k=token_count, dim=1, sorted=True)
        valid_steps = frequencies > 0
        sequence = self.cluster_embedding(cluster_ids) + self.frequency_projection(frequencies.unsqueeze(-1))
        return sequence, valid_steps

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
