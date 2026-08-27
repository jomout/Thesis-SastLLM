from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

Pooling = Literal["last", "mean", "max"]
Truncation = Literal["first", "last"]
TransformerInputEncoding = Literal["ordered_tokens", "cluster_distribution"]


class MLPModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Literal["mlp"] = "mlp"
    hidden_dims: tuple[int, ...] = (512, 256)
    dropout: float = 0.2


class LSTMModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Literal["lstm"] = "lstm"
    embedding_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.2
    bidirectional: bool = False
    pooling: Pooling = "last"
    max_sequence_length: int | None = 512
    truncation: Truncation = "first"


class TransformerModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Literal["transformer"] = "transformer"
    input_encoding: TransformerInputEncoding = "ordered_tokens"
    embedding_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    feedforward_dim: int = 256
    dropout: float = 0.2
    pooling: Pooling = "mean"
    max_sequence_length: int = 256
    truncation: Truncation = "first"

    @model_validator(mode="after")
    def validate_attention_shape(self) -> TransformerModelConfig:
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("transformer embedding_dim must be divisible by num_heads.")
        if self.input_encoding == "cluster_distribution" and self.pooling == "last":
            raise ValueError("transformer pooling='last' is not supported with cluster_distribution input; use 'mean' or 'max'.")
        return self


RepositoryModelConfig = Annotated[
    MLPModelConfig | LSTMModelConfig | TransformerModelConfig,
    Field(discriminator="name"),
]

_MODEL_CONFIG_ADAPTER: TypeAdapter[RepositoryModelConfig] = TypeAdapter(RepositoryModelConfig)


def parse_model_config(name: str, raw: Any) -> RepositoryModelConfig:
    if not isinstance(raw, dict):
        raise TypeError(f"models.{name} must be a mapping.")
    return _MODEL_CONFIG_ADAPTER.validate_python({"name": name, **raw})
