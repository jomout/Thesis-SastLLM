from __future__ import annotations

from argus.configs import get_logger

from .base import RepositoryClassifierModule
from .config import LSTMModelConfig, MLPModelConfig, RepositoryModelConfig, TransformerModelConfig
from .lstm import LSTMRepositoryClassifier
from .mlp import MLPRepositoryClassifier
from .transformer import TransformerRepositoryClassifier

RepositoryModelClass = type[RepositoryClassifierModule]
logger = get_logger(__name__)


def build_model(
    *,
    config: RepositoryModelConfig,
    input_dim: int,
    output_dim: int,
    lr: float,
    weight_decay: float,
    l1_lambda: float,
    class_counts: dict[int, int] | None,
    use_class_weights: bool = True,
) -> RepositoryClassifierModule:
    logger.info(
        "Building repository classifier model",
        model=config.name,
        input_encoding=getattr(config, "input_encoding", None),
        input_dim=input_dim,
        output_dim=output_dim,
        lr=lr,
        weight_decay=weight_decay,
        l1_lambda=l1_lambda,
        use_class_weights=use_class_weights,
        class_counts=class_counts,
    )
    if isinstance(config, MLPModelConfig):
        return MLPRepositoryClassifier(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
            lr=lr,
            weight_decay=weight_decay,
            l1_lambda=l1_lambda,
            class_counts=class_counts,
            use_class_weights=use_class_weights,
        )
    if isinstance(config, LSTMModelConfig):
        return LSTMRepositoryClassifier(
            input_dim=input_dim,
            output_dim=output_dim,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            bidirectional=config.bidirectional,
            pooling=config.pooling,
            lr=lr,
            weight_decay=weight_decay,
            l1_lambda=l1_lambda,
            class_counts=class_counts,
            use_class_weights=use_class_weights,
        )
    if isinstance(config, TransformerModelConfig):
        return TransformerRepositoryClassifier(
            input_dim=input_dim,
            output_dim=output_dim,
            input_encoding=config.input_encoding,
            embedding_dim=config.embedding_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            feedforward_dim=config.feedforward_dim,
            max_sequence_length=config.max_sequence_length,
            dropout=config.dropout,
            pooling=config.pooling,
            lr=lr,
            weight_decay=weight_decay,
            l1_lambda=l1_lambda,
            class_counts=class_counts,
            use_class_weights=use_class_weights,
        )
    logger.exception("Unsupported repository model configuration", config_type=type(config).__name__)
    raise TypeError(f"Unsupported repository model config: {type(config).__name__}.")


def model_class_for(config: RepositoryModelConfig) -> RepositoryModelClass:
    if isinstance(config, MLPModelConfig):
        return MLPRepositoryClassifier
    if isinstance(config, LSTMModelConfig):
        return LSTMRepositoryClassifier
    if isinstance(config, TransformerModelConfig):
        return TransformerRepositoryClassifier
    logger.exception("Unsupported repository model class lookup", config_type=type(config).__name__)
    raise TypeError(f"Unsupported repository model config: {type(config).__name__}.")
