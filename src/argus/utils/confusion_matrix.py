from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.image import AxesImage

from argus.classification import ClassificationConfig, RepositoryClassificationService
from argus.configs import get_logger

logger = get_logger(__name__)

DatasetSplit = Literal["train", "val", "test"]
ImageFormat = Literal["png", "pdf", "svg"]


@dataclass(frozen=True)
class ConfusionMatrixConfig:
    """Validated runtime settings for classifier confusion-matrix plots."""

    classification_config_path: Path = Path("configs/classification.yaml")
    output_dir: Path = Path("plots/confusion_matrix")
    split: DatasetSplit = "test"
    log_scale: bool = True
    image_format: ImageFormat = "png"

    def __post_init__(self) -> None:
        if self.split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test.")
        if self.image_format not in {"png", "pdf", "svg"}:
            raise ValueError("image_format must be one of: png, pdf, svg.")


def run_confusion_matrix(settings: ConfusionMatrixConfig | None = None) -> Path:
    """Evaluate the configured classifier and write confusion-matrix artifacts."""
    settings = settings or ConfusionMatrixConfig()
    classification_config = ClassificationConfig.from_yaml("test", settings.classification_config_path)
    if classification_config.load_model_dir is None:
        raise ValueError("classification.test.load_model_dir must be configured for confusion-matrix evaluation.")

    service = RepositoryClassificationService(config=classification_config)
    metrics = service.evaluate(split=settings.split, persist=False)
    class_names = [service.labels.index_to_label[index] for index in sorted(service.labels.index_to_label)]
    counts = _matrix_from_metrics(metrics, class_count=len(class_names))
    normalized = normalize_confusion_matrix(counts)
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    model_title = classification_config.model.name.upper()
    split_title = settings.split.capitalize()
    counts_path = settings.output_dir / f"{settings.split}_confusion_matrix_counts.{settings.image_format}"
    normalized_path = settings.output_dir / f"{settings.split}_confusion_matrix_normalized.{settings.image_format}"
    plot_confusion_matrix(
        counts,
        class_names=class_names,
        title=f"{model_title} {split_title} Confusion Matrix (Counts)",
        output_path=counts_path,
        normalize=False,
        log_scale=settings.log_scale,
    )
    plot_confusion_matrix(
        counts,
        class_names=class_names,
        title=f"{model_title} {split_title} Confusion Matrix (Normalized)",
        output_path=normalized_path,
        normalize=True,
        log_scale=False,
    )

    metadata = {
        "model": classification_config.model.model_dump(mode="json"),
        "model_dir": str(classification_config.load_model_dir),
        "classification_config": str(settings.classification_config_path),
        "split": settings.split,
        "class_names": class_names,
        "sample_count": int(counts.sum()),
        "counts": counts.astype(int).tolist(),
        "row_normalized": normalized.tolist(),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
        "counts_plot": counts_path.name,
        "normalized_plot": normalized_path.name,
    }
    metadata_path = settings.output_dir / f"{settings.split}_confusion_matrix.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info(
        "Saved classifier confusion matrix",
        model=classification_config.model.name,
        model_dir=str(classification_config.load_model_dir),
        split=settings.split,
        samples=int(counts.sum()),
        counts_path=str(counts_path),
        normalized_path=str(normalized_path),
        metadata_path=str(metadata_path),
    )
    return settings.output_dir


def normalize_confusion_matrix(matrix: np.ndarray) -> np.ndarray:
    """Row-normalize a confusion matrix while preserving empty true-label rows."""
    counts = _validate_confusion_matrix(matrix)
    row_totals = counts.sum(axis=1, keepdims=True)
    return np.divide(
        counts.astype(float),
        row_totals,
        out=np.zeros_like(counts, dtype=float),
        where=row_totals != 0,
    )


def plot_confusion_matrix(
    matrix: np.ndarray,
    *,
    class_names: Sequence[str],
    title: str,
    output_path: Path,
    normalize: bool = False,
    log_scale: bool = False,
) -> Path:
    """Render a validated confusion matrix to an image file."""
    counts = _validate_confusion_matrix(matrix)
    if len(class_names) != counts.shape[0]:
        raise ValueError(f"class_names has {len(class_names)} entries but the matrix has {counts.shape[0]} classes.")

    values = normalize_confusion_matrix(counts) if normalize else counts
    norm = _log_norm(values) if log_scale else None
    fig, axis = plt.subplots(figsize=(5.5, 4.5))
    image = axis.imshow(values, cmap="Blues", norm=norm)
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label("Proportion" if normalize else "Number of samples")

    axis.set_title(title)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks(np.arange(len(class_names)), labels=class_names)
    axis.set_yticks(np.arange(len(class_names)), labels=class_names)
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")

    for true_index in range(values.shape[0]):
        for predicted_index in range(values.shape[1]):
            value = values[true_index, predicted_index]
            axis.text(
                predicted_index,
                true_index,
                f"{value:.3f}" if normalize else f"{int(value)}",
                ha="center",
                va="center",
                fontsize=11,
                color=_annotation_color(image, float(value)),
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _matrix_from_metrics(metrics: dict, *, class_count: int) -> np.ndarray:
    raw_matrix = metrics.get("confusion_matrix")
    if raw_matrix is None:
        raise ValueError("Classification metrics do not contain a confusion_matrix value.")
    matrix = _validate_confusion_matrix(np.asarray(raw_matrix))
    expected_shape = (class_count, class_count)
    if matrix.shape != expected_shape:
        raise ValueError(f"Confusion matrix has shape {matrix.shape}; expected {expected_shape} from the configured labels.")
    return matrix


def _validate_confusion_matrix(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.shape[0] == 0:
        raise ValueError(f"Confusion matrix must be a non-empty square matrix; got shape {values.shape}.")
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError("Confusion matrix must contain only finite numeric values.")
    if (values < 0).any():
        raise ValueError("Confusion matrix values cannot be negative.")
    return values


def _log_norm(values: np.ndarray) -> LogNorm | None:
    positive_values = values[values > 0]
    if not len(positive_values):
        logger.warning("Confusion matrix contains no positive values; using a linear color scale")
        return None
    minimum = float(positive_values.min())
    maximum = float(positive_values.max())
    if minimum == maximum:
        maximum = minimum * 1.01
    return LogNorm(vmin=minimum, vmax=maximum)


def _annotation_color(image: AxesImage, value: float) -> str:
    normalized_value = image.norm(value)
    if np.ma.is_masked(normalized_value) or not np.isfinite(normalized_value):
        return "black"
    return "white" if float(normalized_value) > 0.55 else "black"
