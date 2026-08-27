from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from scripts.utils import load_yaml

ClusteringMode = Literal["search", "train", "test"]


@dataclass(frozen=True)
class SearchConfig:
    grid_search: tuple[int, ...]
    save_model_dir: Path
    random_state: int = 42
    batch_size: int = 1000
    min_samples_per_cluster: int = 20
    num_k_candidates: int = 30


@dataclass(frozen=True)
class EvaluationConfig:
    sample_size: int = 100000
    silhouette_sample_size: int = 5000
    silhouette_samples_per_cluster: int = 5
    silhouette_metric: str = "euclidean"
    random_state: int = 42
    elbow_window_factor: float = 2.0
    max_silhouette_singleton_fraction: float = 0.5


@dataclass(frozen=True)
class TrainConfig:
    k: int
    save_model_dir: Path
    random_state: int = 42
    batch_size: int = 1000


@dataclass(frozen=True)
class TestConfig:
    load_model_dir: Path
    batch_size: int = 1000


@dataclass(frozen=True)
class ClusteringConfig:
    evaluation: EvaluationConfig
    search: SearchConfig
    train: TrainConfig
    test: TestConfig

    @classmethod
    def from_yaml(cls, config_path: str | Path = "configs/clustering.yaml") -> ClusteringConfig:
        raw = load_yaml(config_path).get("clustering", {})
        if not isinstance(raw, dict) or not raw:
            raise ValueError(f"No 'clustering' section found in {config_path}.")

        search_raw = _section(raw, "search")
        train_raw = _section(raw, "train")
        test_raw = _section(raw, "test")
        evaluation_raw = _section(raw, "evaluation")

        random_state = _int(search_raw, "random_state", default=42)

        return cls(
            evaluation=EvaluationConfig(
                sample_size=_int(evaluation_raw, "sample_size", default=100000),
                silhouette_sample_size=_int(evaluation_raw, "silhouette_sample_size", default=5000),
                silhouette_samples_per_cluster=_int(
                    evaluation_raw,
                    "silhouette_samples_per_cluster",
                    default=5,
                ),
                silhouette_metric=_str(evaluation_raw, "silhouette_metric", default="euclidean"),
                random_state=_int(evaluation_raw, "random_state", default=random_state),
                elbow_window_factor=_float(evaluation_raw, "elbow_window_factor", default=2.0, minimum=1.0),
                max_silhouette_singleton_fraction=_float(
                    evaluation_raw,
                    "max_silhouette_singleton_fraction",
                    default=0.5,
                    minimum=0.0,
                    maximum=1.0,
                ),
            ),
            search=SearchConfig(
                grid_search=tuple(_int_list(search_raw, "grid_search")),
                save_model_dir=Path(_str(search_raw, "save_model_dir", default="models/clustering/searching_models")),
                random_state=random_state,
                batch_size=_int(search_raw, "batch_size", default=1000),
                min_samples_per_cluster=_int(search_raw, "min_samples_per_cluster", default=20),
                num_k_candidates=_int(search_raw, "num_k_candidates", default=30),
            ),
            train=TrainConfig(
                k=_int(train_raw, "k"),
                save_model_dir=Path(_str(train_raw, "save_model_dir", default="models/clustering/trained_models")),
                random_state=_int(train_raw, "random_state", default=random_state),
                batch_size=_int(train_raw, "batch_size", default=1000),
            ),
            test=TestConfig(
                load_model_dir=Path(_str(test_raw, "load_model_dir")),
                batch_size=_int(test_raw, "batch_size", default=1000),
            ),
        )


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"'clustering.{key}' must be a mapping.")
    return value


def _str(raw: dict[str, Any], key: str, *, default: str | None = None) -> str:
    value = raw.get(key, default)
    if value is None:
        raise ValueError(f"Missing required clustering config key '{key}'.")
    if not isinstance(value, str):
        raise TypeError(f"Clustering config key '{key}' must be a string.")
    return value


def _int(raw: dict[str, Any], key: str, *, default: int | None = None) -> int:
    value = raw.get(key, default)
    if value is None:
        raise ValueError(f"Missing required clustering config key '{key}'.")
    if not isinstance(value, int):
        raise TypeError(f"Clustering config key '{key}' must be an int.")
    if value <= 0:
        raise ValueError(f"Clustering config key '{key}' must be > 0.")
    return value


def _int_list(raw: dict[str, Any], key: str) -> list[int]:
    value = raw.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise ValueError(f"Clustering config key '{key}' must be a list of ints.")
    if any(item <= 0 for item in value):
        raise ValueError(f"Clustering config key '{key}' must contain only positive ints.")
    return value


def _float(
    raw: dict[str, Any],
    key: str,
    *,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = raw.get(key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"Clustering config key '{key}' must be numeric.")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"Clustering config key '{key}' must be >= {minimum}.")
    if maximum is not None and result > maximum:
        raise ValueError(f"Clustering config key '{key}' must be <= {maximum}.")
    return result
