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
    save_plots_dir: Path
    random_state: int = 42
    batch_size: int = 1000
    min_samples_per_cluster: int = 20
    num_k_candidates: int = 30
    early_stop_patience: int = 4


@dataclass(frozen=True)
class TrainConfig:
    k: int
    save_model_dir: Path
    random_state: int = 42
    batch_size: int = 1000


@dataclass(frozen=True)
class TestConfig:
    load_model_file: Path
    batch_size: int = 1000


@dataclass(frozen=True)
class ClusteringConfig:
    search: SearchConfig
    train: TrainConfig
    test: TestConfig

    @classmethod
    def from_yaml(cls, config_path: str | Path = "configs/clustering.yaml") -> "ClusteringConfig":
        raw = load_yaml(config_path).get("clustering", {})
        if not isinstance(raw, dict) or not raw:
            raise ValueError(f"No 'clustering' section found in {config_path}.")

        search_raw = _section(raw, "search")
        train_raw = _section(raw, "train")
        test_raw = _section(raw, "test")

        random_state = _int(search_raw, "random_state", default=42)

        return cls(
            search=SearchConfig(
                grid_search=tuple(_int_list(search_raw, "grid_search")),
                save_model_dir=Path(_str(search_raw, "save_model_dir", default="models/clustering/searching_models")),
                save_plots_dir=Path(_str(search_raw, "save_plots_dir", default="plots/clustering/searching")),
                random_state=random_state,
                batch_size=_int(search_raw, "batch_size", default=1000),
                min_samples_per_cluster=_int(search_raw, "min_samples_per_cluster", default=20),
                num_k_candidates=_int(search_raw, "num_k_candidates", default=30),
                early_stop_patience=_int(search_raw, "early_stop_patience", default=4),
            ),
            train=TrainConfig(
                k=_int(train_raw, "k"),
                save_model_dir=Path(_str(train_raw, "save_model_dir", default="models/clustering/trained_models")),
                random_state=_int(train_raw, "random_state", default=random_state),
                batch_size=_int(train_raw, "batch_size", default=1000),
            ),
            test=TestConfig(
                load_model_file=Path(_str(test_raw, "load_model_file")),
                batch_size=_int(test_raw, "batch_size", default=1000),
            ),
        )


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"'clustering.{key}' must be a mapping.")
    return value


def _str(raw: dict[str, Any], key: str, *, default: str | None = None) -> str:
    value = raw.get(key, default)
    if value is None:
        raise ValueError(f"Missing required clustering config key '{key}'.")
    if not isinstance(value, str):
        raise ValueError(f"Clustering config key '{key}' must be a string.")
    return value


def _int(raw: dict[str, Any], key: str, *, default: int | None = None) -> int:
    value = raw.get(key, default)
    if value is None:
        raise ValueError(f"Missing required clustering config key '{key}'.")
    if not isinstance(value, int):
        raise ValueError(f"Clustering config key '{key}' must be an int.")
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
