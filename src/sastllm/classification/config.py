from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from scripts.utils import load_yaml

ClassificationMode = Literal["search", "train", "test"]


class TrainingConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    batch_size: int = 32
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-5
    l1_lambda: float = Field(default=1e-3, alias="l1_param")
    seed: int = 42
    k: int = 5000
    validation_size: float = 0.1
    num_workers: int = 4
    use_weighted_sampler: bool = True
    use_class_weights: bool = True


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = "mlp"
    hidden_dims: tuple[int, ...] = (512, 256)
    embedding_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.2
    bidirectional: bool = False
    pooling: Literal["last", "mean", "max"] = "last"
    max_sequence_length: int | None = None
    truncation: Literal["first", "last"] = "first"


@dataclass(frozen=True)
class ClassificationConfig:
    save_model_dir: Path | None
    load_model_dir: Path | None
    save_plots_dir: Path | None
    training: TrainingConfig
    model: ModelConfig
    grid_search: dict[str, tuple[Any, ...]] | None = None

    @classmethod
    def from_yaml(cls, mode: ClassificationMode, config_path: str | Path = "configs/classification.yaml") -> "ClassificationConfig":
        root = load_yaml(config_path)
        raw = root.get("classification", {}).get(mode)
        if not isinstance(raw, dict):
            raise ValueError(f"Missing 'classification.{mode}' section in {config_path}.")

        params = raw.get("params", {})
        model = raw.get("model", {})
        if not isinstance(params, dict):
            raise ValueError(f"'classification.{mode}.params' must be a mapping.")
        if not isinstance(model, dict):
            raise ValueError(f"'classification.{mode}.model' must be a mapping when provided.")

        save_dir = raw.get("save_model_dir")
        load_dir = raw.get("load_model_dir")
        plots_dir = raw.get("save_plots_dir")
        grid_search = _parse_grid_search(raw.get("grid_search", {}))
        if mode in {"search", "train"} and not save_dir:
            raise ValueError(f"'classification.{mode}.save_model_dir' must be set.")
        if mode == "test" and not load_dir:
            raise ValueError("'classification.test.load_model_dir' must be set.")
        if mode == "search" and not plots_dir:
            raise ValueError("'classification.search.save_plots_dir' must be set.")
        if mode == "search" and not grid_search:
            raise ValueError("'classification.search.grid_search' must define at least one parameter.")

        return cls(
            save_model_dir=Path(save_dir) if save_dir else None,
            load_model_dir=Path(load_dir) if load_dir else None,
            save_plots_dir=Path(plots_dir) if plots_dir else None,
            training=TrainingConfig(**params),
            model=ModelConfig(**model),
            grid_search=grid_search if mode == "search" else None,
        )

    def iter_search_configs(self) -> list[tuple[str, "ClassificationConfig", dict[str, Any]]]:
        if not self.grid_search:
            raise ValueError("grid_search is not configured.")
        if self.save_model_dir is None:
            raise ValueError("save_model_dir is required for classification search.")

        keys = list(self.grid_search)
        runs: list[tuple[str, ClassificationConfig, dict[str, Any]]] = []
        base_params = self.training.model_dump(by_alias=True)
        for index, values in enumerate(product(*(self.grid_search[key] for key in keys)), start=1):
            overrides = dict(zip(keys, values))
            params = {**base_params, **overrides}
            run_name = f"search_{index:03d}__" + "__".join(f"{key}_{_slug(value)}" for key, value in overrides.items())
            runs.append(
                (
                    run_name,
                    ClassificationConfig(
                        save_model_dir=self.save_model_dir,
                        load_model_dir=None,
                        save_plots_dir=self.save_plots_dir,
                        training=TrainingConfig(**params),
                        model=self.model,
                        grid_search=None,
                    ),
                    overrides,
                )
            )
        return runs


def load_label_map(config_path: str | Path = "configs/split.yaml") -> dict[int, str]:
    data = load_yaml(config_path)
    raw = data.get("split", {}).get("binary_labels")
    if raw is None:
        return {0: "benign", 1: "malicious"}
    if isinstance(raw, dict):
        return {int(k): str(v) for k, v in raw.items()}
    if isinstance(raw, list):
        out: dict[int, str] = {}
        for item in raw:
            if isinstance(item, dict):
                for key, value in item.items():
                    out[int(key)] = str(value)
        return out or {0: "benign", 1: "malicious"}
    raise ValueError("split.binary_labels must be a mapping or list of mappings.")


def coerce_mode_config(mode: ClassificationMode, params: dict[str, Any], save_or_load_dir: str | Path) -> ClassificationConfig:
    """Compatibility helper for older call sites that already split YAML values."""
    return ClassificationConfig(
        save_model_dir=Path(save_or_load_dir) if mode == "train" else None,
        load_model_dir=Path(save_or_load_dir) if mode == "test" else None,
        save_plots_dir=None,
        training=TrainingConfig(**params),
        model=ModelConfig(),
    )


def _parse_grid_search(raw: Any) -> dict[str, tuple[Any, ...]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("grid_search must be a mapping of parameter name to list of values.")
    grid: dict[str, tuple[Any, ...]] = {}
    for key, values in raw.items():
        if not isinstance(key, str):
            raise ValueError("grid_search parameter names must be strings.")
        if not isinstance(values, list) or not values:
            raise ValueError(f"grid_search.{key} must be a non-empty list.")
        grid[key] = tuple(values)
    return grid


def _slug(value: Any) -> str:
    return str(value).replace("/", "-").replace(" ", "_").replace(".", "p")
