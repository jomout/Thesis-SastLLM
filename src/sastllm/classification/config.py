from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from scripts.utils import load_yaml

ClassificationMode = Literal["train", "test"]


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
    num_workers: int = 1


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = "mlp"
    hidden_dims: tuple[int, ...] = (512, 256)
    dropout: float = 0.2


@dataclass(frozen=True)
class ClassificationConfig:
    save_model_dir: Path | None
    load_model_dir: Path | None
    training: TrainingConfig
    model: ModelConfig

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
        if mode == "train" and not save_dir:
            raise ValueError("'classification.train.save_model_dir' must be set.")
        if mode == "test" and not load_dir:
            raise ValueError("'classification.test.load_model_dir' must be set.")

        return cls(
            save_model_dir=Path(save_dir) if save_dir else None,
            load_model_dir=Path(load_dir) if load_dir else None,
            training=TrainingConfig(**params),
            model=ModelConfig(**model),
        )


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
        training=TrainingConfig(**params),
        model=ModelConfig(),
    )
