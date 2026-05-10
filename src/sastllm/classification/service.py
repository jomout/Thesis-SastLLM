from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from lightning import seed_everything

from sastllm.configs import get_logger
from sastllm.db import RepositoryManager
from sastllm.ml import RepositoryDataModule
from sastllm.ml.models import RepositoryClassifierModule, build_model
from sastllm.ml.training import best_checkpoint_path, build_trainer
from sastllm.utils.observability import count_parameters, log_duration

from .config import ClassificationConfig
from .data import RepositoryDatasetBuilder
from .encoders import ClusterDistributionEncoder, LabelMapping, RepositoryEncoderProtocol
from .metrics import compute_classification_metrics

logger = get_logger(__name__)


class RepositoryClassificationService:
    """
    Application service for repository classification.

    Swappable pieces:
    - repository encoder: any object implementing `RepositoryEncoderProtocol`
    - model: selected through `classification.<mode>.model.name`
    - dataset assembly: `RepositoryDatasetBuilder`
    """

    def __init__(
        self,
        *,
        config: ClassificationConfig,
        labels: LabelMapping | None = None,
        repository_manager: RepositoryManager | None = None,
        encoder: RepositoryEncoderProtocol | None = None,
    ) -> None:
        self.config = config
        self.labels = labels or LabelMapping.from_split_config()
        self.repository_manager = repository_manager or RepositoryManager()
        self.encoder = encoder or ClusterDistributionEncoder(num_clusters=config.training.k, labels=self.labels)
        seed_everything(config.training.seed, workers=True)
        logger.info(
            "Initializing repository classification service",
            model=config.model.name,
            encoder=self.encoder.__class__.__name__,
            k=config.training.k,
            batch_size=config.training.batch_size,
            seed=config.training.seed,
        )

        self.dataset_builder = RepositoryDatasetBuilder(
            repository_manager=self.repository_manager,
            encoder=self.encoder,
            batch_size=config.training.batch_size,
            seed=config.training.seed,
        )
        with log_duration(logger, "classification_dataset_build", validation_size=config.training.validation_size):
            self.bundle = self.dataset_builder.build(validation_size=config.training.validation_size)

    def fit(self) -> Path:
        if self.config.save_model_dir is None:
            raise ValueError("save_model_dir is required for training.")

        datamodule = self._datamodule()
        model = self._build_model()
        trainer = build_trainer(max_epochs=self.config.training.epochs)

        with log_duration(logger, "lightning_fit", model=self.config.model.name, epochs=self.config.training.epochs):
            trainer.fit(model, datamodule=datamodule)

        model_dir = self.config.save_model_dir / f"model_{_timestamp()}"
        model_dir.mkdir(parents=True, exist_ok=True)
        self._persist_checkpoint(trainer, model_dir)
        logger.info("Training complete. Model saved to %s.", model_dir)
        return model_dir

    def test(self, model_dir: str | Path | None = None) -> None:
        resolved_model_dir = Path(model_dir) if model_dir is not None else self._load_model_dir()
        model = self._load_checkpoint(resolved_model_dir)
        with log_duration(logger, "lightning_test", model_dir=str(resolved_model_dir)):
            self._trainer().test(model, datamodule=self._datamodule())

    def predict(
        self,
        *,
        model_dir: str | Path | None = None,
        split: Literal["train", "test"] = "test",
        persist: bool = False,
    ) -> tuple[dict[int, dict], np.ndarray]:
        resolved_model_dir = Path(model_dir) if model_dir is not None else self._load_model_dir()
        model = self._load_checkpoint(resolved_model_dir)
        dataloader = self._datamodule().dataloader_for_split(split)
        with log_duration(logger, "lightning_predict", split=split, model_dir=str(resolved_model_dir)):
            predictions = self._trainer().predict(model, dataloaders=dataloader)

        ids = torch.cat([batch[0].cpu() for batch in predictions]).numpy()  # type: ignore[index]
        logits = torch.cat([batch[1].cpu() for batch in predictions]).numpy()  # type: ignore[index]
        probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()
        predicted_indices = probabilities.argmax(axis=1)
        logger.info(
            "Prediction tensors collected",
            split=split,
            samples=len(ids),
            logits_shape=tuple(logits.shape),
            probabilities_shape=tuple(probabilities.shape),
        )

        results: dict[int, dict] = {}
        for repository_id, predicted_index in zip(ids, predicted_indices):
            repo = self.repository_manager.get_repository(int(repository_id))
            if repo is None:
                raise ValueError(f"Repository ID {repository_id} not found.")
            label = self.labels.normalize_label(repo.label)
            results[int(repository_id)] = {
                "label": label,
                "prediction": self.labels.index_to_label[int(predicted_index)],
            }

        if persist:
            out = resolved_model_dir / f"{split}_predictions.json"
            out.write_text(json.dumps(results, indent=4), encoding="utf-8")
            logger.info("Saved predictions to %s.", out)

        return results, probabilities

    def evaluate(self, *, model_dir: str | Path | None = None, split: Literal["train", "test"] = "test", persist: bool = True) -> dict:
        resolved_model_dir = Path(model_dir) if model_dir is not None else self._load_model_dir()
        results, probabilities = self.predict(model_dir=resolved_model_dir, split=split, persist=persist)
        label_to_index = self.labels.label_to_index
        true_indices = [label_to_index.get(str(row["label"]), -1) for row in results.values()]
        pred_indices = [label_to_index.get(str(row["prediction"]), -1) for row in results.values()]

        metrics = compute_classification_metrics(
            true_indices,
            pred_indices,
            index_to_label=self.labels.index_to_label,
            probabilities=probabilities,
        )
        logger.info(
            "Evaluation (%s) accuracy=%.4f macro_f1=%.4f weighted_f1=%.4f",
            split,
            metrics["accuracy"],
            metrics["macro_f1"],
            metrics["weighted_f1"],
        )

        if persist:
            out = resolved_model_dir / f"{split}_metrics.json"
            out.write_text(json.dumps(metrics, indent=4), encoding="utf-8")
            logger.info("Saved metrics to %s.", out)

        return metrics

    def _datamodule(self) -> RepositoryDataModule:
        return RepositoryDataModule(
            dataset=self.bundle.dataset,
            train_indices=self.bundle.train_indices,
            val_indices=self.bundle.val_indices,
            test_indices=self.bundle.test_indices,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.training.num_workers,
        )

    def datamodule(self) -> RepositoryDataModule:
        return self._datamodule()

    def _build_model(self) -> RepositoryClassifierModule:
        labels = self.bundle.dataset.y.numpy(force=True)
        labels = labels[labels >= 0]
        class_counts = dict(zip(*np.unique(labels, return_counts=True)))
        model = build_model(
            name=self.config.model.name,
            input_dim=int(self.bundle.dataset.X.shape[1]),
            output_dim=len(self.labels.index_to_label),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
            l1_lambda=self.config.training.l1_lambda,
            class_counts=class_counts,
            hidden_dims=self.config.model.hidden_dims,
            dropout=self.config.model.dropout,
        )
        total_parameters, trainable_parameters = count_parameters(model)
        logger.info(
            "Built repository classifier model",
            model=self.config.model.name,
            input_dim=int(self.bundle.dataset.X.shape[1]),
            output_dim=len(self.labels.index_to_label),
            class_counts={int(k): int(v) for k, v in class_counts.items()},
            total_parameters=total_parameters,
            trainable_parameters=trainable_parameters,
        )
        return model

    def _load_checkpoint(self, model_dir: Path) -> RepositoryClassifierModule:
        checkpoint = model_dir / "best.ckpt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        labels = self.bundle.dataset.y.numpy(force=True)
        labels = labels[labels >= 0]
        class_counts = dict(zip(*np.unique(labels, return_counts=True)))
        model_class = self._model_class()
        model = model_class.load_from_checkpoint(str(checkpoint), class_counts=class_counts)
        model.eval()
        total_parameters, trainable_parameters = count_parameters(model)
        logger.info(
            "Loaded classifier checkpoint",
            checkpoint=str(checkpoint),
            model=self.config.model.name,
            total_parameters=total_parameters,
            trainable_parameters=trainable_parameters,
        )
        return model

    def _model_class(self):
        from sastllm.ml.models import MLPRepositoryClassifier

        if self.config.model.name == "mlp":
            return MLPRepositoryClassifier
        raise ValueError(f"Unsupported repository classifier model: {self.config.model.name!r}.")

    def _persist_checkpoint(self, trainer, model_dir: Path) -> None:
        best_src = best_checkpoint_path(trainer)
        destination = model_dir / "best.ckpt"
        if best_src and Path(best_src).is_file():
            shutil.copy2(best_src, destination)
            source_type = "best"
        else:
            trainer.save_checkpoint(destination)
            best_src = str(destination)
            source_type = "last"

        (model_dir / "config.json").write_text(
            json.dumps(self.config.training.model_dump(by_alias=False), indent=2),
            encoding="utf-8",
        )
        (model_dir / "meta.json").write_text(
            json.dumps(
                {
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                    "source": best_src,
                    "source_type": source_type,
                    "monitor": "val_acc",
                    "monitor_mode": "max",
                    "model": self.config.model.model_dump(),
                    "encoder": self.encoder.__class__.__name__,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info(
            "Persisted classifier checkpoint",
            model_dir=str(model_dir),
            checkpoint=str(destination),
            source_type=source_type,
        )

    def _trainer(self):
        return build_trainer(max_epochs=self.config.training.epochs)

    def _load_model_dir(self) -> Path:
        if self.config.load_model_dir is None:
            raise ValueError("load_model_dir is required.")
        return self.config.load_model_dir


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
