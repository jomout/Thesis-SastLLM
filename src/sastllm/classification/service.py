from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import seed_everything

from sastllm.configs import get_logger
from sastllm.db import RepositoryManager
from sastllm.ml import RepositoryDataModule
from sastllm.ml.models import LSTMModelConfig, RepositoryClassifierModule, TransformerModelConfig, build_model, model_class_for
from sastllm.ml.training import AccuracyHistoryCallback, AccuracyHistoryRecord, best_checkpoint_path, build_trainer
from sastllm.utils.observability import count_parameters, log_duration

from .config import ClassificationConfig
from .data import RepositoryDatasetBuilder
from .encoders import ClusterDistributionEncoder, LabelMapping, OrderedFunctionalityTokenSequenceEncoder, RepositoryEncoderProtocol
from .metrics import compute_classification_metrics

logger = get_logger(__name__)
EvaluationSplit = Literal["train", "val", "test"]


class RepositoryClassificationService:
    """
    Application service for repository classification.

    Swappable pieces:
    - repository encoder: any object implementing `RepositoryEncoderProtocol`
    - model: selected through `classification.<mode>.model`
    - dataset assembly: `RepositoryDatasetBuilder`
    """

    def __init__(
        self,
        *,
        config: ClassificationConfig,
        labels: LabelMapping | None = None,
        repository_manager: RepositoryManager | None = None,
        encoder: RepositoryEncoderProtocol | None = None,
        build_bundle: bool = True,
    ) -> None:
        self.config = config
        self.labels = labels or LabelMapping.from_split_config()
        self.repository_manager = repository_manager or RepositoryManager()
        self.encoder = encoder or self._default_encoder()
        seed_everything(config.training.seed, workers=True)
        logger.info(
            "Initializing repository classification service",
            model=config.model.name,
            input_encoding=getattr(config.model, "input_encoding", None),
            encoder=self.encoder.__class__.__name__,
            k=config.training.k,
            batch_size=config.training.batch_size,
            seed=config.training.seed,
        )

        self.dataset_builder: RepositoryDatasetBuilder | None = None
        self.bundle = None
        if build_bundle:
            self.dataset_builder = RepositoryDatasetBuilder(
                repository_manager=self.repository_manager,
                encoder=self.encoder,
                batch_size=config.training.batch_size,
                seed=config.training.seed,
            )
            with log_duration(logger, "classification_dataset_build", validation_size=config.training.validation_size):
                self.bundle = self.dataset_builder.build(validation_size=config.training.validation_size)

    def search(self) -> list[dict]:
        if self.config.save_plots_dir is None:
            raise ValueError("save_plots_dir is required for classification search.")
        self.config.save_plots_dir.mkdir(parents=True, exist_ok=True)

        results: list[dict] = []
        runs = self.config.iter_search_configs()
        logger.info("Starting classification grid search", runs=len(runs), save_plots_dir=str(self.config.save_plots_dir))
        for run_name, run_config, overrides in runs:
            logger.info("Starting classification search run", run_name=run_name, params=overrides)
            history = AccuracyHistoryCallback()
            service = RepositoryClassificationService(
                config=run_config,
                labels=self.labels,
                repository_manager=self.repository_manager,
            )
            model_dir = service.fit(run_name=run_name, history_callback=history)
            train_metrics = service.evaluate(model_dir=model_dir, split="train", persist=True)
            val_metrics = service.evaluate(model_dir=model_dir, split="val", persist=True)
            plot_path = self.config.save_plots_dir / f"{run_name}_accuracy.png"
            _plot_accuracy_history(history.records, plot_path, title=run_name)
            best_val_acc = max((record["val_acc"] for record in history.records if record["val_acc"] is not None), default=None)
            final_train_acc = next((record["train_acc"] for record in reversed(history.records) if record["train_acc"] is not None), None)
            result = {
                "run_name": run_name,
                "params": overrides,
                "model_dir": str(model_dir),
                "plot_path": str(plot_path),
                "best_val_acc": best_val_acc,
                "final_train_acc": final_train_acc,
                "train_metrics": _compact_metrics(train_metrics),
                "val_metrics": _compact_metrics(val_metrics),
                "history": history.records,
            }
            results.append(result)
            logger.info("Completed classification search run", **{k: v for k, v in result.items() if k != "history"})

        summary_path = self.config.save_plots_dir / "search_summary.json"
        summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("Classification grid search completed", runs=len(results), summary_path=str(summary_path))
        return results

    def fit(self, *, run_name: str | None = None, history_callback: AccuracyHistoryCallback | None = None) -> Path:
        if self.config.save_model_dir is None:
            raise ValueError("save_model_dir is required for training.")

        datamodule = self._datamodule()
        model = self._build_model()
        extra_callbacks = [history_callback] if history_callback is not None else None
        trainer = build_trainer(max_epochs=self.config.training.epochs, extra_callbacks=extra_callbacks)

        with log_duration(logger, "lightning_fit", model=self.config.model.name, epochs=self.config.training.epochs):
            trainer.fit(model, datamodule=datamodule)

        model_dir = self.config.save_model_dir / (run_name or f"model_{_timestamp()}")
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
        split: EvaluationSplit = "test",
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

    def evaluate(self, *, model_dir: str | Path | None = None, split: EvaluationSplit = "test", persist: bool = True) -> dict:
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
        if self.bundle is None:
            raise RuntimeError("Classification dataset bundle has not been built.")
        return RepositoryDataModule(
            dataset=self.bundle.dataset,
            train_indices=self.bundle.train_indices,
            val_indices=self.bundle.val_indices,
            test_indices=self.bundle.test_indices,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.training.num_workers,
            use_weighted_sampler=self.config.training.use_weighted_sampler,
        )

    def datamodule(self) -> RepositoryDataModule:
        return self._datamodule()

    def _build_model(self) -> RepositoryClassifierModule:
        if self.bundle is None:
            raise RuntimeError("Classification dataset bundle has not been built.")
        labels = self.bundle.dataset.y.numpy(force=True)
        labels = labels[labels >= 0]
        class_counts = _class_counts(labels)
        model = build_model(
            config=self.config.model,
            input_dim=self._model_input_dim(),
            output_dim=len(self.labels.index_to_label),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
            l1_lambda=self.config.training.l1_lambda,
            class_counts=class_counts,
            use_class_weights=self.config.training.use_class_weights,
        )
        total_parameters, trainable_parameters = count_parameters(model)
        logger.info(
            "Built repository classifier model",
            model=self.config.model.name,
            input_encoding=getattr(self.config.model, "input_encoding", None),
            input_dim=self._model_input_dim(),
            feature_shape=tuple(self.bundle.dataset.X.shape),
            output_dim=len(self.labels.index_to_label),
            class_counts=class_counts,
            use_class_weights=self.config.training.use_class_weights,
            use_weighted_sampler=self.config.training.use_weighted_sampler,
            total_parameters=total_parameters,
            trainable_parameters=trainable_parameters,
        )
        return model

    def _load_checkpoint(self, model_dir: Path) -> RepositoryClassifierModule:
        if self.bundle is None:
            raise RuntimeError("Classification dataset bundle has not been built.")
        checkpoint = model_dir / "best.ckpt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        labels = self.bundle.dataset.y.numpy(force=True)
        labels = labels[labels >= 0]
        class_counts = _class_counts(labels)
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
        return model_class_for(self.config.model)

    def _default_encoder(self) -> RepositoryEncoderProtocol:
        if isinstance(self.config.model, LSTMModelConfig) or (isinstance(self.config.model, TransformerModelConfig) and self.config.model.input_encoding == "ordered_tokens"):
            return OrderedFunctionalityTokenSequenceEncoder(
                num_clusters=self.config.training.k,
                labels=self.labels,
                max_sequence_length=self.config.model.max_sequence_length,
                truncation=self.config.model.truncation,
            )
        if isinstance(self.config.model, TransformerModelConfig):
            return ClusterDistributionEncoder(
                num_clusters=self.config.training.k,
                labels=self.labels,
                matrix_normalization=False,
            )
        return ClusterDistributionEncoder(num_clusters=self.config.training.k, labels=self.labels)

    def _model_input_dim(self) -> int:
        if self.bundle is None:
            raise RuntimeError("Classification dataset bundle has not been built.")
        return int(self.encoder.feature_dim)

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


def _compact_metrics(metrics: dict) -> dict[str, float]:
    keys = ("accuracy", "macro_f1", "weighted_f1", "auc_macro", "auc_weighted")
    return {key: float(metrics[key]) for key in keys if key in metrics}


def _class_counts(labels: np.ndarray) -> dict[int, int]:
    unique_labels, counts = np.unique(labels, return_counts=True)
    return {int(label): int(count) for label, count in zip(unique_labels, counts)}


def _plot_accuracy_history(records: list[AccuracyHistoryRecord], path: Path, *, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    train_epochs: list[int] = []
    train_acc: list[float] = []
    val_epochs: list[int] = []
    val_acc: list[float] = []

    for record in records:
        train_value = record["train_acc"]
        if train_value is not None:
            train_epochs.append(record["epoch"])
            train_acc.append(train_value)

        val_value = record["val_acc"]
        if val_value is not None:
            val_epochs.append(record["epoch"])
            val_acc.append(val_value)

    plt.figure(figsize=(8, 5))
    if train_acc:
        plt.plot(train_epochs, train_acc, marker="o", label="train_acc")
    if val_acc:
        plt.plot(val_epochs, val_acc, marker="o", label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
