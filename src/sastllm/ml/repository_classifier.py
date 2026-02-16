from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from lightning import seed_everything
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from pydantic import BaseModel
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

from sastllm.configs import get_logger
from sastllm.db import RepositoryManager
from sastllm.ml import CodeDataModule, CodeDataset, CodeModel
from sastllm.utils.repository_encoder import (
    BINARY_LABEL_MAP,
    BINARY_LABEL_TO_INDEX,
    RepositoryEncoder,
)

logger = get_logger(__name__)


class ClassifierConfig(BaseModel):
    """
    Configuration for RepositoryClassifier.
    """

    batch_size: int = 32
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-5
    l1_lambda: float = 1e-3
    seed: int = 42
    k: int = 5000


class RepositoryClassifier:
    """
    A thin, kwargs-driven orchestrator that wires:
    - DB -> repositories
    - Encoder -> (X, y, ids)
    - DataModule -> splits/batching (uses kwargs)
    - Model -> hyperparams (uses kwargs)
    - Trainer -> callbacks/logging (uses kwargs)
    """

    def __init__(self, *, config: ClassifierConfig) -> None:
        logger.debug("Initializing RepositoryClassifier.")
        self.repository_db = RepositoryManager()

        self.config = config

        if self.config.seed is not None:
            seed_everything(self.config.seed, workers=True)

        self.preprocessor = RepositoryEncoder(self.config.k, BINARY_LABEL_TO_INDEX)

        # Fetch & encode data
        self.full_dataset = self._fetch_data()

        logger.debug("RepositoryClassifier initialized.")

    def _get_repository_ids_and_labels(self, split: Literal["train", "test"]) -> Tuple[List[int], List[int]]:
        """
        Fetch repository IDs and labels from the database for the given split.
        """
        repo_ids = []
        repo_labels = []
        for repo in self.repository_db.get_repositories(split=split):
            repo_ids.append(repo.repository_id)
            repo_labels.append(repo.label)
        return repo_ids, repo_labels

    def fit(self, save_dir: Path | str) -> str:
        """
        Fit the model using the Trainer and DataModule
        """
        # Setup DataModule
        datamodule: CodeDataModule = self._setup_datamodule(
            full_dataset=self.full_dataset,
            validation_size=0.1,
            batch_size=self.config.batch_size,
        )

        # Setup Trainer
        trainer = self._setup_trainer(epochs=self.config.epochs)

        # Setup Model
        model = self._setup_model()

        # Train
        trainer.fit(model, datamodule=datamodule)

        # Extract best checkpoint
        best_ckpt = self._get_best_checkpoint(trainer)
        logger.info(f"Training complete. Best checkpoint: {best_ckpt}")

        # Save/Persist model with best validation performance into a folder for later use
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_dir = save_dir / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Saving model to directory: {save_dir}")

        self._persist_best_checkpoint_and_metadata(trainer, best_ckpt, save_dir)

        return str(save_dir)

    def fit_k_fold(
        self,
        save_dir: Path | str,
        n_splits: int = 5,
    ) -> List[Dict]:
        """
        Fit the model using k-fold cross-validation.
        """
        logger.info(f"Starting {n_splits}-fold cross-validation.")
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.config.seed,
        )

        # Get test data
        test_ids, test_labels = self._get_repository_ids_and_labels("test")

        # Get train data
        train_ids, train_labels = self._get_repository_ids_and_labels("train")
        X = np.array(train_ids)
        y = np.array(train_labels)

        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\n========== Fold {fold + 1}/{n_splits} ==========\n")

            # Build fold-specific DataModule
            datamodule = CodeDataModule(
                full_dataset=self.full_dataset,
                train_indices=train_idx,
                val_indices=val_idx,
                test_indices=test_ids,
                batch_size=self.config.batch_size,
            )

            # Compute class counts for this fold (training only)
            train_labels = self.full_dataset.y[train_idx].numpy()
            class_counts = dict(zip(*np.unique(train_labels, return_counts=True)))

            # Build a fresh model for this fold
            model = CodeModel(
                input_dim=self.full_dataset.X.shape[1],
                output_dim=len(BINARY_LABEL_MAP),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
                l1_lambda=self.config.l1_lambda,
                class_counts=class_counts,
            )

            # Create a fresh trainer
            trainer = self._setup_trainer(epochs=self.config.epochs)

            # Train
            trainer.fit(model, datamodule=datamodule)

            # Extract best checkpoint for this fold
            best_ckpt = self._get_best_checkpoint(trainer)

            # Evaluate on the fold's validation set
            metrics = self._evaluate_fold(
                trainer=trainer,
                model=model,
                datamodule=datamodule,
            )

            fold_results.append(
                {
                    "fold": fold,
                    "best_ckpt": best_ckpt,
                    "val_metrics": metrics,
                }
            )
        logger.info("K-fold cross-validation complete.")

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        folds_dir = save_dir / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        folds_dir.mkdir(parents=True, exist_ok=True)
        folds_file = folds_dir / "fold_results.json"

        with open(folds_file, "w", encoding="utf-8") as f:
            json.dump(fold_results, f, indent=4)
        return fold_results

    def _evaluate_fold(self, trainer: Trainer, model: CodeModel, datamodule: CodeDataModule):
        """Evaluate a fold and return metrics dict."""
        model.eval()
        outputs = trainer.validate(model, datamodule=datamodule, verbose=False)
        return outputs[0] if outputs else {}

    def predict(
        self,
        model_dir: Path | str,
        split: Literal["train", "test"] = "test",
        persist: bool = False,
    ) -> Tuple[Dict[int, Dict], np.ndarray]:
        # Setup DataModule
        datamodule = self._setup_datamodule(
            full_dataset=self.full_dataset,
            validation_size=0.1,
            batch_size=self.config.batch_size,
        )

        datamodule.setup()

        # Setup trainer
        trainer = self._setup_trainer(epochs=self.config.epochs)

        # Load the correct weights
        ckpt_path = Path(model_dir) / "best.ckpt"
        if ckpt_path.exists():
            logger.info(f"Loading model from checkpoint: {ckpt_path}")
            labels = self.full_dataset.y.numpy(force=True)
            class_counts = dict(zip(*np.unique(labels, return_counts=True)))
            model = CodeModel.load_from_checkpoint(str(ckpt_path), class_counts=class_counts)
            model.eval()
        else:
            logger.error(f"Checkpoint not found ({ckpt_path}). Aborting prediction.")
            raise FileNotFoundError(f"Checkpoint not found ({ckpt_path}). Aborting prediction.")

        # Pick dataset + dataloader
        if split == "train":
            dataset = datamodule.train_ds
            dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        else:
            dataset = datamodule.test_ds
            dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)

        # Run prediction
        preds = trainer.predict(model, dataloaders=dataloader)

        all_ids = []
        all_logits = []

        # each item: (ids_tensor, logits_tensor)
        for batch_ids, batch_logits in preds:  # type: ignore
            all_ids.append(batch_ids.cpu())
            all_logits.append(batch_logits.cpu())

        ids = torch.cat(all_ids, dim=0).numpy()
        logits = torch.cat(all_logits, dim=0).numpy()

        # Convert logits to probabilities for ROC/AUC
        probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()

        # Convert logits to predicted class labels
        predictions = probabilities.argmax(axis=1)

        # Build results dict
        results = {}
        for rid, p in zip(ids, predictions):
            repo = self.repository_db.get_repository(int(rid))

            if repo is None:
                logger.error(f"Repository with ID {rid} not found in the database.")
                raise ValueError(f"Repository with ID {rid} not found in the database.")

            if repo.label != "benign":
                repo.label = "malicious"

            results[int(rid)] = {
                "label": repo.label,
                "prediction": BINARY_LABEL_MAP[p],
            }

        # Optional persistence
        if persist:
            out = Path(model_dir) / f"{split}_predictions.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)
            logger.info(f"Saved predictions to {out}")

        return results, probabilities

    def evaluate(
        self,
        model_dir: Path | str,
        split: Literal["train", "test"] = "test",
        persist: bool = True,
    ) -> None:
        """
        Compute classification metrics (accuracy, precision, recall, f1) for the given split.

        Returns a dictionary with aggregated and per-class metrics.
        If `log_to_tb` is True, logs aggregated metrics to TensorBoard.
        """
        # Use predict to obtain per-repository results (includes true label + prediction)
        results, probabilities = self.predict(split=split, model_dir=model_dir, persist=True)

        true_indices: List[int] = []
        pred_indices: List[int] = []
        for rid, rec in results.items():
            true_label = rec["label"]
            pred_label = rec["prediction"]
            true_indices.append(BINARY_LABEL_TO_INDEX.get(true_label, -1))
            pred_indices.append(BINARY_LABEL_TO_INDEX.get(pred_label, -1))

        if any(i < 0 for i in true_indices):
            logger.warning("Some true labels were unmapped; metrics may be inaccurate")
        if any(i < 0 for i in pred_indices):
            logger.warning("Some predicted labels were unmapped; metrics may be inaccurate")

        num_classes = len(BINARY_LABEL_MAP)
        metrics = self._compute_classification_metrics(true_indices, pred_indices, num_classes, BINARY_LABEL_MAP, probabilities)

        logger.info(
            f"Evaluation ({split}) accuracy={metrics['accuracy']:.4f} \
            macro_f1={metrics['macro_f1']:.4f} weighted_f1={metrics['weighted_f1']:.4f}"
        )

        # Persist metrics
        if persist:
            out = Path(model_dir) / f"{split}_metrics.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Saved metrics to {out}")

    def test(self, model_dir: Path | str) -> None:
        """
        Convenience method to evaluate on the test set.
        """
        # Setup DataModule
        datamodule: CodeDataModule = self._setup_datamodule(
            full_dataset=self.full_dataset,
            validation_size=0.1,
            batch_size=self.config.batch_size,
        )

        # Setup trainer
        trainer = self._setup_trainer(epochs=self.config.epochs)

        ckpt_path = Path(model_dir) / "best.ckpt"
        if ckpt_path.exists():
            logger.info(f"Loading model from checkpoint: {ckpt_path}")
            labels = self.full_dataset.y.numpy(force=True)
            class_counts = dict(zip(*np.unique(labels, return_counts=True)))
            model = CodeModel.load_from_checkpoint(str(ckpt_path), class_counts=class_counts)
            model.eval()
        else:
            logger.error(f"Checkpoint not found ({ckpt_path}). Aborting testing.")
            raise FileNotFoundError(f"Checkpoint not found ({ckpt_path}). Aborting testing.")

        trainer.test(model, datamodule=datamodule)

    # -----------------------------
    # Helpers
    # -----------------------------
    def _fetch_data(self, split: Optional[Literal["train", "test"]] = None) -> CodeDataset:
        """
        Fetch repositories from DB and encode them to (X, ids, y) numpy arrays.
        Uses self.preprocessor to encode.
        """
        repositories = list(self.repository_db.get_repositories_with_cluster_ids(split=split, batch_size=self.config.batch_size))
        if not repositories:
            raise RuntimeError("No repositories returned from DB. Nothing to do.")

        # Make binary classification
        for i in range(len(repositories)):
            if repositories[i].label != "benign":
                repositories[i].label = "malicious"

        X, ids, y = self.preprocessor.encode_repos(repositories)
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(ids, np.ndarray):
            raise TypeError("Encoder must return numpy arrays for X, ids, and y.")

        ids_tensor = torch.tensor(ids, dtype=torch.long)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        return CodeDataset(ids=ids_tensor, X=X_tensor, y=y_tensor)

    @staticmethod
    def _get_best_checkpoint(trainer: Trainer) -> str | None:
        """
        Extract best checkpoint path from Trainer.
        """
        try:
            # lightning exposes the callback
            return getattr(trainer.checkpoint_callback, "best_model_path", None) or None
        except Exception:
            return None

    def _setup_datamodule(
        self,
        *,
        full_dataset: CodeDataset,
        validation_size: float = 0.2,
        batch_size: int = 32,
    ) -> CodeDataModule:
        # Get train/test splits from DB
        train_ids, train_labels = self._get_repository_ids_and_labels("train")
        test_ids, test_labels = self._get_repository_ids_and_labels("test")

        # Create validation split from training ids
        if len(train_ids) == 0:
            test_indices = [i - 1 for i in test_ids]

            return CodeDataModule(
                full_dataset=full_dataset,
                train_indices=[],
                val_indices=[],
                test_indices=test_indices,
                batch_size=batch_size,
            )

        train_ids, val_ids = train_test_split(
            train_ids,
            test_size=validation_size,
            stratify=train_labels,
            random_state=self.config.seed,
        )

        train_indices = [i - 1 for i in train_ids]
        val_indices = [i - 1 for i in val_ids]
        test_indices = [i - 1 for i in test_ids]

        return CodeDataModule(
            full_dataset=full_dataset,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            batch_size=batch_size,
        )

    def _setup_model(self) -> CodeModel:
        features = int(self.full_dataset.X.numpy().shape[1])
        classes = len(BINARY_LABEL_MAP)
        labels = self.full_dataset.y.numpy(force=True)
        class_counts = dict(zip(*np.unique(labels, return_counts=True)))
        return CodeModel(
            input_dim=features,
            output_dim=classes,
            lr=self.config.lr,
            class_counts=class_counts,
            weight_decay=self.config.weight_decay,
            l1_lambda=self.config.l1_lambda,
        )

    @staticmethod
    def _setup_trainer(
        *,
        epochs: int,
        run_dir: str = "tb_logs/repo_classifier",
        monitor: str = "val_acc",
        monitor_mode: str = "max",
        patience: int = 10,
        deterministic: bool = True,
        log_every_n_steps: int = 10,
    ) -> Trainer:
        tb_logger = TensorBoardLogger(save_dir=".", name=run_dir)

        ckpt = ModelCheckpoint(
            monitor=monitor,
            mode=monitor_mode,
            save_top_k=1,
            filename=f"epoch{{epoch:02d}}-{monitor}{{{monitor}:.4f}}",
            auto_insert_metric_name=False,
        )
        early = EarlyStopping(monitor=monitor, mode=monitor_mode, patience=patience, verbose=True)
        lrmon = LearningRateMonitor(logging_interval="epoch")

        precision = "16-mixed" if torch.cuda.is_available() else "32-true"

        return Trainer(
            max_epochs=epochs,
            accelerator="auto",
            devices="auto",
            precision=precision,
            logger=tb_logger,
            callbacks=[ckpt, early, lrmon],
            log_every_n_steps=log_every_n_steps,
            deterministic=deterministic,
        )

    def _persist_best_checkpoint_and_metadata(self, trainer: Trainer, best_ckpt: str | None, save_dir: str | Path) -> None:
        """
        Persist the best checkpoint (or latest if none) to a stable folder along with metadata
        needed to reload later. Sets `self.best_ckpt` to the saved checkpoint path.

        Files saved:
        - best.ckpt (Lightning checkpoint with weights and hyperparams)
        - index_to_label.json (mapping from class index to label)
        - config.json (training configuration)
        - meta.json (auxiliary info: source path, timestamp)
        """
        try:
            save_directory = Path(save_dir)
            save_directory.mkdir(parents=True, exist_ok=True)

            # Destination checkpoint path
            dest_ckpt = save_directory / "best.ckpt"
            if best_ckpt and os.path.isfile(best_ckpt):
                shutil.copy2(best_ckpt, dest_ckpt)
                source = best_ckpt
                source_type = "best"
            else:
                # Fall back to saving the current trainer/model state
                trainer.save_checkpoint(dest_ckpt)
                source = "trainer.save_checkpoint"
                source_type = "last"

            # Save training configuration used
            config_dict = self.config.model_dump()

            config_path = save_directory / "config.json"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, ensure_ascii=False, indent=2)

            # Save meta information
            meta = {
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "source": source,
                "source_type": source_type,
                "monitor": "val_acc",
                "monitor_mode": "max",
            }
            meta_path = save_directory / "meta.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            # Expose saved checkpoint path
            logger.info(f"Persisted best checkpoint to: {dest_ckpt}")
        except Exception as e:
            logger.warning(f"Failed to persist best checkpoint: {e}")

    # -----------------------------
    # Metric computation helpers
    # -----------------------------
    @staticmethod
    def _compute_classification_metrics(
        true_indices: list[int],
        pred_indices: list[int],
        num_classes: int,
        index_to_label: Dict[int, str],
        probabilities: np.ndarray | None = None,
    ) -> Dict[str, object]:
        supports = [0] * num_classes
        tp = [0] * num_classes
        fp = [0] * num_classes
        fn = [0] * num_classes

        for t, p in zip(true_indices, pred_indices):
            if 0 <= t < num_classes:
                supports[t] += 1
            # True positive
            if t == p and 0 <= t < num_classes:
                tp[t] += 1
            # False positive (predicted class p but was different true t)
            if p != t and 0 <= p < num_classes:
                fp[p] += 1
            # False negative (missed true class t)
            if p != t and 0 <= t < num_classes:
                fn[t] += 1

        per_class: Dict[str, Dict[str, float | int]] = {}
        precisions = []
        recalls = []
        f1s = []
        weighted_prec_sum = 0.0
        weighted_rec_sum = 0.0
        weighted_f1_sum = 0.0
        total = len(true_indices)
        correct = sum(1 for t, p in zip(true_indices, pred_indices) if t == p)

        for c in range(num_classes):
            p_den = tp[c] + fp[c]
            r_den = tp[c] + fn[c]
            precision_c = tp[c] / p_den if p_den > 0 else 0.0
            recall_c = tp[c] / r_den if r_den > 0 else 0.0
            f1_c = (2 * precision_c * recall_c / (precision_c + recall_c)) if (precision_c + recall_c) > 0 else 0.0

            label = index_to_label.get(c, str(c))
            per_class[label] = {
                "support": supports[c],
                "precision": precision_c,
                "recall": recall_c,
                "f1": f1_c,
                "tp": tp[c],
                "fp": fp[c],
                "fn": fn[c],
            }
            precisions.append(precision_c)
            recalls.append(recall_c)
            f1s.append(f1_c)
            weight = supports[c]
            weighted_prec_sum += precision_c * weight
            weighted_rec_sum += recall_c * weight
            weighted_f1_sum += f1_c * weight

        macro_precision = sum(precisions) / num_classes if num_classes else 0.0
        macro_recall = sum(recalls) / num_classes if num_classes else 0.0
        macro_f1 = sum(f1s) / num_classes if num_classes else 0.0
        weighted_precision = weighted_prec_sum / total if total else 0.0
        weighted_recall = weighted_rec_sum / total if total else 0.0
        weighted_f1 = weighted_f1_sum / total if total else 0.0
        accuracy = correct / total if total else 0.0

        # ---------------------------------------------------------
        # Confusion Matrix
        # ---------------------------------------------------------
        # We compute the raw N x N matrix.
        # labels parameter ensures the matrix is N x N even if some classes are missing.
        cm = confusion_matrix(true_indices, pred_indices, labels=list(range(num_classes)))
        cm_list = cm.tolist()  # Convert to standard python list for JSON serialization

        # ---------------------------------------------------------
        # AUC / ROC computation
        # ---------------------------------------------------------
        macro_auc = 0.0
        weighted_auc = 0.0
        auc_per_class = {}
        roc_curves_data = {}

        if probabilities is not None:
            try:
                # Ensure y_true is a numpy array for slicing
                y_true = np.array(true_indices)

                # One-hot encode true labels for aggregate AUC computation
                y_true_one_hot = np.eye(num_classes)[y_true]

                # 1. Calculate Aggregate AUCs (Macro and Weighted)
                try:
                    macro_auc = roc_auc_score(
                        y_true_one_hot,
                        probabilities,
                        multi_class="ovr",
                        average="macro",
                        labels=list(range(num_classes)),
                    )
                    weighted_auc = roc_auc_score(
                        y_true_one_hot,
                        probabilities,
                        multi_class="ovr",
                        average="weighted",
                        labels=list(range(num_classes)),
                    )
                except ValueError as e:
                    logger.warning(f"Could not calculate aggregate AUC (missing classes in test set): {e}")

                # 2. Calculate Per-Class AUC & ROC Points
                for c in range(num_classes):
                    label_name = index_to_label.get(c, str(c))

                    # Create binary targets: 1 for class c, 0 for others
                    y_true_binary = (y_true == c).astype(int)
                    y_score_binary = probabilities[:, c]

                    if np.sum(y_true_binary) > 0 and np.sum(y_true_binary) < len(y_true_binary):
                        # A. AUC Score
                        auc_c = roc_auc_score(y_true_binary, y_score_binary)
                        auc_per_class[label_name] = float(auc_c)

                        # B. ROC Curve Points
                        fpr, tpr, thresholds = roc_curve(y_true_binary, y_score_binary)
                        roc_curves_data[label_name] = {
                            "fpr": fpr.tolist(),
                            "tpr": tpr.tolist(),
                            "thresholds": thresholds.tolist(),
                        }
                    else:
                        auc_per_class[label_name] = None
                        roc_curves_data[label_name] = None

            except Exception as e:
                logger.error(f"Error computing AUC metrics: {e}")

        # ---------------------------------------------------------
        # RETURN METRICS
        # ---------------------------------------------------------
        return {
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1,
            "per_class": per_class,
            "confusion_matrix": cm_list,
            "auc_macro": macro_auc,
            "auc_weighted": weighted_auc,
            "auc_per_class": auc_per_class,
            "roc_curves": roc_curves_data,
        }
