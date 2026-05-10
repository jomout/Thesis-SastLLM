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
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
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
    batch_size: int = 32
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-5
    l1_lambda: float = 1e-3
    seed: int = 42
    k: int = 5000


class RepositoryClassifier:
    """
    Orchestrates: DB → encode → DataModule → Model → Trainer → metrics/persistence.
    """

    def __init__(self, *, config: ClassifierConfig) -> None:
        logger.debug("Initializing RepositoryClassifier.")
        self.config = config
        self.repository_db = RepositoryManager()
        self.preprocessor = RepositoryEncoder(config.k, BINARY_LABEL_TO_INDEX)

        seed_everything(config.seed, workers=True)

        self.full_dataset = self._fetch_data()
        logger.debug("RepositoryClassifier initialized.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, save_dir: Path | str) -> str:
        datamodule = self._build_datamodule(validation_size=0.1)
        trainer = self._build_trainer()
        model = self._build_model()

        trainer.fit(model, datamodule=datamodule)

        save_dir = Path(save_dir) / f"model_{_timestamp()}"
        save_dir.mkdir(parents=True, exist_ok=True)
        self._persist_checkpoint(trainer, save_dir)

        best = self._best_ckpt_path(trainer)
        logger.info(f"Training complete. Best checkpoint: {best}")
        return str(save_dir)

    def fit_k_fold(self, save_dir: Path | str, n_splits: int = 5) -> List[Dict]:
        logger.info(f"Starting {n_splits}-fold cross-validation.")

        train_ids, train_labels = self._ids_and_labels("train")
        test_ids, _ = self._ids_and_labels("test")
        X, y = np.array(train_ids), np.array(train_labels)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.config.seed)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\n========== Fold {fold + 1}/{n_splits} ==========\n")

            datamodule = CodeDataModule(
                full_dataset=self.full_dataset,
                train_indices=train_idx,
                val_indices=val_idx,
                test_indices=test_ids,
                batch_size=self.config.batch_size,
            )
            model = self._build_model(indices=train_idx)
            trainer = self._build_trainer()

            trainer.fit(model, datamodule=datamodule)

            metrics = trainer.validate(model, datamodule=datamodule, verbose=False)
            fold_results.append(
                {
                    "fold": fold,
                    "best_ckpt": self._best_ckpt_path(trainer),
                    "val_metrics": metrics[0] if metrics else {},
                }
            )

        save_dir = Path(save_dir) / f"model_{_timestamp()}"
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "fold_results.json").write_text(json.dumps(fold_results, indent=4), encoding="utf-8")
        logger.info("K-fold cross-validation complete.")
        return fold_results

    def predict(
        self,
        model_dir: Path | str,
        split: Literal["train", "test"] = "test",
        persist: bool = False,
    ) -> Tuple[Dict[int, Dict], np.ndarray]:
        model = self._load_checkpoint(model_dir)
        datamodule = self._build_datamodule(validation_size=0.1)
        datamodule.setup()

        dataset = datamodule.train_ds if split == "train" else datamodule.test_ds
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)

        trainer = self._build_trainer()
        preds = trainer.predict(model, dataloaders=dataloader)

        ids = torch.cat([b[0].cpu() for b in preds]).numpy()  # type: ignore
        logits = torch.cat([b[1].cpu() for b in preds]).numpy()  # type: ignore
        probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()
        predictions = probabilities.argmax(axis=1)

        results: Dict[int, Dict] = {}
        for rid, pred_idx in zip(ids, predictions):
            repo = self.repository_db.get_repository(int(rid))
            if repo is None:
                raise ValueError(f"Repository ID {rid} not found in DB.")
            label = "benign" if repo.label == "benign" else "malicious"
            results[int(rid)] = {"label": label, "prediction": BINARY_LABEL_MAP[pred_idx]}

        if persist:
            out = Path(model_dir) / f"{split}_predictions.json"
            out.write_text(json.dumps(results, indent=4), encoding="utf-8")
            logger.info(f"Saved predictions to {out}")

        return results, probabilities

    def evaluate(
        self,
        model_dir: Path | str,
        split: Literal["train", "test"] = "test",
        persist: bool = True,
    ) -> Dict:
        results, probabilities = self.predict(split=split, model_dir=model_dir, persist=True)

        true_indices = [BINARY_LABEL_TO_INDEX.get(r["label"], -1) for r in results.values()]
        pred_indices = [BINARY_LABEL_TO_INDEX.get(r["prediction"], -1) for r in results.values()]

        if any(i < 0 for i in true_indices):
            logger.warning("Some true labels were unmapped; metrics may be inaccurate.")
        if any(i < 0 for i in pred_indices):
            logger.warning("Some predicted labels were unmapped; metrics may be inaccurate.")

        metrics = _compute_metrics(
            true_indices,
            pred_indices,
            num_classes=len(BINARY_LABEL_MAP),
            index_to_label=BINARY_LABEL_MAP,
            probabilities=probabilities,
        )
        logger.info(
            f"Evaluation ({split}) accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} weighted_f1={metrics['weighted_f1']:.4f}"
        )

        if persist:
            out = Path(model_dir) / f"{split}_metrics.json"
            out.write_text(json.dumps(metrics, indent=4), encoding="utf-8")
            logger.info(f"Saved metrics to {out}")

        return metrics

    def test(self, model_dir: Path | str) -> None:
        model = self._load_checkpoint(model_dir)
        datamodule = self._build_datamodule(validation_size=0.1)
        self._build_trainer().test(model, datamodule=datamodule)

    # ------------------------------------------------------------------
    # Private builders  (one responsibility each)
    # ------------------------------------------------------------------

    def _fetch_data(self, split: Optional[Literal["train", "test"]] = None) -> CodeDataset:
        repos = list(self.repository_db.get_repositories_with_cluster_ids(split=split, batch_size=self.config.batch_size))
        if not repos:
            raise RuntimeError("No repositories returned from DB.")

        for r in repos:
            if r.label != "benign":
                r.label = "malicious"

        X, ids, y = self.preprocessor.encode_repos(repos)
        if not all(isinstance(a, np.ndarray) for a in (X, ids, y)):
            raise TypeError("Encoder must return numpy arrays for X, ids, and y.")

        return CodeDataset(
            ids=torch.tensor(ids, dtype=torch.long),
            X=torch.tensor(X, dtype=torch.float32),
            y=torch.tensor(y, dtype=torch.long),
        )

    def _ids_and_labels(self, split: Literal["train", "test"]) -> Tuple[List[int], List[int]]:
        ids, labels = [], []
        for repo in self.repository_db.get_repositories(split=split):
            ids.append(repo.repository_id)
            labels.append(repo.label)
        return ids, labels

    def _build_datamodule(self, validation_size: float = 0.2) -> CodeDataModule:
        train_ids, train_labels = self._ids_and_labels("train")
        test_ids, _ = self._ids_and_labels("test")
        test_indices = [i - 1 for i in test_ids]

        if not train_ids:
            return CodeDataModule(
                full_dataset=self.full_dataset,
                train_indices=[],
                val_indices=[],
                test_indices=test_indices,
                batch_size=self.config.batch_size,
            )

        train_ids, val_ids = train_test_split(
            train_ids,
            test_size=validation_size,
            stratify=train_labels,
            random_state=self.config.seed,
        )
        return CodeDataModule(
            full_dataset=self.full_dataset,
            train_indices=[i - 1 for i in train_ids],
            val_indices=[i - 1 for i in val_ids],
            test_indices=test_indices,
            batch_size=self.config.batch_size,
        )

    def _build_model(self, indices: Optional[np.ndarray] = None) -> CodeModel:
        labels = self.full_dataset.y[indices].numpy() if indices is not None else self.full_dataset.y.numpy(force=True)
        class_counts = dict(zip(*np.unique(labels, return_counts=True)))
        return CodeModel(
            input_dim=int(self.full_dataset.X.shape[1]),
            output_dim=len(BINARY_LABEL_MAP),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            l1_lambda=self.config.l1_lambda,
            class_counts=class_counts,
        )

    def _load_checkpoint(self, model_dir: Path | str) -> CodeModel:
        ckpt_path = Path(model_dir) / "best.ckpt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        labels = self.full_dataset.y.numpy(force=True)
        class_counts = dict(zip(*np.unique(labels, return_counts=True)))
        model = CodeModel.load_from_checkpoint(str(ckpt_path), class_counts=class_counts)
        model.eval()
        logger.info(f"Loaded checkpoint: {ckpt_path}")
        return model

    def _build_trainer(
        self,
        monitor: str = "val_acc",
        monitor_mode: str = "max",
        patience: int = 10,
    ) -> Trainer:
        tb_logger = TensorBoardLogger(save_dir=".", name="tb_logs/repo_classifier")
        ckpt_cb = ModelCheckpoint(
            monitor=monitor,
            mode=monitor_mode,
            save_top_k=1,
            filename=f"epoch{{epoch:02d}}-{monitor}{{{monitor}:.4f}}",
            auto_insert_metric_name=False,
        )
        return Trainer(
            max_epochs=self.config.epochs,
            accelerator="auto",
            devices="auto",
            precision="16-mixed" if torch.cuda.is_available() else "32-true",
            logger=tb_logger,
            callbacks=[
                ckpt_cb,
                EarlyStopping(monitor=monitor, mode=monitor_mode, patience=patience, verbose=True),
                LearningRateMonitor(logging_interval="epoch"),
            ],
            log_every_n_steps=10,
            deterministic=True,
        )

    def _persist_checkpoint(self, trainer: Trainer, save_dir: Path) -> None:
        best_src = self._best_ckpt_path(trainer)
        dest = save_dir / "best.ckpt"

        if best_src and os.path.isfile(best_src):
            shutil.copy2(best_src, dest)
            source_type = "best"
        else:
            trainer.save_checkpoint(dest)
            best_src = str(dest)
            source_type = "last"

        (save_dir / "config.json").write_text(json.dumps(self.config.model_dump(), indent=2), encoding="utf-8")
        (save_dir / "meta.json").write_text(
            json.dumps(
                {
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                    "source": best_src,
                    "source_type": source_type,
                    "monitor": "val_acc",
                    "monitor_mode": "max",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info(f"Persisted checkpoint to {dest}")

    @staticmethod
    def _best_ckpt_path(trainer: Trainer) -> str | None:
        return getattr(trainer.checkpoint_callback, "best_model_path", None) or None


# ------------------------------------------------------------------
# Module-level pure helpers (no class state needed)
# ------------------------------------------------------------------


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _compute_metrics(
    true_indices: List[int],
    pred_indices: List[int],
    num_classes: int,
    index_to_label: Dict[int, str],
    probabilities: Optional[np.ndarray] = None,
) -> Dict:
    supports = [0] * num_classes
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    for t, p in zip(true_indices, pred_indices):
        if 0 <= t < num_classes:
            supports[t] += 1
        if t == p and 0 <= t < num_classes:
            tp[t] += 1
        if p != t and 0 <= p < num_classes:
            fp[p] += 1
        if p != t and 0 <= t < num_classes:
            fn[t] += 1

    total = len(true_indices)
    correct = sum(t == p for t, p in zip(true_indices, pred_indices))
    per_class: Dict[str, Dict] = {}
    precisions, recalls, f1s = [], [], []
    w_prec = w_rec = w_f1 = 0.0

    for c in range(num_classes):
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        label = index_to_label.get(c, str(c))
        per_class[label] = {
            "support": supports[c],
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": tp[c],
            "fp": fp[c],
            "fn": fn[c],
        }
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        w_prec += prec * supports[c]
        w_rec += rec * supports[c]
        w_f1 += f1 * supports[c]

    cm = confusion_matrix(true_indices, pred_indices, labels=list(range(num_classes)))

    # AUC / ROC
    macro_auc = weighted_auc = 0.0
    auc_per_class: Dict[str, Optional[float]] = {}
    roc_curves_data: Dict = {}

    if probabilities is not None:
        y_true = np.array(true_indices)
        y_ohe = np.eye(num_classes)[y_true]
        try:
            macro_auc = roc_auc_score(y_ohe, probabilities, multi_class="ovr", average="macro", labels=list(range(num_classes)))
            weighted_auc = roc_auc_score(y_ohe, probabilities, multi_class="ovr", average="weighted", labels=list(range(num_classes)))
        except ValueError as e:
            logger.warning(f"Aggregate AUC skipped: {e}")

        for c in range(num_classes):
            label_name = index_to_label.get(c, str(c))
            y_bin = (y_true == c).astype(int)
            y_score = probabilities[:, c]
            if 0 < y_bin.sum() < len(y_bin):
                auc_per_class[label_name] = float(roc_auc_score(y_bin, y_score))
                fpr, tpr, thr = roc_curve(y_bin, y_score)
                roc_curves_data[label_name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thr.tolist()}
            else:
                auc_per_class[label_name] = None
                roc_curves_data[label_name] = None

    return {
        "accuracy": correct / total if total else 0.0,
        "macro_precision": sum(precisions) / num_classes if num_classes else 0.0,
        "macro_recall": sum(recalls) / num_classes if num_classes else 0.0,
        "macro_f1": sum(f1s) / num_classes if num_classes else 0.0,
        "weighted_precision": w_prec / total if total else 0.0,
        "weighted_recall": w_rec / total if total else 0.0,
        "weighted_f1": w_f1 / total if total else 0.0,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "auc_macro": macro_auc,
        "auc_weighted": weighted_auc,
        "auc_per_class": auc_per_class,
        "roc_curves": roc_curves_data,
    }
