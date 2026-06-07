from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger


class AccuracyHistoryRecord(TypedDict):
    epoch: int
    train_acc: float | None
    val_acc: float | None


class AccuracyHistoryCallback(Callback):
    def __init__(self) -> None:
        self.records: list[AccuracyHistoryRecord] = []

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        record: AccuracyHistoryRecord = {
            "epoch": int(trainer.current_epoch),
            "train_acc": _metric_value(metrics.get("train_acc")),
            "val_acc": _metric_value(metrics.get("val_acc")),
        }
        self.records.append(record)


def build_trainer(
    *,
    max_epochs: int,
    monitor: str = "val_acc",
    monitor_mode: str = "max",
    patience: int = 10,
    log_dir: str = ".",
    logger_name: str = "tb_logs/repo_classifier",
    extra_callbacks: Sequence[Callback] | None = None,
) -> Trainer:
    tb_logger = TensorBoardLogger(save_dir=log_dir, name=logger_name)
    checkpoint = ModelCheckpoint(
        monitor=monitor,
        mode=monitor_mode,
        save_top_k=1,
        filename=f"epoch{{epoch:02d}}-{monitor}{{{monitor}:.4f}}",
        auto_insert_metric_name=False,
    )
    callbacks: list[Callback] = [
        checkpoint,
        EarlyStopping(monitor=monitor, mode=monitor_mode, patience=patience, verbose=True),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    return Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        logger=tb_logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        deterministic=True,
    )


def best_checkpoint_path(trainer: Trainer) -> str | None:
    return getattr(trainer.checkpoint_callback, "best_model_path", None) or None


def ensure_model_dir(path: str | Path) -> Path:
    model_dir = Path(path)
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def _metric_value(value) -> float | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        return float(value.detach().cpu().item())
    return float(value)
