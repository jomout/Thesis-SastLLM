from __future__ import annotations

from pathlib import Path

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger


def build_trainer(
    *,
    max_epochs: int,
    monitor: str = "val_acc",
    monitor_mode: str = "max",
    patience: int = 10,
    log_dir: str = ".",
    logger_name: str = "tb_logs/repo_classifier",
) -> Trainer:
    tb_logger = TensorBoardLogger(save_dir=log_dir, name=logger_name)
    checkpoint = ModelCheckpoint(
        monitor=monitor,
        mode=monitor_mode,
        save_top_k=1,
        filename=f"epoch{{epoch:02d}}-{monitor}{{{monitor}:.4f}}",
        auto_insert_metric_name=False,
    )
    return Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        logger=tb_logger,
        callbacks=[
            checkpoint,
            EarlyStopping(monitor=monitor, mode=monitor_mode, patience=patience, verbose=True),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        log_every_n_steps=10,
        deterministic=True,
    )


def best_checkpoint_path(trainer: Trainer) -> str | None:
    return getattr(trainer.checkpoint_callback, "best_model_path", None) or None


def ensure_model_dir(path: str | Path) -> Path:
    model_dir = Path(path)
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir
