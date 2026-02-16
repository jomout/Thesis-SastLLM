
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from lightning.pytorch.callbacks import Callback


class ConfusionMatrixCallback(Callback):
    def __init__(self, num_classes, label_map):
        super().__init__()
        self.num_classes = num_classes
        self.label_map = label_map

        self.val_preds = []
        self.val_targets = []

        self.test_preds = []
        self.test_targets = []

    # -------------------------
    # Validation
    # -------------------------
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Model must return logits from validation_step
        logits = outputs
        _, _, y = batch
        preds = logits.argmax(dim=1)

        self.val_preds.append(preds.cpu())
        self.val_targets.append(y.cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.val_preds:
            return

        preds = torch.cat(self.val_preds)
        targets = torch.cat(self.val_targets)
        self.val_preds.clear()
        self.val_targets.clear()

        cm = self._make_confusion_matrix(preds, targets)
        fig = self._plot_confusion_matrix(cm, title="Validation Confusion Matrix")

        trainer.logger.experiment.add_figure(
            "val/confusion_matrix", fig, trainer.current_epoch
        )
        plt.close(fig)

    # -------------------------
    # Test
    # -------------------------
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        logits = outputs
        _, _, y = batch
        preds = logits.argmax(dim=1)

        self.test_preds.append(preds.cpu())
        self.test_targets.append(y.cpu())

    def on_test_epoch_end(self, trainer, pl_module):
        if not self.test_preds:
            return

        preds = torch.cat(self.test_preds)
        targets = torch.cat(self.test_targets)
        self.test_preds.clear()
        self.test_targets.clear()

        cm = self._make_confusion_matrix(preds, targets)
        fig = self._plot_confusion_matrix(cm, title="Test Confusion Matrix")

        trainer.logger.experiment.add_figure(
            "test/confusion_matrix", fig, trainer.current_epoch
        )
        plt.close(fig)

    # -------------------------
    # Helpers
    # -------------------------
    def _make_confusion_matrix(self, preds, targets):
        cm = torch.zeros(self.num_classes, self.num_classes, dtype=torch.int64)
        for t, p in zip(targets, preds):
            cm[t, p] += 1
        return cm

    def _plot_confusion_matrix(self, cm, title):
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(
            cm.numpy(),
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[self.label_map[i] for i in range(self.num_classes)],
            yticklabels=[self.label_map[i] for i in range(self.num_classes)],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        return fig
