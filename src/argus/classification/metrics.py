from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from argus.configs import get_logger

logger = get_logger(__name__)


def compute_classification_metrics(
    true_indices: list[int],
    pred_indices: list[int],
    *,
    index_to_label: dict[int, str],
    probabilities: np.ndarray | None = None,
) -> dict:
    if len(true_indices) != len(pred_indices):
        logger.exception("Classification metric inputs have different lengths", true_labels=len(true_indices), predictions=len(pred_indices))
        raise ValueError("true_indices and pred_indices must have the same length.")
    num_classes = len(index_to_label)
    supports = [0] * num_classes
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    for true, pred in zip(true_indices, pred_indices):
        if 0 <= true < num_classes:
            supports[true] += 1
        if true == pred and 0 <= true < num_classes:
            tp[true] += 1
        if pred != true and 0 <= pred < num_classes:
            fp[pred] += 1
        if pred != true and 0 <= true < num_classes:
            fn[true] += 1

    total = len(true_indices)
    correct = sum(true == pred for true, pred in zip(true_indices, pred_indices))
    per_class: dict[str, dict] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    weighted_precision = weighted_recall = weighted_f1 = 0.0

    for class_index in range(num_classes):
        precision = tp[class_index] / (tp[class_index] + fp[class_index]) if tp[class_index] + fp[class_index] else 0.0
        recall = tp[class_index] / (tp[class_index] + fn[class_index]) if tp[class_index] + fn[class_index] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        label = index_to_label.get(class_index, str(class_index))
        per_class[label] = {
            "support": supports[class_index],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp[class_index],
            "fp": fp[class_index],
            "fn": fn[class_index],
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        weighted_precision += precision * supports[class_index]
        weighted_recall += recall * supports[class_index]
        weighted_f1 += f1 * supports[class_index]

    metrics = {
        "accuracy": correct / total if total else 0.0,
        "macro_precision": sum(precisions) / num_classes if num_classes else 0.0,
        "macro_recall": sum(recalls) / num_classes if num_classes else 0.0,
        "macro_f1": sum(f1s) / num_classes if num_classes else 0.0,
        "weighted_precision": weighted_precision / total if total else 0.0,
        "weighted_recall": weighted_recall / total if total else 0.0,
        "weighted_f1": weighted_f1 / total if total else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(true_indices, pred_indices, labels=list(range(num_classes))).tolist(),
        "auc_macro": 0.0,
        "auc_weighted": 0.0,
        "auc_per_class": {},
        "roc_curves": {},
    }

    if probabilities is not None and total:
        metrics.update(_auc_metrics(np.array(true_indices), probabilities, index_to_label))

    logger.info(
        "Computed classification metrics",
        samples=total,
        classes=num_classes,
        accuracy=metrics["accuracy"],
        macro_f1=metrics["macro_f1"],
        weighted_f1=metrics["weighted_f1"],
        probabilities_supplied=probabilities is not None,
    )
    return metrics


def _auc_metrics(y_true: np.ndarray, probabilities: np.ndarray, index_to_label: dict[int, str]) -> dict:
    num_classes = len(index_to_label)
    y_ohe = np.eye(num_classes)[y_true]
    macro_auc = weighted_auc = 0.0
    auc_per_class: dict[str, float | None] = {}
    roc_curves: dict[str, dict | None] = {}

    try:
        macro_auc = roc_auc_score(y_ohe, probabilities, multi_class="ovr", average="macro", labels=list(range(num_classes)))
        weighted_auc = roc_auc_score(y_ohe, probabilities, multi_class="ovr", average="weighted", labels=list(range(num_classes)))
    except ValueError as e:
        logger.warning("Aggregate AUC skipped: %s", e)

    for class_index in range(num_classes):
        label = index_to_label.get(class_index, str(class_index))
        y_bin = (y_true == class_index).astype(int)
        y_score = probabilities[:, class_index]
        if 0 < y_bin.sum() < len(y_bin):
            auc_per_class[label] = float(roc_auc_score(y_bin, y_score))
            fpr, tpr, thresholds = roc_curve(y_bin, y_score)
            roc_curves[label] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()}
        else:
            auc_per_class[label] = None
            roc_curves[label] = None

    return {
        "auc_macro": macro_auc,
        "auc_weighted": weighted_auc,
        "auc_per_class": auc_per_class,
        "roc_curves": roc_curves,
    }
