"""Classification metrics — one implementation shared by the baseline and
transformer for every task, so numbers are computed identically and are
directly comparable in the model-comparison table."""
from dataclasses import dataclass

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    confusion_matrix: list[list[int]]
    label_names: list[str]
    per_class: dict[str, dict[str, float]]


def compute_metrics(y_true: list[str], y_pred: list[str], label_names: list[str]) -> ClassificationMetrics:
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_names, average=None, zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=label_names, average="macro", zero_division=0
    )

    per_class = {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(label_names)
    }

    cm = confusion_matrix(y_true, y_pred, labels=label_names)

    return ClassificationMetrics(
        accuracy=accuracy,
        precision_macro=float(precision_macro),
        recall_macro=float(recall_macro),
        f1_macro=float(f1_macro),
        confusion_matrix=cm.tolist(),
        label_names=label_names,
        per_class=per_class,
    )


def metrics_to_dict(metrics: ClassificationMetrics) -> dict:
    return {
        "accuracy": metrics.accuracy,
        "precision_macro": metrics.precision_macro,
        "recall_macro": metrics.recall_macro,
        "f1_macro": metrics.f1_macro,
        "confusion_matrix": metrics.confusion_matrix,
        "label_names": metrics.label_names,
        "per_class": metrics.per_class,
    }
