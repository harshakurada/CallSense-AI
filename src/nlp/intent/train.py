"""Trains and saves both the baseline and fine-tuned intent classifier."""
from pathlib import Path

from src.nlp.common.baseline import evaluate_baseline, save_baseline, train_baseline
from src.nlp.common.metrics import metrics_to_dict
from src.nlp.common.transformer import fine_tune_classifier
from src.nlp.intent.dataset import load_intent_dataset
from src.utils.logging import get_logger

logger = get_logger(__name__)

MODEL_DIR = Path("models/nlp/intent")


def train(num_epochs: int = 3) -> dict:
    data = load_intent_dataset()
    train_texts, train_labels = data["train"]
    val_texts, val_labels = data["val"]
    test_texts, test_labels = data["test"]
    label_names = data["label_names"]

    logger.info(
        "Intent: %d train / %d val / %d test, %d classes, full distribution=%s",
        len(train_texts), len(val_texts), len(test_texts), len(label_names), data["full_distribution"],
    )

    baseline = train_baseline(train_texts, train_labels)
    baseline_metrics = evaluate_baseline(baseline, test_texts, test_labels, label_names)
    save_baseline(baseline, MODEL_DIR / "baseline.pkl")
    logger.info("Intent baseline: accuracy=%.4f macro_f1=%.4f", baseline_metrics.accuracy, baseline_metrics.f1_macro)

    transformer_result = fine_tune_classifier(
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels,
        label_names, output_dir=MODEL_DIR / "transformer", num_epochs=num_epochs,
    )
    logger.info(
        "Intent transformer: accuracy=%.4f macro_f1=%.4f",
        transformer_result.metrics.accuracy, transformer_result.metrics.f1_macro,
    )

    return {
        "task": "intent",
        "dataset_size": {"train": len(train_texts), "val": len(val_texts), "test": len(test_texts)},
        "label_names": label_names,
        "full_distribution": data["full_distribution"],
        "baseline": metrics_to_dict(baseline_metrics),
        "transformer": metrics_to_dict(transformer_result.metrics),
    }
