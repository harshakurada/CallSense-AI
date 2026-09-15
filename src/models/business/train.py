"""Trains the resolution and escalation models on the same synthetic
conversation dataset as Module 6 (same labels, same known synthetic-data
caveats — see docs/CONVERSATION_MODEL.md and docs/BUSINESS_MODELS.md)."""
import numpy as np

from src.models.business.escalation import EscalationModel, save_model as save_escalation
from src.models.business.resolution import ResolutionModel, save_model as save_resolution
from src.models.conversation.synthetic import generate_dataset
from src.nlp.common.metrics import metrics_to_dict
from src.utils.logging import get_logger

logger = get_logger(__name__)

N_EXAMPLES = 1200


def _split(examples: list[dict], val_frac: float = 0.0, test_frac: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(examples))
    n_test = int(len(examples) * test_frac)
    test_idx, train_idx = indices[:n_test], indices[n_test:]
    return [examples[i] for i in train_idx], [examples[i] for i in test_idx]


def train(n_examples: int = N_EXAMPLES) -> dict:
    examples = generate_dataset(n_examples)
    train_examples, test_examples = _split(examples)
    logger.info("Business models: %d train / %d test conversations", len(train_examples), len(test_examples))

    train_conversations = [ex["conversation"] for ex in train_examples]
    test_conversations = [ex["conversation"] for ex in test_examples]

    # --- resolution ---
    resolution_train_labels = [ex["labels"]["resolution"] for ex in train_examples]
    resolution_test_labels = [ex["labels"]["resolution"] for ex in test_examples]

    resolution_model = ResolutionModel()
    resolution_model.fit(train_conversations, resolution_train_labels)
    resolution_metrics = resolution_model.evaluate(test_conversations, resolution_test_labels)
    save_resolution(resolution_model)
    logger.info(
        "Resolution: accuracy=%.4f macro_f1=%.4f", resolution_metrics.accuracy, resolution_metrics.f1_macro
    )

    # --- escalation ---
    escalation_train_labels = [ex["labels"]["escalation"] for ex in train_examples]
    escalation_test_labels = [ex["labels"]["escalation"] for ex in test_examples]

    escalation_model = EscalationModel()
    escalation_model.fit(train_conversations, escalation_train_labels)
    escalation_metrics = escalation_model.evaluate(test_conversations, escalation_test_labels)
    save_escalation(escalation_model)
    logger.info(
        "Escalation: f1=%.4f roc_auc=%.4f brier=%.4f",
        escalation_metrics["f1"], escalation_metrics["roc_auc"], escalation_metrics["brier_score"],
    )

    return {
        "dataset_size": {"train": len(train_examples), "test": len(test_examples)},
        "resolution": metrics_to_dict(resolution_metrics),
        "resolution_feature_importances": resolution_model.feature_importances(),
        "escalation": escalation_metrics,
        "escalation_feature_importances": escalation_model.feature_importances(),
    }
