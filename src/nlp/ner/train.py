"""Trains the fine-tuned NER model and evaluates the spaCy+rules baseline
on the same test split, for a real, comparable entity-level P/R/F1 table."""
from pathlib import Path

from seqeval.metrics import classification_report as seqeval_report
from seqeval.metrics import f1_score, precision_score, recall_score

from src.nlp.ner.baseline import extract_baseline_entities
from src.nlp.ner.dataset import load_ner_dataset
from src.nlp.ner.model import fine_tune_ner
from src.utils.logging import get_logger

logger = get_logger(__name__)

MODEL_DIR = Path("models/nlp/ner/transformer")


def _entities_to_bio(tokens: list[str], entities) -> list[str]:
    """Converts baseline entities (character-offset spans over the
    detokenized text) back onto the token sequence, so the baseline can be
    scored with the same seqeval BIO metric as the fine-tuned model."""
    text = " ".join(tokens)
    # recompute each token's character span in the joined string
    spans = []
    cursor = 0
    for tok in tokens:
        start = text.index(tok, cursor)
        end = start + len(tok)
        spans.append((start, end))
        cursor = end

    tags = ["O"] * len(tokens)
    for entity in entities:
        first_inside = None
        for i, (tok_start, tok_end) in enumerate(spans):
            if tok_start < entity.end and tok_end > entity.start:
                tags[i] = f"I-{entity.label}" if first_inside is not None else f"B-{entity.label}"
                first_inside = i
    return tags


def _bio_metrics(true_tags: list[list[str]], pred_tags: list[list[str]]) -> dict:
    return {
        "precision": precision_score(true_tags, pred_tags),
        "recall": recall_score(true_tags, pred_tags),
        "f1": f1_score(true_tags, pred_tags),
        "report": seqeval_report(true_tags, pred_tags, output_dict=True, zero_division=0),
    }


def evaluate_baseline_on_split(test_tokens: list[list[str]], test_tags: list[list[str]]) -> dict:
    pred_tags = [_entities_to_bio(tokens, extract_baseline_entities(" ".join(tokens))) for tokens in test_tokens]
    metrics = _bio_metrics(test_tags, pred_tags)
    metrics["pred_tags"] = pred_tags  # exposed so callers can re-slice by subset (e.g. real vs synthetic)
    return metrics


def train(num_epochs: int = 3) -> dict:
    data = load_ner_dataset()
    train_tokens, train_tags = data["train"]
    val_tokens, val_tags = data["val"]
    test_tokens, test_tags = data["test"]
    label_names = data["label_names"]

    logger.info(
        "NER: %d train / %d val / %d test (real=%d, synthetic=%d)",
        len(train_tokens), len(val_tokens), len(test_tokens), data["counts"]["real"], data["counts"]["synthetic"],
    )

    baseline_metrics = evaluate_baseline_on_split(test_tokens, test_tags)
    logger.info(
        "NER baseline (spaCy+rules): precision=%.4f recall=%.4f f1=%.4f",
        baseline_metrics["precision"], baseline_metrics["recall"], baseline_metrics["f1"],
    )

    transformer_metrics, _ = fine_tune_ner(
        train_tokens, train_tags, val_tokens, val_tags, test_tokens, test_tags,
        label_names, output_dir=MODEL_DIR, num_epochs=num_epochs,
    )
    logger.info(
        "NER transformer: precision=%.4f recall=%.4f f1=%.4f",
        transformer_metrics.precision, transformer_metrics.recall, transformer_metrics.f1,
    )

    # Real vs. synthetic test subsets scored separately — synthetic
    # templates are easier than genuine text, so a blended number alone
    # would overstate real-world performance (see docs/NER.md).
    real_idx = [i for i, s in enumerate(data["test_sources"]) if s == "real"]
    synthetic_idx = [i for i, s in enumerate(data["test_sources"]) if s == "synthetic"]

    def _subset_metrics(true_tags_all, pred_tags_all):
        real = _bio_metrics([true_tags_all[i] for i in real_idx], [pred_tags_all[i] for i in real_idx])
        synthetic = _bio_metrics([true_tags_all[i] for i in synthetic_idx], [pred_tags_all[i] for i in synthetic_idx])
        return {"real": real, "synthetic": synthetic}

    baseline_by_source = _subset_metrics(test_tags, baseline_metrics["pred_tags"])
    transformer_by_source = _subset_metrics(transformer_metrics.true_labels, transformer_metrics.pred_labels)

    return {
        "task": "ner",
        "dataset_size": {"train": len(train_tokens), "val": len(val_tokens), "test": len(test_tokens)},
        "test_composition": {"real": len(real_idx), "synthetic": len(synthetic_idx)},
        "label_names": label_names,
        "baseline": {k: v for k, v in baseline_metrics.items() if k != "pred_tags"},
        "baseline_by_source": baseline_by_source,
        "transformer": {
            "precision": transformer_metrics.precision,
            "recall": transformer_metrics.recall,
            "f1": transformer_metrics.f1,
            "report": transformer_metrics.report,
        },
        "transformer_by_source": transformer_by_source,
    }
