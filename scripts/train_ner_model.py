"""Trains the fine-tuned NER model and evaluates the spaCy+rules baseline,
writing real measured metrics to data/processed/nlp_evaluation/ner.json.

Usage:
    python scripts/train_ner_model.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nlp.ner.train import train
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    start = time.time()
    report = train(num_epochs=8)  # bert-tiny trains in ~2 min; more epochs is cheap — see docs/NER.md
    report["training_seconds"] = round(time.time() - start, 1)

    output_path = Path("data/processed/nlp_evaluation/ner.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        # seqeval's classification_report(output_dict=True) returns numpy
        # int64/float64 for support/score fields, which json can't
        # serialize natively — found when this crashed after a real
        # training run completed, losing nothing but the report write.
        json.dump(report, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    logger.info(
        "NER: baseline f1=%.4f, transformer f1=%.4f (%.1fs). Report: %s",
        report["baseline"]["f1"], report["transformer"]["f1"], report["training_seconds"], output_path,
    )


if __name__ == "__main__":
    main()
