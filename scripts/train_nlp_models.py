"""Trains the baseline + fine-tuned Transformer for intent, sentiment,
and/or emotion, and writes real measured metrics to
data/processed/nlp_evaluation/<task>.json.

Usage:
    python scripts/train_nlp_models.py --task intent
    python scripts/train_nlp_models.py --task all
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging import get_logger

logger = get_logger(__name__)

TASKS = ["intent", "sentiment", "emotion"]


def train_task(task: str, num_epochs: int) -> dict:
    if task == "intent":
        from src.nlp.intent.train import train
    elif task == "sentiment":
        from src.nlp.sentiment.train import train
    elif task == "emotion":
        from src.nlp.emotion.train import train
    else:
        raise ValueError(f"Unknown task '{task}'")

    start = time.time()
    report = train(num_epochs=num_epochs)
    report["training_seconds"] = round(time.time() - start, 1)

    output_path = Path("data/processed/nlp_evaluation") / f"{task}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(
        "%s: baseline macro_f1=%.4f, transformer macro_f1=%.4f (%.1fs). Report: %s",
        task,
        report["baseline"]["f1_macro"],
        report["transformer"]["f1_macro"],
        report["training_seconds"],
        output_path,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS + ["all"], default="all")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    tasks = TASKS if args.task == "all" else [args.task]
    for task in tasks:
        train_task(task, args.epochs)


if __name__ == "__main__":
    main()
