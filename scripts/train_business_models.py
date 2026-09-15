"""Trains the resolution and escalation risk models, writing real
measured metrics to data/processed/nlp_evaluation/business.json.

Usage:
    python scripts/train_business_models.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.business.train import train
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    start = time.time()
    report = train()
    report["training_seconds"] = round(time.time() - start, 1)

    output_path = Path("data/processed/nlp_evaluation/business.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(
        "Business models trained (%.1fs): resolution f1=%.4f, escalation f1=%.4f. Report: %s",
        report["training_seconds"], report["resolution"]["f1_macro"], report["escalation"]["f1"], output_path,
    )


if __name__ == "__main__":
    main()
