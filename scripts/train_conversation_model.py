"""Trains the conversation-level deep model and baseline, writing real
measured metrics to data/processed/nlp_evaluation/conversation.json.

Usage:
    python scripts/train_conversation_model.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.conversation.train import train
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    start = time.time()
    report = train()
    report["training_seconds"] = round(time.time() - start, 1)

    output_path = Path("data/processed/nlp_evaluation/conversation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Conversation model training complete (%.1fs). Report: %s", report["training_seconds"], output_path)


if __name__ == "__main__":
    main()
