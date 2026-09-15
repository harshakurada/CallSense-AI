"""Runs the full unified conversation analysis (intent, sentiment,
emotion, entities, resolution, escalation) on a speaker-attributed
conversation.

Usage:
    python scripts/run_business_pipeline.py --input conversation.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.business.pipeline import analyze_call
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to a conversation JSON file, or '-' for stdin")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    raw = sys.stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8")
    data = json.loads(raw)
    conversation = data["conversation"] if "conversation" in data else data

    result = analyze_call(conversation)
    output_json = json.dumps(result, indent=2)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_json, encoding="utf-8")
        logger.info("Wrote analysis to %s", args.output)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
