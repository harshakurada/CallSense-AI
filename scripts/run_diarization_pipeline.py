"""End-to-end pipeline: audio -> preprocessing -> Whisper -> diarization ->
speaker-attributed transcript.

Usage:
    python scripts/run_diarization_pipeline.py --input path/to/call.wav
    python scripts/run_diarization_pipeline.py --input path/to/call.wav --apply-role-heuristic
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.pipeline import run_pipeline
from src.utils.exceptions import CallSenseError
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--apply-role-heuristic",
        action="store_true",
        help="Apply the unverified 'first speaker = AGENT' heuristic instead of generic SPEAKER_NN labels",
    )
    args = parser.parse_args()

    try:
        result = run_pipeline(args.input, apply_role_heuristic=args.apply_role_heuristic or None)
    except CallSenseError as exc:
        logger.error(str(exc))
        sys.exit(1)

    output_json = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_json, encoding="utf-8")
        logger.info("Wrote conversation to %s", args.output)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
