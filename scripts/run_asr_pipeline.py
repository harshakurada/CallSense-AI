"""End-to-end single-file pipeline: validate -> preprocess -> ASR ->
timestamped transcript -> JSON output.

Usage:
    python scripts/run_asr_pipeline.py --input path/to/call.wav
    python scripts/run_asr_pipeline.py --input path/to/call.wav --output out.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.asr.transcribe import transcribe
from src.audio.validate import validate_audio_file
from src.utils.exceptions import CallSenseError
from src.utils.logging import get_logger

logger = get_logger(__name__)


def run_pipeline(input_path: Path) -> dict:
    validation = validate_audio_file(input_path)
    if not validation.is_valid:
        raise CallSenseError(f"{input_path}: {'; '.join(validation.errors)}")
    for warning in validation.warnings:
        logger.warning("%s: %s", input_path, warning)

    return transcribe(input_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    try:
        result = run_pipeline(args.input)
    except CallSenseError as exc:
        logger.error(str(exc))
        sys.exit(1)

    output_json = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_json, encoding="utf-8")
        logger.info("Wrote transcript to %s", args.output)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
