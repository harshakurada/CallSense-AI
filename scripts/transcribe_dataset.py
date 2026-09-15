"""Batch-transcribes every audio file in a directory to
data/processed/transcripts/<call_id>.json. One failed file is logged and
skipped — it never stops the batch.

Usage:
    python scripts/transcribe_dataset.py --input data/raw/librispeech_dummy
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.settings import get_audio_config
from src.asr.transcribe import transcribe
from src.utils.logging import get_logger

logger = get_logger(__name__)


def transcribe_dataset(input_dir: Path, output_dir: Path) -> dict:
    supported = set(get_audio_config()["io"]["supported_formats"])
    audio_files = sorted(
        p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower().lstrip(".") in supported
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"total": len(audio_files), "succeeded": 0, "failed": 0, "failures": []}

    for path in audio_files:
        call_id = path.stem
        out_path = output_dir / f"{call_id}.json"
        try:
            start = time.time()
            result = transcribe(path, call_id=call_id)
            result["processing_seconds"] = round(time.time() - start, 3)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            summary["succeeded"] += 1
        except Exception as exc:
            logger.error("Failed to transcribe %s: %s", path, exc)
            summary["failed"] += 1
            summary["failures"].append({"path": str(path), "error": str(exc)})

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed/transcripts"))
    args = parser.parse_args()

    summary = transcribe_dataset(args.input, args.output)
    logger.info(
        "Batch transcription complete: %d/%d succeeded, %d failed",
        summary["succeeded"],
        summary["total"],
        summary["failed"],
    )
    if summary["failures"]:
        for failure in summary["failures"]:
            logger.error("  %s: %s", failure["path"], failure["error"])


if __name__ == "__main__":
    main()
