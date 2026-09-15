"""Validates every audio file in a directory and writes a JSON report.

Usage:
    python scripts/validate_dataset.py --input data/raw/librispeech_dummy
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.settings import get_audio_config
from src.audio.validate import validate_audio_file
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _file_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset(input_dir: Path) -> dict:
    all_extensions = set(get_audio_config()["io"]["supported_formats"]) | {
        "ogg", "m4a", "aac", "wma",  # also flag these as unsupported-but-present
    }
    audio_files = sorted(
        p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower().lstrip(".") in all_extensions
    )

    if not audio_files:
        logger.warning("No audio files found under %s", input_dir)

    report = {
        "input_dir": str(input_dir),
        "total_files": len(audio_files),
        "valid_files": 0,
        "invalid_files": 0,
        "duplicate_files": [],
        "sample_rate_distribution": defaultdict(int),
        "channel_distribution": defaultdict(int),
        "format_distribution": defaultdict(int),
        "duration_stats": {"min": None, "max": None, "total_seconds": 0.0},
        "issues": [],
    }

    hashes: dict[str, list[str]] = defaultdict(list)

    for path in audio_files:
        result = validate_audio_file(path)
        file_hash = _file_hash(path)
        hashes[file_hash].append(str(path))

        if result.is_valid:
            report["valid_files"] += 1
        else:
            report["invalid_files"] += 1

        if result.errors or result.warnings:
            report["issues"].append(
                {"path": str(path), "errors": result.errors, "warnings": result.warnings}
            )

        if result.info:
            report["sample_rate_distribution"][str(result.info.sample_rate)] += 1
            report["channel_distribution"][str(result.info.channels)] += 1
            report["format_distribution"][result.info.format] += 1
            d = result.info.duration_seconds
            report["duration_stats"]["total_seconds"] += d
            if report["duration_stats"]["min"] is None or d < report["duration_stats"]["min"]:
                report["duration_stats"]["min"] = d
            if report["duration_stats"]["max"] is None or d > report["duration_stats"]["max"]:
                report["duration_stats"]["max"] = d

    report["duplicate_files"] = [paths for paths in hashes.values() if len(paths) > 1]

    # defaultdicts don't serialize to plain JSON dicts cleanly in all cases
    report["sample_rate_distribution"] = dict(report["sample_rate_distribution"])
    report["channel_distribution"] = dict(report["channel_distribution"])
    report["format_distribution"] = dict(report["format_distribution"])

    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = validate_dataset(args.input)

    output_path = args.output or (args.input / "validation_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(
        "Validated %d files: %d valid, %d invalid, %d duplicate group(s). Report: %s",
        report["total_files"],
        report["valid_files"],
        report["invalid_files"],
        len(report["duplicate_files"]),
        output_path,
    )


if __name__ == "__main__":
    main()
