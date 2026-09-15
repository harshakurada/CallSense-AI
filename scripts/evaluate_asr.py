"""Computes WER for a transcribed dataset against its ground-truth manifest.
Only runs where a manifest with reference_transcript entries exists — never
estimates WER without ground truth.

Usage:
    python scripts/evaluate_asr.py \
        --manifest data/raw/librispeech_dummy/manifest.jsonl \
        --transcripts data/processed/transcripts
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.asr.evaluation import compute_corpus_wer, compute_wer
from src.utils.logging import get_logger

logger = get_logger(__name__)


def evaluate(manifest_path: Path, transcripts_dir: Path) -> dict:
    references, hypotheses, per_file = [], [], []

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            call_id = entry["call_id"]
            transcript_path = transcripts_dir / f"{call_id}.json"
            if not transcript_path.exists():
                logger.warning("No transcript found for %s, skipping", call_id)
                continue

            with open(transcript_path, "r", encoding="utf-8") as tf:
                transcript = json.load(tf)
            hypothesis = " ".join(seg["text"] for seg in transcript["segments"])
            reference = entry["reference_transcript"]

            references.append(reference)
            hypotheses.append(hypothesis)

            per_file_result = compute_wer(reference, hypothesis)
            per_file.append(
                {
                    "call_id": call_id,
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "wer": per_file_result.wer,
                    "substitutions": per_file_result.substitutions,
                    "deletions": per_file_result.deletions,
                    "insertions": per_file_result.insertions,
                }
            )

    corpus_result = compute_corpus_wer(references, hypotheses)
    per_file.sort(key=lambda r: r["wer"], reverse=True)

    return {
        "num_files_evaluated": len(references),
        "corpus_wer": corpus_result.wer,
        "corpus_substitutions": corpus_result.substitutions,
        "corpus_deletions": corpus_result.deletions,
        "corpus_insertions": corpus_result.insertions,
        "corpus_reference_words": corpus_result.reference_word_count,
        "worst_files": per_file[:10],
        "per_file": per_file,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--transcripts", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed/asr_evaluation.json"))
    args = parser.parse_args()

    report = evaluate(args.manifest, args.transcripts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(
        "Evaluated %d files: corpus WER = %.4f (%d sub, %d del, %d ins / %d ref words). Report: %s",
        report["num_files_evaluated"],
        report["corpus_wer"],
        report["corpus_substitutions"],
        report["corpus_deletions"],
        report["corpus_insertions"],
        report["corpus_reference_words"],
        args.output,
    )


if __name__ == "__main__":
    main()
