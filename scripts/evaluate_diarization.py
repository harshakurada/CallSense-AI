"""Diarization Error Rate evaluation.

No real multi-speaker corpus with reference labels is downloaded yet (AMI
is documented but requires manual download — see docs/DATASETS.md), so this
builds a synthetic-but-real test call: two genuine LibriSpeech clips, one
pitch-shifted to stand in for a second voice, concatenated with known
boundaries we control exactly. The audio and the diarization result are
both real; only the *fact that these two segments come from different
people* is synthesized. See docs/DIARIZATION.md for why, and treat this
number as validating the pipeline's mechanics, not as a claim about
performance on genuine two-person conversations.

Usage:
    python scripts/evaluate_diarization.py
"""
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.diarization.diarize import SpeakerSegment, diarize
from src.diarization.evaluation import compute_der
from src.utils.logging import get_logger

logger = get_logger(__name__)

SR = 16000
DUMMY_MANIFEST = Path("data/raw/librispeech_dummy/manifest.jsonl")


def build_synthetic_two_speaker_call(output_path: Path) -> list[SpeakerSegment]:
    with open(DUMMY_MANIFEST, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]
    clip_a_entry, clip_b_entry = entries[0], entries[5]

    dummy_dir = DUMMY_MANIFEST.parent
    clip_a, _ = sf.read(str(dummy_dir / f"{clip_a_entry['call_id']}.flac"))
    clip_b, _ = sf.read(str(dummy_dir / f"{clip_b_entry['call_id']}.flac"))
    clip_a = clip_a.astype(np.float32)
    clip_b_shifted = librosa.effects.pitch_shift(clip_b.astype(np.float32), sr=SR, n_steps=7)

    gap_duration = 0.5
    gap = np.zeros(int(gap_duration * SR), dtype=np.float32)
    call = np.concatenate([clip_a, gap, clip_b_shifted, gap, clip_a])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), call, SR)

    # ground truth we control exactly, since we built the concatenation ourselves
    t1_end = len(clip_a) / SR
    t2_start = t1_end + gap_duration
    t2_end = t2_start + len(clip_b_shifted) / SR
    t3_start = t2_end + gap_duration
    t3_end = t3_start + len(clip_a) / SR

    return [
        SpeakerSegment(speaker="SPEAKER_A", start=0.0, end=t1_end),
        SpeakerSegment(speaker="SPEAKER_B", start=t2_start, end=t2_end),
        SpeakerSegment(speaker="SPEAKER_A", start=t3_start, end=t3_end),
    ]


def main():
    if not DUMMY_MANIFEST.exists():
        logger.error(
            "Missing %s — run: python scripts/download_data.py --dataset librispeech-dummy",
            DUMMY_MANIFEST,
        )
        sys.exit(1)

    test_audio_path = Path("data/processed/diarization_eval/synthetic_two_speaker_call.wav")
    reference = build_synthetic_two_speaker_call(test_audio_path)

    hypothesis = diarize(test_audio_path)
    result = compute_der(reference, hypothesis)

    report = {
        "test_audio": str(test_audio_path),
        "reference": [vars(s) for s in reference],
        "hypothesis": [vars(s) for s in hypothesis],
        "der": result.der,
        "correct": result.correct,
        "confusion": result.confusion,
        "missed_detection": result.missed_detection,
        "false_alarm": result.false_alarm,
        "total_reference_seconds": result.total_reference,
    }
    output_path = Path("data/processed/diarization_evaluation.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(
        "DER = %.4f (correct=%.2fs confusion=%.2fs missed=%.2fs false_alarm=%.2fs / total=%.2fs). Report: %s",
        result.der,
        result.correct,
        result.confusion,
        result.missed_detection,
        result.false_alarm,
        result.total_reference,
        output_path,
    )


if __name__ == "__main__":
    main()
