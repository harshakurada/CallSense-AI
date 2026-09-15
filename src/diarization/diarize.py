"""Diarization pipeline entrypoint: audio -> segmentation -> speaker
embeddings -> clustering -> timestamped speaker segments. This is the only
place that composes src/diarization's stages."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from configs.settings import get_audio_config
from src.audio.io import load_audio
from src.audio.preprocess import preprocess_audio
from src.audio.validate import validate_audio_file
from src.diarization.cluster import cluster_embeddings
from src.diarization.embeddings import embed_segment
from src.diarization.segment import get_diarization_segments
from src.utils.exceptions import DiarizationError
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SpeakerSegment:
    speaker: str
    start: float
    end: float


def diarize(audio_path: str | Path) -> list[SpeakerSegment]:
    """Returns generic SPEAKER_00/SPEAKER_01/... labeled segments for the
    given audio file. Does not attempt CUSTOMER/AGENT role mapping — see
    src/diarization/roles.py for that, applied separately and optionally."""
    audio_path = Path(audio_path)
    validation = validate_audio_file(audio_path)
    if not validation.is_valid:
        raise DiarizationError(f"{audio_path}: {'; '.join(validation.errors)}")

    io_config = get_audio_config()["io"]
    samples, sample_rate = load_audio(audio_path)
    samples, sample_rate = preprocess_audio(
        samples, sample_rate, target_sample_rate=io_config["target_sample_rate"], normalize=True
    )

    diarization_segments = get_diarization_segments(samples, sample_rate)
    if not diarization_segments:
        logger.warning("%s: no speech segments found for diarization", audio_path)
        return []

    embeddings = np.stack(
        [
            embed_segment(
                samples[int(seg.start * sample_rate) : int(seg.end * sample_rate)], sample_rate
            )
            for seg in diarization_segments
        ]
    )
    speaker_labels = cluster_embeddings(embeddings)

    result = [
        SpeakerSegment(speaker=label, start=seg.start, end=seg.end)
        for seg, label in zip(diarization_segments, speaker_labels)
    ]
    logger.info(
        "%s: %d diarization segment(s), %d distinct speaker(s)",
        audio_path,
        len(result),
        len(set(speaker_labels)),
    )
    return result
