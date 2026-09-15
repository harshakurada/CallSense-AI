"""High-level audio preprocessing entrypoint: validate -> load -> mono ->
resample -> normalize -> VAD -> chunk. Used by scripts/run_asr_pipeline.py
and src/asr/transcribe.py; nothing else should re-implement this sequence.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from configs.settings import get_audio_config
from src.audio.chunking import AudioChunk, chunk_audio
from src.audio.io import load_audio
from src.audio.preprocess import preprocess_audio
from src.audio.validate import validate_audio_file
from src.audio.vad import SpeechRegion, detect_speech_regions
from src.utils.exceptions import AudioProcessingError
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PreprocessedAudio:
    path: str
    samples: np.ndarray  # mono, resampled, normalized — the actual signal ASR should consume
    sample_rate: int
    duration_seconds: float
    speech_regions: list[SpeechRegion]
    chunks: list[AudioChunk]


def preprocess_for_asr(path: str | Path) -> PreprocessedAudio:
    path = Path(path)
    validation = validate_audio_file(path)
    if not validation.is_valid:
        raise AudioProcessingError(f"{path}: {'; '.join(validation.errors)}")
    for warning in validation.warnings:
        logger.warning("%s: %s", path, warning)

    io_config = get_audio_config()["io"]
    norm_config = get_audio_config()["normalization"]

    samples, sample_rate = load_audio(path)
    samples, sample_rate = preprocess_audio(
        samples,
        sample_rate,
        target_sample_rate=io_config["target_sample_rate"],
        normalize=norm_config["enabled"],
        target_peak_dbfs=norm_config["target_peak_dbfs"],
    )

    speech_regions = detect_speech_regions(samples, sample_rate)
    if not speech_regions:
        logger.warning("%s: no speech detected by VAD — proceeding with full audio anyway", path)

    chunks = chunk_audio(samples, sample_rate)
    logger.info(
        "%s: preprocessed to %.1fs @ %dHz, %d speech region(s), %d chunk(s)",
        path,
        len(samples) / sample_rate,
        sample_rate,
        len(speech_regions),
        len(chunks),
    )

    return PreprocessedAudio(
        path=str(path),
        samples=samples,
        sample_rate=sample_rate,
        duration_seconds=len(samples) / sample_rate,
        speech_regions=speech_regions,
        chunks=chunks,
    )
