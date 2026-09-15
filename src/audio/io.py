"""Audio loading and saving. All other src/audio modules operate on the
(np.ndarray, sample_rate) pair these functions return/accept — never on a
file path directly — so they stay testable with synthetic in-memory audio."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from src.utils.exceptions import AudioProcessingError
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AudioInfo:
    path: str
    sample_rate: int
    channels: int
    duration_seconds: float
    frames: int
    format: str
    subtype: str


def get_audio_info(path: str | Path) -> AudioInfo:
    """Reads file header metadata only — does not load audio samples."""
    path = Path(path)
    if not path.exists():
        raise AudioProcessingError(f"Audio file not found: {path}")
    try:
        info = sf.info(str(path))
    except Exception as exc:  # soundfile raises its own LibsndfileError subclasses
        raise AudioProcessingError(f"Could not read audio header for {path}: {exc}") from exc

    return AudioInfo(
        path=str(path),
        sample_rate=info.samplerate,
        channels=info.channels,
        duration_seconds=info.duration,
        frames=info.frames,
        format=info.format,
        subtype=info.subtype,
    )


def load_audio(path: str | Path, dtype: str = "float32") -> tuple[np.ndarray, int]:
    """Loads full audio into memory as (samples, sample_rate).

    Returns samples shaped (n_frames,) for mono or (n_frames, n_channels)
    for multi-channel — callers needing mono should go through
    src.audio.preprocess.to_mono first.
    """
    path = Path(path)
    if not path.exists():
        raise AudioProcessingError(f"Audio file not found: {path}")
    try:
        samples, sample_rate = sf.read(str(path), dtype=dtype, always_2d=False)
    except Exception as exc:
        raise AudioProcessingError(f"Failed to decode audio file {path}: {exc}") from exc

    if samples.size == 0:
        raise AudioProcessingError(f"Audio file is empty: {path}")

    return samples, sample_rate


def save_audio(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sample_rate)
    logger.info("Saved audio to %s (%d Hz, %d samples)", path, sample_rate, len(samples))
