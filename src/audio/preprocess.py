"""Reusable, non-destructive audio transforms. Each function takes and
returns (samples, sample_rate) so they compose in any order the pipeline
needs. Deliberately conservative — no compression/limiting/aggressive
denoising that would distort the signal emotion detection (Module 4) later
depends on."""
import numpy as np
import librosa

from src.utils.exceptions import AudioProcessingError
from src.utils.logging import get_logger

logger = get_logger(__name__)


def to_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples
    if samples.ndim == 2:
        return np.mean(samples, axis=1)
    raise AudioProcessingError(f"Unsupported audio array shape: {samples.shape}")


def resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> tuple[np.ndarray, int]:
    if orig_sr == target_sr:
        return samples, orig_sr
    resampled = librosa.resample(samples.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)
    return resampled, target_sr


def normalize_peak(samples: np.ndarray, target_peak_dbfs: float = -3.0) -> np.ndarray:
    """Scales the signal so its peak amplitude sits at target_peak_dbfs.
    A no-op on (near-)silent audio to avoid amplifying noise to full scale."""
    peak = np.max(np.abs(samples))
    if peak < 1e-6:
        logger.warning("Skipping normalization: audio is silent (peak=%.2e)", peak)
        return samples

    target_amplitude = 10 ** (target_peak_dbfs / 20)
    gain = target_amplitude / peak
    return samples * gain


def preprocess_audio(
    samples: np.ndarray,
    sample_rate: int,
    target_sample_rate: int,
    normalize: bool = True,
    target_peak_dbfs: float = -3.0,
) -> tuple[np.ndarray, int]:
    """Standard preprocessing chain: mono -> resample -> (optional) normalize."""
    samples = to_mono(samples)
    samples, sample_rate = resample(samples, sample_rate, target_sample_rate)
    if normalize:
        samples = normalize_peak(samples, target_peak_dbfs)
    return samples, sample_rate
