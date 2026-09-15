import numpy as np
import pytest
import soundfile as sf

from src.audio.chunking import chunk_audio
from src.audio.io import get_audio_info, load_audio
from src.audio.pipeline import preprocess_for_asr
from src.audio.preprocess import normalize_peak, resample, to_mono
from src.audio.validate import validate_audio_file
from src.audio.vad import detect_speech_regions
from src.utils.exceptions import AudioProcessingError

SR = 16000


def _tone(duration_s: float, sr: int = SR, freq: float = 440.0, amplitude: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(duration_s: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(sr * duration_s), dtype=np.float32)


def _write_wav(tmp_path, name: str, samples: np.ndarray, sr: int = SR):
    path = tmp_path / name
    sf.write(str(path), samples, sr)
    return path


# --- valid audio ---


def test_valid_audio_passes_validation(tmp_path):
    path = _write_wav(tmp_path, "valid.wav", _tone(2.0))
    result = validate_audio_file(path)
    assert result.is_valid
    assert not result.errors
    assert result.info.sample_rate == SR


def test_valid_audio_loads(tmp_path):
    path = _write_wav(tmp_path, "valid.wav", _tone(1.0))
    samples, sr = load_audio(path)
    assert sr == SR
    assert len(samples) == SR


# --- invalid / missing audio ---


def test_missing_file_is_invalid(tmp_path):
    result = validate_audio_file(tmp_path / "does_not_exist.wav")
    assert not result.is_valid
    assert "does not exist" in result.errors[0].lower()


def test_empty_file_is_invalid(tmp_path):
    path = tmp_path / "empty.wav"
    path.write_bytes(b"")
    result = validate_audio_file(path)
    assert not result.is_valid


# --- corrupt audio ---


def test_corrupt_audio_is_invalid(tmp_path):
    path = tmp_path / "corrupt.wav"
    path.write_bytes(b"RIFF....WAVEfmt garbage not a real wav file" * 10)
    result = validate_audio_file(path)
    assert not result.is_valid
    assert any("corrupt" in e.lower() for e in result.errors)


def test_load_audio_raises_on_corrupt_file(tmp_path):
    path = tmp_path / "corrupt.wav"
    path.write_bytes(b"not audio data" * 100)
    with pytest.raises(AudioProcessingError):
        load_audio(path)


# --- short / long audio ---


def test_too_short_audio_is_invalid(tmp_path):
    path = _write_wav(tmp_path, "short.wav", _tone(0.05))  # below min_duration_seconds
    result = validate_audio_file(path)
    assert not result.is_valid
    assert any("below minimum" in e for e in result.errors)


def test_long_audio_gets_chunked(tmp_path):
    path = _write_wav(tmp_path, "long.wav", _tone(75.0))  # > 30s chunk_duration
    result = preprocess_for_asr(path)
    assert result.duration_seconds == pytest.approx(75.0, abs=0.1)
    assert len(result.chunks) >= 3


# --- unsupported format ---


def test_unsupported_extension_is_invalid(tmp_path):
    path = tmp_path / "clip.ogg"
    path.write_bytes(b"fake ogg content")
    result = validate_audio_file(path)
    assert not result.is_valid
    assert "unsupported format" in result.errors[0].lower()


# --- preprocessing units ---


def test_to_mono_averages_channels():
    stereo = np.stack([np.ones(100), np.zeros(100)], axis=1).astype(np.float32)
    mono = to_mono(stereo)
    assert mono.shape == (100,)
    assert np.allclose(mono, 0.5)


def test_resample_changes_length_and_rate():
    samples = _tone(1.0, sr=16000)
    resampled, sr = resample(samples, orig_sr=16000, target_sr=8000)
    assert sr == 8000
    assert len(resampled) == pytest.approx(8000, rel=0.01)


def test_normalize_peak_scales_to_target():
    samples = _tone(0.5, amplitude=0.1)
    normalized = normalize_peak(samples, target_peak_dbfs=-3.0)
    expected_peak = 10 ** (-3.0 / 20)
    assert np.max(np.abs(normalized)) == pytest.approx(expected_peak, rel=1e-3)


def test_normalize_peak_is_noop_on_silence():
    samples = _silence(0.5)
    normalized = normalize_peak(samples)
    assert np.max(np.abs(normalized)) == 0.0


# --- VAD ---


def test_vad_detects_speech_in_tone():
    samples = _tone(2.0)
    regions = detect_speech_regions(samples, SR)
    assert len(regions) >= 1
    assert regions[0].start >= 0.0
    assert regions[-1].end <= 2.0 + 1e-6


def test_vad_detects_no_speech_in_silence():
    samples = _silence(2.0)
    regions = detect_speech_regions(samples, SR)
    assert regions == []


def test_vad_separates_two_speech_bursts_with_silence_gap():
    audio = np.concatenate([_tone(1.0), _silence(1.0), _tone(1.0)])
    regions = detect_speech_regions(audio, SR)
    assert len(regions) == 2


# --- chunking ---


def test_chunking_short_audio_returns_single_chunk():
    samples = _tone(5.0)
    chunks = chunk_audio(samples, SR)
    assert len(chunks) == 1
    assert chunks[0].start == 0.0


def test_chunking_covers_full_duration_with_overlap():
    samples = _tone(70.0)
    chunks = chunk_audio(samples, SR)
    assert chunks[0].start == 0.0
    assert chunks[-1].end == pytest.approx(70.0, abs=0.5)
    # consecutive chunks overlap rather than leaving a gap
    for a, b in zip(chunks, chunks[1:]):
        assert b.start <= a.end


# --- batch processing: one bad file must not stop the batch ---


def test_batch_validation_continues_past_bad_file(tmp_path):
    good1 = _write_wav(tmp_path, "good1.wav", _tone(1.0))
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"garbage")
    good2 = _write_wav(tmp_path, "good2.wav", _tone(1.0))

    results = []
    errors = []
    for f in [good1, bad, good2]:
        try:
            results.append(validate_audio_file(f))
        except Exception as exc:  # validate_audio_file itself shouldn't raise, but assert defensively
            errors.append((f, exc))

    assert not errors
    assert results[0].is_valid
    assert not results[1].is_valid
    assert results[2].is_valid
