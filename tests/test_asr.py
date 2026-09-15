"""ASR integration tests. These exercise the real faster-whisper model
(downloaded on first run) against tiny real audio fixtures — slower than a
mocked unit test, but validating the actual JSON contract and WER math is
the point of this module."""
import numpy as np
import pytest
import soundfile as sf

from src.asr.evaluation import compute_corpus_wer, compute_wer
from src.asr.transcribe import transcribe
from src.utils.exceptions import ASRError

SR = 16000


def _tone(duration_s: float, sr: int = SR, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture(scope="module")
def spoken_clip(tmp_path_factory):
    """A synthetic sine tone. Whisper won't produce meaningful *words* from
    it — these tests check the JSON contract/shape, not transcription
    accuracy (that's covered by the real-audio evaluation in
    docs/ASR_EVALUATION.md, generated via scripts/evaluate_asr.py)."""
    tmp_dir = tmp_path_factory.mktemp("asr_fixtures")
    path = tmp_dir / "tone.wav"
    sf.write(str(path), _tone(3.0), SR)
    return path


def test_transcribe_returns_required_schema(spoken_clip):
    result = transcribe(spoken_clip, call_id="test-call-1")

    assert set(result.keys()) == {"call_id", "language", "duration", "segments"}
    assert result["call_id"] == "test-call-1"
    assert isinstance(result["language"], str)
    assert isinstance(result["duration"], float)
    assert isinstance(result["segments"], list)
    for segment in result["segments"]:
        assert set(segment.keys()) == {"start", "end", "text"}
        assert isinstance(segment["start"], float)
        assert isinstance(segment["end"], float)
        assert isinstance(segment["text"], str)
        assert segment["end"] >= segment["start"]


def test_transcribe_defaults_call_id_to_filename(spoken_clip):
    result = transcribe(spoken_clip)
    assert result["call_id"] == spoken_clip.stem


def test_transcribe_no_confidence_field_is_invented(spoken_clip):
    result = transcribe(spoken_clip)
    assert "confidence" not in result
    for segment in result["segments"]:
        assert "confidence" not in segment


def test_transcribe_raises_on_invalid_audio(tmp_path):
    bad_path = tmp_path / "bad.wav"
    bad_path.write_bytes(b"not audio")
    with pytest.raises(Exception):
        transcribe(bad_path)


def test_transcribe_handles_vad_disagreement_gracefully(spoken_clip, monkeypatch):
    """Regression test for the bug documented in docs/ASR_EVALUATION.md:
    with language left to auto-detect, faster-whisper's internal VAD can
    filter out audio that our own energy-based VAD passed through (a pure
    tone, here), and the library raises a bare ValueError instead of
    returning an empty result. transcribe() must not propagate that as a
    pipeline failure."""
    import src.asr.transcribe as transcribe_module

    original_get_config = transcribe_module.get_config

    def config_with_no_language():
        config = original_get_config()
        return {**config, "asr": {**config["asr"], "language": None}}

    monkeypatch.setattr(transcribe_module, "get_config", config_with_no_language)
    transcribe_module._load_model.cache_clear()
    try:
        result = transcribe(spoken_clip, call_id="silent-vad-disagreement")
        assert result["segments"] == []
    finally:
        transcribe_module._load_model.cache_clear()


# --- WER ---


def test_wer_perfect_match_is_zero():
    result = compute_wer("hello world", "hello world")
    assert result.wer == 0.0
    assert result.substitutions == 0


def test_wer_counts_substitution():
    result = compute_wer("the cat sat", "the dog sat")
    assert result.substitutions == 1
    assert result.wer == pytest.approx(1 / 3)


def test_wer_ignores_case_and_punctuation():
    result = compute_wer("Hello, World!", "hello world")
    assert result.wer == 0.0


def test_wer_raises_on_empty_reference():
    with pytest.raises(ValueError):
        compute_wer("", "some hypothesis")


def test_corpus_wer_aggregates_across_files():
    result = compute_corpus_wer(["hello world", "good morning"], ["hello world", "good evening"])
    assert result.substitutions == 1
    assert result.reference_word_count == 4


def test_corpus_wer_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        compute_corpus_wer(["one", "two"], ["one"])
