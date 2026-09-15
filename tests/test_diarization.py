"""Diarization tests. Pure-logic pieces (segmentation math, clustering,
alignment, role heuristics) use synthetic fixtures and need no model or
external data. Real speaker *separation* needs actual speech-like signal —
pure sine tones sit almost on top of each other in ECAPA embedding space
(cosine similarity ~0.95 between very different frequencies, verified
manually), so those tests are skipped if the real LibriSpeech dummy set
(downloaded in Module 2) isn't present on disk, rather than asserting
something a synthetic tone can't actually demonstrate."""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.diarization.align import UNKNOWN_SPEAKER, align_transcript_with_diarization
from src.diarization.cluster import cluster_embeddings
from src.diarization.diarize import SpeakerSegment, diarize
from src.diarization.roles import apply_roles, infer_roles_first_speaker_heuristic
from src.diarization.segment import get_diarization_segments
from src.utils.exceptions import DiarizationError

SR = 16000
DUMMY_MANIFEST = Path("data/raw/librispeech_dummy/manifest.jsonl")

requires_real_audio = pytest.mark.skipif(
    not DUMMY_MANIFEST.exists(),
    reason="requires the real LibriSpeech dummy set: python scripts/download_data.py --dataset librispeech-dummy",
)


def _tone(duration_s: float, sr: int = SR, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(duration_s: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(sr * duration_s), dtype=np.float32)


def _real_clips(n: int = 2) -> list[np.ndarray]:
    clips = []
    dummy_dir = DUMMY_MANIFEST.parent
    with open(DUMMY_MANIFEST, encoding="utf-8") as f:
        for line in f:
            if len(clips) >= n:
                break
            entry = json.loads(line)
            samples, _ = sf.read(str(dummy_dir / f"{entry['call_id']}.flac"))
            clips.append(samples.astype(np.float32))
    return clips


# --- segmentation (no model needed) ---


def test_segmentation_splits_two_bursts_separated_by_silence():
    audio = np.concatenate([_tone(2.0), _silence(1.0), _tone(2.0)])
    segments = get_diarization_segments(audio, SR)
    assert len(segments) == 2


def test_segmentation_returns_nothing_for_pure_silence():
    segments = get_diarization_segments(_silence(3.0), SR)
    assert segments == []


def test_segmentation_splits_long_continuous_speech():
    # one continuous 40s burst, no internal silence to segment on
    audio = _tone(40.0)
    segments = get_diarization_segments(audio, SR)
    assert len(segments) >= 3  # split by max_segment_duration_seconds (12s in config)
    # segments should tile the full duration contiguously
    assert segments[0].start == pytest.approx(0.0, abs=0.01)
    assert segments[-1].end == pytest.approx(40.0, abs=0.5)


def test_segmentation_drops_too_short_regions():
    audio = np.concatenate([_tone(0.1), _silence(1.0), _tone(2.0)])  # first burst below min duration
    segments = get_diarization_segments(audio, SR)
    assert len(segments) == 1


# --- clustering (no model needed — synthetic embeddings) ---


def test_clustering_separates_two_distinct_groups():
    # cosine clustering needs vectors with real magnitude/direction — a
    # loc=0.0 cluster sits at the origin where "direction" is numerically
    # unstable, which broke this test without it being a real bug.
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=10.0, scale=0.05, size=(3, 16))
    cluster_b = rng.normal(loc=-10.0, scale=0.05, size=(3, 16))
    embeddings = np.vstack([cluster_a, cluster_b])

    labels = cluster_embeddings(embeddings)

    assert len(set(labels)) == 2
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]


def test_clustering_labels_by_first_appearance():
    rng = np.random.default_rng(1)
    cluster_a = rng.normal(loc=10.0, scale=0.05, size=(2, 16))
    cluster_b = rng.normal(loc=-10.0, scale=0.05, size=(2, 16))
    # cluster_b's group appears first in the array
    embeddings = np.vstack([cluster_b[:1], cluster_a, cluster_b[1:]])

    labels = cluster_embeddings(embeddings)

    assert labels[0] == "SPEAKER_00"  # first row seen defines SPEAKER_00
    assert labels[1] == labels[2] == "SPEAKER_01"
    assert labels[3] == "SPEAKER_00"


def test_clustering_single_segment_is_single_speaker():
    assert cluster_embeddings(np.zeros((1, 16))) == ["SPEAKER_00"]


def test_clustering_empty_input():
    assert cluster_embeddings(np.zeros((0, 16))) == []


# --- alignment (no model needed) ---


def test_alignment_assigns_majority_overlap_speaker():
    asr_segments = [{"start": 0.0, "end": 5.0, "text": "hello"}]
    speaker_segments = [
        SpeakerSegment(speaker="SPEAKER_00", start=0.0, end=1.0),
        SpeakerSegment(speaker="SPEAKER_01", start=1.0, end=5.0),
    ]
    aligned = align_transcript_with_diarization(asr_segments, speaker_segments)
    assert aligned[0].speaker == "SPEAKER_01"  # 4s overlap vs 1s


def test_alignment_unknown_when_no_diarization_overlap():
    asr_segments = [{"start": 10.0, "end": 12.0, "text": "hello"}]
    speaker_segments = [SpeakerSegment(speaker="SPEAKER_00", start=0.0, end=5.0)]
    aligned = align_transcript_with_diarization(asr_segments, speaker_segments)
    assert aligned[0].speaker == UNKNOWN_SPEAKER


# --- role heuristic (no model needed) ---


def test_role_heuristic_maps_first_speaker_to_agent():
    segments = [
        SpeakerSegment(speaker="SPEAKER_01", start=5.0, end=10.0),
        SpeakerSegment(speaker="SPEAKER_00", start=0.0, end=5.0),
    ]
    roles = infer_roles_first_speaker_heuristic(segments)
    assert roles == {"SPEAKER_00": "AGENT", "SPEAKER_01": "CUSTOMER"}


def test_role_heuristic_empty_for_single_speaker():
    segments = [SpeakerSegment(speaker="SPEAKER_00", start=0.0, end=5.0)]
    assert infer_roles_first_speaker_heuristic(segments) == {}


def test_apply_roles_keeps_unmapped_speakers_generic():
    segments = [SpeakerSegment(speaker="SPEAKER_02", start=0.0, end=1.0)]
    relabeled = apply_roles(segments, {"SPEAKER_00": "AGENT"})
    assert relabeled[0].speaker == "SPEAKER_02"


# --- invalid files ---


def test_diarize_raises_on_missing_file(tmp_path):
    with pytest.raises(DiarizationError):
        diarize(tmp_path / "does_not_exist.wav")


def test_diarize_raises_on_corrupt_file(tmp_path):
    path = tmp_path / "corrupt.wav"
    path.write_bytes(b"not audio" * 20)
    with pytest.raises(DiarizationError):
        diarize(path)


def test_diarize_returns_empty_on_pure_silence(tmp_path):
    path = tmp_path / "silence.wav"
    sf.write(str(path), _silence(3.0), SR)
    assert diarize(path) == []


# --- real-audio integration tests ---


@requires_real_audio
def test_diarize_separates_two_real_speakers(tmp_path):
    import librosa

    clip_a, clip_b = _real_clips(2)
    # A genuinely different embedding-model input is needed to prove
    # separation — a pitch-shifted copy of real speech stands in for a
    # second speaker since no second real speaker ships with the dummy
    # set (documented in docs/DIARIZATION.md).
    clip_b_shifted = librosa.effects.pitch_shift(clip_b, sr=SR, n_steps=7)
    gap = _silence(0.5)
    call = np.concatenate([clip_a, gap, clip_b_shifted, gap, clip_a])
    path = tmp_path / "two_speaker_call.wav"
    sf.write(str(path), call, SR)

    segments = diarize(path)

    assert len(segments) == 3
    speakers = [s.speaker for s in segments]
    assert len(set(speakers)) == 2
    assert speakers[0] == speakers[2]  # first and third turns are the same original speaker
    assert speakers[1] != speakers[0]


@requires_real_audio
def test_diarize_handles_overlapping_speech_without_crashing(tmp_path):
    """Overlapping speech is a documented limitation (docs/DIARIZATION.md):
    the energy-based VAD has no concept of 'two people talking at once', so
    an overlapped region becomes one segment attributed to a single
    speaker. This test only asserts the pipeline doesn't crash on it and
    produces *some* segment — not that it correctly separates the overlap,
    which this approach cannot do."""
    import librosa

    clip_a, clip_b = _real_clips(2)
    clip_b_shifted = librosa.effects.pitch_shift(clip_b, sr=SR, n_steps=7)
    n = min(len(clip_a), len(clip_b_shifted))
    overlapped = clip_a[:n] + clip_b_shifted[:n]

    path = tmp_path / "overlap.wav"
    sf.write(str(path), overlapped, SR)

    segments = diarize(path)  # must not raise
    assert len(segments) >= 1


@requires_real_audio
def test_diarize_long_call_completes(tmp_path):
    clip_a, clip_b = _real_clips(2)
    gap = _silence(0.4)
    long_call = np.concatenate([clip_a, gap, clip_b, gap] * 4)
    path = tmp_path / "long_call.wav"
    sf.write(str(path), long_call, SR)

    segments = diarize(path)
    assert len(segments) >= 6
