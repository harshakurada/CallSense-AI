"""Splits long audio into overlapping chunks for ASR. Whisper-family models
are trained on <=30s windows, so anything longer must be chunked regardless
of VAD; chunk boundaries snap to the nearest detected silence when possible
so a chunk edge doesn't land mid-word."""
from dataclasses import dataclass

import numpy as np

from configs.settings import get_audio_config
from src.audio.vad import detect_speech_regions


@dataclass
class AudioChunk:
    start: float  # seconds, in the original audio's timeline
    end: float
    samples: np.ndarray


def _nearest_silence_boundary(
    target: float, regions: list, duration: float, max_snap_distance: float
) -> float:
    """Given speech regions, returns the boundary point closest to `target`
    that falls in a silence gap between two regions, so the chunk cut
    doesn't land inside detected speech. Only snaps within
    `max_snap_distance` of the target — otherwise there is no nearby
    silence to use (e.g. one continuous speech region) and the fixed cut
    at `target` is kept instead of jumping to a distant gap."""
    # gaps strictly between regions only — the signal's own start/end are
    # not valid "silence" to snap into, they're just where audio stops.
    gaps = []
    for prev_end, next_start in zip([r.end for r in regions], [r.start for r in regions[1:]]):
        if next_start >= prev_end:
            gaps.append((prev_end, next_start))

    candidates = [min(max(target, gs), ge) for gs, ge in gaps]
    if not candidates:
        return target

    best = min(candidates, key=lambda c: abs(c - target))
    return best if abs(best - target) <= max_snap_distance else target


def chunk_audio(samples: np.ndarray, sample_rate: int) -> list[AudioChunk]:
    config = get_audio_config()["chunking"]
    chunk_duration = config["chunk_duration_seconds"]
    overlap = config["overlap_seconds"]
    duration = len(samples) / sample_rate

    if duration <= chunk_duration:
        return [AudioChunk(start=0.0, end=duration, samples=samples)]

    regions = detect_speech_regions(samples, sample_rate) if config["snap_to_silence"] else []
    max_snap_distance = chunk_duration / 4

    chunks: list[AudioChunk] = []
    cursor = 0.0
    while cursor < duration:
        raw_end = min(cursor + chunk_duration, duration)
        end = (
            _nearest_silence_boundary(raw_end, regions, duration, max_snap_distance)
            if raw_end < duration
            else raw_end
        )
        if end <= cursor:
            end = raw_end  # snapping failed to move forward; fall back to the fixed cut

        start_sample = int(cursor * sample_rate)
        end_sample = int(end * sample_rate)
        chunks.append(AudioChunk(start=cursor, end=end, samples=samples[start_sample:end_sample]))

        if end >= duration:
            break
        cursor = max(end - overlap, cursor + 1e-3)

    return chunks
