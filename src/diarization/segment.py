"""Turns speech regions into diarization-ready segments: merges regions
separated by only a brief gap (likely the same speaker pausing, not a
turn change), and splits any region longer than max_segment_duration so a
single long monologue doesn't get one noisy averaged embedding.

This is a classical, silence-gap-based approach — there is no neural
speaker-change segmentation model in play. See docs/DIARIZATION.md for what
that means for overlapping speech and back-to-back turns with no pause."""
from dataclasses import dataclass

import numpy as np

from configs.settings import get_diarization_config
from src.audio.vad import SpeechRegion, detect_speech_regions


@dataclass
class DiarizationSegment:
    start: float
    end: float


def _merge_close_regions(regions: list[SpeechRegion], merge_gap: float) -> list[SpeechRegion]:
    if not regions:
        return []
    merged = [regions[0]]
    for region in regions[1:]:
        if region.start - merged[-1].end <= merge_gap:
            merged[-1] = SpeechRegion(start=merged[-1].start, end=region.end)
        else:
            merged.append(region)
    return merged


def _split_long_region(region: SpeechRegion, max_duration: float) -> list[DiarizationSegment]:
    duration = region.end - region.start
    if duration <= max_duration:
        return [DiarizationSegment(start=region.start, end=region.end)]

    n_parts = int(np.ceil(duration / max_duration))
    part_duration = duration / n_parts
    return [
        DiarizationSegment(
            start=region.start + i * part_duration,
            end=region.start + (i + 1) * part_duration,
        )
        for i in range(n_parts)
    ]


def get_diarization_segments(samples: np.ndarray, sample_rate: int) -> list[DiarizationSegment]:
    config = get_diarization_config()["segmentation"]
    regions = detect_speech_regions(samples, sample_rate)
    regions = _merge_close_regions(regions, config["merge_gap_seconds"])

    segments: list[DiarizationSegment] = []
    for region in regions:
        duration = region.end - region.start
        if duration < config["min_segment_duration_seconds"]:
            continue  # too short for a reliable embedding
        segments.extend(_split_long_region(region, config["max_segment_duration_seconds"]))

    return segments
