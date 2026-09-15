"""Aligns Whisper's transcript segments (Module 2) with diarization's
speaker segments (this module) by temporal overlap — the two are produced
by entirely separate processes with different segment boundaries, so this
is a many-to-one join, not a simple zip."""
from dataclasses import dataclass

from src.diarization.diarize import SpeakerSegment

UNKNOWN_SPEAKER = "UNKNOWN"


@dataclass
class AlignedSegment:
    speaker: str
    start: float
    end: float
    text: str


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def align_transcript_with_diarization(
    asr_segments: list[dict], speaker_segments: list[SpeakerSegment]
) -> list[AlignedSegment]:
    """For each ASR segment {start, end, text}, assigns the speaker whose
    diarization segment(s) overlap it the most. If no diarization segment
    overlaps at all (e.g. diarization found no speech there), the segment
    is labeled UNKNOWN_SPEAKER rather than guessed."""
    aligned = []
    for asr_seg in asr_segments:
        best_speaker = UNKNOWN_SPEAKER
        best_overlap = 0.0
        overlap_by_speaker: dict[str, float] = {}

        for speaker_seg in speaker_segments:
            overlap = _overlap(asr_seg["start"], asr_seg["end"], speaker_seg.start, speaker_seg.end)
            if overlap > 0:
                overlap_by_speaker[speaker_seg.speaker] = (
                    overlap_by_speaker.get(speaker_seg.speaker, 0.0) + overlap
                )

        if overlap_by_speaker:
            best_speaker, best_overlap = max(overlap_by_speaker.items(), key=lambda kv: kv[1])

        aligned.append(
            AlignedSegment(
                speaker=best_speaker,
                start=asr_seg["start"],
                end=asr_seg["end"],
                text=asr_seg["text"],
            )
        )

    return aligned
