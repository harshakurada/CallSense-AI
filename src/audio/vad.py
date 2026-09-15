"""Frame-based energy voice-activity detection.

Deliberately classical (RMS-in-dB per frame + hangover smoothing) rather
than a neural VAD (silero/webrtcvad) — this keeps Module 2 free of a torch
dependency and fully unit-testable with synthetic tone/silence audio. A
neural VAD is a documented upgrade path, not required here; faster-whisper's
own `vad_filter` option (bundled Silero, used in src/asr/transcribe.py) adds
a second layer of robustness at the ASR stage regardless.
"""
from dataclasses import dataclass

import numpy as np

from configs.settings import get_audio_config


@dataclass
class SpeechRegion:
    start: float  # seconds
    end: float    # seconds


def _frame_rms_db(samples: np.ndarray, frame_len: int) -> np.ndarray:
    n_frames = max(1, len(samples) // frame_len)
    trimmed = samples[: n_frames * frame_len]
    frames = trimmed.reshape(n_frames, frame_len) if trimmed.size else np.zeros((0, frame_len))
    rms = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)
    return 20 * np.log10(np.maximum(rms, 1e-9))


def detect_speech_regions(samples: np.ndarray, sample_rate: int) -> list[SpeechRegion]:
    """Returns the list of (start, end) second ranges classified as speech.
    An all-silent or empty signal returns an empty list."""
    config = get_audio_config()["vad"]
    frame_len = max(1, int(sample_rate * config["frame_duration_ms"] / 1000))

    if len(samples) < frame_len:
        return []

    frame_db = _frame_rms_db(samples, frame_len)
    is_speech_frame = frame_db > config["silence_threshold_db"]

    frame_duration_s = frame_len / sample_rate
    min_silence_frames = max(1, int(config["min_silence_duration_ms"] / 1000 / frame_duration_s))
    min_speech_frames = max(1, int(config["min_speech_duration_ms"] / 1000 / frame_duration_s))
    pad_s = config["speech_pad_ms"] / 1000

    regions: list[SpeechRegion] = []
    in_speech = False
    region_start_frame = 0
    silence_run = 0

    for i, speech in enumerate(is_speech_frame):
        if speech:
            if not in_speech:
                in_speech = True
                region_start_frame = i
            silence_run = 0
        elif in_speech:
            silence_run += 1
            if silence_run >= min_silence_frames:
                region_end_frame = i - silence_run + 1
                if region_end_frame - region_start_frame >= min_speech_frames:
                    regions.append(
                        SpeechRegion(
                            start=region_start_frame * frame_duration_s,
                            end=region_end_frame * frame_duration_s,
                        )
                    )
                in_speech = False
                silence_run = 0

    if in_speech:
        region_end_frame = len(is_speech_frame)
        if region_end_frame - region_start_frame >= min_speech_frames:
            regions.append(
                SpeechRegion(
                    start=region_start_frame * frame_duration_s,
                    end=region_end_frame * frame_duration_s,
                )
            )

    total_duration = len(samples) / sample_rate
    padded = [
        SpeechRegion(start=max(0.0, r.start - pad_s), end=min(total_duration, r.end + pad_s))
        for r in regions
    ]
    return padded


def is_silent(samples: np.ndarray, sample_rate: int) -> bool:
    return len(detect_speech_regions(samples, sample_rate)) == 0
