"""Full pipeline: audio -> preprocessing -> ASR (Module 2) -> diarization
(Module 3) -> speaker-attributed transcript. The only place these two
modules are composed — nothing else should re-implement this sequence."""
from pathlib import Path

from configs.settings import get_diarization_config
from src.asr.transcribe import transcribe
from src.diarization.align import align_transcript_with_diarization
from src.diarization.diarize import diarize
from src.diarization.roles import apply_roles, infer_roles_first_speaker_heuristic
from src.utils.logging import get_logger

logger = get_logger(__name__)


def run_pipeline(audio_path: str | Path, call_id: str | None = None, apply_role_heuristic: bool | None = None) -> dict:
    """Returns the final conversation format:
    {
      "call_id": str,
      "duration": float,
      "conversation": [{"speaker": str, "start": float, "end": float, "text": str}, ...]
    }

    `speaker` is a generic SPEAKER_NN label unless apply_role_heuristic is
    True, in which case the unverified first-speaker-is-agent heuristic
    (see src/diarization/roles.py) relabels them CUSTOMER/AGENT. Defaults
    to configs/diarization.yaml's roles.apply_first_speaker_heuristic
    (false) rather than guessing roles by default.
    """
    audio_path = Path(audio_path)
    call_id = call_id or audio_path.stem
    if apply_role_heuristic is None:
        apply_role_heuristic = get_diarization_config()["roles"]["apply_first_speaker_heuristic"]

    asr_result = transcribe(audio_path, call_id=call_id)
    speaker_segments = diarize(audio_path)

    if apply_role_heuristic:
        role_mapping = infer_roles_first_speaker_heuristic(speaker_segments)
        speaker_segments = apply_roles(speaker_segments, role_mapping)

    aligned = align_transcript_with_diarization(asr_result["segments"], speaker_segments)

    # faster-whisper's last segment can slightly overrun the audio's actual
    # duration (a known upstream timestamp quirk) — clip rather than report
    # a segment that ends after the call does.
    duration = asr_result["duration"]
    conversation = [
        {
            "speaker": seg.speaker,
            "start": min(seg.start, duration),
            "end": min(seg.end, duration),
            "text": seg.text,
        }
        for seg in aligned
    ]

    logger.info(
        "%s: pipeline complete — %d conversation turn(s), roles_applied=%s",
        call_id,
        len(conversation),
        apply_role_heuristic,
    )

    return {
        "call_id": call_id,
        "duration": asr_result["duration"],
        "conversation": conversation,
    }
