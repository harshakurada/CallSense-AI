"""CUSTOMER/AGENT role inference — deliberately separate from diarize()
and NOT applied by default. Speaker clustering alone (src/diarization/
cluster.py) has no signal for which cluster is the agent; the only
available heuristic (first speaker = agent, since inbound calls are
typically answered/greeted by the agent) is domain-typical, not reliable,
and is never asserted as ground truth. See docs/DIARIZATION.md."""
from src.diarization.diarize import SpeakerSegment
from src.utils.logging import get_logger

logger = get_logger(__name__)


def infer_roles_first_speaker_heuristic(segments: list[SpeakerSegment]) -> dict[str, str]:
    """Maps the earliest-speaking distinct speaker to AGENT and all others
    to CUSTOMER. This is a weak, unverified heuristic — callers must treat
    the result as low-confidence, not ground truth. Returns {} if there are
    fewer than 2 distinct speakers (nothing to map)."""
    if not segments:
        return {}

    ordered_unique_speakers = list(dict.fromkeys(seg.speaker for seg in sorted(segments, key=lambda s: s.start)))
    if len(ordered_unique_speakers) < 2:
        logger.warning(
            "Only %d distinct speaker(s) found — role mapping is not meaningful",
            len(ordered_unique_speakers),
        )
        return {}

    logger.warning(
        "Applying unverified 'first speaker = AGENT' heuristic — do not treat as ground truth"
    )
    roles = {ordered_unique_speakers[0]: "AGENT"}
    for speaker in ordered_unique_speakers[1:]:
        roles[speaker] = "CUSTOMER"
    return roles


def apply_roles(segments: list[SpeakerSegment], role_mapping: dict[str, str]) -> list[SpeakerSegment]:
    """Relabels segments using role_mapping; speakers absent from the
    mapping keep their generic SPEAKER_NN label rather than being guessed."""
    return [
        SpeakerSegment(
            speaker=role_mapping.get(seg.speaker, seg.speaker), start=seg.start, end=seg.end
        )
        for seg in segments
    ]
