"""Diarization Error Rate via pyannote.metrics — the standard
implementation, not a hand-rolled one. Only ever called where real
reference speaker labels exist; never estimated."""
from dataclasses import dataclass

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

from src.diarization.diarize import SpeakerSegment


@dataclass
class DERResult:
    der: float
    correct: float
    confusion: float
    missed_detection: float
    false_alarm: float
    total_reference: float


def _to_annotation(segments: list[SpeakerSegment]) -> Annotation:
    annotation = Annotation()
    for seg in segments:
        if seg.end > seg.start:
            annotation[Segment(seg.start, seg.end)] = seg.speaker
    return annotation


def compute_der(reference: list[SpeakerSegment], hypothesis: list[SpeakerSegment]) -> DERResult:
    """DER = (missed detection + false alarm + speaker confusion) / total
    reference speech time. Speaker *labels* don't need to match between
    reference and hypothesis — pyannote.metrics finds the optimal
    label-to-label mapping internally, since clustering output (SPEAKER_00,
    SPEAKER_01, ...) is arbitrary and unrelated to reference label names.
    """
    if not reference:
        raise ValueError("Reference diarization is empty — DER is undefined")

    ref_annotation = _to_annotation(reference)
    hyp_annotation = _to_annotation(hypothesis)

    metric = DiarizationErrorRate()
    detail = metric(ref_annotation, hyp_annotation, detailed=True)

    return DERResult(
        der=detail["diarization error rate"],
        correct=detail["correct"],
        confusion=detail["confusion"],
        missed_detection=detail["missed detection"],
        false_alarm=detail["false alarm"],
        total_reference=detail["total"],
    )
