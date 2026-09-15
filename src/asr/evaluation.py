"""Word Error Rate evaluation. Only ever called where a real ground-truth
transcript exists — WER is never estimated or guessed."""
from dataclasses import dataclass

import jiwer

_TRANSFORM = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)


@dataclass
class WERResult:
    wer: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    reference_word_count: int


def compute_wer(reference: str, hypothesis: str) -> WERResult:
    """WER = (S + D + I) / N, where N is the reference word count.

    Uses jiwer's edit-distance alignment rather than a hand-rolled one, to
    avoid subtle alignment bugs; text is lowercased and stripped of
    punctuation before alignment since ASR casing/punctuation is not the
    thing being measured here.
    """
    if not reference.strip():
        raise ValueError("Reference transcript is empty — WER is undefined")

    measures = jiwer.process_words(
        reference,
        hypothesis,
        reference_transform=_TRANSFORM,
        hypothesis_transform=_TRANSFORM,
    )
    reference_word_count = measures.hits + measures.substitutions + measures.deletions

    return WERResult(
        wer=measures.wer,
        substitutions=measures.substitutions,
        deletions=measures.deletions,
        insertions=measures.insertions,
        hits=measures.hits,
        reference_word_count=reference_word_count,
    )


def compute_corpus_wer(references: list[str], hypotheses: list[str]) -> WERResult:
    """Aggregate WER over a whole dataset (S/D/I summed across all
    utterances, then divided by total reference words) — not the mean of
    per-utterance WERs, which would over-weight short utterances."""
    if len(references) != len(hypotheses):
        raise ValueError(
            f"references ({len(references)}) and hypotheses ({len(hypotheses)}) length mismatch"
        )
    if not references:
        raise ValueError("No reference/hypothesis pairs provided")

    measures = jiwer.process_words(
        references,
        hypotheses,
        reference_transform=_TRANSFORM,
        hypothesis_transform=_TRANSFORM,
    )
    reference_word_count = measures.hits + measures.substitutions + measures.deletions

    return WERResult(
        wer=measures.wer,
        substitutions=measures.substitutions,
        deletions=measures.deletions,
        insertions=measures.insertions,
        hits=measures.hits,
        reference_word_count=reference_word_count,
    )
