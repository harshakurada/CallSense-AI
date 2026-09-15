"""NER inference: fine-tuned Transformer entities, backed up by regex
rules for anything the model missed. The model is primary (it has context
the regex patterns don't), but on an overlapping span the **longer** match
wins regardless of source.

Found necessary, not assumed: with more preceding context, the model
sometimes predicts a truncated span for a structured identifier it
otherwise gets right in isolation — e.g. "regarding order 45821" ->
ORDER_ID "45821" (correct), but "...my order 45821 never arrived..." ->
ORDER_ID "45" only. The original model-always-wins-on-overlap logic then
discarded the regex's correct, longer "45821" match specifically because
it overlapped with the model's incorrect partial one. Preferring the
longer span fixes this without hardcoding a type-based priority list,
and doesn't disturb cases where the model's span is the more complete
one (verified in tests/test_ner.py)."""
from pathlib import Path

from src.nlp.ner.baseline import Entity
from src.nlp.ner.model import predict_entities
from src.nlp.ner.rules import extract_rule_based_entities

MODEL_DIR = Path("models/nlp/ner/transformer")


def extract_entities(text: str) -> list[Entity]:
    model_entities = [
        Entity(label=e["label"], text=e["text"], start=e["start"], end=e["end"], source="transformer")
        for e in predict_entities(MODEL_DIR, text)
    ]
    rule_entities = [
        Entity(label=e.label, text=e.text, start=e.start, end=e.end, source="rule")
        for e in extract_rule_based_entities(text)
    ]

    def overlaps(a: Entity, b: Entity) -> bool:
        return a.start < b.end and a.end > b.start

    kept_model = []
    for m in model_entities:
        overlapping_rules = [r for r in rule_entities if overlaps(m, r)]
        longer_rule_exists = any((r.end - r.start) > (m.end - m.start) for r in overlapping_rules)
        if not longer_rule_exists:
            kept_model.append(m)

    kept_rules = []
    for r in rule_entities:
        overlapping_models = [m for m in model_entities if overlaps(m, r)]
        shorter_or_equal_than_all = all((r.end - r.start) >= (m.end - m.start) for m in overlapping_models)
        if not overlapping_models or shorter_or_equal_than_all:
            kept_rules.append(r)

    # a span can now appear in both kept lists (equal length, different
    # source) — deduplicate by (start, end), preferring the model's label
    # since it has context the regex doesn't
    combined: dict[tuple[int, int], Entity] = {}
    for r in kept_rules:
        combined[(r.start, r.end)] = r
    for m in kept_model:
        combined[(m.start, m.end)] = m

    return sorted(combined.values(), key=lambda e: e.start)
