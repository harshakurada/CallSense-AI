"""NER inference: fine-tuned Transformer entities, backed up by regex
rules for anything the model missed. The model is primary (it has context
the regex patterns don't), but a missed ORDER_ID/EMAIL/PHONE that a rule
catches is still worth surfacing — same rationale as the baseline's
rule-priority-on-overlap design in src/nlp/ner/baseline.py, just with the
model given first claim on a span instead of the rules."""
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
    model_spans = [(e.start, e.end) for e in model_entities]

    rule_entities = [
        Entity(label=e.label, text=e.text, start=e.start, end=e.end, source="rule")
        for e in extract_rule_based_entities(text)
        if not any(e.start < m_end and e.end > m_start for m_start, m_end in model_spans)
    ]

    return sorted(model_entities + rule_entities, key=lambda e: e.start)
