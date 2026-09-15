"""Baseline NER: pretrained spaCy (en_core_web_sm, not fine-tuned) for
general entities + regex rules for structured business identifiers. This
combination — a general-purpose statistical NER plus pattern rules for
IDs/contacts — is standard practice, not a placeholder; it is what the
fine-tuned Transformer (src/nlp/ner/train.py) is benchmarked against."""
from dataclasses import dataclass
from functools import lru_cache

import spacy

from src.nlp.ner.rules import extract_rule_based_entities

# spaCy's OntoNotes-based labels mapped onto this project's entity schema.
# GPE (countries/cities/states) and LOC (non-GPE locations) both become
# our single LOCATION type; everything spaCy doesn't map to is dropped
# here — it isn't in this project's target schema (see docs/NER.md).
_SPACY_LABEL_MAP = {
    "PERSON": "PERSON",
    "ORG": "ORGANIZATION",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "PRODUCT": "PRODUCT",
    "DATE": "DATE",
    "MONEY": "MONEY",
}


@dataclass
class Entity:
    label: str
    text: str
    start: int
    end: int
    source: str  # "spacy" or "rule"


@lru_cache
def _load_spacy():
    return spacy.load("en_core_web_sm")


def extract_baseline_entities(text: str) -> list[Entity]:
    nlp = _load_spacy()
    doc = nlp(text)

    entities = [
        Entity(label=_SPACY_LABEL_MAP[ent.label_], text=ent.text, start=ent.start_char, end=ent.end_char, source="spacy")
        for ent in doc.ents
        if ent.label_ in _SPACY_LABEL_MAP
    ]

    # Rules take priority over spaCy in overlapping spans — spaCy is known
    # to mislabel business identifiers (e.g. an order number as DATE; see
    # docs/NER.md for a real example), and the regex patterns are far more
    # precise for exactly the entity types they target.
    rule_entities = extract_rule_based_entities(text)
    rule_spans = [(e.start, e.end) for e in rule_entities]

    entities = [
        e for e in entities if not any(e.start < r_end and e.end > r_start for r_start, r_end in rule_spans)
    ]
    entities.extend(
        Entity(label=e.label, text=e.text, start=e.start, end=e.end, source="rule") for e in rule_entities
    )

    return sorted(entities, key=lambda e: e.start)
