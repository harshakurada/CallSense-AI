"""Structured field extraction from entities. Only fields NER can actually
support are populated; a field with no matching entity is null, never
guessed. "issue" is always null — describing what the customer's problem
actually is requires summarization/intent understanding, not entity
recognition, and is out of this module's scope (Module 4's intent
classifier is the closest existing signal, not a substitute)."""
from src.nlp.ner.baseline import Entity


def extract_structured_fields(entities: list[Entity]) -> dict:
    """Returns {"customer", "order_id", "product", "issue", "date", "amount"}.
    When multiple entities of the same type exist, the first one found
    (by position) is used; there is no reliable signal here for picking a
    "more correct" one among several."""

    def first(label: str) -> str | None:
        matches = [e.text for e in sorted(entities, key=lambda e: e.start) if e.label == label]
        return matches[0] if matches else None

    return {
        "customer": first("PERSON"),
        "order_id": first("ORDER_ID"),
        "product": first("PRODUCT"),
        "issue": None,  # not an NER task — see module docstring
        "date": first("DATE"),
        "amount": first("MONEY"),
    }
