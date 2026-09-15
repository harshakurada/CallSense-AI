"""Regex-based extraction for structured, pattern-matchable entities.

PHONE, EMAIL, MONEY, and business identifiers (ORDER_ID, ACCOUNT_ID,
INVOICE_ID) are far more reliably extracted by pattern matching than by a
learned model trained on a few thousand synthetic examples — this is
standard practice in production entity extraction, not a shortcut. Used as
part of the baseline (src/nlp/ner/baseline.py) and as a second signal
alongside the fine-tuned model's output (src/nlp/ner/inference.py).
"""
import re
from dataclasses import dataclass

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"(?<!\w)(\+?\d{1,2}[\s.-]?)?(\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})(?!\w)")
_MONEY_RE = re.compile(r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b")
_INVOICE_ID_RE = re.compile(r"\bINV-\d{4}-\d{3,6}\b", re.IGNORECASE)
_ACCOUNT_ID_RE = re.compile(r"\b(?:ACC\d{5,8}|A-\d{4,8})\b", re.IGNORECASE)
_ORDER_ID_STRICT_RE = re.compile(r"\bORD-\d{5,8}\b", re.IGNORECASE)
# A fixed-width lookbehind (?<=order\s) only matches "order 45821" — real
# (especially spoken/noisy) text puts filler between the word and the
# number ("order number is uh 45821"). Matched separately below by
# proximity instead of requiring immediate adjacency.
_ORDER_KEYWORD_RE = re.compile(r"\border\b", re.IGNORECASE)
_BARE_NUMBER_RE = re.compile(r"\b\d{4,8}\b")
_ORDER_ID_PROXIMITY_WINDOW = 30  # characters after "order" to look for the number in


@dataclass
class RuleEntity:
    label: str
    text: str
    start: int
    end: int


_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", _EMAIL_RE),
    ("INVOICE_ID", _INVOICE_ID_RE),
    ("ACCOUNT_ID", _ACCOUNT_ID_RE),
    ("ORDER_ID", _ORDER_ID_STRICT_RE),
    ("PHONE", _PHONE_RE),
    ("MONEY", _MONEY_RE),
]


def _find_order_ids_by_proximity(text: str, already_claimed: list[tuple[int, int]]) -> list[RuleEntity]:
    """Finds the nearest bare 4-8 digit number after the word "order"
    within _ORDER_ID_PROXIMITY_WINDOW characters — handles realistic filler
    ("order number is uh 45821") that a fixed-width lookbehind can't."""
    found = []
    for keyword_match in _ORDER_KEYWORD_RE.finditer(text):
        window_start = keyword_match.end()
        window_end = min(len(text), window_start + _ORDER_ID_PROXIMITY_WINDOW)
        number_match = _BARE_NUMBER_RE.search(text, window_start, window_end)
        if not number_match:
            continue
        start, end = number_match.span()
        if any(start < c_end and end > c_start for c_start, c_end in already_claimed):
            continue
        found.append(RuleEntity(label="ORDER_ID", text=number_match.group(), start=start, end=end))
    return found


def extract_rule_based_entities(text: str) -> list[RuleEntity]:
    """Runs every pattern and returns non-overlapping matches, preferring
    earlier patterns in _PATTERNS (more specific identifiers like
    INVOICE_ID/ACCOUNT_ID/ORDER_ID are checked before the generic PHONE
    pattern, since a bare order number could otherwise look phone-shaped)."""
    claimed: list[tuple[int, int]] = []
    entities: list[RuleEntity] = []

    for label, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(start < c_end and end > c_start for c_start, c_end in claimed):
                continue
            claimed.append((start, end))
            entities.append(RuleEntity(label=label, text=match.group().strip(), start=start, end=end))

    for entity in _find_order_ids_by_proximity(text, claimed):
        claimed.append((entity.start, entity.end))
        entities.append(entity)

    return sorted(entities, key=lambda e: e.start)
