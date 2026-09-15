"""Entity normalization: standardizes raw entity text into a consistent
value per type, where that can be done unambiguously without guessing.
Anything that can't be parsed confidently keeps its raw text rather than
being forced into a wrong normalized value."""
import re
from datetime import datetime

from src.nlp.ner.baseline import Entity

_MONEY_CLEAN_RE = re.compile(r"[^\d.]")
_PHONE_CLEAN_RE = re.compile(r"[^\d]")

_DATE_FORMATS = ["%B %d, %Y", "%B %d", "%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d"]


def _normalize_money(text: str) -> float | None:
    cleaned = _MONEY_CLEAN_RE.sub("", text)
    try:
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


def _normalize_phone(text: str) -> str | None:
    digits = _PHONE_CLEAN_RE.sub("", text)
    return digits if 7 <= len(digits) <= 15 else None


def _normalize_date(text: str) -> str | None:
    for fmt in _DATE_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.strftime("%Y-%m-%d") if "%Y" in fmt or "%y" in fmt else parsed.strftime("--%m-%d")
        except ValueError:
            continue
    return None  # not confidently parseable — caller keeps the raw entity text instead


_NORMALIZERS = {"MONEY": _normalize_money, "PHONE": _normalize_phone, "DATE": _normalize_date}


def normalize_entity(entity: Entity) -> dict:
    """Returns {"label", "text", "normalized"} — "normalized" is None when
    the type has no normalizer (e.g. PERSON) or the value didn't parse."""
    normalizer = _NORMALIZERS.get(entity.label)
    normalized = normalizer(entity.text) if normalizer else None
    return {"label": entity.label, "text": entity.text, "normalized": normalized}
