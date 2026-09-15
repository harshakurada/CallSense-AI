"""PII masking: replaces entity spans with bracketed placeholders.

Default masked types are exactly what the spec's own worked example masks
(PERSON, ORDER_ID) plus the rest of section 4's list (phone, email,
account numbers). LOCATION, ORGANIZATION, PRODUCT, DATE, and MONEY are
NOT masked by default — a business needs the product/amount/date for
analytics, and "address" (section 4) isn't reliably separable from a
general place mention without extra classification this project doesn't
have; that limitation is documented here rather than either masking every
location (destroying useful data) or silently not masking addresses at
all.
"""
from src.nlp.ner.baseline import Entity

DEFAULT_PII_TYPES = {"PERSON", "PHONE", "EMAIL", "ACCOUNT_ID", "ORDER_ID", "INVOICE_ID"}


def mask_pii(text: str, entities: list[Entity], mask_types: set[str] = DEFAULT_PII_TYPES) -> str:
    """Replaces each entity of a masked type with "[LABEL]". Processes
    spans right-to-left so earlier offsets stay valid as the string
    shrinks/grows."""
    to_mask = sorted((e for e in entities if e.label in mask_types), key=lambda e: e.start, reverse=True)

    masked = text
    for entity in to_mask:
        masked = masked[: entity.start] + f"[{entity.label}]" + masked[entity.end :]
    return masked
