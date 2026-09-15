"""Full Module 5 pipeline: transcript -> NER -> PII masking -> entity
normalization -> structured information. Integrates with Module 3's
speaker-attributed conversation format."""
from src.nlp.ner.extract import extract_structured_fields
from src.nlp.ner.inference import extract_entities
from src.nlp.ner.normalize import normalize_entity
from src.nlp.ner.pii import mask_pii


def analyze_text(text: str) -> dict:
    """Runs the full pipeline on one piece of text (one utterance or a
    whole transcript). Returns:
    {
      "entities": [{"label", "text", "normalized"}, ...],
      "masked_text": str,
      "structured_fields": {"customer", "order_id", "product", "issue", "date", "amount"}
    }
    """
    entities = extract_entities(text)
    return {
        "entities": [normalize_entity(e) for e in entities],
        "masked_text": mask_pii(text, entities),
        "structured_fields": extract_structured_fields(entities),
    }


def analyze_conversation(conversation: list[dict]) -> dict:
    """conversation: [{"speaker": str, "start": float, "end": float, "text": str}, ...]
    (Module 3's output format). Runs NER per turn, and separately over the
    whole concatenated transcript for the call-level structured_fields
    (an order/invoice/amount is usually mentioned once, not per-turn)."""
    per_turn = []
    for turn in conversation:
        result = analyze_text(turn["text"])
        per_turn.append(
            {
                "speaker": turn["speaker"],
                "start": turn["start"],
                "end": turn["end"],
                "masked_text": result["masked_text"],
                "entities": result["entities"],
            }
        )

    full_text = " ".join(turn["text"] for turn in conversation)
    call_level = analyze_text(full_text)

    return {"per_turn": per_turn, "structured_fields": call_level["structured_fields"]}
