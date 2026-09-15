"""Intent inference using the saved fine-tuned model."""
from pathlib import Path

from src.nlp.common.transformer import predict_one

MODEL_DIR = Path("models/nlp/intent/transformer")


def predict_intent(text: str) -> dict:
    """Returns {"label": str, "confidence": float}. `text` is typically the
    customer's utterance(s) for a call — see src/nlp/inference.py for how
    a whole conversation is reduced to intent-classifier input."""
    result = predict_one(MODEL_DIR, text)
    return {"label": result["label"], "confidence": result["confidence"]}
