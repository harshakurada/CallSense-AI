"""Sentiment inference: per-utterance prediction plus customer-level and
conversation-level aggregation (Module 4 spec section 5)."""
from pathlib import Path

from src.nlp.common.transformer import predict_batch, predict_one

MODEL_DIR = Path("models/nlp/sentiment/transformer")

_SCORE_MAP = {"positive": 1, "neutral": 0, "negative": -1}


def predict_sentiment(text: str) -> dict:
    result = predict_one(MODEL_DIR, text)
    return {"label": result["label"], "confidence": result["confidence"]}


def predict_sentiment_batch(texts: list[str]) -> list[dict]:
    results = predict_batch(MODEL_DIR, texts)
    return [{"label": r["label"], "confidence": r["confidence"]} for r in results]


def aggregate_sentiment(per_utterance: list[dict]) -> dict:
    """Majority-vote label (ties broken by mean confidence), plus a
    continuous score (positive=+1, neutral=0, negative=-1) averaged across
    utterances — the score is a simple, transparent trend indicator, not a
    model output, and is reported as such."""
    if not per_utterance:
        return {"label": None, "confidence": None, "score": None}

    label_confidences: dict[str, list[float]] = {}
    for pred in per_utterance:
        label_confidences.setdefault(pred["label"], []).append(pred["confidence"])

    majority_label = max(label_confidences.items(), key=lambda kv: (len(kv[1]), sum(kv[1]) / len(kv[1])))[0]
    mean_confidence = sum(label_confidences[majority_label]) / len(label_confidences[majority_label])
    score = sum(_SCORE_MAP.get(p["label"], 0) for p in per_utterance) / len(per_utterance)

    return {"label": majority_label, "confidence": mean_confidence, "score": score}


def analyze_conversation_sentiment(conversation: list[dict]) -> dict:
    """conversation: [{"speaker": str, "text": str, ...}, ...] (Module 3's
    output format). Returns per-utterance sentiment plus CUSTOMER-only and
    whole-conversation aggregates.

    customer_level only has data if turns are actually labeled "CUSTOMER"
    — Module 3's default output uses generic SPEAKER_00/SPEAKER_01 unless
    its opt-in role heuristic was applied. With generic labels,
    customer_level comes back as {"label": None, ...} rather than silently
    aggregating the wrong speaker's turns."""
    texts = [turn["text"] for turn in conversation]
    per_utterance_preds = predict_sentiment_batch(texts) if texts else []
    per_utterance = [
        {**turn_pred, "speaker": turn["speaker"], "start": turn["start"], "end": turn["end"]}
        for turn, turn_pred in zip(conversation, per_utterance_preds)
    ]

    customer_preds = [p for p, turn in zip(per_utterance_preds, conversation) if turn["speaker"] == "CUSTOMER"]

    return {
        "per_utterance": per_utterance,
        "customer_level": aggregate_sentiment(customer_preds),
        "conversation_level": aggregate_sentiment(per_utterance_preds),
    }
