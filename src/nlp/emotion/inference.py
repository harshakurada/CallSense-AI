"""Emotion inference using the saved fine-tuned model."""
from pathlib import Path

from src.nlp.common.transformer import predict_batch, predict_one

MODEL_DIR = Path("models/nlp/emotion/transformer")


def predict_emotion(text: str) -> dict:
    result = predict_one(MODEL_DIR, text)
    return {"label": result["label"], "confidence": result["confidence"]}


def predict_emotion_batch(texts: list[str]) -> list[dict]:
    results = predict_batch(MODEL_DIR, texts)
    return [{"label": r["label"], "confidence": r["confidence"]} for r in results]
