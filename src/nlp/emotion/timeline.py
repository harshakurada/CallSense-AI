"""Emotion timeline: one emotion label per conversation turn, ordered by
time — feeds the dashboard's emotion-over-time view (Module 4 spec
section 7)."""
from src.nlp.emotion.inference import predict_emotion_batch


def build_emotion_timeline(conversation: list[dict]) -> list[dict]:
    """conversation: [{"speaker": str, "start": float, "end": float, "text": str}, ...]
    Returns [{"time": float, "speaker": str, "label": str, "confidence": float}, ...]
    ordered by `start`, one entry per turn — not resampled to fixed
    intervals, since a turn is the only unit of text this model actually
    saw."""
    if not conversation:
        return []

    texts = [turn["text"] for turn in conversation]
    predictions = predict_emotion_batch(texts)

    timeline = [
        {
            "time": turn["start"],
            "speaker": turn["speaker"],
            "label": pred["label"],
            "confidence": pred["confidence"],
        }
        for turn, pred in zip(conversation, predictions)
    ]
    return sorted(timeline, key=lambda entry: entry["time"])


def format_timeline_text(timeline: list[dict]) -> str:
    """Renders the timeline as the human-readable format from the spec:
    '00:20 Frustrated' — using the model's actual predicted label, not a
    remapped display name."""
    lines = []
    for entry in timeline:
        minutes, seconds = divmod(int(entry["time"]), 60)
        lines.append(f"{minutes:02d}:{seconds:02d} {entry['label'].capitalize()}")
    return "\n".join(lines)
