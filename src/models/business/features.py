"""Named, explicit conversation-level features for resolution and
escalation prediction. Deliberately explicit rather than raw embeddings
(Module 6's approach) — Module 7 needs per-prediction explanations
(spec section 3), and you cannot name which of 768 frozen embedding
dimensions "caused" a prediction, but you can name which of these features
did.

Reuses Module 4's real fine-tuned sentiment/emotion/intent classifiers as
signal, the same way Module 6's baseline does — this is legitimate feature
reuse (all computed independently of the resolution/escalation label),
not data leakage. See docs/BUSINESS_MODELS.md.
"""
import re

import numpy as np

from src.nlp.emotion.inference import predict_emotion_batch
from src.nlp.intent.inference import predict_intent
from src.nlp.sentiment.inference import predict_sentiment_batch

_SENTIMENT_SCORE = {"positive": 1, "neutral": 0, "negative": -1}
_NEGATIVE_EMOTIONS = {"anger", "fear", "sadness"}

_SUPERVISOR_RE = re.compile(r"\b(manager|supervisor|someone (else|in charge))\b", re.IGNORECASE)
_TRANSFER_RE = re.compile(r"\b(transfer(ring)?|connect(ing)? you)\b", re.IGNORECASE)

FEATURE_NAMES = [
    "num_turns",
    "num_customer_turns",
    "num_agent_turns",
    "conversation_duration_seconds",
    "avg_customer_utterance_length",
    "first_customer_sentiment_score",
    "last_customer_sentiment_score",
    "mean_customer_sentiment_score",
    "sentiment_trend",
    "negative_sentiment_turn_count",
    "negative_sentiment_fraction",
    "negative_emotion_count",
    "anger_or_frustration_present",
    "supervisor_request_mentioned",
    "transfer_mentioned",
    "repeated_complaint_signal",
    "ends_on_negative_sentiment",
]

# Human-readable names for the same features, used when building
# structured "reasons" (spec section 3) — kept separate from the raw
# feature keys so the explanation reads naturally.
FEATURE_DISPLAY_NAMES = {
    "negative_sentiment_fraction": "High proportion of negative customer sentiment",
    "negative_emotion_count": "Multiple negative-emotion customer turns",
    "anger_or_frustration_present": "Anger or frustration detected",
    "supervisor_request_mentioned": "Customer requested a supervisor",
    "transfer_mentioned": "Call was transferred",
    "repeated_complaint_signal": "Repeated complaints in the conversation",
    "ends_on_negative_sentiment": "Conversation ends on negative sentiment",
    "sentiment_trend": "Sentiment worsened during the call",
    "conversation_duration_seconds": "Unusually long conversation",
}


def extract_business_features(conversation: list[dict]) -> dict:
    customer_turns = [t for t in conversation if t["speaker"] == "CUSTOMER"]
    agent_turns = [t for t in conversation if t["speaker"] == "AGENT"]
    customer_texts = [t["text"] for t in customer_turns]
    all_text = " ".join(t["text"] for t in conversation)

    sentiments = predict_sentiment_batch(customer_texts) if customer_texts else []
    emotions = predict_emotion_batch(customer_texts) if customer_texts else []
    scores = [_SENTIMENT_SCORE.get(p["label"], 0) for p in sentiments]

    duration = 0.0
    if conversation:
        duration = max(t["end"] for t in conversation) - min(t["start"] for t in conversation)

    negative_count = sum(1 for s in scores if s < 0)
    negative_emotion_count = sum(1 for e in emotions if e["label"] in _NEGATIVE_EMOTIONS)

    return {
        "num_turns": len(conversation),
        "num_customer_turns": len(customer_turns),
        "num_agent_turns": len(agent_turns),
        "conversation_duration_seconds": duration,
        "avg_customer_utterance_length": float(np.mean([len(t.split()) for t in customer_texts])) if customer_texts else 0.0,
        "first_customer_sentiment_score": scores[0] if scores else 0,
        "last_customer_sentiment_score": scores[-1] if scores else 0,
        "mean_customer_sentiment_score": float(np.mean(scores)) if scores else 0.0,
        "sentiment_trend": (scores[-1] - scores[0]) if len(scores) >= 2 else 0,
        "negative_sentiment_turn_count": negative_count,
        "negative_sentiment_fraction": negative_count / len(scores) if scores else 0.0,
        "negative_emotion_count": negative_emotion_count,
        "anger_or_frustration_present": 1 if any(e["label"] == "anger" for e in emotions) else 0,
        "supervisor_request_mentioned": 1 if _SUPERVISOR_RE.search(all_text) else 0,
        "transfer_mentioned": 1 if _TRANSFER_RE.search(all_text) else 0,
        # a simple, honest proxy: 2+ negative-sentiment customer turns —
        # not true repeated-complaint detection (that would need topic
        # clustering across turns), documented as a heuristic in
        # docs/BUSINESS_MODELS.md
        "repeated_complaint_signal": 1 if negative_count >= 2 else 0,
        "ends_on_negative_sentiment": 1 if (scores and scores[-1] < 0) else 0,
    }


def extract_business_intent(conversation: list[dict]) -> str:
    customer_turns = [t for t in conversation if t["speaker"] == "CUSTOMER"]
    if not customer_turns:
        return "UNKNOWN"
    return predict_intent(customer_turns[0]["text"])["label"]


def features_to_vector(features: dict) -> list[float]:
    return [features[name] for name in FEATURE_NAMES]
