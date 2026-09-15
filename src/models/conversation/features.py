"""Classical conversation-level features for the baseline model. Reuses
Modules 4 and 5's already-trained/available components as feature
extractors — genuine integration, not reimplementation — computed
independently of the conversation-level labels being predicted, so using
them is legitimate signal, not leakage (see docs/CONVERSATION_MODEL.md
"Avoiding Data Leakage" for what would and wouldn't count as leakage here).
"""
import numpy as np

from src.nlp.emotion.inference import predict_emotion_batch
from src.nlp.intent.inference import predict_intent
from src.nlp.ner.baseline import extract_baseline_entities
from src.nlp.sentiment.inference import predict_sentiment_batch

_SENTIMENT_SCORE = {"positive": 1, "neutral": 0, "negative": -1}

FEATURE_NAMES = [
    "num_turns",
    "num_customer_turns",
    "num_agent_turns",
    "avg_utterance_length",
    "customer_avg_sentiment_score",
    "customer_negative_emotion_fraction",
    "num_entities",
    "ends_on_negative_sentiment",
]


def extract_conversation_features(conversation: list[dict]) -> dict:
    customer_turns = [t for t in conversation if t["speaker"] == "CUSTOMER"]
    agent_turns = [t for t in conversation if t["speaker"] == "AGENT"]
    all_texts = [t["text"] for t in conversation]

    customer_texts = [t["text"] for t in customer_turns]
    customer_sentiments = predict_sentiment_batch(customer_texts) if customer_texts else []
    customer_emotions = predict_emotion_batch(customer_texts) if customer_texts else []

    avg_sentiment_score = (
        float(np.mean([_SENTIMENT_SCORE.get(p["label"], 0) for p in customer_sentiments]))
        if customer_sentiments
        else 0.0
    )
    negative_emotion_fraction = (
        float(np.mean([1.0 if p["label"] in ("anger", "fear", "sadness") else 0.0 for p in customer_emotions]))
        if customer_emotions
        else 0.0
    )

    num_entities = sum(len(extract_baseline_entities(text)) for text in all_texts)

    last_customer_sentiment = customer_sentiments[-1]["label"] if customer_sentiments else "neutral"

    return {
        "num_turns": len(conversation),
        "num_customer_turns": len(customer_turns),
        "num_agent_turns": len(agent_turns),
        "avg_utterance_length": float(np.mean([len(t.split()) for t in all_texts])) if all_texts else 0.0,
        "customer_avg_sentiment_score": avg_sentiment_score,
        "customer_negative_emotion_fraction": negative_emotion_fraction,
        "num_entities": num_entities,
        "ends_on_negative_sentiment": 1.0 if last_customer_sentiment == "negative" else 0.0,
    }


def extract_conversation_intent(conversation: list[dict]) -> str:
    """The first customer turn's intent — a real, independently-computed
    feature (Module 4's fine-tuned classifier), not the conversation
    category label itself."""
    customer_turns = [t for t in conversation if t["speaker"] == "CUSTOMER"]
    if not customer_turns:
        return "UNKNOWN"
    return predict_intent(customer_turns[0]["text"])["label"]


def features_to_vector(features: dict) -> list[float]:
    return [features[name] for name in FEATURE_NAMES]
