"""Combined NLP inference: speaker-attributed transcript (Module 3) ->
intent -> sentiment -> emotion -> structured predictions (Module 4 spec
section 12). The only place all three task models are composed."""
from src.nlp.emotion.timeline import build_emotion_timeline
from src.nlp.intent.inference import predict_intent
from src.nlp.sentiment.inference import analyze_conversation_sentiment
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _intent_input_text(conversation: list[dict]) -> str:
    """Intent is classified at call level, not per-utterance (see the
    spec's example JSON, which has one top-level intent). Concatenates
    CUSTOMER turns if the conversation has role labels; otherwise falls
    back to the whole conversation's text, since intent still needs some
    input even with generic SPEAKER_NN labels."""
    customer_turns = [turn["text"] for turn in conversation if turn["speaker"] == "CUSTOMER"]
    if customer_turns:
        return " ".join(customer_turns)
    return " ".join(turn["text"] for turn in conversation)


def analyze_conversation(conversation: list[dict]) -> dict:
    """conversation: [{"speaker": str, "start": float, "end": float, "text": str}, ...]

    Returns:
    {
      "intent": {"label": str, "confidence": float},
      "sentiment": {
        "per_utterance": [...],
        "customer_level": {...},
        "conversation_level": {...}
      },
      "emotion": {
        "timeline": [{"time": float, "speaker": str, "label": str, "confidence": float}, ...]
      }
    }
    All confidence values are the models' own softmax probabilities —
    nothing here is estimated or invented.
    """
    if not conversation:
        return {
            "intent": {"label": None, "confidence": None},
            "sentiment": {"per_utterance": [], "customer_level": {}, "conversation_level": {}},
            "emotion": {"timeline": []},
        }

    intent_text = _intent_input_text(conversation)
    intent_result = predict_intent(intent_text)

    sentiment_result = analyze_conversation_sentiment(conversation)
    emotion_timeline = build_emotion_timeline(conversation)

    logger.info(
        "Analyzed %d-turn conversation: intent=%s sentiment(conv)=%s",
        len(conversation),
        intent_result["label"],
        sentiment_result["conversation_level"]["label"],
    )

    return {
        "intent": intent_result,
        "sentiment": sentiment_result,
        "emotion": {"timeline": emotion_timeline},
    }
