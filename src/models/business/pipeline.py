"""Unified conversation analysis: combines every prior module's output
into one JSON, matching the spec's example structure exactly
(intent, sentiment, emotion, entities, resolution, escalation,
agent_metrics)."""
from src.models.business.escalation import EscalationModel, load_model as load_escalation
from src.models.business.resolution import ResolutionModel, load_model as load_resolution
from src.nlp.inference import analyze_conversation as analyze_nlp
from src.nlp.ner.pipeline import analyze_conversation as analyze_entities


def analyze_call(conversation: list[dict]) -> dict:
    """conversation: [{"speaker", "start", "end", "text"}, ...] (Module 3's
    format). Returns the full business analysis for one call — no
    per-agent aggregation (see src/models/business/agent_metrics.py for
    that, which needs multiple calls, not one)."""
    resolution_model: ResolutionModel = load_resolution()
    escalation_model: EscalationModel = load_escalation()

    nlp_result = analyze_nlp(conversation)
    entity_result = analyze_entities(conversation)
    resolution_result = resolution_model.predict_one(conversation) if conversation else None
    escalation_result = escalation_model.predict_one(conversation) if conversation else None

    return {
        "intent": nlp_result["intent"],
        "sentiment": nlp_result["sentiment"],
        "emotion": nlp_result["emotion"],
        "entities": entity_result,
        "resolution": resolution_result,
        "escalation": escalation_result,
    }
