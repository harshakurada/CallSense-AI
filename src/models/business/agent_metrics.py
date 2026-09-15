"""Agent analytics computed over a list of conversations attributed to one
agent. Only metrics genuinely computable from this project's data are
implemented — per the spec's own instruction, nothing is invented for data
that doesn't exist. See docs/BUSINESS_MODELS.md for exactly what's
skipped and why.

**Interruption frequency is deliberately NOT implemented.** Module 3's
diarization explicitly does not detect overlapping speech (documented in
docs/DIARIZATION.md) — the energy-based VAD it's built on has no concept
of two people talking at once. Without real overlap detection, there is
no honest way to count interruptions; fabricating one from turn-taking
patterns alone would be exactly the kind of invented metric the spec
warns against.
"""
from src.models.conversation.inference import predict_conversation
from src.models.business.escalation import EscalationModel
from src.models.business.features import extract_business_features
from src.models.business.resolution import ResolutionModel

_SENTIMENT_SCORE = {"positive": 1, "neutral": 0, "negative": -1}


def _talk_listen_ratio(conversation: list[dict]) -> float | None:
    """AGENT talk time / CUSTOMER talk time, from real turn start/end
    timestamps — genuinely measurable, unlike interruptions."""
    agent_time = sum(t["end"] - t["start"] for t in conversation if t["speaker"] == "AGENT")
    customer_time = sum(t["end"] - t["start"] for t in conversation if t["speaker"] == "CUSTOMER")
    if customer_time <= 0:
        return None
    return agent_time / customer_time


def analyze_single_call(
    conversation: list[dict], resolution_model: ResolutionModel, escalation_model: EscalationModel
) -> dict:
    """Per-call metrics that later get aggregated across an agent's calls."""
    features = extract_business_features(conversation)
    resolution = resolution_model.predict_one(conversation)
    escalation = escalation_model.predict_one(conversation)

    return {
        "handling_time_seconds": features["conversation_duration_seconds"],
        "resolved": resolution["label"] == "Resolved",
        "escalated": escalation["risk_tier"] == "High",
        "sentiment_improvement": features["last_customer_sentiment_score"] - features["first_customer_sentiment_score"],
        "talk_listen_ratio": _talk_listen_ratio(conversation),
        "customer_satisfaction": predict_conversation(conversation)["satisfaction"]["label"],
    }


def aggregate_agent_metrics(per_call_metrics: list[dict]) -> dict:
    """Aggregates analyze_single_call() outputs across an agent's calls
    into the metrics the spec asks for."""
    n = len(per_call_metrics)
    if n == 0:
        return {
            "num_calls": 0,
            "resolution_rate": None,
            "avg_handling_time_seconds": None,
            "escalation_rate": None,
            "avg_sentiment_improvement": None,
            "avg_talk_listen_ratio": None,
            "satisfaction_distribution": {},
        }

    talk_listen_values = [m["talk_listen_ratio"] for m in per_call_metrics if m["talk_listen_ratio"] is not None]
    satisfaction_counts: dict[str, int] = {}
    for m in per_call_metrics:
        label = m["customer_satisfaction"]
        satisfaction_counts[label] = satisfaction_counts.get(label, 0) + 1

    return {
        "num_calls": n,
        "resolution_rate": sum(m["resolved"] for m in per_call_metrics) / n,
        "avg_handling_time_seconds": sum(m["handling_time_seconds"] for m in per_call_metrics) / n,
        "escalation_rate": sum(m["escalated"] for m in per_call_metrics) / n,
        "avg_sentiment_improvement": sum(m["sentiment_improvement"] for m in per_call_metrics) / n,
        "avg_talk_listen_ratio": (sum(talk_listen_values) / len(talk_listen_values)) if talk_listen_values else None,
        "satisfaction_distribution": satisfaction_counts,
        # interruption_frequency intentionally omitted — see module docstring
    }
