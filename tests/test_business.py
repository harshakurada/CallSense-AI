"""Business model tests. Risk tiering, agent-score math, and metric
aggregation are pure logic and always run. Resolution/escalation model
tests are skipped until trained (python scripts/train_business_models.py)."""
from pathlib import Path

import pytest

from src.models.business.agent_metrics import aggregate_agent_metrics
from src.models.business.agent_score import compute_agent_score, _normalize_sentiment_delta
from src.models.business.escalation import DEFAULT_RISK_THRESHOLDS, risk_tier

RESOLUTION_MODEL = Path("models/business/resolution.pkl")
ESCALATION_MODEL = Path("models/business/escalation.pkl")
requires_business_models = pytest.mark.skipif(
    not (RESOLUTION_MODEL.exists() and ESCALATION_MODEL.exists()),
    reason="run: python scripts/train_business_models.py",
)


# --- risk tiering ---


def test_risk_tier_boundaries():
    assert risk_tier(0.0) == "Low"
    assert risk_tier(DEFAULT_RISK_THRESHOLDS["low_max"]) == "Low"
    assert risk_tier(DEFAULT_RISK_THRESHOLDS["low_max"] + 0.01) == "Medium"
    assert risk_tier(DEFAULT_RISK_THRESHOLDS["medium_max"]) == "Medium"
    assert risk_tier(DEFAULT_RISK_THRESHOLDS["medium_max"] + 0.01) == "High"
    assert risk_tier(1.0) == "High"


def test_risk_tier_respects_custom_thresholds():
    custom = {"low_max": 0.1, "medium_max": 0.2}
    assert risk_tier(0.15, custom) == "Medium"
    assert risk_tier(0.25, custom) == "High"


# --- agent metrics aggregation ---


def test_aggregate_agent_metrics_computes_rates():
    per_call = [
        {"handling_time_seconds": 100, "resolved": True, "escalated": False, "sentiment_improvement": 1, "talk_listen_ratio": 1.2, "customer_satisfaction": "Satisfied"},
        {"handling_time_seconds": 200, "resolved": False, "escalated": True, "sentiment_improvement": -1, "talk_listen_ratio": 0.8, "customer_satisfaction": "Dissatisfied"},
    ]
    result = aggregate_agent_metrics(per_call)
    assert result["num_calls"] == 2
    assert result["resolution_rate"] == 0.5
    assert result["escalation_rate"] == 0.5
    assert result["avg_handling_time_seconds"] == 150
    assert result["avg_sentiment_improvement"] == 0.0
    assert result["satisfaction_distribution"] == {"Satisfied": 1, "Dissatisfied": 1}


def test_aggregate_agent_metrics_empty_input():
    result = aggregate_agent_metrics([])
    assert result["num_calls"] == 0
    assert result["resolution_rate"] is None


def test_aggregate_agent_metrics_handles_missing_talk_listen_ratio():
    per_call = [
        {"handling_time_seconds": 100, "resolved": True, "escalated": False, "sentiment_improvement": 0, "talk_listen_ratio": None, "customer_satisfaction": "Neutral"},
    ]
    result = aggregate_agent_metrics(per_call)
    assert result["avg_talk_listen_ratio"] is None


def test_agent_metrics_never_reports_interruption_frequency():
    # explicit regression test for the spec's "do not invent metrics from
    # unavailable data" instruction — Module 3 can't detect overlapping
    # speech, so this must never appear as a computed metric
    result = aggregate_agent_metrics([{"handling_time_seconds": 1, "resolved": True, "escalated": False, "sentiment_improvement": 0, "talk_listen_ratio": 1.0, "customer_satisfaction": "Neutral"}])
    assert "interruption_frequency" not in result


# --- agent score ---


def test_normalize_sentiment_delta_clamps_and_scales():
    assert _normalize_sentiment_delta(0.0) == 0.5
    assert _normalize_sentiment_delta(2.0) == 1.0
    assert _normalize_sentiment_delta(-2.0) == 0.0
    assert _normalize_sentiment_delta(10.0) == 1.0  # clamped, not out of range
    assert _normalize_sentiment_delta(None) == 0.5


def test_compute_agent_score_is_bounded_and_auditable():
    metrics = {
        "resolution_rate": 0.8,
        "escalation_rate": 0.1,
        "avg_sentiment_improvement": 1.0,
        "satisfaction_distribution": {"Satisfied": 8, "Neutral": 1, "Dissatisfied": 1},
    }
    result = compute_agent_score(metrics)
    assert 0.0 <= result["score"] <= 1.0
    assert set(result["components"].keys()) == {"resolution_rate", "non_escalation_rate", "sentiment_improvement", "satisfaction"}
    assert result["weights"]  # weights are always reported, never hidden


def test_compute_agent_score_handles_missing_data_neutrally():
    metrics = {"resolution_rate": None, "escalation_rate": None, "avg_sentiment_improvement": None, "satisfaction_distribution": {}}
    result = compute_agent_score(metrics)
    # missing data should not silently produce a 0 or a fabricated high score
    assert result["components"]["resolution_rate"] == 0.5
    assert result["components"]["satisfaction"] == 0.5


# --- resolution/escalation models (skipped until trained) ---


@requires_business_models
def test_resolution_model_predict_schema():
    from src.models.business.resolution import load_model

    model = load_model()
    conversation = [
        {"speaker": "AGENT", "start": 0, "end": 2, "text": "How can I help?"},
        {"speaker": "CUSTOMER", "start": 2, "end": 5, "text": "I want a refund for order 12345."},
        {"speaker": "AGENT", "start": 5, "end": 8, "text": "I've processed that for you."},
        {"speaker": "CUSTOMER", "start": 8, "end": 10, "text": "Thank you so much!"},
    ]
    result = model.predict_one(conversation)
    assert result["label"] in {"Resolved", "Partially Resolved", "Unresolved"}
    assert 0.0 <= result["confidence"] <= 1.0


@requires_business_models
def test_escalation_model_predict_schema_and_explanation():
    from src.models.business.escalation import load_model

    model = load_model()
    conversation = [
        {"speaker": "AGENT", "start": 0, "end": 2, "text": "How can I help?"},
        {"speaker": "CUSTOMER", "start": 2, "end": 5, "text": "This is ridiculous, I want to speak to a manager right now!"},
    ]
    result = model.predict_one(conversation)
    assert result["risk_tier"] in {"Low", "Medium", "High"}
    assert 0.0 <= result["probability"] <= 1.0
    assert "top_factors" in result["model_explanation"]
    assert isinstance(result["natural_language_explanation"], str)
    # the explicit supervisor request should show up as a factor given this text
    assert "supervisor_request_mentioned" in result["model_explanation"]["top_factors"]


@requires_business_models
def test_escalation_explanation_distinguishes_model_from_nl():
    from src.models.business.escalation import load_model

    model = load_model()
    conversation = [{"speaker": "CUSTOMER", "start": 0, "end": 2, "text": "Get me a manager!"}]
    result = model.predict_one(conversation)
    # model_explanation is structured data; natural_language_explanation is prose built from it
    assert isinstance(result["model_explanation"], dict)
    assert isinstance(result["natural_language_explanation"], str)
    assert result["risk_tier"] in result["natural_language_explanation"] or result["risk_tier"].lower() in result["natural_language_explanation"].lower()


@requires_business_models
def test_full_pipeline_matches_spec_schema():
    from src.models.business.pipeline import analyze_call

    conversation = [
        {"speaker": "AGENT", "start": 0, "end": 2, "text": "Thank you for calling, how can I help?"},
        {"speaker": "CUSTOMER", "start": 2, "end": 6, "text": "I want to cancel my subscription, this is frustrating"},
        {"speaker": "AGENT", "start": 6, "end": 9, "text": "I understand, let me help you with that"},
    ]
    result = analyze_call(conversation)
    assert set(result.keys()) == {"intent", "sentiment", "emotion", "entities", "resolution", "escalation"}
    assert result["resolution"]["label"] in {"Resolved", "Partially Resolved", "Unresolved"}
    assert result["escalation"]["risk_tier"] in {"Low", "Medium", "High"}


@requires_business_models
def test_agent_metrics_full_integration():
    from src.models.business.agent_metrics import analyze_single_call
    from src.models.business.escalation import load_model as load_escalation
    from src.models.business.resolution import load_model as load_resolution

    resolution_model = load_resolution()
    escalation_model = load_escalation()
    conversation = [
        {"speaker": "AGENT", "start": 0, "end": 3, "text": "How can I help?"},
        {"speaker": "CUSTOMER", "start": 3, "end": 6, "text": "My order never arrived."},
        {"speaker": "AGENT", "start": 6, "end": 9, "text": "I've resent it, apologies."},
        {"speaker": "CUSTOMER", "start": 9, "end": 11, "text": "Thank you!"},
    ]
    result = analyze_single_call(conversation, resolution_model, escalation_model)
    assert isinstance(result["resolved"], bool)
    assert isinstance(result["escalated"], bool)
    assert result["talk_listen_ratio"] is not None
