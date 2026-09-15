"""Transparent, configurable agent quality score. Every weight lives in
configs/agent_score.yaml with its rationale documented there — never
hardcoded here, and never presented as an empirically-fit weighting
(no labeled ground truth for "good agent" exists to fit against; see
docs/BUSINESS_MODELS.md)."""
from configs.settings import get_agent_score_config

# sentiment_improvement ranges roughly [-2, +2] (last score - first score,
# each in {-1, 0, +1}) — rescaled to [0, 1] like the other three components.
_SENTIMENT_DELTA_MIN = -2.0
_SENTIMENT_DELTA_MAX = 2.0


def _normalize_sentiment_delta(delta: float | None) -> float:
    if delta is None:
        return 0.5  # no data — neutral midpoint, not a guessed direction
    clamped = max(_SENTIMENT_DELTA_MIN, min(_SENTIMENT_DELTA_MAX, delta))
    return (clamped - _SENTIMENT_DELTA_MIN) / (_SENTIMENT_DELTA_MAX - _SENTIMENT_DELTA_MIN)


def compute_agent_score(agent_metrics: dict) -> dict:
    """agent_metrics: the dict returned by
    src.models.business.agent_metrics.aggregate_agent_metrics(). Returns
    {"score": float in [0,1], "components": {...}, "weights": {...}} —
    the full breakdown, not just the final number, so the score is always
    auditable."""
    weights = get_agent_score_config()["weights"]

    satisfaction_dist = agent_metrics.get("satisfaction_distribution") or {}
    total_satisfaction_calls = sum(satisfaction_dist.values())
    satisfied_fraction = (
        satisfaction_dist.get("Satisfied", 0) / total_satisfaction_calls if total_satisfaction_calls else 0.5
    )

    resolution_rate = agent_metrics.get("resolution_rate")
    escalation_rate = agent_metrics.get("escalation_rate")

    components = {
        "resolution_rate": resolution_rate if resolution_rate is not None else 0.5,
        "non_escalation_rate": (1 - escalation_rate) if escalation_rate is not None else 0.5,
        "sentiment_improvement": _normalize_sentiment_delta(agent_metrics.get("avg_sentiment_improvement")),
        "satisfaction": satisfied_fraction,
    }

    score = sum(weights[name] * value for name, value in components.items())

    return {"score": score, "components": components, "weights": weights}
