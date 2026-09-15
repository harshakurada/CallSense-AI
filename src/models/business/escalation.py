"""Escalation risk: a genuine ML model (RandomForestClassifier) predicts
an escalation probability; a configurable threshold maps it to Low/Medium/
High. Explanations come in two explicitly separate forms (spec section 3):

- `model_explanation`: which named features were both (a) globally
  important to the trained model and (b) actually active/flagged for this
  specific conversation. This is a real, if approximate, model-derived
  signal — not SHAP-grade per-instance attribution (adding that dependency
  wasn't worth it on this memory-constrained machine), and documented as
  such in docs/BUSINESS_MODELS.md.
- `natural_language_explanation`: a templated sentence built FROM
  model_explanation's factors. This is text generation, not a second
  model — never presented as an independent judgment.
"""
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.models.business.features import FEATURE_DISPLAY_NAMES, FEATURE_NAMES, extract_business_features, features_to_vector

MODEL_PATH = Path("models/business/escalation.pkl")

# Boolean/flag-style features are "active" at 1; numeric ones need a
# domain threshold — chosen to mean "notably elevated", not just nonzero.
_ACTIVE_THRESHOLDS = {
    "negative_sentiment_fraction": 0.5,
    "negative_emotion_count": 1,
    "conversation_duration_seconds": 60.0,
    "sentiment_trend": -1,  # trend more negative than -1 point
}
_BOOLEAN_FEATURES = {
    "anger_or_frustration_present",
    "supervisor_request_mentioned",
    "transfer_mentioned",
    "repeated_complaint_signal",
    "ends_on_negative_sentiment",
}

DEFAULT_RISK_THRESHOLDS = {"low_max": 0.35, "medium_max": 0.65}  # >medium_max is High


def _is_active(feature_name: str, value: float) -> bool:
    if feature_name in _BOOLEAN_FEATURES:
        return value >= 1
    if feature_name in _ACTIVE_THRESHOLDS:
        threshold = _ACTIVE_THRESHOLDS[feature_name]
        return value <= threshold if feature_name == "sentiment_trend" else value >= threshold
    return False


def risk_tier(probability: float, thresholds: dict = DEFAULT_RISK_THRESHOLDS) -> str:
    if probability <= thresholds["low_max"]:
        return "Low"
    if probability <= thresholds["medium_max"]:
        return "Medium"
    return "High"


class EscalationModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(
            n_estimators=200, max_depth=6, class_weight="balanced", random_state=42
        )

    def fit(self, conversations: list[list[dict]], labels: list[str]) -> None:
        """labels: "Yes"/"No" ground truth (the binary event the
        probability is calibrated against — Low/Medium/High is a
        downstream bucketing of that probability, not a separately
        trained 3-class target)."""
        feature_dicts = [extract_business_features(c) for c in conversations]
        X = self.scaler.fit_transform([features_to_vector(f) for f in feature_dicts])
        self.classifier.fit(X, labels)

    def predict_proba(self, conversations: list[list[dict]]) -> np.ndarray:
        feature_dicts = [extract_business_features(c) for c in conversations]
        X = self.scaler.transform([features_to_vector(f) for f in feature_dicts])
        yes_idx = list(self.classifier.classes_).index("Yes")
        return self.classifier.predict_proba(X)[:, yes_idx]

    def feature_importances(self) -> dict[str, float]:
        return dict(zip(FEATURE_NAMES, self.classifier.feature_importances_.tolist()))

    def predict_one(self, conversation: list[dict], thresholds: dict = DEFAULT_RISK_THRESHOLDS) -> dict:
        features = extract_business_features(conversation)
        X = self.scaler.transform([features_to_vector(features)])
        yes_idx = list(self.classifier.classes_).index("Yes")
        probability = float(self.classifier.predict_proba(X)[0][yes_idx])
        tier = risk_tier(probability, thresholds)

        importances = self.feature_importances()
        active_and_important = sorted(
            (
                (name, importances[name])
                for name in FEATURE_NAMES
                if _is_active(name, features[name])
            ),
            key=lambda kv: kv[1],
            reverse=True,
        )
        top_reasons = [name for name, _ in active_and_important[:4]]

        return {
            "risk_tier": tier,
            "probability": probability,
            "model_explanation": {
                "top_factors": top_reasons,
                "feature_importances": {name: importances[name] for name, _ in active_and_important},
            },
            "natural_language_explanation": _render_explanation(tier, top_reasons),
        }

    def evaluate(self, conversations: list[list[dict]], labels: list[str]) -> dict:
        from sklearn.metrics import (
            average_precision_score,
            brier_score_loss,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        probs = self.predict_proba(conversations)
        y_true = [1 if label == "Yes" else 0 for label in labels]
        preds = [1 if p >= 0.5 else 0 for p in probs]

        return {
            "precision": precision_score(y_true, preds, zero_division=0),
            "recall": recall_score(y_true, preds, zero_division=0),
            "f1": f1_score(y_true, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_true, probs),
            "average_precision": average_precision_score(y_true, probs),
            # calibration: how close predicted probabilities are to actual
            # outcome frequencies — lower is better, 0 is perfect
            "brier_score": brier_score_loss(y_true, probs),
        }


def _render_explanation(tier: str, top_reasons: list[str]) -> str:
    if not top_reasons:
        return f"Escalation risk: {tier}. No strong contributing factors identified."
    factors = "; ".join(FEATURE_DISPLAY_NAMES.get(r, r) for r in top_reasons)
    return f"Escalation risk: {tier}. Contributing factors: {factors}."


def save_model(model: EscalationModel, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: Path = MODEL_PATH) -> EscalationModel:
    with open(path, "rb") as f:
        return pickle.load(f)
