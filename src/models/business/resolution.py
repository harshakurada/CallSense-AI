"""Resolution prediction: Resolved / Partially Resolved / Unresolved.

A genuine ML model (RandomForestClassifier — real non-linear learning,
not a lookup table), trained on the named features in features.py rather
than raw embeddings, specifically so predictions can be explained by
feature importance (spec section 3) rather than being a black box.
"""
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.business.features import FEATURE_NAMES, extract_business_features, extract_business_intent, features_to_vector
from src.nlp.common.metrics import ClassificationMetrics, compute_metrics

MODEL_PATH = Path("models/business/resolution.pkl")


class ResolutionModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.intent_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.classifier = RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced", random_state=42
        )
        self.combined_feature_names: list[str] = []

    def _build_matrix(self, conversations: list[list[dict]], fit: bool) -> np.ndarray:
        numeric = np.array([features_to_vector(extract_business_features(c)) for c in conversations])
        intents = np.array([extract_business_intent(c) for c in conversations]).reshape(-1, 1)

        if fit:
            scaled = self.scaler.fit_transform(numeric)
            intent_encoded = self.intent_encoder.fit_transform(intents)
            self.combined_feature_names = FEATURE_NAMES + [
                f"intent={c}" for c in self.intent_encoder.categories_[0]
            ]
        else:
            scaled = self.scaler.transform(numeric)
            intent_encoded = self.intent_encoder.transform(intents)

        return np.hstack([scaled, intent_encoded])

    def fit(self, conversations: list[list[dict]], labels: list[str]) -> None:
        X = self._build_matrix(conversations, fit=True)
        self.classifier.fit(X, labels)

    def predict(self, conversations: list[list[dict]]) -> list[str]:
        X = self._build_matrix(conversations, fit=False)
        return list(self.classifier.predict(X))

    def predict_one(self, conversation: list[dict]) -> dict:
        """Returns {"label", "confidence", "probabilities"} — confidence
        is the forest's own vote fraction for the predicted class."""
        X = self._build_matrix([conversation], fit=False)
        probs = self.classifier.predict_proba(X)[0]
        classes = self.classifier.classes_
        pred_idx = int(np.argmax(probs))
        return {
            "label": classes[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {c: float(p) for c, p in zip(classes, probs)},
        }

    def evaluate(self, conversations: list[list[dict]], labels: list[str]) -> ClassificationMetrics:
        predictions = self.predict(conversations)
        label_names = sorted(set(labels) | set(predictions))
        return compute_metrics(labels, predictions, label_names)

    def feature_importances(self) -> dict[str, float]:
        return dict(zip(self.combined_feature_names, self.classifier.feature_importances_.tolist()))


def save_model(model: ResolutionModel, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: Path = MODEL_PATH) -> ResolutionModel:
    with open(path, "rb") as f:
        return pickle.load(f)
