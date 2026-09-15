"""Statistical baseline: classical conversation-level features (turn
counts, aggregated sentiment/emotion, entity counts, first-turn intent) +
Logistic Regression per task — the benchmark the deep conversation model
is measured against."""
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.conversation.features import extract_conversation_features, extract_conversation_intent, features_to_vector
from src.nlp.common.metrics import ClassificationMetrics, compute_metrics


def _build_feature_matrix(conversations: list[list[dict]]) -> tuple[np.ndarray, list[str]]:
    numeric = np.array([features_to_vector(extract_conversation_features(c)) for c in conversations])
    intents = [extract_conversation_intent(c) for c in conversations]
    return numeric, intents


class ConversationBaseline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.intent_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.classifiers: dict[str, LogisticRegression] = {}

    def _combined_features(self, numeric: np.ndarray, intents: list[str], fit: bool) -> np.ndarray:
        if fit:
            scaled = self.scaler.fit_transform(numeric)
            intent_encoded = self.intent_encoder.fit_transform(np.array(intents).reshape(-1, 1))
        else:
            scaled = self.scaler.transform(numeric)
            intent_encoded = self.intent_encoder.transform(np.array(intents).reshape(-1, 1))
        return np.hstack([scaled, intent_encoded])

    def fit(self, conversations: list[list[dict]], labels: dict[str, list[str]]) -> None:
        numeric, intents = _build_feature_matrix(conversations)
        X = self._combined_features(numeric, intents, fit=True)
        for task, task_labels in labels.items():
            clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
            clf.fit(X, task_labels)
            self.classifiers[task] = clf

    def predict(self, conversations: list[list[dict]]) -> dict[str, list[str]]:
        numeric, intents = _build_feature_matrix(conversations)
        X = self._combined_features(numeric, intents, fit=False)
        return {task: list(clf.predict(X)) for task, clf in self.classifiers.items()}

    def evaluate(self, conversations: list[list[dict]], labels: dict[str, list[str]]) -> dict[str, ClassificationMetrics]:
        predictions = self.predict(conversations)
        return {
            task: compute_metrics(labels[task], predictions[task], label_names=sorted(set(labels[task]) | set(predictions[task])))
            for task in labels
        }


def save_baseline(model: ConversationBaseline, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_baseline(path: str | Path) -> ConversationBaseline:
    with open(path, "rb") as f:
        return pickle.load(f)
