"""TF-IDF + Logistic Regression baseline — the benchmark every fine-tuned
Transformer is measured against in docs/NLP_MODELS.md's comparison table."""
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.nlp.common.metrics import ClassificationMetrics, compute_metrics


def train_baseline(train_texts: list[str], train_labels: list[str]) -> Pipeline:
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",  # same imbalance strategy as the transformer's weighted loss
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(train_texts, train_labels)
    return pipeline


def evaluate_baseline(
    pipeline: Pipeline, texts: list[str], labels: list[str], label_names: list[str]
) -> ClassificationMetrics:
    predictions = pipeline.predict(texts)
    return compute_metrics(labels, list(predictions), label_names)


def save_baseline(pipeline: Pipeline, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)


def load_baseline(path: str | Path) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)
