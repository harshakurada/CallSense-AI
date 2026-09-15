"""NLP tests. Pure-logic pieces (subsampling/splitting math, metrics,
sentiment aggregation, timeline formatting, the baseline trainer) need no
fine-tuned model and always run. Transformer inference tests are skipped if
the corresponding model hasn't been trained yet (python scripts/
train_nlp_models.py) — trained by this same Module 4 change, but the test
suite must stay runnable in a fresh checkout before training is re-run."""
from pathlib import Path

import pytest

from src.nlp.common.baseline import evaluate_baseline, train_baseline
from src.nlp.common.data import class_distribution, stratified_split, stratified_subsample
from src.nlp.common.metrics import compute_metrics
from src.nlp.emotion.timeline import format_timeline_text
from src.nlp.sentiment.inference import aggregate_sentiment

INTENT_MODEL = Path("models/nlp/intent/transformer")
SENTIMENT_MODEL = Path("models/nlp/sentiment/transformer")
EMOTION_MODEL = Path("models/nlp/emotion/transformer")

# Check config.json specifically, not just the directory — TrainingArguments
# creates the transformer/ dir (for its checkpoints/ subfolder) well before
# trainer.save_model() actually finishes, so the bare directory existing
# doesn't mean training completed.
requires_intent_model = pytest.mark.skipif(not (INTENT_MODEL / "config.json").exists(), reason="run: python scripts/train_nlp_models.py --task intent")
requires_sentiment_model = pytest.mark.skipif(not (SENTIMENT_MODEL / "config.json").exists(), reason="run: python scripts/train_nlp_models.py --task sentiment")
requires_emotion_model = pytest.mark.skipif(not (EMOTION_MODEL / "config.json").exists(), reason="run: python scripts/train_nlp_models.py --task emotion")


# --- data utilities (no model needed) ---


def test_stratified_subsample_preserves_proportions():
    texts = [f"t{i}" for i in range(100)]
    labels = ["A"] * 80 + ["B"] * 20
    sub_texts, sub_labels = stratified_subsample(texts, labels, max_total=20)
    dist = class_distribution(sub_labels)
    assert dist["A"] > dist["B"]
    assert sum(dist.values()) <= 22  # rounding may push slightly over max_total


def test_stratified_subsample_keeps_rare_class_present():
    texts = [f"t{i}" for i in range(1000)]
    labels = ["A"] * 990 + ["B"] * 10
    _, sub_labels = stratified_subsample(texts, labels, max_total=50)
    assert "B" in sub_labels


def test_stratified_split_respects_fractions_per_class():
    texts = [f"t{i}" for i in range(200)]
    labels = ["A"] * 100 + ["B"] * 100
    splits = stratified_split(texts, labels, val_frac=0.1, test_frac=0.1)
    train_texts, train_labels = splits["train"]
    val_texts, val_labels = splits["val"]
    test_texts, test_labels = splits["test"]

    assert len(train_texts) + len(val_texts) + len(test_texts) == 200
    # both classes present in every split
    for split_labels in (train_labels, val_labels, test_labels):
        assert set(split_labels) == {"A", "B"}


def test_class_distribution_counts_correctly():
    dist = class_distribution(["A", "B", "A", "A", "C"])
    assert dist == {"A": 3, "B": 1, "C": 1}


# --- metrics (no model needed) ---


def test_compute_metrics_perfect_predictions():
    y_true = ["A", "B", "A", "B"]
    y_pred = ["A", "B", "A", "B"]
    metrics = compute_metrics(y_true, y_pred, label_names=["A", "B"])
    assert metrics.accuracy == 1.0
    assert metrics.f1_macro == 1.0
    assert metrics.confusion_matrix == [[2, 0], [0, 2]]


def test_compute_metrics_with_errors():
    y_true = ["A", "A", "B", "B"]
    y_pred = ["A", "B", "B", "B"]
    metrics = compute_metrics(y_true, y_pred, label_names=["A", "B"])
    assert metrics.accuracy == 0.75
    assert metrics.per_class["A"]["recall"] == 0.5
    assert metrics.per_class["B"]["recall"] == 1.0


# --- baseline (sklearn only, no transformer) ---


def test_baseline_trains_and_evaluates_on_separable_data():
    train_texts = ["refund my order please"] * 20 + ["my account is locked out"] * 20
    train_labels = ["REFUND"] * 20 + ["ACCOUNT"] * 20
    test_texts = ["I need a refund", "cannot access my account"]
    test_labels = ["REFUND", "ACCOUNT"]

    model = train_baseline(train_texts, train_labels)
    metrics = evaluate_baseline(model, test_texts, test_labels, label_names=["ACCOUNT", "REFUND"])

    assert metrics.accuracy == 1.0  # trivially separable vocabulary


# --- sentiment aggregation (no model needed) ---


def test_aggregate_sentiment_majority_vote():
    per_utterance = [
        {"label": "negative", "confidence": 0.9},
        {"label": "negative", "confidence": 0.8},
        {"label": "positive", "confidence": 0.6},
    ]
    result = aggregate_sentiment(per_utterance)
    assert result["label"] == "negative"
    assert result["score"] == pytest.approx((-1 - 1 + 1) / 3)


def test_aggregate_sentiment_empty_input():
    result = aggregate_sentiment([])
    assert result == {"label": None, "confidence": None, "score": None}


# --- emotion timeline formatting (no model needed) ---


def test_format_timeline_text():
    timeline = [
        {"time": 0, "speaker": "CUSTOMER", "label": "neutral", "confidence": 0.9},
        {"time": 65, "speaker": "CUSTOMER", "label": "anger", "confidence": 0.8},
    ]
    text = format_timeline_text(timeline)
    assert text == "00:00 Neutral\n01:05 Anger"


# --- transformer inference integration (skipped until trained) ---


@requires_intent_model
def test_predict_intent_returns_valid_schema():
    from src.nlp.intent.inference import predict_intent

    result = predict_intent("I want to cancel my subscription")
    assert isinstance(result["label"], str)
    assert 0.0 <= result["confidence"] <= 1.0


@requires_sentiment_model
def test_predict_sentiment_returns_valid_schema():
    from src.nlp.sentiment.inference import predict_sentiment

    result = predict_sentiment("This is terrible, I am very upset")
    assert isinstance(result["label"], str)
    assert 0.0 <= result["confidence"] <= 1.0


@requires_emotion_model
def test_predict_emotion_returns_valid_schema():
    from src.nlp.emotion.inference import predict_emotion

    result = predict_emotion("I am so happy about this")
    assert isinstance(result["label"], str)
    assert 0.0 <= result["confidence"] <= 1.0


@requires_intent_model
@requires_sentiment_model
@requires_emotion_model
def test_analyze_conversation_full_integration():
    from src.nlp.inference import analyze_conversation

    conversation = [
        {"speaker": "AGENT", "start": 0.0, "end": 3.0, "text": "Thank you for calling, how can I help?"},
        {"speaker": "CUSTOMER", "start": 3.0, "end": 8.0, "text": "I want to cancel my subscription, this is frustrating"},
        {"speaker": "AGENT", "start": 8.0, "end": 12.0, "text": "I understand, let me help you with that"},
    ]
    result = analyze_conversation(conversation)

    assert isinstance(result["intent"]["label"], str)
    assert len(result["sentiment"]["per_utterance"]) == 3
    assert result["sentiment"]["customer_level"]["label"] is not None
    assert len(result["emotion"]["timeline"]) == 3
    assert result["emotion"]["timeline"][0]["time"] == 0.0


def test_analyze_conversation_empty_input():
    from src.nlp.inference import analyze_conversation

    result = analyze_conversation([])
    assert result["intent"]["label"] is None
    assert result["sentiment"]["per_utterance"] == []
    assert result["emotion"]["timeline"] == []
