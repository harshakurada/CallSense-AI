"""Conversation-level model tests. The ConversationModel's forward pass
is tested with random tensors (no encoder/download needed) since its
architecture is independent of how embeddings are produced. Synthetic
data generation and the attention-pooling mask are pure logic. Anything
needing the frozen text encoder, Module 4's trained classifiers (for
baseline features), or the trained conversation model itself is skipped
until those exist."""
from pathlib import Path

import pytest
import torch

from src.models.conversation.model import SPEAKERS, TASKS, AttentionPooling, ConversationModel
from src.models.conversation.synthetic import ESCALATION_MAP, OUTCOMES, RESOLUTION_MAP, SATISFACTION_MAP, generate_dataset

CONVERSATION_MODEL = Path("models/conversation/model.pt")
INTENT_MODEL = Path("models/nlp/intent/transformer/config.json")
SENTIMENT_MODEL = Path("models/nlp/sentiment/transformer/config.json")
EMOTION_MODEL = Path("models/nlp/emotion/transformer/config.json")

requires_conversation_model = pytest.mark.skipif(not CONVERSATION_MODEL.exists(), reason="run: python scripts/train_conversation_model.py")
requires_nlp_models = pytest.mark.skipif(
    not (INTENT_MODEL.exists() and SENTIMENT_MODEL.exists() and EMOTION_MODEL.exists()),
    reason="baseline features need Module 4's trained intent/sentiment/emotion models",
)


# --- synthetic data generation ---


def test_synthetic_generates_all_outcome_labels_are_consistent():
    examples = generate_dataset(60)
    for ex in examples:
        outcome = ex["outcome_template"]
        assert ex["labels"]["resolution"] == RESOLUTION_MAP[outcome]
        assert ex["labels"]["satisfaction"] == SATISFACTION_MAP[outcome]
        assert ex["labels"]["escalation"] == ESCALATION_MAP[outcome]


def test_synthetic_conversations_have_no_leftover_placeholders():
    examples = generate_dataset(100)
    for ex in examples:
        for turn in ex["conversation"]:
            assert "{{" not in turn["text"]


def test_synthetic_covers_short_and_long_conversations():
    examples = generate_dataset(80)
    lengths = [len(ex["conversation"]) for ex in examples]
    assert min(lengths) <= 7  # "short" — no extra clarification round
    assert max(lengths) >= 9  # "long" — extra clarification round added


def test_synthetic_alternates_speakers():
    examples = generate_dataset(10)
    for ex in examples:
        speakers = [t["speaker"] for t in ex["conversation"]]
        assert speakers[0] == "AGENT"  # agent always opens
        for a, b in zip(speakers, speakers[1:]):
            assert a != b  # strict alternation — no same-speaker back-to-back turns


def test_synthetic_escalated_outcome_produces_escalation_yes():
    rng_examples = [ex for ex in generate_dataset(200) if ex["outcome_template"] == "UNRESOLVED_ESCALATED"]
    assert rng_examples  # sanity: this outcome actually got sampled
    assert all(ex["labels"]["escalation"] == "Yes" for ex in rng_examples)


def test_synthetic_reuses_real_bitext_categories():
    examples = generate_dataset(30)
    categories = {ex["labels"]["category"] for ex in examples}
    # real Bitext categories (Module 4), not invented ones
    assert categories <= {"ACCOUNT", "ORDER", "REFUND", "INVOICE", "CONTACT", "PAYMENT", "FEEDBACK", "DELIVERY", "SHIPPING", "SUBSCRIPTION", "CANCEL"}


# --- ConversationModel architecture (random tensors, no encoder needed) ---


def _dummy_batch(batch_size=2, seq_len=5, dim=768):
    embeddings = torch.randn(batch_size, seq_len, dim)
    speaker_ids = torch.randint(0, len(SPEAKERS), (batch_size, seq_len))
    padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    return embeddings, speaker_ids, padding_mask


def test_conversation_model_forward_shapes():
    model = ConversationModel(label_counts={"category": 11, "resolution": 3, "satisfaction": 3, "escalation": 2})
    embeddings, speaker_ids, padding_mask = _dummy_batch()
    outputs = model(embeddings, speaker_ids, padding_mask)

    assert set(outputs.keys()) == {"category", "resolution", "satisfaction", "escalation"}
    assert outputs["category"].shape == (2, 11)
    assert outputs["resolution"].shape == (2, 3)
    assert outputs["escalation"].shape == (2, 2)


def test_conversation_model_handles_variable_length_via_padding():
    model = ConversationModel(label_counts={"resolution": 3})
    embeddings, speaker_ids, padding_mask = _dummy_batch(batch_size=1, seq_len=3)
    padding_mask[0, 2] = True  # last position is padding
    outputs = model(embeddings, speaker_ids, padding_mask)
    assert outputs["resolution"].shape == (1, 3)
    assert not torch.isnan(outputs["resolution"]).any()


def test_attention_pooling_ignores_padded_positions():
    pool = AttentionPooling(dim=4)
    hidden = torch.zeros(1, 3, 4)
    hidden[0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    hidden[0, 1] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    hidden[0, 2] = torch.tensor([999.0, 999.0, 999.0, 999.0])  # would dominate if not masked
    mask = torch.tensor([[False, False, True]])

    output = pool(hidden, mask)
    assert torch.all(output.abs() < 10)  # the padded 999s must not leak into the pooled result


def test_conversation_model_single_utterance_conversation():
    """Shortest possible interaction: one turn."""
    model = ConversationModel(label_counts={"resolution": 3})
    embeddings, speaker_ids, padding_mask = _dummy_batch(batch_size=1, seq_len=1)
    outputs = model(embeddings, speaker_ids, padding_mask)
    assert outputs["resolution"].shape == (1, 3)


# --- feature extraction (needs Module 4's trained models + Module 5's spaCy baseline) ---


@requires_nlp_models
def test_features_detect_negative_sentiment_conversation():
    from src.models.conversation.features import extract_conversation_features

    conversation = [
        {"speaker": "AGENT", "start": 0, "end": 2, "text": "How can I help?"},
        {"speaker": "CUSTOMER", "start": 2, "end": 5, "text": "This is absolutely terrible, I am furious and disgusted."},
    ]
    features = extract_conversation_features(conversation)
    assert features["customer_avg_sentiment_score"] < 0
    assert features["ends_on_negative_sentiment"] == 1.0


@requires_nlp_models
def test_features_multiple_intents_conversation_does_not_crash():
    from src.models.conversation.features import extract_conversation_intent

    conversation = [
        {"speaker": "AGENT", "start": 0, "end": 2, "text": "How can I help?"},
        {"speaker": "CUSTOMER", "start": 2, "end": 5, "text": "I want a refund and also need to update my account."},
    ]
    # only the first customer turn's intent is used — documented in features.py
    intent = extract_conversation_intent(conversation)
    assert isinstance(intent, str)


@requires_nlp_models
def test_features_no_customer_turns_returns_neutral_defaults():
    from src.models.conversation.features import extract_conversation_features

    conversation = [{"speaker": "AGENT", "start": 0, "end": 2, "text": "Hello?"}]
    features = extract_conversation_features(conversation)
    assert features["customer_avg_sentiment_score"] == 0.0
    assert features["num_customer_turns"] == 0


# --- full trained pipeline (skipped until trained) ---


@requires_conversation_model
def test_predict_conversation_returns_all_four_tasks():
    from src.models.conversation.inference import predict_conversation

    conversation = [
        {"speaker": "AGENT", "start": 0, "end": 2, "text": "Thank you for calling, how can I help?"},
        {"speaker": "CUSTOMER", "start": 2, "end": 5, "text": "I want a refund for order 12345."},
        {"speaker": "AGENT", "start": 5, "end": 8, "text": "I've processed that for you."},
        {"speaker": "CUSTOMER", "start": 8, "end": 10, "text": "Thank you so much!"},
    ]
    result = predict_conversation(conversation)
    assert set(result.keys()) == set(TASKS)
    for task in TASKS:
        assert 0.0 <= result[task]["confidence"] <= 1.0


@requires_conversation_model
def test_encode_conversation_returns_fixed_size_vector():
    from src.models.conversation.inference import encode_conversation
    from src.models.conversation.model import HIDDEN_DIM, TASKS

    conversation = [{"speaker": "AGENT", "start": 0, "end": 2, "text": "Hello?"}]
    vector = encode_conversation(conversation)
    # one pooled view per task, concatenated — see ConversationModel's docstring
    assert vector.shape == (HIDDEN_DIM * len(TASKS),)


@requires_conversation_model
def test_predict_conversation_empty_input():
    from src.models.conversation.inference import predict_conversation

    result = predict_conversation([])
    assert all(result[task]["label"] is None for task in TASKS)
