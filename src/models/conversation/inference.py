"""Reusable conversation-level inference. `encode_conversation()` exposes
the pooled representation itself (not just the 4 task predictions) so a
later module can consume it directly — e.g. Module 7's agent-quality
scoring — without retraining or reimplementing the encoder stack."""
from functools import lru_cache
from pathlib import Path

import torch

from src.models.conversation.encoder import encode_utterances
from src.models.conversation.model import SPEAKERS, TASKS, ConversationModel

MODEL_PATH = Path("models/conversation/model.pt")


@lru_cache
def _load_model():
    checkpoint = torch.load(MODEL_PATH, weights_only=False)
    label_names = checkpoint["label_names"]
    model = ConversationModel(label_counts={task: len(names) for task, names in label_names.items()})
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["id2label"]


def _prepare_batch(conversation: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    texts = [t["text"] for t in conversation]
    embeddings = encode_utterances(texts).unsqueeze(0)  # (1, seq, dim)
    speaker_ids = torch.tensor([[SPEAKERS.index(t["speaker"]) for t in conversation]], dtype=torch.long)
    padding_mask = torch.zeros(1, len(conversation), dtype=torch.bool)
    return embeddings, speaker_ids, padding_mask


def encode_conversation(conversation: list[dict]) -> torch.Tensor:
    """Returns the reusable conversation representation (shape:
    (hidden_dim * len(TASKS),)) for consumption by a later module — e.g.
    Module 7's agent-quality scoring.

    The model pools per task (see ConversationModel's docstring for why:
    one shared pooled vector for all four heads caused task interference
    in training), so "the" representation is the concatenation of every
    task's pooled view of the same context-encoded sequence — a later
    consumer gets all four perspectives, not just one arbitrarily chosen
    task's.
    """
    model, _ = _load_model()
    embeddings, speaker_ids, padding_mask = _prepare_batch(conversation)

    hidden = model.project(embeddings)
    seq_len = hidden.shape[1]
    positions = torch.arange(seq_len).unsqueeze(0)
    hidden = hidden + model.speaker_embedding(speaker_ids) + model.position_embedding(positions)
    hidden = model.layer_norm(hidden)
    hidden = model.context_encoder(hidden, src_key_padding_mask=padding_mask)

    with torch.no_grad():
        pooled = [model.pools[task](hidden, padding_mask).squeeze(0) for task in TASKS]
        return torch.cat(pooled, dim=-1)


def predict_conversation(conversation: list[dict]) -> dict:
    """Returns {"category": {"label", "confidence"}, "resolution": {...},
    "satisfaction": {...}, "escalation": {...}} — confidence is the
    model's own softmax probability for its predicted class."""
    if not conversation:
        return {task: {"label": None, "confidence": None} for task in TASKS}

    model, id2label = _load_model()
    embeddings, speaker_ids, padding_mask = _prepare_batch(conversation)

    with torch.no_grad():
        outputs = model(embeddings, speaker_ids, padding_mask)

    result = {}
    for task in TASKS:
        probs = torch.softmax(outputs[task][0], dim=-1)
        pred_id = int(probs.argmax())
        result[task] = {"label": id2label[task][pred_id], "confidence": float(probs[pred_id])}
    return result
