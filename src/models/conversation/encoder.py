"""Utterance encoding for the conversation-level model.

The encoder (distilbert-base-uncased) is used **frozen** — no gradient,
no fine-tuning — and only its [CLS] embedding per utterance is kept. This
is a deliberate choice, not a shortcut: fine-tuning a full encoder here
would mean backpropagating through DistilBERT for every utterance in every
conversation, which on this project's 8GB CPU-only machine already caused
an OOM kill during Module 4's much simpler per-utterance fine-tuning (see
docs/NLP_MODELS.md). A frozen encoder + a small trainable conversation
Transformer on top is a standard, practical pattern for exactly this
resource-constrained setting, and keeps the trainable parameter count to
a few hundred thousand instead of 66 million.
"""
from functools import lru_cache

import torch
from transformers import AutoModel, AutoTokenizer

ENCODER_NAME = "distilbert-base-uncased"
EMBEDDING_DIM = 768
MAX_LENGTH = 48  # conversation utterances are short — see docs/CONVERSATION_MODEL.md


@lru_cache
def _load_encoder():
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
    model = AutoModel.from_pretrained(ENCODER_NAME)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return tokenizer, model


def encode_utterances(texts: list[str]) -> torch.Tensor:
    """Returns a (len(texts), EMBEDDING_DIM) tensor of frozen [CLS]
    embeddings — one per utterance, in order."""
    if not texts:
        return torch.zeros((0, EMBEDDING_DIM))

    tokenizer, model = _load_encoder()
    with torch.no_grad():
        encoded = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")
        output = model(**encoded)
        return output.last_hidden_state[:, 0, :]  # [CLS] token per utterance
