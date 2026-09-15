"""Conversation-level architecture:

    Utterances
      -> frozen DistilBERT [CLS] embeddings (src/models/conversation/encoder.py)
      -> + learned speaker embedding + learned positional embedding
      -> Transformer encoder (context aggregation across utterances)
      -> attention pooling -> one conversation vector
      -> 4 independent linear heads: category, resolution, satisfaction, escalation

Speaker and positional embeddings are the "context" features the spec asks
for (speaker, utterance order) that a per-utterance classifier has no way
to use — this is the actual point of a conversation-level model over
independently classifying each utterance. Intent/sentiment/emotion/entity
features are deliberately NOT concatenated into this deep model's input;
see docs/CONVERSATION_MODEL.md for why (they're used in the baseline
instead, and mixing them into the frozen-embedding path here would makes
it unclear which representation is actually driving predictions).
"""
from dataclasses import dataclass

import torch
import torch.nn as nn

from src.models.conversation.encoder import EMBEDDING_DIM

HIDDEN_DIM = 128
MAX_UTTERANCES = 40
SPEAKERS = ["AGENT", "CUSTOMER"]
TASKS = ["category", "resolution", "satisfaction", "escalation"]


class AttentionPooling(nn.Module):
    """A single learned query vector attends over the utterance sequence
    to produce one fixed-size conversation vector — lets the model weigh
    a decisive late utterance ("I want a manager!") more than a routine
    greeting, unlike mean pooling."""

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim))

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # hidden: (batch, seq, dim), mask: (batch, seq) — True where padded
        scores = hidden @ self.query  # (batch, seq)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (batch, seq, 1)
        return (hidden * weights).sum(dim=1)  # (batch, dim)


class ConversationModel(nn.Module):
    """Per-task attention pooling, not one pooled vector shared by all four
    heads. A single shared pool was tried first and failed: category needs
    the opening utterance's content, resolution/satisfaction/escalation
    need the closing turns — one pooled vector serving both ended up
    starved for whichever task's gradient didn't dominate (see
    docs/CONVERSATION_MODEL.md for two real training runs that
    demonstrated this: category either collapsed to never predicting most
    classes, or the other three did, depending on which task's signal was
    stronger that run). Separate query vectors, one per task, let each
    head attend to a different part of the same shared context-encoded
    sequence instead of fighting over one summary."""

    def __init__(self, label_counts: dict[str, int], hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.project = nn.Linear(EMBEDDING_DIM, hidden_dim)
        self.speaker_embedding = nn.Embedding(len(SPEAKERS), hidden_dim)
        self.position_embedding = nn.Embedding(MAX_UTTERANCES, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2, dropout=0.1, batch_first=True
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pools = nn.ModuleDict({task: AttentionPooling(hidden_dim) for task in label_counts})

        self.heads = nn.ModuleDict({task: nn.Linear(hidden_dim, n) for task, n in label_counts.items()})

    def forward(self, utterance_embeddings: torch.Tensor, speaker_ids: torch.Tensor, padding_mask: torch.Tensor) -> dict:
        # utterance_embeddings: (batch, seq, EMBEDDING_DIM)
        # speaker_ids: (batch, seq) int64, 0=AGENT/1=CUSTOMER
        # padding_mask: (batch, seq) bool, True where padded
        batch_size, seq_len, _ = utterance_embeddings.shape
        positions = torch.arange(seq_len, device=utterance_embeddings.device).unsqueeze(0).expand(batch_size, -1)

        hidden = self.project(utterance_embeddings)
        hidden = hidden + self.speaker_embedding(speaker_ids) + self.position_embedding(positions)
        hidden = self.layer_norm(hidden)

        hidden = self.context_encoder(hidden, src_key_padding_mask=padding_mask)

        return {task: head(self.pools[task](hidden, padding_mask)) for task, head in self.heads.items()}


@dataclass
class ConversationBatch:
    utterance_embeddings: torch.Tensor
    speaker_ids: torch.Tensor
    padding_mask: torch.Tensor
    labels: dict[str, torch.Tensor] | None = None
