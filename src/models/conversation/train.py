"""Trains the conversation-level deep model and the classical baseline on
the same synthetic conversations and test split, for a real, comparable
per-task metrics table (docs/CONVERSATION_MODEL.md)."""
import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.conversation.baseline import ConversationBaseline, save_baseline
from src.models.conversation.encoder import _load_encoder, encode_utterances
from src.models.conversation.model import MAX_UTTERANCES, SPEAKERS, TASKS, ConversationModel
from src.models.conversation.synthetic import generate_dataset
from src.nlp.common.metrics import compute_metrics, metrics_to_dict
from src.nlp.common.transformer import _load_classifier
from src.nlp.ner.baseline import _load_spacy
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _release_cached_models(*cached_functions) -> None:
    """Baseline feature extraction (sentiment/emotion/intent/NER, each its
    own model kept alive by @lru_cache) and the frozen conversation
    encoder are never needed at the same time as each other, but both
    stayed resident simultaneously — several models' worth of weights at
    once on an 8GB machine that had already OOM-killed training more than
    once. Explicitly dropping whichever phase just finished is a real fix,
    not a defensive nicety."""
    for fn in cached_functions:
        fn.cache_clear()
    gc.collect()

MODEL_DIR = Path("models/conversation")
N_EXAMPLES = 1200


def _split(examples: list[dict], val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(examples))
    n_test = int(len(examples) * test_frac)
    n_val = int(len(examples) * val_frac)
    test_idx, val_idx, train_idx = indices[:n_test], indices[n_test : n_test + n_val], indices[n_test + n_val :]
    return [examples[i] for i in train_idx], [examples[i] for i in val_idx], [examples[i] for i in test_idx]


class ConversationDataset(Dataset):
    """Pre-encodes every conversation's utterances once at construction
    time (frozen encoder, no gradient) rather than re-encoding every
    epoch — the encoder never changes, so this is a real compute saving,
    not just a convenience."""

    def __init__(self, examples: list[dict], label2id: dict[str, dict[str, int]]):
        self.samples = []
        for ex in examples:
            conversation = ex["conversation"][:MAX_UTTERANCES]
            texts = [t["text"] for t in conversation]
            embeddings = encode_utterances(texts)
            speaker_ids = torch.tensor([SPEAKERS.index(t["speaker"]) for t in conversation], dtype=torch.long)
            labels = {task: label2id[task][ex["labels"][task]] for task in TASKS}
            self.samples.append((embeddings, speaker_ids, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _collate(batch):
    max_len = max(emb.shape[0] for emb, _, _ in batch)
    dim = batch[0][0].shape[1]
    batch_size = len(batch)

    embeddings = torch.zeros(batch_size, max_len, dim)
    speaker_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)  # True = padded
    labels = {task: torch.zeros(batch_size, dtype=torch.long) for task in TASKS}

    for i, (emb, spk, lab) in enumerate(batch):
        n = emb.shape[0]
        embeddings[i, :n] = emb
        speaker_ids[i, :n] = spk
        padding_mask[i, :n] = False
        for task in TASKS:
            labels[task][i] = lab[task]

    return embeddings, speaker_ids, padding_mask, labels


def train(num_epochs: int = 20, n_examples: int = N_EXAMPLES) -> dict:
    examples = generate_dataset(n_examples)
    train_examples, val_examples, test_examples = _split(examples)
    logger.info("Conversations: %d train / %d val / %d test", len(train_examples), len(val_examples), len(test_examples))

    label_names = {task: sorted({ex["labels"][task] for ex in examples}) for task in TASKS}
    label2id = {task: {label: i for i, label in enumerate(names)} for task, names in label_names.items()}
    id2label = {task: {i: label for label, i in mapping.items()} for task, mapping in label2id.items()}

    # --- baseline ---
    train_conversations = [ex["conversation"] for ex in train_examples]
    test_conversations = [ex["conversation"] for ex in test_examples]
    train_labels = {task: [ex["labels"][task] for ex in train_examples] for task in TASKS}
    test_labels = {task: [ex["labels"][task] for ex in test_examples] for task in TASKS}

    baseline = ConversationBaseline()
    baseline.fit(train_conversations, train_labels)
    baseline_metrics = baseline.evaluate(test_conversations, test_labels)
    save_baseline(baseline, MODEL_DIR / "baseline.pkl")
    for task, m in baseline_metrics.items():
        logger.info("Baseline %s: accuracy=%.4f macro_f1=%.4f", task, m.accuracy, m.f1_macro)

    # Baseline feature extraction loaded Module 4's sentiment/emotion/intent
    # classifiers and Module 5's spaCy model — none needed again until the
    # deep model's own frozen encoder is done. See _release_cached_models.
    _release_cached_models(_load_classifier, _load_spacy)

    # --- deep model ---
    train_dataset = ConversationDataset(train_examples, label2id)
    val_dataset = ConversationDataset(val_examples, label2id)
    test_dataset = ConversationDataset(test_examples, label2id)

    # The frozen encoder's job (turning text into embeddings) is done now
    # that all three datasets are built — freed before the training loop
    # starts, which needs memory for the trainable model/optimizer instead.
    _release_cached_models(_load_encoder)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=_collate)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=_collate)

    model = ConversationModel(label_counts={task: len(names) for task, names in label_names.items()})
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # category (11 classes, driven by diverse real text) was found to be
    # starved out by the other three tasks (2-3 classes each, and largely
    # solvable from a small closed set of template endings): a first run's
    # confusion matrix showed the shared trunk had collapsed to never
    # predicting 6 of 11 categories at all, while resolution/satisfaction/
    # escalation hit 100%. Upweighting category's loss and — more
    # importantly — selecting the "best" checkpoint by mean per-task val
    # *accuracy* rather than summed val *loss* (which the easy tasks
    # dominate once their loss floors near zero) are both aimed at this.
    # See docs/CONVERSATION_MODEL.md for the full incident.
    task_weights = {"category": 2.0, "resolution": 1.0, "satisfaction": 1.0, "escalation": 1.0}

    best_val_score = -1.0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for embeddings, speaker_ids, padding_mask, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(embeddings, speaker_ids, padding_mask)
            loss = sum(task_weights[task] * loss_fn(outputs[task], labels[task]) for task in TASKS)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = {task: 0 for task in TASKS}
        total = 0
        with torch.no_grad():
            for embeddings, speaker_ids, padding_mask, labels in val_loader:
                outputs = model(embeddings, speaker_ids, padding_mask)
                val_loss += sum(task_weights[task] * loss_fn(outputs[task], labels[task]) for task in TASKS).item()
                total += embeddings.shape[0]
                for task in TASKS:
                    correct[task] += (outputs[task].argmax(dim=-1) == labels[task]).sum().item()

        val_accuracy = {task: correct[task] / total for task in TASKS}
        mean_val_accuracy = sum(val_accuracy.values()) / len(TASKS)

        logger.info(
            "Epoch %d: train_loss=%.4f val_loss=%.4f val_acc=%s (mean=%.4f)",
            epoch + 1, train_loss / len(train_loader), val_loss / len(val_loader), val_accuracy, mean_val_accuracy,
        )
        if mean_val_accuracy > best_val_score:
            best_val_score = mean_val_accuracy
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "label_names": label_names, "id2label": id2label},
        MODEL_DIR / "model.pt",
    )

    model.eval()
    all_preds = {task: [] for task in TASKS}
    all_true = {task: [] for task in TASKS}
    with torch.no_grad():
        for embeddings, speaker_ids, padding_mask, labels in test_loader:
            outputs = model(embeddings, speaker_ids, padding_mask)
            for task in TASKS:
                preds = outputs[task].argmax(dim=-1).tolist()
                all_preds[task].extend(id2label[task][p] for p in preds)
                all_true[task].extend(id2label[task][t] for t in labels[task].tolist())

    deep_metrics = {task: compute_metrics(all_true[task], all_preds[task], label_names[task]) for task in TASKS}
    for task, m in deep_metrics.items():
        logger.info("Deep model %s: accuracy=%.4f macro_f1=%.4f", task, m.accuracy, m.f1_macro)

    return {
        "dataset_size": {"train": len(train_examples), "val": len(val_examples), "test": len(test_examples)},
        "label_names": label_names,
        "baseline": {task: metrics_to_dict(m) for task, m in baseline_metrics.items()},
        "deep_model": {task: metrics_to_dict(m) for task, m in deep_metrics.items()},
    }
