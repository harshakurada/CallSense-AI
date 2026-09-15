"""Token-classification (NER) fine-tuning and inference.

Default model is `prajjwal1/bert-tiny` (~4.4M params), not
distilbert-base-uncased. Module 4's emotion training was killed twice by
the OS for low memory on this 8GB machine even after batch-size/Adafactor
mitigations (see src/nlp/common/transformer.py) — NER is trained after
that was discovered, so it starts with the memory-safe model directly
rather than repeating the same failure. See docs/NER.md for what this
trades away versus DistilBERT."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from seqeval.metrics import classification_report as seqeval_report
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.utils.data import Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments

from src.utils.logging import get_logger

logger = get_logger(__name__)

MODEL_NAME = "prajjwal1/bert-tiny"
MAX_LENGTH = 64


class TokenClassificationDataset(Dataset):
    def __init__(self, tokenized_inputs, label_ids: list[list[int]]):
        self.tokenized_inputs = tokenized_inputs
        self.label_ids = label_ids

    def __len__(self):
        return len(self.label_ids)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_inputs.items()}
        item["labels"] = torch.tensor(self.label_ids[idx])
        return item


def _tokenize_and_align_labels(tokenizer, tokens_list: list[list[str]], tags_list: list[list[str]], label2id: dict):
    tokenized = tokenizer(
        tokens_list, truncation=True, padding="max_length", max_length=MAX_LENGTH, is_split_into_words=True
    )
    all_label_ids = []
    for i, tags in enumerate(tags_list):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        prev_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # special tokens ([CLS]/[SEP]/padding) — ignored by the loss
            elif word_idx != prev_word_idx:
                label_ids.append(label2id[tags[word_idx]])
            else:
                # subsequent subword of the same word — repeat as I- if it
                # was B-/I-, since the word itself didn't restart
                tag = tags[word_idx]
                label_ids.append(label2id["I-" + tag[2:]] if tag.startswith("B-") else label2id[tag])
            prev_word_idx = word_idx
        all_label_ids.append(label_ids)
    return tokenized, all_label_ids


@dataclass
class NERMetrics:
    precision: float
    recall: float
    f1: float
    report: dict
    true_labels: list[list[str]]  # per-test-example BIO tag sequences — lets callers re-slice by subset (e.g. real vs synthetic)
    pred_labels: list[list[str]]


def _decode_predictions(predictions: np.ndarray, label_ids: list[list[int]], id2label: dict):
    pred_ids = np.argmax(predictions, axis=2)
    true_labels, pred_labels = [], []
    for pred_row, label_row in zip(pred_ids, label_ids):
        true_seq, pred_seq = [], []
        for p, l in zip(pred_row, label_row):
            if l == -100:
                continue
            true_seq.append(id2label[l])
            pred_seq.append(id2label[p])
        true_labels.append(true_seq)
        pred_labels.append(pred_seq)
    return true_labels, pred_labels


def fine_tune_ner(
    train_tokens, train_tags, val_tokens, val_tags, test_tokens, test_tags, label_names: list[str], output_dir: str | Path,
    num_epochs: int = 3, learning_rate: float = 5e-4,
) -> tuple[NERMetrics, Path]:
    output_dir = Path(output_dir)
    label2id = {label: i for i, label in enumerate(label_names)}
    id2label = {i: label for label, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label_names), id2label=id2label, label2id=label2id
    )

    train_encodings, train_label_ids = _tokenize_and_align_labels(tokenizer, train_tokens, train_tags, label2id)
    val_encodings, val_label_ids = _tokenize_and_align_labels(tokenizer, val_tokens, val_tags, label2id)
    test_encodings, test_label_ids = _tokenize_and_align_labels(tokenizer, test_tokens, test_tags, label2id)

    train_dataset = TokenClassificationDataset(train_encodings, train_label_ids)
    val_dataset = TokenClassificationDataset(val_encodings, val_label_ids)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=num_epochs,
        # See src/nlp/common/transformer.py — this machine's 8GB RAM killed
        # a batch-16 fine-tune under memory pressure from other running
        # applications. Same effective batch size (8*2=16), lower peak.
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="no",
        dataloader_pin_memory=False,
        logging_steps=50,
        # 3e-5 (DistilBERT-appropriate) scored far below the baseline on
        # bert-tiny (F1 0.029 vs. baseline's 0.525) — same failure mode as
        # Module 4's emotion classifier, same fix: a smaller, weaker-
        # pretrained model needs a higher learning rate. See docs/NER.md.
        learning_rate=learning_rate,
        weight_decay=0.01,
        report_to=[],
        use_cpu=not torch.cuda.is_available(),
        optim="adafactor",  # see src/nlp/common/transformer.py — AdamW's optimizer state OOM'd this machine
        seed=42,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset)

    logger.info("Fine-tuning NER model on %d train / %d val examples", len(train_tokens), len(val_tokens))
    trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    test_dataset = TokenClassificationDataset(test_encodings, test_label_ids)
    raw_predictions = trainer.predict(test_dataset)
    true_labels, pred_labels = _decode_predictions(raw_predictions.predictions, test_label_ids, id2label)

    metrics = NERMetrics(
        precision=precision_score(true_labels, pred_labels),
        recall=recall_score(true_labels, pred_labels),
        f1=f1_score(true_labels, pred_labels),
        report=seqeval_report(true_labels, pred_labels, output_dict=True, zero_division=0),
        true_labels=true_labels,
        pred_labels=pred_labels,
    )
    return metrics, output_dir


def predict_entities(model_dir: str | Path, text: str) -> list[dict]:
    """Runs the fine-tuned model on raw text and returns character-offset
    entities: [{"label", "text", "start", "end", "confidence"}, ...].
    confidence is the mean softmax probability of the predicted tags across
    the entity's tokens — the model's own output, not invented."""
    model_dir = str(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()
    id2label = model.config.id2label

    encoding = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    offset_mapping = encoding.pop("offset_mapping")[0].tolist()

    with torch.no_grad():
        logits = model(**encoding).logits[0]
    probs = torch.softmax(logits, dim=-1).numpy()
    pred_ids = np.argmax(probs, axis=-1)

    entities = []
    current = None
    for (start, end), pred_id, prob_row in zip(offset_mapping, pred_ids, probs):
        if start == end:  # special token
            continue
        tag = id2label[int(pred_id)]
        confidence = float(prob_row[pred_id])

        if tag.startswith("B-"):
            if current:
                entities.append(current)
            current = {"label": tag[2:], "text": text[start:end], "start": start, "end": end, "_confidences": [confidence]}
        elif tag.startswith("I-") and current and current["label"] == tag[2:]:
            current["text"] = text[current["start"] : end]
            current["end"] = end
            current["_confidences"].append(confidence)
        else:
            if current:
                entities.append(current)
            current = None
    if current:
        entities.append(current)

    for e in entities:
        e["confidence"] = sum(e["_confidences"]) / len(e["_confidences"])
        del e["_confidences"]

    return entities
