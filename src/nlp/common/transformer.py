"""Generic Hugging Face Transformer fine-tuning + inference for text
classification — used identically by intent, sentiment, and emotion so
their results are comparable and none of them re-implements training.

Default model: distilbert-base-uncased, not full BERT/RoBERTa/DeBERTa.
Module 1 fixed this as a CPU-only-by-default project; DistilBERT keeps ~97%
of BERT's benchmark performance at roughly 60% of the parameters and
noticeably faster CPU inference/training. Intent and sentiment were both
trained on it successfully. See docs/NLP_MODELS.md for the reasoning and
measured numbers.

`model_name` is a parameter, not hardcoded, because emotion's training run
was killed twice by the OS for low memory on this 8GB machine even after
the batch-size/Adafactor mitigations below — a real, repeated hardware
constraint, not a one-off. Emotion (and NER) fall back to
`prajjwal1/bert-tiny` (~4.4M params vs. DistilBERT's ~66M) instead of
retrying the same configuration a third time; see docs/NLP_MODELS.md for
what that trades away.
"""
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.nlp.common.metrics import ClassificationMetrics, compute_metrics
from src.utils.logging import get_logger

logger = get_logger(__name__)

MODEL_NAME = "distilbert-base-uncased"


class TextClassificationDataset(Dataset):
    def __init__(self, encodings, label_ids: list[int]):
        self.encodings = encodings
        self.label_ids = label_ids

    def __len__(self):
        return len(self.label_ids)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.label_ids[idx])
        return item


class WeightedLossTrainer(Trainer):
    """Applies inverse-frequency class weights to the loss instead of
    resampling the data — the class-imbalance strategy used consistently
    across baseline (class_weight='balanced') and transformer here."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def _compute_class_weights(label_ids: list[int], num_labels: int) -> torch.Tensor:
    counts = Counter(label_ids)
    total = len(label_ids)
    weights = [total / (num_labels * counts.get(i, 1)) for i in range(num_labels)]
    return torch.tensor(weights, dtype=torch.float32)


@dataclass
class TrainedClassifier:
    metrics: ClassificationMetrics
    output_dir: Path


def fine_tune_classifier(
    train_texts: list[str],
    train_labels: list[str],
    val_texts: list[str],
    val_labels: list[str],
    test_texts: list[str],
    test_labels: list[str],
    label_names: list[str],
    output_dir: str | Path,
    num_epochs: int = 3,
    max_length: int = 64,
    model_name: str = MODEL_NAME,
    learning_rate: float = 2e-5,
) -> TrainedClassifier:
    output_dir = Path(output_dir)
    label2id = {label: i for i, label in enumerate(label_names)}
    id2label = {i: label for label, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label_names), id2label=id2label, label2id=label2id
    )

    def tokenize(texts):
        return tokenizer(texts, truncation=True, padding="max_length", max_length=max_length)

    train_label_ids = [label2id[l] for l in train_labels]
    val_label_ids = [label2id[l] for l in val_labels]

    train_dataset = TextClassificationDataset(tokenize(train_texts), train_label_ids)
    val_dataset = TextClassificationDataset(tokenize(val_texts), val_label_ids)

    class_weights = _compute_class_weights(train_label_ids, len(label_names))
    logger.info("Class weights (inverse frequency): %s", dict(zip(label_names, class_weights.tolist())))

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=num_epochs,
        # Small batch + accumulation instead of batch=16 directly: this
        # machine has 8GB RAM shared with other running applications, and
        # a batch-16 sentiment fine-tune was killed by the OS for low
        # memory mid-run. Same effective batch size (8*2=16), lower peak.
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        learning_rate=learning_rate,
        weight_decay=0.01,
        report_to=[],
        use_cpu=not torch.cuda.is_available(),
        dataloader_pin_memory=False,
        # Adafactor instead of the Trainer default AdamW: AdamW keeps two
        # extra full-size moment buffers per parameter (~3x model size in
        # optimizer state alone), which is what actually exhausted this
        # 8GB machine's memory, not the batch size. Adafactor factors that
        # state down to roughly a third the memory at the cost of somewhat
        # noisier convergence — an acceptable trade given the alternative
        # is the process being killed.
        optim="adafactor",
        seed=42,
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        class_weights=class_weights,
    )

    logger.info("Fine-tuning %s on %d train / %d val examples", model_name, len(train_texts), len(val_texts))
    trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    test_predictions = predict_batch(output_dir, test_texts, max_length=max_length)
    predicted_labels = [p["label"] for p in test_predictions]
    metrics = compute_metrics(test_labels, predicted_labels, label_names)

    return TrainedClassifier(metrics=metrics, output_dir=output_dir)


@lru_cache
def _load_classifier(model_dir: str):
    """Cached per model_dir — found reloading from disk on every single
    predict_batch() call (no caching at all originally), which meant
    Module 6's baseline feature extraction — one predict call per
    conversation, ~1200 conversations — repeatedly allocated and freed a
    full model. That alloc/dealloc churn was a real contributor to this
    8GB machine's OOM kills, not just wasted time."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def predict_batch(model_dir: str | Path, texts: list[str], max_length: int = 64, batch_size: int = 32) -> list[dict]:
    """Returns [{"label": str, "confidence": float, "probabilities": {label: prob}}, ...].
    Confidence is the model's own softmax probability for its predicted
    class — never a fabricated or estimated value."""
    tokenizer, model = _load_classifier(str(model_dir))
    id2label = model.config.id2label
    results = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).numpy()
            for row in probs:
                pred_id = int(np.argmax(row))
                results.append(
                    {
                        "label": id2label[pred_id],
                        "confidence": float(row[pred_id]),
                        "probabilities": {id2label[j]: float(row[j]) for j in range(len(row))},
                    }
                )
    return results


def predict_one(model_dir: str | Path, text: str, max_length: int = 64) -> dict:
    return predict_batch(model_dir, [text], max_length=max_length)[0]
