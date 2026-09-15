"""Emotion dataset: dair-ai/emotion (see docs/NLP_MODELS.md for source/
license/size). Real labels are {sadness, joy, love, anger, fear, surprise}
— not the spec's suggested {anger, frustration, happiness, sadness, fear,
neutral, satisfaction}. No public conversational corpus labeled with
exactly frustration/satisfaction/neutral under a usable license was found,
so per the spec's own instruction ("use the actual dataset labels and
create a documented mapping if necessary"), the real 6 labels are used
directly; `joy` is the nearest available stand-in for "happiness" and
"satisfaction" is not distinguished from it. Documented, not hidden."""
from datasets import load_dataset

from src.nlp.common.data import class_distribution, stratified_subsample

DATASET_ID = "dair-ai/emotion"
MAX_TOTAL_TRAIN = 4000
MAX_TOTAL_EVAL = 800  # applied to the dataset's own val/test splits


def load_emotion_dataset(max_train: int = MAX_TOTAL_TRAIN, max_eval: int = MAX_TOTAL_EVAL) -> dict:
    ds = load_dataset(DATASET_ID, "split")
    label_names = ds["train"].features["label"].names

    def _texts_labels(split):
        texts = list(ds[split]["text"])
        labels = [label_names[i] for i in ds[split]["label"]]
        return texts, labels

    train_texts, train_labels = _texts_labels("train")
    val_texts, val_labels = _texts_labels("validation")
    test_texts, test_labels = _texts_labels("test")

    full_distribution = class_distribution(train_labels)

    train_texts, train_labels = stratified_subsample(train_texts, train_labels, max_train)
    val_texts, val_labels = stratified_subsample(val_texts, val_labels, max_eval)
    test_texts, test_labels = stratified_subsample(test_texts, test_labels, max_eval)

    return {
        "train": (train_texts, train_labels),
        "val": (val_texts, val_labels),
        "test": (test_texts, test_labels),
        "label_names": sorted(label_names),
        "full_distribution": full_distribution,
    }
