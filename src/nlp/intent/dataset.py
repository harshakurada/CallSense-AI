"""Intent dataset: Bitext customer support dataset (see docs/NLP_MODELS.md
for source/license/size). Uses the real `category` field (11 classes) as
the intent label — not the spec's suggested 10-class list, since that list
doesn't correspond to any dataset's actual labels; using real labels
instead of inventing a mapping follows the same "use actual dataset
labels" instruction given explicitly for emotion."""
from datasets import load_dataset

from src.nlp.common.data import class_distribution, stratified_split, stratified_subsample

DATASET_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
MAX_TOTAL_EXAMPLES = 4000  # CPU fine-tuning budget — see docs/NLP_MODELS.md


def load_intent_dataset(max_total: int = MAX_TOTAL_EXAMPLES) -> dict:
    ds = load_dataset(DATASET_ID, split="train")
    texts = list(ds["instruction"])
    labels = list(ds["category"])

    label_names = sorted(set(labels))
    texts, labels = stratified_subsample(texts, labels, max_total)
    splits = stratified_split(texts, labels)
    splits["label_names"] = label_names
    splits["full_distribution"] = class_distribution(labels)
    return splits
