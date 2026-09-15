"""Sentiment dataset: Twitter US Airline Sentiment (see docs/NLP_MODELS.md
for source/license/size). Real customer complaints/praise about airline
service — the closest available public proxy for customer-service
sentiment; a genuine call-center transcript corpus with sentiment labels
was not found under a usable license."""
from datasets import load_dataset

from src.nlp.common.data import class_distribution, stratified_split, stratified_subsample

DATASET_ID = "osanseviero/twitter-airline-sentiment"
MAX_TOTAL_EXAMPLES = 4000


def load_sentiment_dataset(max_total: int = MAX_TOTAL_EXAMPLES) -> dict:
    ds = load_dataset(DATASET_ID, split="train")
    texts = list(ds["text"])
    labels = list(ds["airline_sentiment"])

    label_names = sorted(set(labels))
    texts, labels = stratified_subsample(texts, labels, max_total)
    splits = stratified_split(texts, labels)
    splits["label_names"] = label_names
    splits["full_distribution"] = class_distribution(labels)
    return splits
