"""NER training data: a real dataset (Few-NERD, CC BY-SA 4.0) for general
entities, augmented with synthetic examples (src/nlp/ner/synthetic.py) for
business-specific entities no public dataset labels. See docs/NER.md for
why CoNLL-2003 was not used (requires an NIST/Reuters license agreement)."""
import random

from datasets import load_dataset

from src.nlp.ner.synthetic import generate_dataset

DATASET_ID = "DFKI-SLT/few-nerd"
MAX_REAL_EXAMPLES = 4000
N_SYNTHETIC_EXAMPLES = 2000

# Few-NERD's coarse types we keep — "art", "building", "event", "other" are
# outside this project's target schema (see docs/NER.md) and are relabeled O.
_FEW_NERD_LABEL_MAP = {
    "person": "PERSON",
    "organization": "ORGANIZATION",
    "location": "LOCATION",
    "product": "PRODUCT",
}

ALL_ENTITY_TYPES = ["PERSON", "ORGANIZATION", "LOCATION", "PRODUCT", "DATE", "MONEY", "ORDER_ID", "ACCOUNT_ID", "INVOICE_ID", "PHONE", "EMAIL"]


def _flat_tags_to_bio(tokens: list[str], flat_labels: list[str]) -> list[str]:
    """Few-NERD's ner_tags are per-token entity types with no B-/I- prefix
    — a run of consecutive identical labels is one entity. Converts that
    to standard BIO so it can be mixed with the synthetic BIO data."""
    bio = []
    prev_label = None
    for label in flat_labels:
        if label == "O":
            bio.append("O")
        elif label == prev_label:
            bio.append(f"I-{label}")
        else:
            bio.append(f"B-{label}")
        prev_label = label
    return bio


def _load_few_nerd_subset(max_examples: int, seed: int = 42) -> list[tuple[list[str], list[str]]]:
    ds = load_dataset(DATASET_ID, "supervised", split="train")
    label_names = ds.features["ner_tags"].feature.names

    examples = []
    for row in ds:
        tokens = row["tokens"]
        raw_labels = [label_names[i] for i in row["ner_tags"]]
        mapped = [_FEW_NERD_LABEL_MAP.get(label, "O") for label in raw_labels]
        if any(label != "O" for label in mapped):  # keep only sentences with a target entity
            examples.append((tokens, _flat_tags_to_bio(tokens, mapped)))

    random.Random(seed).shuffle(examples)
    return examples[:max_examples]


def load_ner_dataset(
    max_real: int = MAX_REAL_EXAMPLES, n_synthetic: int = N_SYNTHETIC_EXAMPLES, seed: int = 42
) -> dict:
    real_examples = _load_few_nerd_subset(max_real, seed=seed)
    synthetic_examples = generate_dataset(n_synthetic, seed=seed)

    all_examples = [(tok, tags, "real") for tok, tags in real_examples] + [
        (tok, tags, "synthetic") for tok, tags in synthetic_examples
    ]
    random.Random(seed).shuffle(all_examples)

    n = len(all_examples)
    n_test = max(1, int(n * 0.15))
    n_val = max(1, int(n * 0.15))

    test = all_examples[:n_test]
    val = all_examples[n_test : n_test + n_val]
    train = all_examples[n_test + n_val :]

    def _split(rows):
        return [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]

    train_tokens, train_tags, _ = _split(train)
    val_tokens, val_tags, _ = _split(val)
    test_tokens, test_tags, test_sources = _split(test)

    return {
        "train": (train_tokens, train_tags),
        "val": (val_tokens, val_tags),
        "test": (test_tokens, test_tags),
        "test_sources": test_sources,  # "real" or "synthetic" per test example — see docs/NER.md eval split
        "label_names": _build_label_list(),
        "counts": {"real": len(real_examples), "synthetic": len(synthetic_examples)},
    }


def _build_label_list() -> list[str]:
    labels = ["O"]
    for entity_type in ALL_ENTITY_TYPES:
        labels.append(f"B-{entity_type}")
        labels.append(f"I-{entity_type}")
    return labels
