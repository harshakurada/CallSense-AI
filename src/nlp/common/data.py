"""Dataset-shape utilities shared by intent/sentiment/emotion — stratified
subsampling (so CPU-only fine-tuning stays practical, per Module 1's
hardware constraints) and splitting, without disturbing real class
imbalance, which each task's error analysis specifically examines."""
import random
from collections import defaultdict


def stratified_subsample(
    texts: list[str], labels: list[str], max_total: int, seed: int = 42
) -> tuple[list[str], list[str]]:
    """Samples up to max_total examples, keeping each class's original
    proportion of the full set — so imbalance is preserved, not erased,
    while keeping CPU fine-tuning time practical."""
    rng = random.Random(seed)
    by_label: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        by_label[label].append(i)

    total = len(texts)
    keep_indices = []
    for label, indices in by_label.items():
        n_keep = max(1, round(len(indices) / total * max_total))
        rng.shuffle(indices)
        keep_indices.extend(indices[:n_keep])

    rng.shuffle(keep_indices)
    return [texts[i] for i in keep_indices], [labels[i] for i in keep_indices]


def stratified_split(
    texts: list[str], labels: list[str], val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42
) -> dict:
    """Returns {"train": (texts, labels), "val": (...), "test": (...)},
    splitting within each class so every split reflects the same class
    distribution as the whole set."""
    rng = random.Random(seed)
    by_label: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        by_label[label].append(i)

    train_idx, val_idx, test_idx = [], [], []
    for label, indices in by_label.items():
        rng.shuffle(indices)
        n = len(indices)
        n_test = max(1, int(n * test_frac)) if n >= 3 else 0
        n_val = max(1, int(n * val_frac)) if n >= 3 else 0
        test_idx.extend(indices[:n_test])
        val_idx.extend(indices[n_test : n_test + n_val])
        train_idx.extend(indices[n_test + n_val :])

    def _subset(indices):
        rng.shuffle(indices)
        return [texts[i] for i in indices], [labels[i] for i in indices]

    return {"train": _subset(train_idx), "val": _subset(val_idx), "test": _subset(test_idx)}


def class_distribution(labels: list[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for label in labels:
        counts[label] += 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
