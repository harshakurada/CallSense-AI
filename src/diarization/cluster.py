"""Clusters segment-level speaker embeddings into speaker labels.
Speaker count is a configured assumption (customer-service calls are
two-party by convention), not auto-estimated — see docs/DIARIZATION.md."""
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from configs.settings import get_diarization_config


def cluster_embeddings(embeddings: np.ndarray) -> list[str]:
    """Returns one 'SPEAKER_NN' label per row of `embeddings`, numbered by
    order of first appearance (SPEAKER_00 is whichever cluster's segment
    starts first) so labels are deterministic and stable run to run."""
    config = get_diarization_config()["clustering"]
    n_segments = embeddings.shape[0]

    if n_segments == 0:
        return []
    if n_segments == 1:
        return ["SPEAKER_00"]

    n_clusters = min(config["expected_speakers"], n_segments)
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric=config["metric"],
        linkage=config["linkage"],
    )
    raw_labels = clustering.fit_predict(embeddings)

    # Renumber clusters by first-appearance order instead of sklearn's
    # arbitrary cluster ids, so SPEAKER_00 is always whoever spoke first.
    first_seen_order = []
    for label in raw_labels:
        if label not in first_seen_order:
            first_seen_order.append(label)
    relabel_map = {old: new for new, old in enumerate(first_seen_order)}

    return [f"SPEAKER_{relabel_map[label]:02d}" for label in raw_labels]
