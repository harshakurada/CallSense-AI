"""Speaker embeddings via speechbrain's ECAPA-TDNN (spkrec-ecapa-voxceleb) —
ungated, no Hugging Face auth required. See docs/DIARIZATION.md for why
this replaces pyannote.audio's pretrained pipeline, which is gated."""
from functools import lru_cache

import numpy as np
import torch

from configs.settings import get_diarization_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


@lru_cache
def _load_model():
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    config = get_diarization_config()["embedding"]
    logger.info("Loading speaker embedding model '%s'", config["model_name"])
    return EncoderClassifier.from_hparams(
        source=config["model_name"],
        savedir="models/spkrec-ecapa-voxceleb",
        run_opts={"device": config["device"]},
        # Windows without Developer Mode/admin can't create symlinks, which
        # speechbrain's default caching strategy relies on.
        local_strategy=LocalStrategy.COPY,
    )


def embed_segment(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Returns a single fixed-length embedding vector for one audio segment.
    Callers should pass audio already resampled to what the embedding model
    expects (16kHz, mono) — see src/audio/preprocess.py."""
    if sample_rate != 16000:
        raise ValueError(f"Embedding model expects 16kHz audio, got {sample_rate}Hz")

    model = _load_model()
    waveform = torch.from_numpy(samples).float().unsqueeze(0)  # (1, n_samples)
    with torch.no_grad():
        embedding = model.encode_batch(waveform)
    return embedding.squeeze().numpy()


def embed_segments(segments: list[np.ndarray], sample_rate: int) -> np.ndarray:
    """Batched convenience wrapper. Returns an (n_segments, embedding_dim) array."""
    return np.stack([embed_segment(seg, sample_rate) for seg in segments])
