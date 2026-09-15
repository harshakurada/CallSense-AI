"""Speech-to-text via faster-whisper. transcribe() is the pipeline's ASR
entrypoint and the only place a Whisper model is loaded/called from."""
from functools import lru_cache
from pathlib import Path

from faster_whisper import WhisperModel

from configs.settings import get_config
from src.audio.pipeline import preprocess_for_asr
from src.utils.exceptions import ASRError
from src.utils.logging import get_logger

logger = get_logger(__name__)


@lru_cache
def _load_model() -> WhisperModel:
    """Cached so repeated transcribe() calls (batch processing) reuse one
    loaded model instead of reloading it per file."""
    config = get_config()["asr"]
    device = config["device"]  # "auto" lets ctranslate2 pick cuda if available, else cpu
    compute_type = config["compute_type"]
    try:
        model = WhisperModel(config["model_size"], device=device, compute_type=compute_type)
    except ValueError as exc:
        # e.g. int8 requested but unsupported on the resolved device
        logger.warning(
            "compute_type=%s unsupported on this device (%s); falling back to 'default'",
            compute_type,
            exc,
        )
        model = WhisperModel(config["model_size"], device=device, compute_type="default")
    logger.info(
        "Loaded faster-whisper '%s' (device=%s, compute_type=%s)",
        config["model_size"],
        device,
        compute_type,
    )
    return model


def transcribe(audio_path: str | Path, call_id: str | None = None) -> dict:
    """Preprocesses audio_path and transcribes it end to end.

    Returns exactly:
    {
      "call_id": str,
      "language": str,
      "duration": float,
      "segments": [{"start": float, "end": float, "text": str}, ...]
    }

    No confidence value is included — faster-whisper's segment-level
    avg_logprob is a log-probability, not a calibrated confidence, and
    reporting it as "confidence" would misrepresent what it measures.
    """
    audio_path = Path(audio_path)
    call_id = call_id or audio_path.stem
    config = get_config()["asr"]

    preprocessed = preprocess_for_asr(audio_path)

    def _empty_result(language: str | None) -> dict:
        return {
            "call_id": call_id,
            "language": language,
            "duration": round(preprocessed.duration_seconds, 3),
            "segments": [],
        }

    if not preprocessed.speech_regions:
        # Our own VAD already found nothing to transcribe (dead air, hold
        # music, silence) — skip the model call entirely rather than pay
        # for inference on audio known to be empty.
        logger.warning("%s: no speech detected — returning empty transcript", call_id)
        return _empty_result(language=None)

    model = _load_model()
    configured_language = config.get("language") or None

    try:
        # Pass the already mono/resampled/normalized array, not the raw file
        # path — faster-whisper would otherwise re-decode the original file
        # itself and our preprocessing stage would have no effect.
        segments_iter, info = model.transcribe(
            preprocessed.samples,
            language=configured_language,
            vad_filter=True,  # faster-whisper's own bundled Silero VAD, at decode time
        )
        segments = [
            {"start": round(seg.start, 3), "end": round(seg.end, 3), "text": seg.text.strip()}
            for seg in segments_iter
        ]
    except ValueError as exc:
        # faster-whisper raises a bare ValueError ("max() iterable argument
        # is empty") when its own internal VAD filters out every frame and
        # language is left to auto-detect — i.e. our energy-based VAD and
        # its neural VAD disagree (e.g. a loud non-speech tone). Treated the
        # same as "no speech found", not as a pipeline failure.
        if "max()" in str(exc):
            logger.warning(
                "%s: faster-whisper's VAD filtered out all audio — returning empty transcript",
                call_id,
            )
            return _empty_result(language=configured_language)
        raise ASRError(f"Transcription failed for {audio_path}: {exc}") from exc
    except Exception as exc:
        raise ASRError(f"Transcription failed for {audio_path}: {exc}") from exc

    result = {
        "call_id": call_id,
        "language": info.language,
        "duration": round(preprocessed.duration_seconds, 3),
        "segments": segments,
    }
    logger.info(
        "%s: transcribed %d segment(s), language=%s (p=%.2f)",
        call_id,
        len(segments),
        info.language,
        info.language_probability,
    )
    return result
