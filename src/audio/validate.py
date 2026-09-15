"""Single-file audio validation. Dataset-wide checks (missing/duplicate
files across a directory) live in scripts/validate_dataset.py, which calls
into this module per-file."""
from dataclasses import dataclass, field
from pathlib import Path

from configs.settings import get_audio_config
from src.audio.io import AudioInfo, get_audio_info
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    path: str
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: AudioInfo | None = None


def validate_audio_file(path: str | Path) -> ValidationResult:
    path = Path(path)
    config = get_audio_config()["io"]
    result = ValidationResult(path=str(path), is_valid=True)

    if not path.exists():
        result.is_valid = False
        result.errors.append("File does not exist")
        return result

    if path.stat().st_size == 0:
        result.is_valid = False
        result.errors.append("File is empty (0 bytes)")
        return result

    extension = path.suffix.lower().lstrip(".")
    if extension not in config["supported_formats"]:
        result.is_valid = False
        result.errors.append(
            f"Unsupported format '.{extension}' — supported: {config['supported_formats']}"
        )
        return result

    try:
        info = get_audio_info(path)
    except Exception as exc:
        result.is_valid = False
        result.errors.append(f"Corrupted or unreadable audio: {exc}")
        return result

    result.info = info

    if info.frames == 0 or info.duration_seconds <= 0:
        result.is_valid = False
        result.errors.append("Audio contains zero frames")
        return result

    if info.duration_seconds < config["min_duration_seconds"]:
        result.is_valid = False
        result.errors.append(
            f"Duration {info.duration_seconds:.3f}s is below minimum "
            f"{config['min_duration_seconds']}s"
        )

    if info.duration_seconds > config["max_duration_seconds"]:
        result.is_valid = False
        result.errors.append(
            f"Duration {info.duration_seconds:.1f}s exceeds maximum "
            f"{config['max_duration_seconds']}s"
        )

    if info.sample_rate != config["target_sample_rate"]:
        result.warnings.append(
            f"Sample rate {info.sample_rate}Hz differs from target "
            f"{config['target_sample_rate']}Hz — will be resampled"
        )

    if info.channels != config["target_channels"]:
        result.warnings.append(
            f"{info.channels} channel(s) found — will be downmixed to "
            f"{config['target_channels']}"
        )

    return result
