"""Audio EDA: waveform, spectrogram, mel spectrogram for one file, and a
duration-distribution histogram across a directory.

Usage:
    python scripts/eda_audio.py --input data/raw/librispeech_dummy/1272-128104-0000.flac
    python scripts/eda_audio.py --input data/raw/librispeech_dummy --output-dir reports/eda
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — this script never opens a display window
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.settings import get_audio_config
from src.audio.io import get_audio_info, load_audio
from src.utils.logging import get_logger

logger = get_logger(__name__)


def plot_single_file(path: Path, output_dir: Path) -> None:
    samples, sr = load_audio(path)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem

    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(samples, sr=sr, ax=ax)
    ax.set(title=f"Waveform — {path.name}", xlabel="Time (s)", ylabel="Amplitude")
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}_waveform.png", dpi=120)
    plt.close(fig)

    stft = librosa.stft(samples)
    db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(db, sr=sr, x_axis="time", y_axis="log", ax=ax)
    ax.set(title=f"Spectrogram — {path.name}")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}_spectrogram.png", dpi=120)
    plt.close(fig)

    mel = librosa.feature.melspectrogram(y=samples, sr=sr, n_mels=80)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    ax.set(title=f"Mel Spectrogram — {path.name}")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}_melspectrogram.png", dpi=120)
    plt.close(fig)

    logger.info("Wrote waveform/spectrogram/mel plots for %s to %s", path.name, output_dir)


def plot_duration_distribution(input_dir: Path, output_dir: Path) -> None:
    supported = set(get_audio_config()["io"]["supported_formats"])
    audio_files = [
        p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower().lstrip(".") in supported
    ]
    if not audio_files:
        logger.warning("No audio files found under %s for duration distribution", input_dir)
        return

    durations = [get_audio_info(p).duration_seconds for p in audio_files]

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(durations, bins=min(20, len(durations)))
    ax.set(
        title=f"Duration distribution — {input_dir.name} (n={len(durations)})",
        xlabel="Duration (s)",
        ylabel="Count",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "duration_distribution.png", dpi=120)
    plt.close(fig)

    logger.info(
        "Duration stats: min=%.2fs max=%.2fs mean=%.2fs total=%.1fs (n=%d)",
        min(durations),
        max(durations),
        sum(durations) / len(durations),
        sum(durations),
        len(durations),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/eda"))
    args = parser.parse_args()

    if args.input.is_dir():
        plot_duration_distribution(args.input, args.output_dir)
        first_file = next(
            (
                p
                for p in sorted(args.input.rglob("*"))
                if p.suffix.lower().lstrip(".") in get_audio_config()["io"]["supported_formats"]
            ),
            None,
        )
        if first_file:
            plot_single_file(first_file, args.output_dir)
    else:
        plot_single_file(args.input, args.output_dir)


if __name__ == "__main__":
    main()
