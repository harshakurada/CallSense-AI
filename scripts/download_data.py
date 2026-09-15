"""Dataset download. See docs/DATASETS.md for what each dataset is, its
license, and why it was chosen.

Usage:
    python scripts/download_data.py --dataset librispeech-dummy
    python scripts/download_data.py --dataset librispeech --subset dev-clean
    python scripts/download_data.py --dataset ami
    python scripts/download_data.py --dataset common-voice
"""
import argparse
import hashlib
import io
import sys
import tarfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging import get_logger

logger = get_logger(__name__)

OPENSLR_BASE = "https://www.openslr.org/resources/12"
OPENSLR_SUBSETS = {"dev-clean", "dev-other", "test-clean", "test-other"}


def _download_with_progress(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    """(connect_timeout, read_timeout) — found hanging indefinitely on a
    stalled connection during development with only a single connect
    timeout; a read timeout ensures a stall errors out instead."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=(30, 60)) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100 * downloaded / total
                    print(f"\r  {dest.name}: {downloaded / 1e6:.1f}/{total / 1e6:.1f} MB ({pct:.0f}%)", end="")
        print()


def _md5(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_librispeech(subset: str, output_dir: Path) -> None:
    """Downloads one official LibriSpeech subset from OpenSLR and verifies
    it against OpenSLR's own published md5sum.txt (fetched live, not
    hardcoded, so it stays correct if OpenSLR ever updates an archive)."""
    if subset not in OPENSLR_SUBSETS:
        raise ValueError(f"Unknown LibriSpeech subset '{subset}'. Choose from: {OPENSLR_SUBSETS}")

    archive_name = f"{subset}.tar.gz"
    archive_url = f"{OPENSLR_BASE}/{archive_name}"
    dest_dir = output_dir / "librispeech"
    archive_path = dest_dir / archive_name

    logger.info("Fetching official checksums from %s/md5sum.txt", OPENSLR_BASE)
    checksums_text = requests.get(f"{OPENSLR_BASE}/md5sum.txt", timeout=30).text
    expected_md5 = None
    for line in checksums_text.splitlines():
        if line.strip().endswith(archive_name):
            expected_md5 = line.split()[0].strip()
    if not expected_md5:
        raise RuntimeError(f"Could not find checksum for {archive_name} in OpenSLR's md5sum.txt")

    if archive_path.exists() and _md5(archive_path) == expected_md5:
        logger.info("%s already downloaded and verified, skipping", archive_name)
    else:
        logger.info("Downloading %s from %s", archive_name, archive_url)
        _download_with_progress(archive_url, archive_path)
        actual_md5 = _md5(archive_path)
        if actual_md5 != expected_md5:
            archive_path.unlink()
            raise RuntimeError(
                f"Checksum mismatch for {archive_name}: expected {expected_md5}, got {actual_md5}"
            )
        logger.info("Checksum verified for %s", archive_name)

    logger.info("Extracting %s", archive_path)
    with tarfile.open(archive_path) as tar:
        tar.extractall(dest_dir)
    logger.info("LibriSpeech %s ready at %s", subset, dest_dir / "LibriSpeech" / subset)


def download_librispeech_dummy(output_dir: Path) -> None:
    """Downloads hf-internal-testing/librispeech_asr_dummy (73 samples,
    ~9MB) — real LibriSpeech audio+transcript pairs, small enough for fast
    pipeline testing/CI. See docs/DATASETS.md for why this is used instead
    of the full corpus during development."""
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    dest_dir = output_dir / "librispeech_dummy"
    dest_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading hf-internal-testing/librispeech_asr_dummy parquet file")
    parquet_path = hf_hub_download(
        repo_id="hf-internal-testing/librispeech_asr_dummy",
        repo_type="dataset",
        filename="clean/validation-00000-of-00001.parquet",
    )
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()

    manifest = []
    for row in rows:
        sample_id = row["id"]
        audio_bytes = row["audio"]["bytes"]
        text = row["text"]

        audio_path = dest_dir / f"{sample_id}.flac"
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
        # sanity-decode so a corrupt write is caught at download time, not later
        info = sf.info(str(audio_path))

        manifest.append(
            {
                "call_id": sample_id,
                "audio_path": str(audio_path.relative_to(output_dir.parent)),
                "reference_transcript": text,
                "duration_seconds": info.duration,
                "sample_rate": info.samplerate,
            }
        )

    manifest_path = dest_dir / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        import json

        for entry in manifest:
            f.write(json.dumps(entry) + "\n")

    logger.info("%d samples written to %s (manifest: %s)", len(manifest), dest_dir, manifest_path)


def download_ami(output_dir: Path) -> None:
    raise NotImplementedError(
        "AMI Meeting Corpus is not bulk-downloadable from a single fixed URL — "
        "meetings are selected individually from https://groups.inf.ed.ac.uk/ami/download/ "
        "which builds a signed manifest per selection. See docs/DATASETS.md for the "
        "manual steps; this is intentionally not scripted to avoid guessing at URL "
        "patterns that may silently break."
    )


def download_common_voice(output_dir: Path) -> None:
    raise NotImplementedError(
        "Common Voice requires agreeing to terms on commonvoice.mozilla.org/en/datasets "
        "to get a time-limited signed download link — there is no unauthenticated bulk "
        "endpoint to script against. See docs/DATASETS.md for the manual download steps."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["librispeech", "librispeech-dummy", "ami", "common-voice"],
    )
    parser.add_argument("--subset", default="dev-clean", help="LibriSpeech subset name")
    parser.add_argument("--output-dir", default="data/raw", type=Path)
    args = parser.parse_args()

    if args.dataset == "librispeech":
        download_librispeech(args.subset, args.output_dir)
    elif args.dataset == "librispeech-dummy":
        download_librispeech_dummy(args.output_dir)
    elif args.dataset == "ami":
        download_ami(args.output_dir)
    elif args.dataset == "common-voice":
        download_common_voice(args.output_dir)


if __name__ == "__main__":
    main()
