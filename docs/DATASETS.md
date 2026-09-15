# CallSense AI — Datasets (Module 2)

## Why there is no single "customer service call" audio dataset

The realistic customer-service call corpora that exist (Switchboard, Fisher
English, CallHome/CallFriend) are LDC-licensed and require a paid LDC
membership or institutional subscription — they are not freely downloadable,
and this project does not have that access. No adequately-licensed public
dataset of real customer-service call audio was identified; genuine customer
call recordings are also proprietary/privacy-restricted by nature, which is
exactly why Module 1's constraints rule out using real customer data anyway.

The resulting strategy is a **multi-dataset approach**: use freely-licensed,
verifiable public speech corpora to build and validate the technical
pipeline (ASR, VAD, chunking, diarization in Module 3) correctly, while
domain-specific customer-service *language* (intents, entities, escalation
patterns) is addressed later, in the NLP modules, using text data and/or
synthetic conversations authored for that purpose — never fabricated
statistics about audio corpora that don't exist.

## Datasets used

### 1. LibriSpeech (technical development & full-scale ASR evaluation)

| | |
|---|---|
| Source | OpenSLR (SLR12), derived from LibriVox audiobooks / Project Gutenberg texts |
| URL | https://www.openslr.org/12/ |
| License | CC BY 4.0 |
| Size | `dev-clean.tar.gz` 322M, `dev-other.tar.gz` 300M, `test-clean.tar.gz` 331M, `test-other.tar.gz` 314M (larger `train-*` subsets also available, up to 30G) |
| Total corpus duration | ~1000 hours of 16kHz read English speech |
| Audio format | WAV, 16kHz mono |
| Language | English |
| Labels | Verified ground-truth transcripts per utterance |
| Intended use | Real, verifiable WER evaluation; validating the audio preprocessing and ASR code paths against clean, well-understood speech before pointing them at anything domain-specific |
| Limitations | Read audiobook speech, not spontaneous conversation — no speaker turns, no customer-service register, no overlapping speech. Not representative of the target domain; used for technical validation only. |

Downloaded via `scripts/download_data.py --dataset librispeech --subset dev-clean`,
which verifies the archive against OpenSLR's own published `md5sum.txt`
(fetched live at download time, not hardcoded).

### 2. LibriSpeech ASR Dummy (used for this module's actual test runs)

| | |
|---|---|
| Source | `hf-internal-testing/librispeech_asr_dummy` on Hugging Face Hub |
| URL | https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy |
| License | Not explicitly declared on the dataset card; derived from LibriSpeech (CC BY 4.0) |
| Size | 73 samples, ~9.2MB, FLAC audio + text, single validation split |
| Audio format | FLAC, 16kHz mono, 1.6–29.4s per clip |
| Language | English |
| Labels | Ground-truth transcript per clip |
| Intended use | Small, fast, real (not synthetic) audio+transcript pairs for developing and testing this module's code without downloading the full multi-hundred-MB corpus. All of Module 2's actual pipeline runs and the measured WER in `docs/ASR_EVALUATION.md` used this set. |
| Limitations | Same domain limitations as LibriSpeech, plus its small size means the measured WER is indicative, not a rigorous benchmark. |

Downloaded via `scripts/download_data.py --dataset librispeech-dummy`, which
pulls the dataset's parquet file via `huggingface_hub`, decodes each row's
embedded audio bytes to `.flac`, and writes a `manifest.jsonl` with
`{call_id, audio_path, reference_transcript, duration_seconds, sample_rate}`.

### 3. AMI Meeting Corpus (documented for Module 3 — not yet downloaded)

| | |
|---|---|
| Source | University of Edinburgh |
| URL | https://groups.inf.ed.ac.uk/ami/corpus/ |
| License | CC BY 4.0 (signals, transcripts, and some annotations) |
| Size | 100 hours of meeting recordings |
| Audio format | Close-talking and far-field microphone recordings (multi-channel) |
| Language | English |
| Labels | Transcripts, speaker segmentation |
| Intended use | The closest freely-available proxy to real spontaneous, multi-speaker, overlapping conversation — earmarked for Module 3 (speaker diarization), where its speaker-segmented multi-party audio is directly useful. |
| Limitations | Meetings, not phone calls — different acoustic and turn-taking characteristics than customer-service calls. |

**Manual step required:** AMI is not a single fixed download URL — specific
meetings are selected from https://groups.inf.ed.ac.uk/ami/download/, which
builds a signed download manifest per selection. `scripts/download_data.py
--dataset ami` deliberately raises `NotImplementedError` with these
instructions rather than guessing at URL patterns that could silently break.

### 4. Mozilla Common Voice (documented — not yet downloaded)

| | |
|---|---|
| Source | Mozilla |
| URL | https://commonvoice.mozilla.org/en/datasets |
| License | CC0 (public domain) |
| Size | Initial English release (Nov 2017): ~500 hours from 20,000+ contributors; current releases are substantially larger — check the site at download time rather than trusting a fixed figure here |
| Audio format | Short read-sentence clips |
| Language | Many languages, including English |
| Labels | Transcript per clip; some demographic metadata (optional, self-reported) |
| Intended use | Speaker/accent diversity for ASR robustness testing, once the core pipeline is validated. |
| Limitations | Single-speaker read clips, not conversational — same domain gap as LibriSpeech. |

**Manual step required:** Common Voice requires agreeing to terms on the
website to receive a time-limited signed download link — there is no
unauthenticated bulk endpoint to script against. `scripts/download_data.py
--dataset common-voice` raises `NotImplementedError` pointing here rather
than attempting to bypass that.

## Summary table

| Dataset | License | Downloadable now | Used in this module's actual runs |
|---|---|---|---|
| LibriSpeech (dev-clean etc.) | CC BY 4.0 | Yes, scripted | Available, not run (large; dummy set used instead) |
| LibriSpeech ASR Dummy | Derived from CC BY 4.0 | Yes, scripted | **Yes — all Module 2 results are on this set** |
| AMI Meeting Corpus | CC BY 4.0 | Manual (documented) | No — earmarked for Module 3 |
| Common Voice | CC0 | Manual (documented) | No — earmarked for later robustness testing |
