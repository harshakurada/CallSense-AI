# CallSense AI

**Intelligent Customer Conversation Analytics & Quality Intelligence Platform**

CallSense AI turns customer-service call recordings into structured, actionable
insight — intent, sentiment, emotion, resolution status, escalation risk, entity
extraction, agent quality scoring, and semantic search — built on real Deep
Learning and NLP models, not an LLM wrapper.

## Features (target — see Current Status)

- Audio → text via faster-whisper, with speaker diarization (CUSTOMER/AGENT)
- Intent, sentiment, and emotion classification (fine-tuned Transformers)
- Named entity recognition with PII masking
- Conversation-level resolution and escalation-risk prediction, with reasons
- Transparent agent quality scoring
- Semantic search across historical conversations
- REST API (FastAPI) + analytics dashboard (Streamlit)

## Architecture

Full detail in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

```text
Audio → Preprocessing → ASR → Diarization → Transcript
      → Intent / Sentiment / Emotion / NER
      → Conversation-Level Understanding
      → Resolution / Escalation / Agent Quality
      → Analytics → Dashboard
```

## Technology Stack

| Layer | Choice |
|---|---|
| Language / DL | Python, PyTorch |
| NLP | Hugging Face Transformers |
| ASR | faster-whisper |
| Diarization | pyannote.audio |
| Audio | librosa |
| Search | sentence-transformers |
| API | FastAPI |
| Dashboard | Streamlit |
| Database | PostgreSQL |
| Experiment tracking | MLflow |
| Containerization | Docker |

Rationale for each choice: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#5-technology-stack).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # then edit values as needed
```

### Run the API

```bash
uvicorn api.main:app --reload
```

### Run the dashboard

```bash
streamlit run app/app.py
```

### Run tests

```bash
pytest
```

### Run the audio + ASR pipeline (Module 2)

```bash
python scripts/download_data.py --dataset librispeech-dummy
python scripts/run_asr_pipeline.py --input data/raw/librispeech_dummy/1272-128104-0000.flac
python scripts/transcribe_dataset.py --input data/raw/librispeech_dummy
python scripts/validate_dataset.py --input data/raw/librispeech_dummy
python scripts/evaluate_asr.py --manifest data/raw/librispeech_dummy/manifest.jsonl --transcripts data/processed/transcripts
python scripts/eda_audio.py --input data/raw/librispeech_dummy --output-dir reports/eda
```

## Project Structure

```text
CallSense AI/
├── data/               # raw / interim / processed (git-ignored, folders kept)
├── src/
│   ├── audio/          # preprocessing, VAD, chunking (Module 2)
│   ├── asr/            # speech-to-text + WER evaluation (Module 2)
│   ├── diarization/    # speaker diarization (Module 3)
│   ├── nlp/            # intent / sentiment / emotion / NER (Modules 4-5)
│   ├── models/         # conversation-level architectures (Modules 6-7)
│   ├── inference/       # pipeline orchestration
│   └── utils/           # logging, exceptions
├── scripts/            # download_data, validate_dataset, transcribe_dataset,
│                       # evaluate_asr, eda_audio, run_asr_pipeline
├── tests/              # unit + API + audio + ASR tests
├── configs/            # config.yaml, audio.yaml (non-secret) + settings.py (env-based)
├── models/             # saved model artifacts (git-ignored)
├── notebooks/          # exploratory notebooks per module
├── docs/               # ARCHITECTURE.md, PROJECT_PLAN.md, DATASETS.md, ASR_EVALUATION.md
├── app/                # Streamlit dashboard
├── api/                # FastAPI backend
├── requirements.txt
├── .env.example
└── .gitignore
```

## Current Status

**Modules 1–2 complete.**

- **Module 1 (Foundation)**: repository structure, centralized configuration,
  a working FastAPI backend and Streamlit dashboard wired via a health check,
  logging/exception scaffolding, tests.
- **Module 2 (Data + Audio + ASR)**: dataset research and download scripts
  (`docs/DATASETS.md`), a real audio preprocessing pipeline (validation,
  resampling, normalization, VAD, silence-aware chunking), faster-whisper
  transcription returning the exact `{call_id, language, duration, segments}`
  schema, batch transcription, WER evaluation, and an EDA script. Measured on
  73 real LibriSpeech clips: **8.70% corpus WER** — full breakdown and error
  analysis in `docs/ASR_EVALUATION.md`. No customer-service-domain metrics
  exist yet or are claimed; that data doesn't exist until later modules.

Full requirements and constraints: [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md).

## Roadmap

| Module | Scope | Status |
|---|---|---|
| 1 | Foundation | ✅ Done |
| 2 | Data + Audio + ASR | ✅ Done |
| 3 | Speaker Diarization | Not started |
| 4 | Intent + Sentiment + Emotion | Not started |
| 5 | NER + Information Extraction | Not started |
| 6 | Conversation-Level Deep Learning | Not started |
| 7 | Resolution + Escalation + Agent Intelligence | Not started |
| 8 | Semantic Search + Analytics + Optional LLM Layer | Not started |
| 9 | Backend + Dashboard + Database | Not started |
| 10 | Testing + MLOps + Docker + Deployment | Not started |

Full module-by-module scope: [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md#11-development-roadmap).
