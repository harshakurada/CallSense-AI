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

## Project Structure

```text
CallSense AI/
├── data/               # raw / interim / processed (git-ignored, folders kept)
├── src/
│   ├── audio/          # preprocessing (Module 2)
│   ├── asr/            # speech-to-text (Module 2)
│   ├── diarization/    # speaker diarization (Module 3)
│   ├── nlp/            # intent / sentiment / emotion / NER (Modules 4-5)
│   ├── models/         # conversation-level architectures (Modules 6-7)
│   ├── inference/       # pipeline orchestration
│   └── utils/           # logging, exceptions
├── scripts/            # one-off / CLI scripts
├── tests/              # unit + API tests
├── configs/            # config.yaml (non-secret) + settings.py (env-based)
├── models/             # saved model artifacts (git-ignored)
├── notebooks/          # exploratory notebooks per module
├── docs/               # ARCHITECTURE.md, PROJECT_PLAN.md
├── app/                # Streamlit dashboard
├── api/                # FastAPI backend
├── requirements.txt
├── .env.example
└── .gitignore
```

## Current Status

**Module 1 (Foundation) complete.** Repository structure, centralized
configuration (`configs/config.yaml` + `.env`), a working FastAPI backend and
Streamlit dashboard wired to each other via a health check, logging and
exception scaffolding, and test setup are in place. No modeling work has
started yet — no metrics exist, and none are claimed.

Full requirements and constraints: [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md).

## Roadmap

| Module | Scope | Status |
|---|---|---|
| 1 | Foundation | ✅ Done |
| 2 | Data + Audio + ASR | Not started |
| 3 | Speaker Diarization | Not started |
| 4 | Intent + Sentiment + Emotion | Not started |
| 5 | NER + Information Extraction | Not started |
| 6 | Conversation-Level Deep Learning | Not started |
| 7 | Resolution + Escalation + Agent Intelligence | Not started |
| 8 | Semantic Search + Analytics + Optional LLM Layer | Not started |
| 9 | Backend + Dashboard + Database | Not started |
| 10 | Testing + MLOps + Docker + Deployment | Not started |

Full module-by-module scope: [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md#11-development-roadmap).
