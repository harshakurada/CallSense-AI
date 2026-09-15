# CallSense AI

**Intelligent Customer Conversation Analytics & Quality Intelligence Platform**

CallSense AI turns customer-service call recordings into structured, actionable
insight: intent, sentiment, emotion, resolution status, escalation risk, entity
extraction, agent quality scoring, summarization, and semantic search — built as a
genuine Deep Learning / NLP system, not an LLM wrapper.

Full problem statement and architecture: [`docs/module_01_problem_definition_and_architecture.md`](docs/module_01_problem_definition_and_architecture.md).

## Tech Stack

- **Backend**: Python, FastAPI
- **Frontend**: React
- **ML/NLP**: Transformers (BERT/RoBERTa family), Whisper (ASR), spaCy, sentence-transformers
- **Database**: PostgreSQL (+ vector store for semantic search)
- **Infra**: Docker / Docker Compose, MLflow (experiment tracking)

## Repository Structure

```text
CallSense AI/
├── backend/     # FastAPI app + inference pipeline
├── frontend/    # React dashboard
├── ml/          # Training code, notebooks, experiments
├── models/      # Saved model artifacts (git-ignored)
├── tests/       # Unit / integration / API / model tests
└── docs/        # Module-by-module documentation
```

## Build Approach

This project is built **one module at a time** — each module is defined, implemented,
and validated independently before the next begins. Progress:

- [x] Module 1 — Problem Definition & System Architecture
- [ ] Module 2 — Dataset Research & Data Collection
- [ ] Module 3 — Audio Preprocessing
- [ ] Module 4 — Speech-to-Text (Whisper)
- [ ] Module 5 — Speaker Diarization
- [ ] Module 6 — Transcript Preprocessing
- [ ] Module 7 — Intent Classification
- [ ] Module 8 — Sentiment Analysis
- [ ] Module 9 — Emotion Detection
- [ ] Module 10 — Named Entity Recognition
- [ ] Module 11 — Conversation-Level Understanding
- [ ] Module 12 — Resolution Prediction
- [ ] Module 13 — Escalation Risk Prediction
- [ ] Module 14 — Agent Quality Analysis
- [ ] Module 15 — Conversation Summarization
- [ ] Module 16 — Conversation Search
- [ ] Module 17 — Analytics Engine
- [ ] Module 18 — Dashboard
- [ ] Module 19 — Backend API
- [ ] Module 20 — Database
- [ ] Module 21 — Model Serving
- [ ] Module 22 — Testing
- [ ] Module 23 — MLOps
- [ ] Module 24 — Docker
- [ ] Module 25 — Monitoring
- [ ] Module 26 — Security & Privacy
- [ ] Module 27 — Model Comparison & Experimentation
- [ ] Module 28 — Error Analysis
- [ ] Module 29 — Final System Integration
- [ ] Module 30 — Final Documentation
- [ ] Module 31 — Resume & Interview Preparation

## Getting Started

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```
