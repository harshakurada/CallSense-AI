# CallSense AI — Project Plan

## 1. Business Problem

Customer support organizations generate large volumes of call recordings that are
almost never fully reviewed. Manual QA samples a small fraction of calls, so most
customer frustration, unresolved issues, and escalation risk surface only after
they've already become churn or complaints elsewhere.

## 2. Technical Problem

Turning unstructured call audio into structured, per-conversation and
cross-conversation insight requires chaining several genuinely hard ML/DL problems:
robust speech recognition on informal conversational audio, speaker attribution,
multi-task NLP classification (intent/sentiment/emotion/NER) on noisy transcripts,
and conversation-level modeling that reasons over a whole exchange rather than
isolated sentences.

## 3. Motivation

This is a B.Tech AIML major project, built to be defensible in ML Engineer / Deep
Learning Engineer / NLP Engineer / AI Engineer interviews. The core intelligence
must come from trained/fine-tuned ML and DL models with measured, non-fabricated
metrics — an LLM may only ever sit on top of that as an optional summarization
layer, never as the system's primary intelligence.

## 4. Objectives

1. Build a modular, independently-testable pipeline: audio → ASR → diarization →
   NLP → conversation-level prediction → analytics.
2. Use real Deep Learning (fine-tuned Transformers, not just prompted LLMs) for
   intent, sentiment, emotion, and NER.
3. Produce explainable, evidence-backed predictions for resolution status,
   escalation risk, and agent quality — not opaque scores.
4. Ship a working API + dashboard on top of the pipeline, with a real database.
5. Follow production ML engineering practice: config-driven, tested, logged,
   versioned, containerized.

## 5. Target Users

| User | Need |
|---|---|
| Support Team Lead / QA Manager | Find conversations that need review; measure agent quality |
| Operations / CX Analyst | Track trends in complaint categories, sentiment, resolution rate |
| Support Agent | Understand their own performance feedback |
| Engineer (interview reviewer) | A system that demonstrates genuine ML/DL/NLP/Speech competence |

## 6. Real-World Use Cases

- QA manager reviews only the calls flagged as high escalation-risk instead of a
  random sample.
- Ops analyst pulls "what are customers complaining about this week" from
  aggregated intent + entity trends.
- Agent performance is scored on transparent criteria (talk ratio, resolution
  rate, sentiment trajectory) instead of anecdote.
- Semantic search finds conversations about "delayed delivery" even when that
  exact phrase was never used.

## 7. Functional Requirements

- Accept a call recording (WAV/MP3) or a raw transcript.
- Transcribe to text with timestamps; diarize into CUSTOMER/AGENT turns.
- Classify intent, sentiment (per-utterance and conversation-level), and emotion.
- Extract named entities with PII masking.
- Predict resolution status and escalation risk with supporting reasons.
- Score agent quality against transparent, documented criteria.
- Summarize the conversation.
- Support semantic search across historical conversations.
- Expose everything through a REST API and a dashboard.

## 8. Non-Functional Requirements

- **Modularity** — every pipeline stage is independently trainable/testable
  before integration (`src/audio`, `src/asr`, `src/diarization`, `src/nlp`,
  `src/models`, `src/inference`).
- **Config-driven** — no hard-coded paths, model names, thresholds, or audio
  parameters (`configs/config.yaml` + `.env`).
- **Explainability** — risk/quality scores expose the structured signals that
  produced them.
- **Privacy** — PII detected and maskable before storage; development uses
  only public/synthetic data, never real customer data.
- **Reproducibility** — fixed seeds, versioned models, tracked experiments
  (MLflow).
- **Testability** — unit tests per module, integration tests across pipeline
  stages, API tests.

## 9. Constraints

- Local development on a single machine, CPU-only by default (ASR/NLP model
  choices in `configs/config.yaml` must have viable CPU inference paths, e.g.
  `faster-whisper` int8 on CPU).
- No real customer call data — only public datasets and synthetic data.
- Solo-developer project — architecture favors clarity and modularity over
  premature scaling infrastructure (e.g. no Kubernetes; Docker Compose is enough).

## 10. Limitations

- ASR accuracy on noisy/accented conversational audio will not match
  studio-quality benchmarks; Word Error Rate is measured and reported honestly,
  not assumed.
- Emotion/intent taxonomies are fixed sets chosen at design time; the system
  does not discover novel categories on its own.
- Escalation-risk and resolution predictions are trained on whatever labeled
  data is available/constructed in Module 2 — reported metrics reflect that
  data's coverage, not universal accuracy.

## 11. Development Roadmap

| Module | Scope |
|---|---|
| **1. Foundation** | Problem definition, architecture, tech stack, repo structure, config system, docs — *this module*. |
| **2. Data + Audio + ASR** | Dataset research, audio preprocessing pipeline, faster-whisper transcription, WER evaluation. |
| **3. Speaker Diarization** | pyannote.audio diarization, CUSTOMER/AGENT role mapping, timestamped transcript. |
| **4. Intent + Sentiment + Emotion** | Baselines (TF-IDF+LR) then fine-tuned Transformers for each; per-utterance and conversation-level aggregation. |
| **5. NER + Information Extraction** | spaCy baseline vs. Transformer NER; PII masking. |
| **6. Conversation-Level Deep Learning** | Encoder over utterance sequence; conversation-level representation and classification head. |
| **7. Resolution + Escalation + Agent Intelligence** | Resolution status, escalation risk with reasons, transparent agent quality scoring. |
| **8. Semantic Search + Analytics + Optional LLM Layer** | Sentence-transformer embeddings + vector search; cross-conversation analytics; optional LLM summarization layer clearly separated from the ML/DL core. |
| **9. Backend + Dashboard + Database** | Full FastAPI surface, Streamlit dashboard pages, PostgreSQL schema. |
| **10. Testing + MLOps + Docker + Deployment** | Full test suite, MLflow tracking, Docker Compose, monitoring. |

Each module is defined, implemented, and validated independently before the
next begins — Modules 2–10 are not started until explicitly requested.

## 12. Current Status

Module 1 complete: repository structure, configuration system, FastAPI +
Streamlit skeletons (health-check wired end-to-end), test scaffolding, and
this documentation. No modeling work has started — no metrics exist yet, and
none are claimed.
