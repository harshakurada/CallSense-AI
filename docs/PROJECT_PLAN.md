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
next begins — Modules 6–10 are not started until explicitly requested.

## 12. Current Status

**Module 1** complete: repository structure, configuration system, FastAPI +
Streamlit skeletons (health-check wired end-to-end), test scaffolding, and
this documentation.

**Module 2** complete: dataset strategy documented (`docs/DATASETS.md`) —
LibriSpeech + a 73-sample real subset for development, AMI and Common Voice
documented with their manual-auth requirements rather than scripted around;
a real audio pipeline (`src/audio/`: validation, resampling, normalization,
energy-based VAD, silence-aware chunking) driven by `configs/audio.yaml`;
faster-whisper transcription (`src/asr/transcribe.py`) returning the exact
required JSON schema; batch transcription and dataset validation scripts
with per-file failure isolation; WER evaluation (`src/asr/evaluation.py`)
measured at **8.70% corpus WER on 73 real clips** with a full error
analysis in `docs/ASR_EVALUATION.md`, including a real bug found and fixed
during testing (documented there). 36 tests passing.

**Module 3** complete: pyannote.audio's pretrained diarization pipeline is
gated (verified against the HF Hub API — no token/accepted terms available
here), so `src/diarization/` implements an ungated alternative instead:
Module 2's VAD reused for segmentation, speechbrain ECAPA-TDNN speaker
embeddings, scikit-learn agglomerative clustering, overlap-based alignment
with Whisper's transcript, and an explicitly opt-in (not default)
CUSTOMER/AGENT role heuristic. `src/inference/pipeline.py` integrates
Modules 2–3 into one call producing the final speaker-attributed
conversation format. Measured DER on a synthetically-labeled but real
two-speaker test call: **11.68%, zero speaker confusion** — full
methodology, the gating decision, and the overlapping-speech limitation are
in `docs/DIARIZATION.md`. 56 tests passing (19 new for diarization).

**Module 4** complete: TF-IDF+LogisticRegression baseline and a fine-tuned
Transformer for each of intent (`src/nlp/intent/`), sentiment
(`src/nlp/sentiment/`), and emotion (`src/nlp/emotion/`), sharing common
infrastructure (`src/nlp/common/`: stratified splitting, metrics,
weighted-loss fine-tuning). Real datasets throughout — Bitext customer
support (11 intent classes), Twitter US Airline sentiment (3 classes),
dair-ai emotion (6 classes) — documented with verified source/license/size
in `docs/NLP_MODELS.md`, including why each differs from the spec's
suggested label lists. Measured: intent 99.66%/0.9972 macro F1, sentiment
82.80%/0.7748, emotion 86.38%/0.8277 (bert-tiny, after DistilBERT was
killed twice by the OS for low memory on this 8GB machine — the full
incident, including a learning-rate mistake found and fixed along the way,
is in `docs/NLP_MODELS.md`). Sentiment aggregation and an emotion timeline
(`src/nlp/emotion/timeline.py`) are built; `src/nlp/inference.py`
integrates all three into one call. 71 tests passing.

**Module 5** complete: `src/nlp/ner/` — spaCy (`en_core_web_sm`) + regex
rules baseline, and a fine-tuned `prajjwal1/bert-tiny` token-classification
Transformer, trained on Few-NERD (CC BY-SA 4.0; CoNLL-2003 ruled out —
requires a signed NIST/Reuters license agreement) augmented with
clearly-documented synthetic examples for business identifiers no public
dataset labels (ORDER_ID, ACCOUNT_ID, INVOICE_ID — generated via
templates + Faker, never presented as real). PII masking
(`src/nlp/ner/pii.py`) reproduces the spec's own worked example exactly.
Real and synthetic test data scored separately specifically so the easy
synthetic entities can't inflate the real number: overall F1 0.6482,
**real-only F1 0.4894** (synthetic-only 0.9968) — full per-entity
breakdown in `docs/NER.md`. bert-tiny was chosen from the start here
(not as an emergency fallback) given Module 4's repeated OOM incidents at
DistilBERT's scale, and hit the identical too-low-learning-rate failure
Module 4's emotion classifier did, fixed the same way. 95 tests passing.

No customer-service domain data or metrics exist beyond what's listed
above — nothing is claimed without a real measured number behind it.
