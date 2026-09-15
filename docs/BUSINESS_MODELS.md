# Resolution, Escalation Risk & Agent Quality (Module 7)

## 1. Why Explicit Features, Not Module 6's Embeddings

Module 6 already predicts resolution and escalation from frozen-embedding
conversation representations. Module 7 rebuilds both with **named,
explicit features** instead (`src/models/business/features.py`) for one
concrete reason the spec makes explicit: section 3 requires structured
per-prediction *reasons* ("Repeated complaint", "Strong negative
sentiment", "Customer requested supervisor"). You cannot name which of
768 opaque embedding dimensions caused a prediction; you can name which
of 17 explicit features did. This is a genuine interpretability/
performance trade-off, not a strict improvement — see §3's honest
comparison against Module 6's numbers.

## 2. Features

All computed from the conversation text and timestamps, reusing Module
4's real fine-tuned sentiment/emotion/intent classifiers as signal —
independently computed from the resolution/escalation labels, so this is
legitimate feature reuse, not leakage (same reasoning as Module 6's
baseline, see `docs/CONVERSATION_MODEL.md` §4):

| Feature | Source |
|---|---|
| num_turns, num_customer_turns, num_agent_turns | conversation structure |
| conversation_duration_seconds | real turn timestamps |
| avg_customer_utterance_length | conversation text |
| first/last/mean_customer_sentiment_score, sentiment_trend | Module 4 sentiment, per customer turn |
| negative_sentiment_turn_count, negative_sentiment_fraction | Module 4 sentiment |
| negative_emotion_count, anger_or_frustration_present | Module 4 emotion |
| supervisor_request_mentioned, transfer_mentioned | regex keyword match |
| repeated_complaint_signal | heuristic: ≥2 negative-sentiment customer turns |
| ends_on_negative_sentiment | Module 4 sentiment, last turn |

**`repeated_complaint_signal` is an honest heuristic, not real repeated-
complaint detection** — genuine repeated-complaint detection would need
topic clustering across turns (e.g. "is turn 2's complaint about the same
issue as turn 6's?"), which this project doesn't build. Counting
negative-sentiment turns is a defensible proxy, documented as such rather
than presented as more sophisticated than it is.

## 3. Resolution Prediction

`src/models/business/resolution.py`: RandomForestClassifier (200 trees,
max depth 8, class-weight balanced) on the features above + one-hot
intent. Genuine non-linear ML, not a lookup table.

**Result**: 71.67% accuracy / 0.6876 macro F1 (960 train / 240 test
synthetic conversations).

**Read honestly against Module 6**: Module 6's deep model (frozen
DistilBERT embeddings + learned context Transformer) reached 90.00% /
0.8966 on the same 3-class task. Module 7's explicit-feature model is
**~18 points of accuracy worse**. This is the real cost of
interpretability here — 17 hand-picked summary statistics discard
information that a full-sequence embedding representation keeps. Both
numbers are reported, not just the better one.

**Per-class**, the pattern is clear:

| Class | F1 | Support |
|---|---|---|
| Resolved | 0.772 | 78 |
| Unresolved | 0.766 | 107 |
| **Partially Resolved** | **0.524** | 55 |

Partially Resolved is the confusable middle category — the confusion
matrix shows it's most often predicted as Resolved (20 of 55 true
Partially-Resolved cases). This matches intuition: a genuinely
in-between outcome is harder to separate from a clean "Resolved" using
coarse aggregate sentiment statistics than the two more extreme classes
are from each other.

**Top features** (by RandomForest importance): `last_customer_sentiment_score`
(0.148), `avg_customer_utterance_length` (0.120), `supervisor_request_mentioned`
(0.111), `ends_on_negative_sentiment` (0.102) — the model leans heavily on
how the conversation *ends*, which matches how the synthetic data itself
was constructed (see `docs/CONVERSATION_MODEL.md` §1).

## 4. Escalation Risk

`src/models/business/escalation.py`: RandomForestClassifier predicts a
calibrated probability of escalation; `risk_tier()` buckets it into
Low/Medium/High via **configurable thresholds**
(`DEFAULT_RISK_THRESHOLDS = {"low_max": 0.35, "medium_max": 0.65}`) —
not a separately-trained 3-class model, since the underlying event
(escalation, Yes/No) is binary and genuinely has a probability; the tiers
are a presentation layer on top of it, adjustable without retraining.

**Result**: F1 0.9247, ROC-AUC 0.9422, **Brier score 0.0618** (calibration
— how close predicted probabilities track actual outcome frequency; 0 is
perfect, this is good). All three requested metric types (F1, ROC-AUC,
calibration) are real numbers from the same held-out test set.

**Top features**: `supervisor_request_mentioned` (0.438) and
`transfer_mentioned` (0.222) alone account for two-thirds of the model's
decision weight. **Read honestly**: this is both a strength and a
limitation. Strength — it matches real intuition (an explicit request
for a manager is a strong escalation signal) and makes the model highly
interpretable. Limitation — the model may be over-reliant on these two
keyword features specifically *because* the synthetic data's escalated-
outcome closing lines reliably contain them (§1 of
`docs/CONVERSATION_MODEL.md`); a real deployment would need to verify the
model doesn't miss escalations that don't use these exact words, or
false-positive on non-escalating mentions of "manager" (e.g. "my manager
suggested I call you").

### Explanation: model vs. natural language, explicitly separated

```python
>>> model.predict_one([{"speaker": "CUSTOMER", "start": 0, "end": 2,
...                      "text": "Get me a manager!"}])
{
  "risk_tier": "High",
  "probability": 0.87,
  "model_explanation": {
    "top_factors": ["supervisor_request_mentioned", "ends_on_negative_sentiment", ...],
    "feature_importances": {...}
  },
  "natural_language_explanation": "Escalation risk: High. Contributing factors: Customer requested a supervisor; Conversation ends on negative sentiment."
}
```

`model_explanation` is real model output: which named features were both
globally important to the trained forest *and* actually active for this
specific conversation (`_is_active()`). This is **not SHAP-grade
per-instance attribution** — a true additive explanation would need the
`shap` package, which wasn't added given this project's repeated memory
constraints on an 8GB machine (`docs/NLP_MODELS.md`,
`docs/CONVERSATION_MODEL.md`). The importance-intersected-with-active-
features approach is a real, honest, but coarser approximation, documented
as such rather than presented as equivalent to per-instance attribution.

`natural_language_explanation` is template text generated *from*
`model_explanation`'s factors — explicitly not a second model or an LLM
call, and never presented as an independent judgment.

## 5. Agent Analytics

`src/models/business/agent_metrics.py` computes metrics over a list of
conversations attributed to one agent:

| Metric | Computed from |
|---|---|
| resolution_rate | fraction where `resolution.label == "Resolved"` |
| escalation_rate | fraction where `escalation.risk_tier == "High"` |
| avg_handling_time_seconds | real turn timestamps |
| avg_sentiment_improvement | last − first customer sentiment score |
| avg_talk_listen_ratio | real AGENT turn time ÷ real CUSTOMER turn time |
| satisfaction_distribution | Module 6's satisfaction prediction, per call |

**`interruption_frequency` is deliberately not implemented.** Module 3's
diarization explicitly does not detect overlapping speech (its
energy-based VAD has no concept of two people talking at once —
`docs/DIARIZATION.md`). Without real overlap detection there is no
honest way to count interruptions; inventing one from turn-taking
patterns alone is exactly what the spec's "do not invent metrics from
unavailable data" instruction rules out. Talk/listen ratio *is*
implemented because it only needs turn durations, which are real.

This project's synthetic conversations don't carry a real `agent_id`
linking multiple calls to one agent — in a real deployment, calls would
be grouped by the system's own agent identifier before calling
`aggregate_agent_metrics()`; the function itself is agent-agnostic and
just aggregates whatever list of per-call metrics it's given.

## 6. Agent Score

`src/models/business/agent_score.py` + `configs/agent_score.yaml`: a
weighted sum of four components, each rescaled to [0, 1]:

```yaml
weights:
  resolution_rate: 0.25
  non_escalation_rate: 0.25
  sentiment_improvement: 0.25
  satisfaction: 0.25
```

**Equal weighting, and why that's not arbitrary**: no labeled ground
truth exists for "what makes a good agent score" on this synthetic/demo
data to fit weights against. Equal weighting is the documented,
least-assumption default rather than a set of numbers invented to look
precise. The weights live in a config file specifically so a real
deployment with actual business priorities (e.g. a team focused on
de-escalation might weight `non_escalation_rate` higher) can change them
without touching code. Every score returned includes its full
`components` and `weights` breakdown — never just the final number — so
it's always auditable.

## 7. Integration

`src/models/business/pipeline.py::analyze_call()` returns exactly the
spec's example structure:

```json
{
  "intent": {...},
  "sentiment": {...},
  "emotion": {...},
  "entities": {...},
  "resolution": {...},
  "escalation": {...}
}
```

(`agent_metrics` is per-agent, computed across multiple calls via
`aggregate_agent_metrics()` — not part of a single call's analysis, so
it's a separate function rather than a key here that would be `null` for
every single-call analysis.)

## 8. Limitations

- Synthetic training data throughout — same caveat as Modules 5-6.
- Resolution's explicit-feature model trades ~18 points of accuracy for
  interpretability versus Module 6's embedding-based model (§3) — a real,
  measured cost, not assumed.
- Escalation's explanation method is a coarser approximation of "why",
  not true per-instance attribution (§4).
- `repeated_complaint_signal` and `transfer_mentioned` are heuristic
  proxies (negative-turn counting, keyword matching), not genuine
  topic-repetition or call-transfer-metadata detection (§2).
- `interruption_frequency` is not computed at all (§5) — an honest
  omission given Module 3's overlap-detection limitation, not a gap
  filled with an invented number.
- Agent analytics assume calls are pre-grouped by agent identity, which
  this project's synthetic data doesn't model.
