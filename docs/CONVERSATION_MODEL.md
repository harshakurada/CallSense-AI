# Conversation-Level Deep Learning (Module 6)

## 1. Why Synthetic Conversations

No public dataset has real resolution/customer-satisfaction/escalation
labels for customer-service conversations — checked directly, not assumed:

| Dataset | What it has | Why it doesn't fit |
|---|---|---|
| MultiWOZ | Task-oriented dialogues with a computable "task success" signal | Booking-domain (hotel/restaurant/taxi), not customer-service complaint resolution |
| SPADE (MultiWOZ-derived) | 616 real + ~14k synthetic hotel dialogues, Apache 2.0 | Goal descriptions, not resolution/satisfaction/escalation labels |
| DailyDialog | Per-utterance emotion (7 classes) + dialogue act (5 classes) | No dialogue-level label at all |

Module 6, unlike Modules 2/4/5, doesn't include an explicit "select a
public dataset" instruction — the emphasis is architecture, context
features, and honest evaluation. Given no real labels exist to select,
`src/models/conversation/synthetic.py` generates labeled multi-turn
conversations instead, following the same disclosure standard as Module
5's synthetic NER data: **explicitly documented as synthetic, never
presented as real**, and never blended into a real-data metric.

**What's real inside the synthetic data**: the customer's opening
complaint is genuine text from the Bitext dataset (Module 4) for a real
`category` label — 26,872 real customer-support instructions, cleaned of
Bitext's own unfilled `{{Order Number}}`-style template slots (verified
against the full dataset: `{{Invoice Number}}`, `{{Order Number}}`,
`{{Person Name}}`, `{{Account Category}}`, `{{Account Type}}`,
`{{Currency Symbol}}`, `{{Delivery City}}`, `{{Delivery Country}}`,
`{{Refund Amount}}` are the complete set that occur). What's synthetic:
the rest of the conversation and the resolution/satisfaction/escalation
outcome.

**Avoiding a trivial task — this took two real fixes, not one**: each
conversation is built from one of five outcome templates
(`RESOLVED_SATISFIED`, `RESOLVED_NEUTRAL`, `PARTIALLY_RESOLVED`,
`UNRESOLVED_DISSATISFIED`, `UNRESOLVED_ESCALATED`), each with several
phrasing variants, and a fraction of conversations get a deliberately
mismatched ending — a resolved conversation closing flatly, an unresolved
one closing politely — so the label isn't a perfect surface-level
giveaway. The first version of this (12% noise rate, 2 phrasings per
outcome, and — the actual bug — noise applied only to the *customer's*
line while the *agent's* closing always stayed outcome-specific) let the
deep model reach literal 100% accuracy on resolution/satisfaction/
escalation, because the agent's final turn ("I'm connecting you to a
supervisor now" is an unambiguous escalation tell) was always honest
regardless of the customer noise. Fixed by noising **both** the customer's
reaction and the agent's closing together (25% rate, 4 phrasings per
outcome) — see §5 for the training run that caught this and the one that
confirmed the fix.

## 2. Architecture

```text
Utterances (speaker-attributed, Module 3's format)
  -> frozen DistilBERT [CLS] embedding per utterance   (src/models/conversation/encoder.py)
  -> + learned speaker embedding + learned position embedding
  -> 2-layer Transformer encoder                        (context aggregation across utterances — SHARED)
  -> 4 separate attention-pooling heads, one per task    (each its own learned query vector)
  -> 4 independent linear classification heads: category, resolution, satisfaction, escalation
```

**Per-task pooling, not one shared pooled vector — found necessary, not
designed in from the start.** The first version used a single shared
attention-pooled vector feeding all four heads. Two training runs showed
why that fails: category needs the *opening* utterance's content
(the real Bitext complaint); resolution/satisfaction/escalation need the
*closing* turns. One pooled summary serving both interests ended up
starved for whichever task's gradient was weaker that run — one run's
confusion matrix showed category collapsed to never predicting 6 of 11
classes at all while the other three hit 100%; a hyperparameter-only fix
(loss reweighting + picking the checkpoint by mean per-task val accuracy)
flipped the *same* failure onto the other three tasks instead of fixing
it. Giving each task its own attention query vector — still attending
over the same shared, context-encoded sequence — let every head find the
part of the conversation actually relevant to it, without competing for
one shared summary. See §5 for the concrete numbers from all three runs.

**Why a frozen encoder, not end-to-end fine-tuning**: this project's own
CPU-only, 8GB-RAM machine already OOM-killed a *simpler* per-utterance
fine-tune during Module 4 (see `docs/NLP_MODELS.md`). Backpropagating
through DistilBERT for every utterance of every conversation here would be
substantially more expensive. Freezing the encoder and training only a
small Transformer + attention pool + heads on top (a few hundred thousand
parameters instead of 66 million) is the practical choice for this
hardware — a standard pattern, not a shortcut.

**Input representation**: each utterance becomes one 768-dim frozen [CLS]
vector, projected to 128 dims, then summed with a learned per-speaker
embedding (AGENT/CUSTOMER — 2 rows) and a learned positional embedding
(utterance order, up to 40 turns). This is literally the "speaker" and
"utterance order" context the spec asks for, and is exactly the thing a
per-utterance classifier (Modules 4-5) structurally cannot use.

**Attention aggregation**: each task's learned query vector attends over
the (masked) utterance sequence independently — lets category's head
weigh the opening complaint heavily while escalation's head weighs a late
"I want a manager!" turn heavily, from the same underlying sequence.

**Loss**: weighted sum of 4 cross-entropy losses — category weighted 2x
the other three, since it's the harder task (11 classes, diverse real
text) and was the one found starved under equal weighting. Checkpoint
selection uses **mean per-task validation accuracy**, not summed
validation loss — summed loss is dominated by whichever task's loss has
the largest absolute scale once the easy tasks floor out near zero, which
was actively misleading during debugging (see §5).

**Optimizer**: AdamW, lr=1e-3. Unlike Modules 4-5's fine-tuning, AdamW's
memory overhead here is trivial — the trainable parameter count is a few
hundred thousand, not 66 million, so the Adafactor swap those modules
needed isn't necessary here.

**A real memory-efficiency fix along the way**: `src/nlp/common/transformer.py`'s
`predict_batch()` (used by the baseline's sentiment/emotion/intent
features) reloaded its model from disk on *every single call* — no
caching at all. Since baseline feature extraction calls it once per
conversation (~1,200 times), this meant repeatedly allocating and freeing
a full model, real churn that contributed to this machine's memory
pressure. Fixed with an `@lru_cache`'d loader
(`src/nlp/common/transformer.py::_load_classifier`), and
`src/models/conversation/train.py` now explicitly releases the
baseline-phase models before building the deep model's datasets, and
releases the frozen text encoder before the training loop starts, since
neither phase needs the other's models in memory at the same time. This
also made training substantially faster (353s vs. ~850-1270s in earlier
runs), not just safer.

## 3. Baseline

`src/models/conversation/baseline.py`: classical conversation-level
features + Logistic Regression per task. Features
(`src/models/conversation/features.py`):

- Turn counts (total, per speaker), average utterance length
- **Customer average sentiment score** — Module 4's real fine-tuned
  sentiment classifier, run per customer turn and averaged
- **Customer negative-emotion fraction** — Module 4's real fine-tuned
  emotion classifier
- **Entity count** — Module 5's spaCy+rules baseline (doesn't require the
  fine-tuned NER model, always available)
- First customer turn's **intent** — Module 4's real fine-tuned intent
  classifier, one-hot encoded as a categorical feature
- Whether the conversation ends on negative sentiment

This is genuine integration of Modules 4-5, not reimplementation.

## 4. Avoiding Data Leakage

Two distinct senses of "leakage" apply here, handled differently:

1. **The literal one** (a feature that already encodes the answer): the
   synthetic generator's internal `outcome_template` field (which
   `RESOLVED_SATISFIED`/etc. template built the conversation) is carried
   in the dataset only for this documentation's error analysis — it is
   **never passed to either model as an input feature**. Both the
   baseline and the deep model only ever see the conversation text/speaker
   sequence and labels; the mapping table that generated the labels is
   not accessible to either model at train or inference time.
2. **The engineering-judgment one** (is a feature reused from another
   module illegitimate to use?): No — sentiment/emotion/intent/entities
   are all computed independently of the resolution/satisfaction/
   escalation labels, using models trained on entirely separate data
   (Twitter airline sentiment, dair-ai emotion, Bitext intent). A real
   deployed system would have exactly this same information available at
   prediction time. Using them as engineered features is standard
   practice, not leakage.

## 5. Evaluation

1,200 synthetic conversations (840 train / 180 val / 180 test).

| Task | Baseline (acc / macro F1) | Deep model (acc / macro F1) | Best |
|---|---|---|---|
| Category | 99.44% / 0.9929 | 88.89% / 0.8927 | **Baseline** |
| Resolution | 65.00% / 0.6142 | **90.00% / 0.8966** | Deep model |
| Satisfaction | 67.22% / 0.6631 | **88.89% / 0.8863** | Deep model |
| Escalation | 71.11% / 0.6662 | **96.67% / 0.9468** | Deep model |

**Category — baseline wins, honestly**: the baseline's category feature
*is* Module 4's fine-tuned intent classifier's own prediction, one-hot
encoded (§3) — it's not really a from-scratch baseline for this one task,
it's reusing an already-excellent classifier. The deep model has to learn
category from scratch, from generic frozen embeddings of diverse real
text, with 840 training examples across 11 classes (~76/class) — 88.89%
is a solid result for that, just not a fair fight against a model that
gets to cheat by calling Module 4.

**Resolution/satisfaction/escalation — the deep model wins clearly**,
by 20-25 points of accuracy each. This is the actual point of Module 6:
the baseline's features (aggregate sentiment score, negative-emotion
fraction) are coarse summaries that discard *where in the conversation*
things happened; the deep model's per-utterance sequence + speaker +
position + attention can find the specific late turn that reveals the
outcome, which is exactly the "context aggregation" and "speaker/order"
signal a per-utterance classifier structurally cannot use.

### Three training runs, not one — the real debugging history

| Run | Change | category | resolution | satisfaction | escalation |
|---|---|---|---|---|---|
| 1 | Shared pooling, 12% noise, 2 phrasings, noise on customer line only | 11.11% / 0.066 (collapsed) | 100% / 1.000 | 100% / 1.000 | 100% / 1.000 |
| 2 | + category loss 2x, checkpoint by mean val accuracy (still shared pooling) | 12.22% / 0.070 (still collapsed) | 89.44% / 0.892 | 90.00% / 0.902 | 97.22% / 0.953 |
| 3 (final) | + 25% noise on **both** customer and agent lines, 4 phrasings, **per-task pooling** | **88.89% / 0.893** | **90.00% / 0.897** | **88.89% / 0.886** | **96.67% / 0.947** |

Run 1's 100%s were the tell that something was wrong (§1) — a model that
solves resolution/satisfaction/escalation perfectly on a task designed to
be non-trivial hasn't learned to understand conversations, it's found a
shortcut. Run 2 confirmed the shortcut was in the shared-pooling
architecture, not just the hyperparameters: fixing the loss weighting and
checkpoint selection made the other three tasks slightly more realistic
but left category exactly as broken. Run 3's architecture change (§2) is
what actually resolved it. None of this is hidden — reporting only run 3
would have looked like a clean success on the first attempt, which is not
what happened.

Full per-class precision/recall/F1: `data/processed/nlp_evaluation/conversation.json`
(regenerate via `python scripts/train_conversation_model.py`, not committed).

## 6. Limitations

- **Synthetic data**: results here validate the architecture and training
  pipeline work correctly, not real-world conversation understanding —
  the same caveat Module 5's synthetic NER entities carry. The 3-run
  debugging history in §5 is itself evidence of this: it took real effort
  to make the synthetic task non-trivial, and there's no guarantee every
  remaining shortcut has been found.
- **Frozen encoder**: the utterance representations are generic DistilBERT
  embeddings, never adapted to the customer-service domain — an
  end-to-end fine-tune would likely do better, at a memory cost this
  machine doesn't have.
- **Category loss weight (2x) was tuned on this dataset**: it's not
  derived from a principled per-task difficulty measure, just the value
  that worked once per-task pooling was also in place. A different
  dataset/task mix might need a different weight.
- **First-customer-turn-only intent feature**: a conversation raising
  multiple intents (spec's "multiple intents" test case) only has its
  *first* customer turn's intent captured by the baseline — verified not
  to crash (`tests/test_conversation.py`), but the second intent isn't
  used by either model.

## 7. Integration

`src/models/conversation/inference.py::encode_conversation()` exposes the
reusable representation — now the concatenation of all four tasks' pooled
views of the same context-encoded sequence (per-task pooling means there
is no longer one single canonical "the" vector; concatenating gives a
later consumer every task's perspective at once) — specifically so a
later module (e.g. Module 7's agent-quality scoring) can consume it
directly without retraining or reimplementing the encoder stack.

## 8. Status

Agent scoring (Module 7) not started, per the spec's instruction to stop
after Module 6.
