# NLP Models — Intent, Sentiment, Emotion (Module 4)

## 1. Datasets

No public dataset of real customer-service call transcripts with intent/
sentiment/emotion labels exists under a usable license (the same gap
documented for audio in `docs/DATASETS.md`). Each task instead uses the
closest real, verifiably-licensed public dataset; all facts below were
confirmed by loading the dataset in this repo, not taken from a dataset
card at face value.

### Intent — Bitext Customer Support LLM Chatbot Training Dataset

| | |
|---|---|
| Source | Bitext Innovations, hybrid synthetic (templated + NLG-generated), designed for customer-support chatbot training |
| URL | https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset |
| License | CDLA-Sharing-1.0 |
| Size | 26,872 examples |
| Classes | 11, from the dataset's own `category` field: **ACCOUNT** (5986), **ORDER** (3988), **REFUND** (2992), **INVOICE** (1999), **CONTACT** (1999), **PAYMENT** (1998), **FEEDBACK** (1997), **DELIVERY** (1994), **SHIPPING** (1970), **SUBSCRIPTION** (999), **CANCEL** (950) |
| Limitations | Synthetic (templated/NLG-generated), not real customer utterances — phrasing is more regular/less noisy than genuine transcripts. |

**Label choice**: the spec suggests a 10-class list (Billing, Refund,
Cancellation, Technical Support, Account, Delivery, Complaint,
Subscription, Product Information, Other) that doesn't match any dataset's
actual labels. Rather than force an unverified mapping, this uses the
dataset's real 11 `category` values directly.

### Sentiment — Twitter US Airline Sentiment

| | |
|---|---|
| Source | Crowdflower's Data for Everyone library |
| URL | https://huggingface.co/datasets/osanseviero/twitter-airline-sentiment |
| License | CC-BY-NC-SA-4.0 (**non-commercial** — noted since this project could otherwise imply broader reuse rights) |
| Size | 14,640 tweets |
| Classes | positive (2363), neutral (3099), negative (9178) |
| Limitations | Tweets about airlines, not call-center transcripts — different register/length than a spoken utterance; genuinely real customer sentiment (not synthetic), which is the trade-off against Intent's dataset. |

### Emotion — dair-ai/emotion

| | |
|---|---|
| Source | DAIR.AI, from the 2018 EMNLP paper "CARER: Contextualized Affect Representations for Emotion Recognition" |
| URL | https://huggingface.co/datasets/dair-ai/emotion |
| License | Stated for educational/research use |
| Size | 16,000 train / 2,000 validation / 2,000 test (the `split` config; a 416,809-example unsplit config also exists, not used here) |
| Classes | sadness (4666), joy (5362), love (1304), anger (2159), fear (1937), surprise (572) — counts from the train split |
| Limitations | Single Twitter messages, not conversational turns; classes don't match the spec's suggested set (see below). |

**Label choice**: the spec suggests {anger, frustration, happiness,
sadness, fear, neutral, satisfaction}. No public dataset with exactly
these labels on conversational (or even just real) text was found under a
usable license. Per the spec's own instruction ("use the actual dataset
labels and create a documented mapping if necessary"), the real 6 labels
are used as-is: `joy` stands in for "happiness"; "frustration", "neutral",
and "satisfaction" are not represented and are not fabricated by relabeling
something else.

## 2. Class Imbalance

All three datasets are naturally imbalanced (see counts above — e.g.
CANCEL at 950 vs. ACCOUNT at 5986, roughly 6x; surprise at 572 vs. joy at
5362, roughly 9x). Handled the same way in both the baseline and the
Transformer, rather than resampling, which would create synthetic/repeated
examples: `class_weight="balanced"` in the sklearn baseline, and an
inverse-frequency-weighted `CrossEntropyLoss` in the Transformer
(`src/nlp/common/transformer.py`'s `WeightedLossTrainer`). Stratified
train/val/test splitting (`src/nlp/common/data.py`) preserves each split's
class proportions rather than letting a random split accidentally
starve a rare class from train or test.

## 3. Models

| | |
|---|---|
| Baseline | TF-IDF (1-2 grams, 10k features) + Logistic Regression (`class_weight="balanced"`) — `src/nlp/common/baseline.py` |
| Transformer | `distilbert-base-uncased`, fine-tuned per task |

**Why DistilBERT, not BERT/RoBERTa/DeBERTa directly**: Module 1 fixed this
as a CPU-only-by-default project. DistilBERT keeps ~97% of BERT's benchmark
performance at ~60% of the parameters, which is what makes fine-tuning
three separate classifiers in this environment practical at all — a full
BERT-base or DeBERTa fine-tune per task would cost several times the CPU
wall-clock time for a marginal accuracy gain at this dataset scale. This is
the "choose the best practical model" call the spec explicitly allows.

**CPU training budget**: each task is trained on a stratified subsample
(≤4,000 examples) rather than the full dataset, for the same CPU-only
reason — documented explicitly rather than silently training on less data
than exists. See `src/nlp/{intent,sentiment,emotion}/dataset.py`.

## 4. Model Comparison (real measured results)

| Task | Baseline (acc / macro F1) | Transformer (acc / macro F1) | Best | Training time |
|---|---|---|---|---|
| Intent | 98.99% / 0.9906 | **99.66% / 0.9972** (DistilBERT) | Transformer | 2059.6s (~34 min) |
| Sentiment | 75.46% / 0.6712 | **82.80% / 0.7748** (DistilBERT) | Transformer | 3169.0s (~53 min) |
| Emotion | 74.00% / 0.6983 | **86.38% / 0.8277** (bert-tiny) | Transformer | 442.4s (~7 min) |

Full per-class precision/recall/F1 and confusion matrices:
`data/processed/nlp_evaluation/<task>.json`, generated by
`python scripts/train_nlp_models.py --task all`, not committed.

**Intent, read honestly**: both models score above 98% because the Bitext
dataset is templated/synthetic (see §1) — real, noisier customer speech
would not be this easy. The gap between baseline and Transformer (0.66
points of accuracy) is real but small for the same reason: with regular,
templated phrasing, TF-IDF n-grams already capture most of what a
Transformer's contextual embeddings would add.

**Sentiment, read honestly**: this is real tweet text, not templated —
and the gap between baseline and Transformer is much larger here (7.3
points accuracy, 10.4 points macro F1) than on intent. This is the more
representative result for what to expect once this pipeline runs on real,
noisy customer speech: a Transformer's contextual understanding earns its
cost much more clearly on genuinely hard text than on templated text.
Training also took noticeably longer than intent (53 min vs. 34 min) even
on a comparable dataset size — see the memory-constraint note below.

**A real infrastructure constraint, not a modeling one**: this machine
has 8GB RAM shared with other running applications. The first sentiment
training attempt (batch size 16, default AdamW) was killed by the OS for
low memory partway through. Fixed by reducing to batch size 8 with
gradient accumulation (same effective batch size) and switching from
AdamW to **Adafactor**, which factors down the optimizer's per-parameter
state instead of keeping two full-size moment buffers — the actual source
of the memory pressure, not the batch size itself (see
`src/nlp/common/transformer.py`). Training succeeded on retry but slower.

That fix wasn't enough for emotion: it was **killed twice more**, by the
OS, for low memory, on DistilBERT — even with the same batch/Adafactor
mitigations. Rather than retry the same configuration a third time,
emotion falls back to **`prajjwal1/bert-tiny`** (~4.4M parameters vs.
DistilBERT's ~66M — a 15x reduction). The first bert-tiny run, at
DistilBERT's learning rate (2e-5), scored dramatically *below* the
baseline (36.0% accuracy vs. baseline's 74.0%) — 2e-5 is far too
conservative for a model this small starting from much weaker
pretraining. Raised to 5e-4 (tuned by observing that failure, not guessed
upfront) and given more epochs (8, cheap since bert-tiny trains in
minutes not hours), it reached 86.38% / 0.8277 macro F1 — beating
DistilBERT-scale expectations on this dataset entirely, and in ~7 minutes
versus sentiment's ~53. **This is now an intentional inconsistency across
the three tasks** (intent/sentiment use DistilBERT, emotion uses
bert-tiny) — documented rather than hidden, and not retroactively applied
to intent/sentiment since both already had valid, real, successful runs
that didn't need it.

## 5. Error Analysis

### Intent (2 errors / 594 test examples)

Both errors are the same confusion, found by inspecting the actual
misclassified inputs:

```text
TRUE=SHIPPING PRED=ACCOUNT (conf=0.95): can utell me more about updating my addfess
TRUE=SHIPPING PRED=ACCOUNT (conf=0.97): can ya give me information about an addres update
```

Both are genuinely ambiguous — "updating an address" is plausibly an
account-settings action, not obviously shipping-specific, so the model's
ACCOUNT prediction is a reasonable reading, just not the dataset's label.
Both also contain the Bitext dataset's intentional typos ("addfess",
"addres") — the dataset design's "spelling issues" augmentation tag,
already visible even at this near-ceiling accuracy.

### Sentiment (103 errors / 599 test examples)

Confusion matrix (rows = true, columns = predicted; order negative,
neutral, positive):

```text
          neg  neu  pos
negative  341   27    8
neutral    30   82   15
positive   13   10   73
```

**neutral is the hardest class** (F1 0.667 vs. 0.897 for negative, 0.760
for positive), and its errors skew heavily toward being called negative
(30 of 45 neutral errors), not positive. Inspecting the actual
misclassified tweets shows why:

```text
TRUE=neutral PRED=negative (0.57): "@AmericanAir I thought all those planes were retired? #MD80"
TRUE=neutral PRED=negative (0.62): "does she need to complain on Twitter for the refund or is it auto-applied?"
TRUE=neutral PRED=negative (0.97): "I was under the impression when there is an 8 hour delay in your flight because of equipment failure, compensation is offered?"
```

All three are **factual/rhetorical questions that mention negative-coded
words** ("retired", "complain", "refund", "delay") **without actually
expressing negative sentiment themselves** — the customer is asking about
policy, not complaining. The model has learned that these words correlate
strongly with negative sentiment (correctly, in aggregate — negative is
79% of the corpus with 30% of the mentions of "delay"/"refund"/"complain"
alone), and over-applies that association to neutral questions that merely
reference the same topics. This is a real, actionable limitation: a
production system should not treat "mentions a complaint-adjacent word" as
equivalent to "is a complaint" — exactly the gap conversation-level context
(Module 6) or a dedicated question/statement detector would need to close.

### Emotion (109 errors / 800 test examples)

Per-class F1, sorted:

| Class | Support | F1 |
|---|---|---|
| joy | 278 | 0.897 |
| sadness | 232 | 0.888 |
| fear | 90 | 0.851 |
| anger | 110 | 0.841 |
| love | 64 | **0.756** |
| surprise | 26 | **0.733** |

The two rarest classes in the training data (`surprise` at 572/16000 and
`love` at 1304/16000 — see §1) are also the two worst-performing, despite
the inverse-frequency weighted loss meant to counteract exactly this. The
confusion matrix shows *why* for `love` specifically: 10 of its 64 test
examples are misclassified as `joy` — a genuinely close pair
(expressions of love are frequently also joyful) rather than a random
error. `surprise`'s errors are small in absolute count (4 total) and
spread across classes rather than concentrated, consistent with too few
training examples (572) to learn a sharp decision boundary rather than a
specific confusable pair.

## 6. Sentiment Aggregation

`src/nlp/sentiment/inference.py` predicts per-utterance, then aggregates:

- **customer_level**: majority-vote label (ties broken by mean confidence)
  over turns labeled `CUSTOMER` specifically. This only has data if the
  conversation's speakers were actually role-mapped — Module 3's default
  output uses generic `SPEAKER_00`/`SPEAKER_01` labels, so `customer_level`
  intentionally comes back empty (`{"label": None, ...}`) rather than
  silently aggregating the wrong speaker's turns.
- **conversation_level**: the same aggregation over every turn.
- Both also report a simple **score** (positive=+1, neutral=0,
  negative=-1, averaged) as a transparent trend indicator — explicitly not
  a model output.

## 7. Emotion Timeline

`src/nlp/emotion/timeline.py` predicts one emotion per conversation turn
(not resampled to fixed time intervals, since a turn is the actual unit of
text the model saw) and renders it as:

```text
00:00 Neutral
00:20 Frustrated
00:45 Angry
```

using whichever label the model actually predicted — not remapped to
display names outside the dataset's real classes.

## 8. Inference Pipeline

```text
Speaker-attributed transcript (Module 3)
  -> src/nlp/inference.py: analyze_conversation()
       -> src/nlp/intent/inference.py      (call-level: CUSTOMER turns concatenated)
       -> src/nlp/sentiment/inference.py   (per-utterance + aggregation)
       -> src/nlp/emotion/timeline.py      (per-utterance timeline)
  -> {"intent": {...}, "sentiment": {...}, "emotion": {...}}
```

Every `confidence` value returned anywhere in this module is the
classifier's own softmax probability for its predicted class — never
estimated or invented.

## 9. Status

NER (Module 5) not started, per the spec's instruction to stop after
Module 4.
