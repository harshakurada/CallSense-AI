# Named Entity Recognition & Information Extraction (Module 5)

## 1. Entity Schema

11 types, matching the spec's list exactly:

| General (learned) | Structured (rule-based + augmented) |
|---|---|
| PERSON, ORGANIZATION, LOCATION, PRODUCT | DATE, MONEY, ORDER_ID, ACCOUNT_ID, INVOICE_ID, PHONE, EMAIL |

## 2. Datasets

### Real: Few-NERD (general entities)

| | |
|---|---|
| Source | https://ningding97.github.io/fewnerd/ (ACL 2021) |
| URL | https://huggingface.co/datasets/DFKI-SLT/few-nerd |
| License | CC BY-SA 4.0 |
| Size | 131,767 training sentences (supervised split); a stratified subset (≤4,000, sentences containing a target entity) is used for CPU-practical training |
| Classes used | `person`→PERSON, `organization`→ORGANIZATION, `location`→LOCATION, `product`→PRODUCT (of Few-NERD's 8 coarse types; `art`, `building`, `event`, `other` are outside this project's schema and relabeled O) |
| Format | Flat per-token entity type (no B-/I- prefix) — converted to standard BIO by `src/nlp/ner/dataset.py::_flat_tags_to_bio` before use |
| Limitations | General-domain text (Wikipedia-sourced), not customer-service conversation. |

**CoNLL-2003 was considered and rejected**: it's the standard NER
benchmark, but redistribution requires a signed NIST agreement and the
underlying Reuters Corpus is copyrighted, requiring separate licensing —
the same restrictive-licensing problem this project has avoided for every
other dataset (Switchboard/Fisher/CallHome for audio, gated pyannote
models for diarization). Few-NERD's CC BY-SA 4.0 license and comparable
entity coverage (PERSON/ORGANIZATION/LOCATION/PRODUCT) made it the
practical, freely-usable choice instead.

### Synthetic: business-specific entities

No public dataset labels ORDER_ID, ACCOUNT_ID, or INVOICE_ID — these are
business-specific identifiers, not general named entities. Per the spec's
own instruction, `src/nlp/ner/synthetic.py` generates template sentences
("Can you check the status of invoice {INVOICE_ID} for account
{ACCOUNT_ID}?") with values filled in by the `Faker` library (fake names,
phone numbers, emails, monetary amounts, dates, and synthetic
order/account/invoice numbers in a few plausible formats).

**This is 100% synthetic data — stated explicitly, not implied.** No real
customer name, phone number, email, order, account, or invoice number
appears anywhere in it. It exists solely to teach the model what these
entity types look like in a sentence, since no real labeled examples are
available under any license.

2,000 synthetic examples are combined with the Few-NERD subset for
training. The **test set keeps real and synthetic examples separate**
(`src/nlp/ner/dataset.py`'s `test_sources`) specifically so a synthetic
score can never be blended into and inflate the real-data score — see §6.

## 3. Baseline

`src/nlp/ner/baseline.py`: pretrained **spaCy** (`en_core_web_sm`, not
fine-tuned) for PERSON/ORGANIZATION/LOCATION/PRODUCT/DATE/MONEY, combined
with the regex rules below for structured identifiers spaCy has no concept
of. This combination is the actual baseline system, not a placeholder —
and it surfaces a real, worth-documenting failure mode:

```python
>>> [(e.text, e.label_) for e in nlp("...regarding order 45821...").ents]
[('45821', 'DATE')]  # spaCy alone misreads a bare order number as a date
```

The combined baseline's regex rules take priority over spaCy on an
overlapping span for exactly this reason (verified fixed in
`tests/test_ner.py::test_baseline_rule_wins_over_spacy_misclassification`).

## 3a. Transformer Model

Fine-tuned token classification (`src/nlp/ner/model.py`), trained on the
Few-NERD + synthetic combined dataset (§2). **Model: `prajjwal1/bert-tiny`
(~4.4M parameters), not DistilBERT** — chosen from the start, not as a
fallback, because Module 4's emotion classifier had already been killed
twice by the OS for low memory on this 8GB machine at DistilBERT's ~66M
parameters (see `docs/NLP_MODELS.md`); NER, trained after that was
discovered, starts with the memory-safe model directly.

**A repeat of the exact same lesson Module 4 hit**: the first NER training
run, at a DistilBERT-appropriate learning rate (3e-5), scored *far* below
the baseline (F1 0.029 vs. baseline's 0.525) — a small, weakly-pretrained
model needs a much higher learning rate than a full-size one. Raised to
5e-4 (the same fix, and the same value, as Module 4's emotion classifier)
and given more epochs (8, cheap since bert-tiny trains in a few minutes),
it reached 0.6482 overall F1, clearly beating the baseline. Documented
here because it's a second real occurrence of the same failure mode, not
a one-off — see §6 for what that F1 actually breaks down to.

## 4. Rule-Based Extraction

`src/nlp/ner/rules.py` — regex patterns for EMAIL, PHONE, MONEY,
INVOICE_ID (`INV-YYYY-NNNN`), ACCOUNT_ID (`ACCNNNNNN` / `A-NNNNN`), and
ORDER_ID. These are deliberately rule-based rather than learned: a regex
is more reliable than a model trained on a few thousand synthetic examples
for exactly the entities that already have a rigid, known shape.

**A real bug found and fixed during testing**: the initial ORDER_ID
pattern used a fixed-width lookbehind (`(?<=order\s)\d{4,8}`), which only
matches "order 45821" — realistic (especially spoken/transcribed) text
puts filler in between: "order **number is uh** 45821". Python's `re`
doesn't support variable-length lookbehind, so this is instead handled by
`_find_order_ids_by_proximity`: find "order", then search up to 30
characters ahead for the nearest bare 4-8 digit number. Caught by
`tests/test_ner.py::test_baseline_noisy_transcript_does_not_crash`, which
failed on the first implementation.

## 5. PII Detection

`src/nlp/ner/pii.py::mask_pii` replaces entity spans with `[LABEL]`.
Verified against the spec's own worked example exactly:

```text
"John Smith called regarding order 45821."
  -> "[PERSON] called regarding order [ORDER_ID]."
```

Default masked types: PERSON, PHONE, EMAIL, ACCOUNT_ID, ORDER_ID,
INVOICE_ID. **LOCATION, ORGANIZATION, PRODUCT, DATE, and MONEY are not
masked by default** — a business needs the product/amount/date for
analytics. The spec's "addresses" isn't masked either: a personal address
isn't reliably distinguishable from a general place mention (a city, a
store location) without a classifier this project doesn't have — masking
every LOCATION would destroy useful data, and masking none silently misses
addresses. Documented as an open limitation rather than picking a silently
wrong default in either direction.

## 6. Entity Evaluation

Entity-level Precision/Recall/F1 via **seqeval** (the standard library for
sequence-labeling evaluation — not a hand-rolled token-overlap metric),
computed separately for:

- The **real** (Few-NERD-derived) portion of the test set
- The **synthetic** portion of the test set

kept apart specifically so a high synthetic score (templated data is
easier) can't make the real-data score look better than it is.

| | Overall F1 | Real-only F1 | Synthetic-only F1 |
|---|---|---|---|
| Baseline (spaCy + rules) | 0.5249 | 0.3961 | 0.8310 |
| Transformer (fine-tuned) | 0.6482 | **0.4894** | 0.9968 |

**The gap this split was built to catch actually shows up**: synthetic
F1 is near-perfect (0.9968) for both — expected, since the synthetic
templates are far more regular than genuine text. The **real-only F1
(0.489) is the honest number** for how this model performs on actual
sentences, and a blended overall score (0.648) would have overstated it
by 16 points. Full per-file breakdown:
`data/processed/nlp_evaluation/ner.json` (regenerate via
`python scripts/train_ner_model.py`, not committed).

**Per-entity F1** (fine-tuned model, all test data):

| Entity | F1 | Support | Source |
|---|---|---|---|
| ACCOUNT_ID | 1.000 | 113 | synthetic |
| EMAIL | 1.000 | 76 | synthetic |
| MONEY | 1.000 | 94 | synthetic |
| ORDER_ID | 1.000 | 98 | synthetic |
| INVOICE_ID | 1.000 | 107 | synthetic |
| PHONE | 0.977 | 86 | synthetic |
| DATE | 0.996 | 131 | synthetic |
| PERSON | 0.655 | 522 | real (Few-NERD) |
| LOCATION | 0.562 | 552 | real (Few-NERD) |
| ORGANIZATION | 0.335 | 392 | real (Few-NERD) |
| PRODUCT | 0.268 | 142 | real (Few-NERD) |

**Every synthetic-only entity type scores ≥0.977; every real-only type
scores ≤0.655.** This is the clearest possible confirmation that the
model has learned the synthetic patterns well but genuinely struggles
with the harder, real entity types — ORGANIZATION and PRODUCT especially,
which are the two Few-NERD coarse types with the most fine-grained
internal variety (companies, media, sports teams, political parties for
ORGANIZATION; food, software, vehicles, weapons for PRODUCT — see Few-NERD's
fine-grained label list in the dataset card), making them intrinsically
harder for a 4.4M-parameter model with only ~4,000 real training examples.

## 7. Information Extraction

`src/nlp/ner/extract.py` maps entities to
`{customer, order_id, product, issue, date, amount}`. Only `PERSON`,
`ORDER_ID`, `PRODUCT`, `DATE`, and `MONEY` entities populate their
corresponding field; **`issue` is always `null`** — describing what the
customer's actual problem is requires summarization or intent
understanding (Module 4's classifier is the nearest existing signal, not a
substitute), not entity recognition, and is out of this module's scope.
When an entity type has no match at all, its field is `null`, never
guessed.

## 8. Entity Normalization

`src/nlp/ner/normalize.py` converts MONEY to a float, PHONE to digits-only,
and DATE to ISO-8601 where the format is confidently parseable. A DATE
string that doesn't match a known format (e.g. "next Tuesday") keeps its
raw text with `normalized: null` rather than being forced into a guess.

## 9. Integration

```text
Transcript (Module 2/3)
  -> src/nlp/ner/inference.py: fine-tuned Transformer entities + rule fallback
  -> src/nlp/ner/normalize.py: MONEY/PHONE/DATE standardization
  -> src/nlp/ner/pii.py: mask_pii()
  -> src/nlp/ner/extract.py: extract_structured_fields()
  -> src/nlp/ner/pipeline.py::analyze_conversation()
```

`analyze_conversation()` runs NER per turn (for per-turn masked text) and
once over the full concatenated transcript (for call-level
`structured_fields`, since an order/invoice number is usually stated once,
not repeated every turn). Takes Module 3's speaker-attributed conversation
format directly.

## 10. Status

Conversation-level modeling (Module 6) not started, per the spec's
instruction to stop after Module 5.
