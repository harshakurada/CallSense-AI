# ASR Evaluation & Error Analysis (Module 2)

All numbers on this page are from an actual run of this repository's code —
none are estimated or assumed. Reproduce with:

```bash
python scripts/download_data.py --dataset librispeech-dummy
python scripts/transcribe_dataset.py --input data/raw/librispeech_dummy
python scripts/evaluate_asr.py \
    --manifest data/raw/librispeech_dummy/manifest.jsonl \
    --transcripts data/processed/transcripts
```

## Setup

- **Dataset**: LibriSpeech ASR Dummy, 73 real clips (see `docs/DATASETS.md`)
- **Model**: faster-whisper `base`, `device=auto` (resolved to CPU on this
  machine), `compute_type=int8`, `language=en` (from `configs/config.yaml`,
  not auto-detected — see note below)
- **WER normalization**: lowercase, punctuation stripped, whitespace
  collapsed (`src/asr/evaluation.py`) — casing and punctuation are not what
  this measures

## Result

| Metric | Value |
|---|---|
| Files evaluated | 73 |
| Corpus WER | **8.70%** |
| Substitutions | 77 |
| Deletions | 13 |
| Insertions | 10 |
| Reference words | 1150 |

Full per-file breakdown: `data/processed/asr_evaluation.json` (generated,
not committed — regenerate with the commands above).

## Error Analysis

Ranked by per-file WER, the worst cases fall into clear, explainable
categories:

**1. Invented/fantasy proper nouns** — the dominant error source in this
sample. Several clips are from an Oz-book excerpt featuring the name
"Kaliko", which the model never renders correctly:
```
ref: NOT EXACTLY RETURNED KALIKO       (WER 0.75)
hyp: Not exactly, we've turned Calico.
ref: KALIKO HESITATED                  (WER 0.50)
hyp: Calico hesitated
```
"Kaliko" isn't a real English word, so the model substitutes the phonetically
closest real word or name it knows ("Calico"), and in one case mis-hears
"returned" as "we've turned" in the same clip. This is a known,
expected weakness of general-purpose ASR on out-of-vocabulary/invented
names — a real customer-service deployment would face the same failure mode
on product names, brand names, and uncommon surnames, and would need a
custom vocabulary/biasing list (faster-whisper supports `initial_prompt` /
hotword biasing) to correct it.

**2. Number normalization mismatch, not a real transcription error** —
Whisper's default text normalization renders spoken numbers as digits:
```
ref: THE TWENTIES     hyp: the 20s.       (WER 0.50)
ref: TEN SECONDS       hyp: 10 seconds     (WER 0.50)
```
The audio was transcribed correctly — "ten seconds" is not wrong — but our
WER computation treats "ten" and "10" as different words because it only
lowercases and strips punctuation, it doesn't apply number-word
normalization. This inflates the measured WER above the model's actual
transcription accuracy. **This is a measurement artifact, not a model
error**, and is the single clearest actionable fix: add a number-normalizing
step (e.g. `whisper_normalizer`'s `EnglishTextNormalizer`) to
`src/asr/evaluation.py` before the next evaluation run, rather than
reporting a WER that partly measures formatting convention instead of
transcription correctness.

**3. Rare/compound word mis-segmentation**:
```
ref: IROLG LOOKED AMAZED AT THE SUDDEN FURY OF THE ATTACK THEN SMILED
hyp: I rolled click the maze at the sudden fury of the attack, then smiled.
```
"Irolg" (another invented name) gets split into "I rolled click", and
"AMAZED" is misheard as "the maze at" — a cascading effect where one
unrecognized token throws off segmentation of the following words too.

**4. Formatting/abbreviation differences** (minor, same category as #2):
```
ref: BY HARRY QUILTER M A     hyp: by Harry Quilter MA.   (WER 0.40)
```
"M A" (spelled out) vs "MA" (the model's own reasonable rendering of a
post-nominal abbreviation) — again a normalization mismatch rather than a
mis-transcription.

### Categories the spec asks about that this dataset cannot speak to

Accent robustness, background noise, and overlapping speech are explicitly
in scope for later evaluation but LibriSpeech (read, studio-quality,
single-speaker audiobook narration) contains essentially none of these
conditions — reporting numbers for them here would mean inventing results.
These need the AMI Meeting Corpus (overlapping multi-speaker speech) and/or
Common Voice (accent diversity), both documented but not yet downloaded in
`docs/DATASETS.md`. Product terminology and customer-service-specific
named entities similarly require domain data that doesn't exist yet — that
gap is exactly why Module 1 scoped it out until real/synthetic
customer-service transcripts exist.

## Engineering note: VAD disagreement edge case

During testing, a synthetic pure-tone clip (used to unit-test the ASR JSON
contract without needing real speech) crashed `faster-whisper` with a bare
`ValueError: max() iterable argument is empty`. Cause: our own energy-based
VAD (`src/audio/vad.py`) classified the loud tone as "speech" (it's above
the silence threshold), but faster-whisper's bundled neural VAD correctly
recognized it as non-speech and filtered out every frame — and the library
crashes rather than returning an empty result when that happens combined
with language auto-detection. Fixed in `src/asr/transcribe.py` two ways:
pinning `language` from config (skips auto-detection, the actual trigger),
and a `ValueError` fallback that treats this specific failure as "no speech
found" rather than propagating it as a pipeline error. This is exactly the
kind of case a real system must handle gracefully (dead air, hold music,
non-speech noise), not an artifact of the test only.

## Takeaways feeding into later modules

- Whisper's built-in `initial_prompt` / hotword biasing should be evaluated
  once a real customer-service vocabulary (product names, account
  terminology) exists — Module 4+.
- `src/asr/evaluation.py`'s normalization should add number-word
  normalization before it's used as the reported metric in any later module.
- Diarization (Module 3) will need overlapping-speech test cases; AMI is the
  candidate source, per `docs/DATASETS.md`.
