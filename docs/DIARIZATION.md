# Speaker Diarization (Module 3)

## 1. Model Selection

### Why not pyannote.audio's pretrained pipeline

`pyannote/speaker-diarization-3.1` (and the `pyannote/segmentation-3.0`
model it depends on) are **gated** on Hugging Face: using them requires a
Hugging Face account, accepting each model's usage terms in a browser, and
an access token with read permission. Verified directly against the Hub API
during development:

```python
>>> HfApi().model_info("pyannote/speaker-diarization-3.1").gated
'auto'
>>> HfApi().model_info("speechbrain/spkrec-ecapa-voxceleb").gated
False
```

This project has no such token, and this environment has no browser to
accept the terms with. Module 1's constraints already rule out anything
requiring interactive setup outside the codebase, so rather than write
integration code that can't actually be run or tested here, diarization is
built on an **ungated, equally-real alternative**.

### Chosen approach: energy-VAD segmentation + ECAPA-TDNN embeddings + clustering

| Stage | Component | Why |
|---|---|---|
| Segmentation | Module 2's energy-based VAD (`src/audio/vad.py`), reused | No neural speaker-change segmenter is in play; silence gaps are used as segment boundaries instead — see Limitations. |
| Speaker embeddings | `speechbrain/spkrec-ecapa-voxceleb` (ECAPA-TDNN) | Ungated, standard architecture for speaker embeddings (also what pyannote's own pipelines use internally), 192-dim, CPU-viable. |
| Clustering | scikit-learn `AgglomerativeClustering`, cosine metric, average linkage | Simple, deterministic, no training required; speaker count is a configured assumption (2, customer-service calls are two-party) rather than auto-estimated, which this approach has no reliable signal for. |

**Requirements**: `torch`, `torchaudio`, `speechbrain` (CPU wheels — no
GPU required). **Hardware**: CPU-only tested; the embedding model is small
enough that CPU inference per segment is sub-second. **Authentication**:
**none** — `speechbrain/spkrec-ecapa-voxceleb` downloads anonymously from
the Hub, verified working in this environment.

### A genuine Windows-specific fix along the way

`speechbrain`'s model download defaults to symlinking from its Hub cache
into the local `savedir`, which fails on Windows without Developer Mode or
admin rights (`OSError: [WinError 1314]`, hit and fixed during development).
`src/diarization/embeddings.py` passes `local_strategy=LocalStrategy.COPY`
to avoid it.

## 2. Pipeline

```text
Audio
  -> src/audio (Module 2): validate, load, mono, resample, normalize
  -> src/audio/vad.py: speech regions (reused from Module 2)
  -> src/diarization/segment.py: merge close regions, split long ones
  -> src/diarization/embeddings.py: one ECAPA embedding per segment
  -> src/diarization/cluster.py: agglomerative clustering -> SPEAKER_NN labels
  -> src/diarization/diarize.py: [{speaker, start, end}, ...]
```

## 3. Transcript Alignment

`src/diarization/align.py` joins Whisper's ASR segments (Module 2) with
diarization's speaker segments by temporal overlap — the two processes
produce segments with completely different boundaries, so this is a
many-to-one overlap join, not a zip. Each ASR segment is assigned whichever
speaker's diarization segment(s) overlap it most; an ASR segment with zero
diarization overlap is labeled `UNKNOWN` rather than guessed.

## 4. Agent/Customer Role Mapping

Clustering alone produces arbitrary `SPEAKER_00`/`SPEAKER_01` labels with no
inherent meaning — there is no signal in embeddings or clustering that says
which one is the agent. The only available heuristic, "the first speaker to
talk is the agent" (inbound calls are typically answered/greeted by the
agent), is:

- Implemented in `src/diarization/roles.py`
  (`infer_roles_first_speaker_heuristic`)
- **Not applied by default** (`configs/diarization.yaml`:
  `roles.apply_first_speaker_heuristic: false`) — the pipeline's default
  output keeps generic `SPEAKER_00`/`SPEAKER_01` labels
- Logged with an explicit warning whenever it is used, and documented here
  as unverified, not ground truth

This directly follows the instruction not to make unsupported assumptions:
generic labels are the honest default; the heuristic is opt-in and
clearly flagged wherever it's used (`src/inference/pipeline.py`'s
`apply_role_heuristic` parameter, `scripts/run_diarization_pipeline.py
--apply-role-heuristic`).

## 5. Overlapping Speech

**Not handled as overlap.** The energy-based VAD classifies each frame as
speech/silence only — it has no concept of "two people talking
simultaneously." When speech overlaps, the region becomes a single
segment, and its blended embedding gets assigned to whichever cluster it's
closest to (usually the louder/dominant voice). This was verified directly:
`tests/test_diarization.py::test_diarize_handles_overlapping_speech_without_crashing`
constructs real overlapping audio and confirms the pipeline produces a
result without crashing — it does **not** assert correct separation, which
this approach cannot do. Pyannote's neural segmentation models support
multi-label overlap detection; that capability doesn't exist in the ungated
alternative used here. Documented as a known limitation, not silently
dropped.

## 6. Quality Evaluation (DER)

No real multi-speaker corpus with reference speaker labels is downloaded
yet — AMI (documented in `docs/DATASETS.md`) requires a manual per-meeting
download step not yet performed. Rather than skip evaluation or fabricate a
number, `scripts/evaluate_diarization.py` builds a **synthetic-but-real**
test: two genuine LibriSpeech clips, one pitch-shifted (+7 semitones via
`librosa.effects.pitch_shift`) to produce audibly distinct vocal
characteristics, concatenated with boundaries this project controls exactly
(so the reference labels are exact, not estimated). The audio and the
diarization output are both real; only the fact that segments 1/3 and
segment 2 "belong to different people" is synthesized rather than sourced
from two actual humans.

**Result** (`python scripts/evaluate_diarization.py`, reproducible):

| Metric | Value |
|---|---|
| DER | **11.68%** |
| Correct | 18.30s |
| Speaker confusion | **0.00s** |
| Missed detection | 2.42s |
| False alarm | 0.00s |
| Total reference speech | 20.72s |

**All error is missed detection, zero speaker confusion** — the clustering
correctly separated the two voices (and correctly recognized the 1st and
3rd segments as the same speaker) every time this was run. The error comes
entirely from the energy-based VAD's boundary imprecision (it trims a bit
of speech at segment edges relative to the exact synthetic ground truth),
not from misidentifying who's speaking. This is consistent with Module 2's
own documented VAD limitations, not a new failure mode.

**What this number does and doesn't show**: it validates that the
segmentation -> embedding -> clustering -> DER-computation pipeline is
implemented correctly end to end. It is not a claim about performance on
genuine spontaneous two-person conversation (different turn-taking
dynamics, real overlap, closer vocal similarity between two real speakers
than between a real clip and its pitch-shifted copy). Real DER on AMI
remains a documented follow-up once that corpus is downloaded.

## 7. Integration

`src/inference/pipeline.py`'s `run_pipeline()` composes Module 2 and
Module 3 into one call: preprocess -> transcribe (Whisper) -> diarize ->
align -> (optional role heuristic) -> final conversation format:

```json
{
  "call_id": "...",
  "duration": 21.72,
  "conversation": [
    {"speaker": "SPEAKER_00", "start": 0.14, "end": 7.14, "text": "..."},
    {"speaker": "SPEAKER_01", "start": 7.14, "end": 16.14, "text": "..."}
  ]
}
```

Verified end to end on a real synthetic two-speaker call via
`python scripts/run_diarization_pipeline.py --input <file>` — see
`tests/test_diarization.py` for the automated version. Intent, sentiment,
and emotion classification (Module 4) are explicitly not started.
