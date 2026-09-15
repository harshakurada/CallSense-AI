"""End-to-end inference pipeline orchestration (audio in, structured
prediction out) — composes src/audio, src/asr, src/diarization, src/nlp,
src/models. Used by api/, never duplicated there."""
