# End-to-End Testing

This document explains how to run the voicebot end-to-end (E2E) tests, record WAV fixtures, and what each test scenario covers.

## Overview

E2E tests live in `src/e2e_tests.rs` and are loaded as a `#[cfg(test)]` submodule of `main.rs`. This gives them direct access to `run_pipeline` and all internal types without needing a separate integration test binary.

All E2E tests are marked `#[ignore]` so they never run during `cargo test`. They must be invoked explicitly by the developer.

The tests are split into two categories:

| Category | STT | LLM | Requires |
|----------|-----|-----|---------|
| **Mocked** (fast) | `SttStream::mock(transcript)` | wiremock | Audio output device |
| **Real STT** | Real Whisper on a WAV file | wiremock | Audio output device + Whisper model + WAV fixture |

## Running the tests

```bash
# Run all E2E tests
cargo test e2e -- --ignored --nocapture

# Run a single scenario
cargo test e2e::basic_conversation_mocked_transcript -- --ignored --nocapture

# Run only real-STT tests (require Whisper model + WAV fixtures)
cargo test e2e::stt_ -- --ignored --nocapture
```

> **Note:** All E2E tests require a working audio output device (CPAL). They use `TtsEngine::Mock` which returns 1 sample of silence, so audio playback completes in microseconds — but the CPAL device handle must open successfully.

## Test scenarios

### Mocked transcript tests (no Whisper needed)

| Test | What it verifies |
|------|-----------------|
| `basic_conversation_mocked_transcript` | LLM response reaches TTS and is saved to DB |
| `empty_transcript_is_discarded` | Empty STT output → no LLM call, no DB write |
| `ambient_mode_discards_utterance_without_wake_word` | Ambient mode silences the bot when wake word is absent |
| `ambient_mode_responds_when_wake_word_present` | Ambient mode lets through utterances containing the wake word |
| `multi_sentence_response_splits_into_sentences` | SentenceSplitter emits ≥ 2 TTS chunks for a multi-sentence reply |
| `db_persists_multiple_turns` | DB accumulates 2 assistant messages across 2 pipeline calls |

### Real STT tests (require Whisper + WAV fixtures)

| Test | What it verifies |
|------|-----------------|
| `stt_transcribes_wav_file` | Whisper produces a non-empty transcript for `tests/fixtures/hola.wav` |
| `full_pipeline_wav_to_db` | WAV → Whisper → mock LLM → DB: full pipeline with real STT |

## WAV fixture requirements

Whisper requires **16kHz mono** WAV files. Use the scripts below to record or convert.

### Recording on macOS

```bash
# Record 5 seconds of audio (say "Hola, ¿qué hora es?" while it records)
sox -d -r 16000 -c 1 tests/fixtures/hola.wav trim 0 5

# Or with ffmpeg from mic input (macOS default device index is usually 0)
ffmpeg -f avfoundation -i ":0" -t 5 -ar 16000 -ac 1 tests/fixtures/hola.wav
```

### Converting an existing file

```bash
ffmpeg -i input.m4a -ar 16000 -ac 1 tests/fixtures/hola.wav

# Verify the output
soxi tests/fixtures/hola.wav   # should show: Rate=16000, Channels=1
```

### Required fixtures

| File | Content | Used by |
|------|---------|---------|
| `tests/fixtures/hola.wav` | "Hola, ¿qué hora es?" | `stt_transcribes_wav_file`, `full_pipeline_wav_to_db` |

Additional fixtures can be added for tool-call or ambient-mode scenarios as those tests are written.

## Environment variables for STT tests

```bash
# Path to the Whisper GGML model (default: models/ggml-large-v3-turbo.bin)
export WHISPER_MODEL=models/ggml-large-v3-turbo.bin

cargo test e2e::stt_ -- --ignored --nocapture
```

If the model file or WAV fixture is missing, the tests print a `SKIP:` message and exit cleanly — they do not fail.

## Architecture

```
src/e2e_tests.rs          ← test module (declared via #[cfg(test)] mod in main.rs)
tests/fixtures/           ← pre-recorded WAV files (gitkeep, record manually)
```

### What the harness does

1. Starts a wiremock HTTP server to stand in for llama-server.
2. Creates a `TtsEngine::Mock` that captures synthesized sentence text instead of playing audio.
3. Opens a real SQLite database in a `tempfile::TempDir`.
4. Calls `run_pipeline()` with `SttStream::mock(transcript)` (bypasses Whisper).
5. After the pipeline returns, asserts on captured TTS sentences and DB rows.

### Why not temperature=0 against a live llama-server?

wiremock is more reliable: it defines exactly what the LLM says, requires no external process, and never varies across model versions. Temperature=0 is useful for manual smoke-testing but not for automated assertions.
