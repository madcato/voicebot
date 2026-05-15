# Testing Strategy for Voicebot

## Test Layers Overview

The project has 4 distinct test layers:

1. **Unit tests** — Inline `#[cfg(test)]` modules inside each source file
2. **Pipeline unit tests** — Isolated tests for individual pipeline tasks
3. **E2E tests** — Full STT → LLM → TTS → DB pipeline (`src/e2e_tests.rs`)
4. **Feature-gated tests** — Tests that require specific build features (`avspeech`, `kokoro`, `tui`, `remote`)

---

## Where tests live

```
src/audio/          →  #[cfg(test)] modules in vad.rs, buffer.rs, output.rs, audio_capture.rs
src/stt/            →  #[cfg(test)] module in mod.rs (WhisperSTTVAD)
src/tts/            →  #[cfg(test)] module in sentence.rs (SentenceSplitter)
src/llm/            →  #[cfg(test)] modules in client.rs, session.rs
src/tools/          →  #[cfg(test)] in web_search.rs, conversation_mode.rs
src/db/             →  #[cfg(test)] in database.rs
src/e2e_tests.rs    →  Full pipeline E2E tests (all #[ignore])
```

---

## Running tests

```bash
# All tests (skips #[ignore] E2E tests)
cargo test

# All tests including E2E
cargo test e2e -- --ignored --nocapture

# A single E2E scenario
cargo test e2e::basic_conversation_mocked_transcript -- --ignored --nocapture

# STT-only E2E tests (require Whisper model)
cargo test e2e::stt_ -- --ignored --nocapture

# Tests for a specific package
cargo test -p voicebot

# Feature-gated: with TTS backends
cargo test --features avspeech   # macOS only
cargo test --features kokoro     # Linux
```

---

## Key testing patterns

### 1. Audio pipeline tests

**Target components**: `src/audio/`, `src/stt/mod.rs` (WhisperSTTVAD)

VAD is integrated into `WhisperSTTVAD` inside `src/stt/mod.rs` — not a separate module. Tests use synthetic audio (sine waves, silence) to avoid hardware dependency.

```rust
// Example: VAD integrated with WhisperSTTVAD
use crate::stt::{WhisperSTTVAD, WhisperSTTVADConfig, SpeechEvent};

#[tokio::test]
async fn test_vad_detects_silence() {
    // Feed silence samples — should emit SpeechEnd with empty transcript
    let silence = vec![0.0f32; 32000]; // 2 seconds at 16kHz
    // ... process via WhisperSTTVAD and verify SpeechEnd
}
```

**Key concerns**:
- Use synthetic/silent audio to avoid hardware dependency
- Test VAD state machine transitions (Silence → SpeechStart → SpeechEnd)
- Mock `cpal` for output testing via `AudioOutput::null()`

### 2. Database tests

**Target components**: `src/db/database.rs`

Database tests use real SQLite with `tempfile::TempDir` for isolation:

```rust
use crate::db::Database;

#[tokio::test]
async fn test_message_persistence() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("test.db");
    let db = Database::new(db_path.to_str().unwrap()).await.unwrap();

    let session_id = db.get_or_create_session().await.unwrap();
    // ... insert messages, verify counts, assert schema
}
```

### 3. LLM client tests

**Target components**: `src/llm/client.rs`, `src/llm/session.rs`

Use `wiremock` to stand in for the LLM server. This is the pattern used across all E2E tests.

```rust
use wiremock::{Mock, MockServer, ResponseTemplate};
use wiremock::matchers::{method, path};
use crate::llm::OpenAIClient;

#[tokio::test]
async fn test_streaming_parsing() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200)
            .set_body_string(make_sse("Hello world"))
            .append_header("content-type", "text/event-stream"))
        .mount(&server)
        .await;

    let client = OpenAIClient::new(&server.uri(), "model", 100, 0.5);
    // ... verify streaming tokens
}
```

### 4. Tool tests

**Target components**: `src/tools/` (web_search, clipboard, current_time, screenshot, etc.)

Each tool implements the `Tool` trait in `src/tools/mod.rs`. Tests verify the `run()` method returns expected output.

```rust
use crate::tools::{Tool, CurrentTimeTool};

#[tokio::test]
async fn test_current_time_tool() {
    let tool = CurrentTimeTool;
    let result = tool.run("").await;
    assert!(result.contains(':')); // Contains time component
}
```

Available tools: `CurrentTimeTool`, `ReadClipboardTool`, `SetClipboardTool`, `OpenAppTool`, `TakeScreenshotTool`, `WebSearchTool`, `RunShellTool`, `ReadFileTool`, `SetConversationModeTool`, `RunAgentTool`.

### 5. TTS sentence splitting

**Target components**: `src/tts/sentence.rs` (`SentenceSplitter`)

Tests verify correct splitting on punctuation boundaries:

```rust
use crate::tts::SentenceSplitter;

#[test]
fn test_sentence_splitter() {
    let splitter = SentenceSplitter::new();
    // Feed tokens, verify splits on ".", "!", "?"
}
```

---

## E2E test harness

Located in `src/e2e_tests.rs`. Provides `E2eHarness` that sets up:

- wiremock HTTP server for OpenAI-compatible LLM
- `TtsEngine::Mock` to capture synthesized sentences
- Real SQLite in a `tempfile::TempDir`
- Direct transcript injection (bypasses Whisper for deterministic tests)

Run: `cargo test e2e -- --ignored --nocapture`

See `doc/e2e.md` for full details and scenario catalog.

---

## Testing quirks

- **VAD/audio tests**: Use synthetic sine waves / silence. See `src/audio/` tests.
- **STT tests**: Skip if model file missing. Uses `whisper-cpp-plus`.
- **TTS tests**: macOS requires `avspeech` voices installed. Kokoro needs `espeak-ng`.
- **Parallel tests**: Use `temp-env` crate to safely override env vars.
- **All E2E tests are `#[ignore]`** — they require audio hardware and are slow.
