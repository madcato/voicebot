# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
cargo build --release      # Release build
cargo run                  # Development run
cargo run --release        # Production run
cargo test                 # Run all tests
cargo fmt                  # Format code
cargo clippy               # Lint code

# List available audio devices
cargo run -- --list-devices
# or
LIST_AUDIO_DEVICES=1 cargo run
```

To run a single test:
```bash
cargo test <test_name>
cargo test -p voicebot <test_name>
```

## Architecture

Voicebot is a mono-user voice AI chatbot in Rust using a streaming STT→LLM→TTS pipeline.
All components run in a single process connected by tokio channels (no inter-service WebSockets).

### Data flow

```
Microphone
  → AudioCapture (CPAL)
  → VAD (Silero, voice_activity_detector crate)
  → STT (whisper-rs, embedded whisper.cpp)
      partial transcripts accumulated in-memory
  → LLM client (llama.cpp HTTP, streaming SSE, cache_prompt=true)
      tokens streamed as they arrive
  → SentenceSplitter (buffer until punctuation boundary)
  → TTS (piper-rs, embedded Piper ONNX)
      synthesizes sentence by sentence
  → AudioOutput (CPAL speaker)
```

### Key design decisions

- **Single binary**: no inter-service communication; all stages connected by `tokio::sync` channels
- **STT→LLM latency trick**: partial Whisper transcripts are accumulated in a `String`; when VAD signals end-of-speech the full transcript is sent to llama.cpp with `cache_prompt=true`. The KV-cache already holds previous turns, so only the new user turn needs prefill.
- **LLM→TTS streaming**: LLM tokens arrive via SSE and are buffered until a sentence boundary (`.`, `!`, `?`, `;`, `:`) — then that sentence is synthesized immediately. While sentence N plays, sentence N+1 is being generated and synthesized.
- **Language**: Spanish by default, English supported. Configurable via `VOICEBOT_LANGUAGE` env var (`es` or `en`). Affects both Whisper transcription hint and the Piper voice model selected.
- **LLM backend**: external llama.cpp server (`llama-server`). The voicebot maintains the accumulated prompt string in-memory and passes `slot_id` + `cache_prompt=true` for KV reuse across turns (mirrors `stateful-llm-server.py` from the butler project but in-process).

### Key Modules

**`src/audio/`** — Audio pipeline (keep as-is)
- `audio_capture.rs`: CPAL microphone input; normalizes I16/U16/F32 to f32 (-1.0..1.0)
- `vad.rs`: Silero VAD; emits `SpeechStart/Speech/SpeechEnd/Silence`; 8 speech frames to start (~250ms), 48 silence frames to end (~1.5s)
- `buffer.rs`: Circular VecDeque buffer accumulating samples with duration tracking
- `audio_transform.rs`: Rubato-based resampling (FftFixedIn, 1024-chunk)
- `output.rs`: CPAL speaker playback with condvar drain (400ms silence tail to avoid CoreAudio cutoff)

**`src/stt/`** — Speech-to-Text (to be implemented)
- `whisper.rs`: whisper-rs FFI wrapper; transcribes f32 mono 16kHz audio; returns text + detected language
- Language hint passed to whisper for faster decoding when language is known

**`src/llm/`** — LLM client (to be implemented)
- `client.rs`: async HTTP client to llama.cpp `/completion` endpoint
- `session.rs`: `LlmSession` struct holding `accumulated_prompt: String` and `slot_id: u8`; appends user/assistant turns in ChatML format (`<|im_start|>role\n...<|im_end|>\n`)
- Streams SSE tokens via `reqwest` with `stream` feature; yields `String` tokens through a tokio channel

**`src/tts/`** — Text-to-Speech (to be implemented)
- `piper.rs`: piper-rs wrapper; loads ONNX voice model at startup; synthesizes `&str` → `Vec<i16>` PCM at model sample rate (22050 Hz for medium models)
- `sentence_splitter.rs`: buffers incoming token stream; emits complete sentences on punctuation boundaries (`. ! ? ; :` followed by space or end)

**`src/session/`** — Conversation state (simplified)
- `context.rs`: `ConversationContext` with message history (User/Assistant/System roles)

**`src/config.rs`** — Environment-based config
- `AUDIO_SAMPLE_RATE` (default 16000), `AUDIO_CHANNELS` (default 1), `AUDIO_DEVICE`, `LIST_AUDIO_DEVICES`
- `VOICEBOT_LANGUAGE` — `es` (default) or `en`
- `LLM_URL` — llama.cpp server URL (default `http://localhost:8080`)
- `LLM_SLOT_ID` — llama.cpp KV-cache slot (default 0)
- `LLM_MAX_TOKENS` — max tokens per response (default 400)
- `LLM_SYSTEM_PROMPT` — system prompt text
- `WHISPER_MODEL` — path to whisper GGML model file
- `PIPER_MODEL_ES` — path to Spanish Piper ONNX model
- `PIPER_MODEL_EN` — path to English Piper ONNX model

**`src/db/`** — SQLite persistence (keep and extend)
- Chat history **must** survive process restarts — SQLite is the source of truth
- On startup: load the last session's accumulated prompt from DB to restore LLM KV-cache context
- On each turn: persist user transcript and assistant response to DB
- Tables: `sessions`, `messages` (role, content, timestamp)

### Legacy modules (to be removed or gutted)

The following were part of the S2S approach and will be replaced:
- `src/s2s/` — S2S adapter + LFM model (replaced by `src/stt/` + `src/llm/`)
- `src/tools/`, `src/mcp/`, `src/agents/` — not needed for MVP
- `src/websocket_client.rs` — no longer needed
- `provider/` — Python LFM2.5-Audio HTTP server (no longer used)

### Design Patterns
- **`anyhow::Result`** for error propagation with context; `thiserror` for custom error types
- **`tracing`** for structured logging throughout
- **tokio channels** (`mpsc`, `broadcast`) for inter-stage communication within the pipeline
- Cancellation via `CancellationToken` (tokio-util) — barge-in support in future

### Testing Approach
- Generate synthetic audio (sine waves / silence) for VAD and buffer tests
- Mock LLM/TTS via trait objects for pipeline integration tests
- Whisper and Piper tests require model files; skip in CI if not present (`#[ignore]`)

### Reference project
`/Users/danielvela/projects/ai/butler` — the working Python equivalent.
Key files to reference:
- `llm/zosia/stateful-llm-server.py` — stateful LLM session + llama.cpp KV-cache pattern
- `text-to-speech/main.py` — sentence splitting + Piper streaming pattern
- `speech-to-text/singleuser/main.py` — faster-whisper + VAD integration pattern
