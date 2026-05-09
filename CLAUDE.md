# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ Legal Notice

**Jarvis® is a trademark of Marvel Studios/Disney.** This is an independent fan project with no commercial intent. See [LICENSE-VOICEBOT.md](LICENSE-VOICEBOT.md) for full details.

## Project Name

This project is called **Jarvis Voicebot**. All documentation and code should refer to it as "Jarvis" (not "Hive" or "Voicebot" alone).

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

# List available TTS voices (for the active TTS_PROVIDER)
cargo run -- --list-voices
# or
LIST_VOICES=1 cargo run
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
  → WhisperSTTVAD (whisper-cpp-plus + Silero VAD integrated)
      partial transcripts accumulated in-memory
  → LLM client (mlx-lm / oMLX HTTP, streaming SSE)
      tokens streamed as they arrive
  → SentenceSplitter (buffer until punctuation boundary)
  → TTS (macOS AVSpeechSynthesizer via avspeech.rs, or kokoro ONNX)
      synthesizes sentence by sentence
  → AudioOutput (CPAL speaker)
```

### Key design decisions

- **Single binary**: no inter-service communication; all stages connected by `tokio::sync` channels
- **STT→LLM latency trick**: partial Whisper transcripts are accumulated in a `String`; when VAD signals end-of-speech the full transcript is sent to the LLM server. The server (mlx-lm or oMLX) maintains its own KV-cache implicitly across requests within a session.
- **LLM→TTS streaming**: LLM tokens arrive via SSE and are buffered until a sentence boundary (`.`, `!`, `?`, `;`, `:`) — then that sentence is synthesized immediately. While sentence N plays, sentence N+1 is being generated and synthesized.
- **Language**: Spanish by default, English supported. Configurable via `VOICEBOT_LANGUAGE` env var (`es` or `en`). Affects the Whisper transcription hint and the `AVSPEECH_VOICE` selected.
- **LLM backend**: external mlx-lm or oMLX server (OpenAI-compatible `/v1/chat/completions`). Both are substantially faster than llama.cpp on Apple Silicon due to the MLX framework and Apple unified memory.

### Key Modules

**`src/audio/`** — Audio pipeline (keep as-is)
- `audio_capture.rs`: CPAL microphone input; normalizes I16/U16/F32 to f32 (-1.0..1.0)
- `vad.rs`: Silero VAD; emits `SpeechStart/Speech/SpeechEnd/Silence`; 8 speech frames to start (~250ms), 48 silence frames to end (~1.5s)
- `buffer.rs`: Circular VecDeque buffer accumulating samples with duration tracking
- `audio_transform.rs`: Rubato-based resampling (FftFixedIn, 1024-chunk)
- `output.rs`: CPAL speaker playback with condvar drain (400ms silence tail to avoid CoreAudio cutoff)

**`src/stt/`** — Speech-to-Text
- `mod.rs`: `WhisperSTTVAD` — integrated STT+VAD on top of `whisper-cpp-plus`; Silero VAD for voice activity detection; streaming transcription with language hint
- `whisper.rs`: **DEPRECATED** — legacy whisper-rs wrapper; replaced by `whisper-cpp-plus` in `mod.rs`

**`src/llm/`** — LLM client
- `client.rs`: async HTTP client to `/v1/chat/completions` (OpenAI-compatible; works with mlx-lm and oMLX); `stream()` for conversation, `complete()` / `complete_short()` for background tasks (summarization, profile/memory extraction)
- `session.rs`: `LlmSession` holding `messages: Vec<Value>` + `original_system_prompt` + `summary`; `set_system_prompt()` for runtime prompt rebuild; `needs_consolidation(tokens, pct)` for threshold check

**`src/tts/`** — Text-to-Speech
- `avspeech.rs`: macOS AVSpeechSynthesizer (objc2 bindings); voice configured via `AVSPEECH_VOICE` env var (default `"Jorge (Enhanced)"`), rate via `AVSPEECH_RATE` (0.0–1.0, default 0.55)
- `kokoro.rs`: Kokoro TTS via ONNX runtime (higher quality, offline; enables with `--features kokoro`)
- `sentence.rs`: buffers incoming token stream; emits complete sentences on punctuation boundaries (`. ! ? ; :` followed by space or end). First sentence of each response uses aggressive early splitting
- `piper.rs`: Piper subprocess wrapper (kept for reference; not active)

**`src/session/`** — Conversation state
- Conversation context with message history (User/Assistant/System roles); managed via `LlmSession` in `src/llm/session.rs`

**`src/config.rs`** — Environment-based config
- `AUDIO_SAMPLE_RATE` (default 16000), `AUDIO_CHANNELS` (default 1), `AUDIO_DEVICE`, `LIST_AUDIO_DEVICES`
- `VOICEBOT_LANGUAGE` — `es` (default) or `en`
- `LLM_URL` — LLM server URL (default `http://127.0.0.1:8000` for mlx-lm; oMLX default is `8001`)
- `LLM_MAX_TOKENS` — max tokens per response (default 200)
- `LLM_CONTEXT_TOKENS` — context window size in tokens (default 8192)
- `LLM_CONSOLIDATION_THRESHOLD_PCT` — % of context window that triggers consolidation (default 80)
- `LLM_SUMMARY_KEEP_TURNS` — recent turns to keep after consolidation (default 6)
- `WHISPER_THREADS` — CPU threads for Whisper decoding (default 0 = auto)
- `LLM_SYSTEM_PROMPT` — system prompt text
- `WHISPER_MODEL` — path to whisper GGML model file (default `models/ggml-large-v3-turbo.bin`)
- `AVSPEECH_VOICE` — macOS AVSpeech voice name (default `"Jorge (Enhanced)"`)
- `AVSPEECH_RATE` — normalized speech rate 0.0–1.0 (default 0.55)
- `SAY_VOICE` (deprecated, ignored) — legacy name for `AVSPEECH_VOICE`
- `AUDIO_OUTPUT_DEVICE` — substring match of output device name; leave unset to use system default
- `SEARXNG_URL` — base URL of SearXNG instance; enables `web_search` tool when set
- `SEARXNG_SECRET` — Bearer token for SearXNG API authentication

**`src/memory/`** — Persistent memory system
- `mod.rs`: `extract_memories()` asks LLM to extract persistent notes from conversation history; `build_memory_context()` builds the `[MEMORIES]` block for the system prompt
- Memories are free-form notes (projects, decisions, preferences) that persist across sessions
- LLM can also archive outdated memories during extraction

**`src/db/`** — SQLite persistence (keep and extend)
- Chat history **must** survive process restarts — SQLite is the source of truth
- On startup: load the last session's messages, summary, profile facts, and memories from DB
- On each turn: persist user transcript and assistant response to DB
- Tables: `sessions`, `messages`, `user_profile`, `memories`

**`src/pipeline/`** — Pipeline orchestration with FSM (Finite State Machine)
- `fsm.rs`: `PipelineState` enum and `PauseReason` — tracks idle/busy/paused states for barge-in and state machine logic
- `mod.rs`: Orchestrates the STT→LLM→TTS pipeline loop
- `llm_task.rs` / `tts_task.rs` / `sen_task.rs`: Per-stage async tasks
- `frames.rs`: Audio frame handling for streaming pipeline
- `state.rs`: Shared pipeline state management
- `consolidation.rs`: Context window consolidation when threshold exceeded

**`src/daemon.rs`** — InferenceDaemon
- Long-running background daemon that loops: listen VAD → STT → LLM → TTS. Manages the main inference lifecycle.

**`src/eyes.rs`** — EyesDaemon
- Background daemon for visual/status monitoring. Periodically observes system state and reacts to changes.

**`src/control/`** — Control API
- `api.rs`: HTTP/WebSocket API for external control (start/stop pipeline, status queries)
- `state.rs`: Shared control state (running, paused, error)
- `broadcast.rs`: Event broadcast channel for state change notifications

**`src/mcp/`** — MCP (Model Context Protocol) Integration
- `mod.rs`: `McpClient` for talking to external MCP servers; `McpToolDef` for tool definitions; `call_tool()` for remote tool invocation

**`src/audio/speaker.rs`** — Speaker Verification (feature flag `speaker`)
- `SpeakerVerifier`: Identifies known speakers using ONNX speaker embeddings
- `SpeakerProfile`: Stores enrolled voice profiles
- `SpeakerVerdict`: Match/no-match/unknown verdict enum

### Legacy modules

The following modules are deprecated or removed:
- `src/s2s/` — **REMOVED** (directory no longer exists). Was the S2S adapter + LFM model, replaced by `src/stt/` + `src/llm/`
- `src/stt/whisper.rs` — **DEPRECATED** — legacy whisper-rs wrapper; replaced by `whisper-cpp-plus` in `src/stt/mod.rs`
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
- Whisper tests require model file; skip in CI if not present (`#[ignore]`)
- AVSpeech TTS tests require macOS with voices installed, kokoro for Linux CI
