# Voicebot Architecture

## Overview

Voicebot is a mono-user voice AI assistant in Rust. It runs as a single binary using a
**streaming STT → LLM → TTS pipeline** where every stage is connected by `tokio` channels.
There is no inter-service communication; everything runs in-process.

```
Microphone → AudioCapture (CPAL)
           → VAD (Silero)
           → STT (Whisper)          ← speaks transcription as soon as speech ends
           → LLM (OpenAI-compat.)   ← starts while STT is still running (speculative prefill)
           → SentenceSplitter       ← fires per sentence as tokens arrive
           → TTS synthesizer        ← synthesis of sentence N overlaps playback of N-1
           → AudioOutput (CPAL)
```

All GPU/CPU-heavy work runs in `tokio::task::spawn_blocking` threads so the async event
loop stays unblocked.

---

## Project Structure

```
src/
├── main.rs                    # Pipeline orchestration, VAD loop, barge-in logic
├── config.rs                  # All config (env-var driven)
│
├── audio/
│   ├── audio_capture.rs       # CPAL mic input; normalises I16/U16/F32 → f32
│   ├── audio_transform.rs     # Rubato resampling (FftFixedIn, 1024-chunk)
│   ├── vad.rs                 # Silero VAD; emits SpeechStart/Speech/SpeechEnd/Silence
│   ├── buffer.rs              # Circular VecDeque buffer with duration tracking
│   ├── output.rs              # CPAL speaker playback (play_blocking + cancel support)
│   └── speaker.rs             # Optional speaker verification (ONNX embedding model)
│
├── stt/
│   ├── whisper.rs             # WhisperStt — whisper-rs FFI, Metal GPU state cached
│   └── stream.rs              # SttStream — always-on background Whisper worker
│
├── llm/
│   ├── client.rs              # LlamaClient — OpenAI-compatible streaming SSE client
│   └── session.rs             # LlmSession — message history + all_messages_api()
│
├── tts/
│   ├── mod.rs                 # TtsEngine enum (Say | Kokoro | Mock)
│   ├── say.rs                 # SayTts — macOS `say` subprocess → raw PCM
│   ├── piper.rs               # PiperTts — Piper subprocess (kept for reference)
│   ├── kokoro.rs              # KokoroTts — ONNX model (--features kokoro)
│   └── sentence.rs            # SentenceSplitter — buffers tokens; emits on punctuation
│
├── tools/                     # Tool registry + individual tool implementations
│   ├── mod.rs                 # ToolRegistry, ToolDefinition, format_history
│   ├── current_time.rs
│   ├── calendar.rs
│   ├── clipboard.rs
│   ├── read_file.rs
│   ├── open_app.rs
│   ├── run_shell.rs           # Enabled via SHELL_ENABLED=1
│   ├── send_notification.rs
│   ├── take_screenshot.rs     # Enabled via VISION_URL
│   ├── run_agent.rs           # Delegates to external agent binary (AGENT_COMMAND)
│   └── conversation_mode.rs   # SetConversationModeTool (Active / Ambient)
│
├── agents/
│   └── mod.rs                 # ProactiveEvent — agent result delivery to main loop
│
├── daemon.rs                  # InferenceDaemon — periodic "anything worth saying?" check
├── profile/mod.rs             # User profile: fact extraction + context injection
└── db/
    └── database.rs            # SQLite: sessions, messages, summaries, user_profile
```

---

## Provider Interfaces

The pipeline uses three pluggable provider layers: **STT**, **LLM**, and **TTS**.
Each is designed so the rest of the pipeline (`stream_and_tts`, `run_pipeline`, etc.)
is completely backend-agnostic.

### TTS — `TtsEngine` (enum dispatch)

Location: `src/tts/mod.rs`

```rust
pub enum TtsEngine {
    Say(SayTts),                  // macOS `say`         — no build flags needed
    Kokoro(KokoroTts),            // ONNX model          — requires --features kokoro
    Mock(MockTts),                // test only
}

impl TtsEngine {
    pub fn synthesize(&self, text: &str) -> Result<Vec<f32>>;
    pub fn sample_rate(&self) -> u32;
}
```

**Adding a new TTS backend** (e.g. Piper, Coqui, ElevenLabs):

1. Create `src/tts/my_backend.rs` with a struct implementing `synthesize(&self, text) -> Result<Vec<f32>>` and `sample_rate()`.
2. Add a variant to `TtsEngine` and its two `match` arms.
3. Add the branch to the `match config.tts_provider.as_str()` block in `main.rs`.
4. Add `TTS_PROVIDER=my_backend` to the env-var docs.

| Backend | Provider key | Feature flag | Notes |
|---------|-------------|--------------|-------|
| macOS say | `say` | — | default; macOS only |
| Kokoro ONNX | `kokoro` | `kokoro` | offline, high quality |
| Piper | `piper` | — | subprocess; `piper.rs` is a starting point |

---

### STT — `WhisperStt` + `SttStream`

Location: `src/stt/`

```rust
// Core transcription — one blocking call per audio snapshot
pub struct WhisperStt { ... }
impl WhisperStt {
    pub fn new(model_path: &str, language: &str) -> Result<Self>;
    pub fn transcribe(&self, audio: &[f32]) -> Result<String>;   // 16kHz mono f32
}

// Always-on streaming wrapper — submits audio snapshots and returns
// results keyed by sequence number. Decouples VAD timing from Whisper latency.
pub struct SttStream { ... }
impl SttStream {
    pub fn new(stt: Arc<WhisperStt>) -> Arc<Self>;
    pub fn submit(&self, audio: Vec<f32>) -> u64;                // returns seq number
    pub async fn await_result(&self, min_seq: u64) -> String;
}
```

**Adding a new STT backend** (e.g. faster-whisper via HTTP, Deepgram, Apple Speech):

1. Create `src/stt/my_backend.rs` with a struct that has `transcribe(&self, audio: &[f32]) -> Result<String>`.
2. Wrap it in `SttStream::new_with(Arc<dyn SttTranscriber>)` — or replace the concrete
   `WhisperStt` reference in `SttStream` with a trait object:

```rust
pub trait SttTranscriber: Send + Sync {
    fn transcribe(&self, audio: &[f32]) -> Result<String>;
}
```

3. Update `main.rs` to construct the chosen backend based on `STT_PROVIDER` env var.

| Backend | Notes |
|---------|-------|
| whisper-rs (whisper.cpp) | default; embedded; Metal GPU on Apple Silicon |
| faster-whisper HTTP | call a Python sidecar via HTTP |
| Apple Speech (AVSpeechRecognizer) | macOS native; requires `CoreML` bindings |

---

### LLM — `LlamaClient`

Location: `src/llm/client.rs`

```rust
pub struct LlamaClient { ... }

impl LlamaClient {
    // Streaming completion — yields StreamToken::Content or StreamToken::ToolCall
    pub async fn stream(
        &self,
        messages: &[serde_json::Value],
        tools: &[serde_json::Value],
    ) -> Result<mpsc::Receiver<StreamToken>>;

    // One-shot completion — used for summarisation / profile extraction
    pub async fn complete(&self, messages: &[Message]) -> Result<String>;
    pub async fn complete_short(&self, messages: &[Message]) -> Result<String>;

    // KV-cache prefill (max_tokens=0) — llama.cpp only; disabled for other providers
    pub async fn prefill_cache(&self, messages: Vec<serde_json::Value>) -> Result<()>;
    pub fn supports_prefill_warm(&self) -> bool;
}
```

`LlamaClient` targets **any OpenAI-compatible endpoint**.
Backend-specific behaviour is selected at construction via `.with_provider(name)`:

| `LLM_PROVIDER` | Backend | Extra fields sent |
|----------------|---------|-------------------|
| `llama` (default) | llama.cpp | `cache_prompt`, `slot_id` |
| `mlx` | mlx-lm | `enable_thinking: false`, `chat_template_kwargs`, `repetition_penalty` |
| *(any)* | LM Studio, OpenAI, Ollama | no extra fields (standard OpenAI body) |

**Adding a non-OpenAI LLM backend** (e.g. a gRPC model, custom HTTP API):

Define a trait and make both the existing client and the new one implement it:

```rust
pub trait LlmProvider: Send + Sync {
    async fn stream(
        &self,
        messages: &[serde_json::Value],
        tools: &[serde_json::Value],
    ) -> Result<mpsc::Receiver<StreamToken>>;

    async fn complete(&self, messages: &[Message]) -> Result<String>;
    fn supports_prefill_warm(&self) -> bool { false }
}
```

Then replace `LlamaClient` in `run_pipeline` / `run_text_pipeline` signatures with
`Arc<dyn LlmProvider>`.

---

## Data Flow

### Normal turn

```
SpeechStart
  ├─ pre-roll flushed into speech_buffer
  ├─ SttStream.submit(snapshot) → seq = N   ← Whisper starts immediately
  └─ barge-in: if pipeline running → cancel=true, abort handle

Speech (every ~500ms)
  └─ SttStream.submit(snapshot) → seq = N+k  ← keeps Whisper current

SpeechEnd
  ├─ SttStream.submit(final_audio) → min_seq  (or re-use last Silence seq)
  ├─ cancel old handle, sleep 25ms, cancel=false
  └─ spawn run_pipeline(min_seq, ...)

run_pipeline
  ├─ await_result(min_seq)                   ← Whisper result (usually already ready)
  ├─ llm_client.stream(messages, tools)      ← streaming SSE
  ├─ stream_and_tts(token_rx, ...)
  │   ├─ SentenceSplitter buffers tokens
  │   ├─ on sentence boundary → spawn synthesize(sentence N)
  │   ├─ await previous play_blocking
  │   └─ spawn play_blocking(samples N)      ← CPAL output
  ├─ db.save_message(User + Assistant)
  └─ spawn maybe_summarize()
```

### Barge-in

When `SpeechStart` fires while a pipeline is playing:
1. `cancel.store(true)` — CPAL callback sees this and stops playback
2. `pipeline_handle.abort()` — cancels the async task wrapper
3. `cancel.store(false)` in `SpeechEnd` after a 25ms grace period

> Note: `spawn_blocking` threads cannot be preempted by `abort()`. They check
> `cancel` on each CPAL callback (~10ms period) and stop themselves.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Single binary, tokio channels | No IPC overhead; easy to reason about cancellation |
| Always-on Whisper (`SttStream`) | Transcript is ready ~0ms after SpeechEnd |
| During-speech KV-cache prefill (`prefill_cache`) | llama.cpp: prefills partial user transcript during speech; cache warm by SpeechEnd |
| Sentence-by-sentence TTS | First word heard in <1s; synthesis of N overlaps playback of N-1 |
| `TtsEngine` enum | Zero-cost dispatch; each variant owns its state; easy to add variants |
| SQLite for history | Survives restarts; restored to `LlmSession` on startup |
| `cancel: Arc<AtomicBool>` | Single barge-in flag shared between async tasks and blocking threads |

---

## Configuration (env vars)

All config is loaded from environment variables (`.env` file supported via `dotenvy`).

| Variable | Default | Description |
|----------|---------|-------------|
| `AUDIO_SAMPLE_RATE` | `16000` | Microphone sample rate |
| `AUDIO_DEVICE` | — | Substring match for input device name |
| `AUDIO_OUTPUT_DEVICE` | — | Substring match for output device name |
| `VAD_SILENCE_MS` | `1500` | ms of silence before SpeechEnd fires |
| `VOICEBOT_LANGUAGE` | `es` | `es` or `en` — Whisper hint + TTS voice selection |
| `WHISPER_MODEL` | `models/ggml-large-v3-turbo.bin` | Path to GGML model |
| `LLM_URL` | `http://localhost:8080` | OpenAI-compatible endpoint base URL |
| `LLM_MODEL` | — | Model name sent in API requests |
| `LLM_PROVIDER` | `llama` | `llama` (llama.cpp) or `mlx` (mlx-lm) |
| `LLM_SLOT_ID` | `0` | KV-cache slot (llama.cpp only) |
| `LLM_MAX_TOKENS` | `400` | Max tokens per response |
| `LLM_TEMPERATURE` | `0.7` | Sampling temperature |
| `LLM_SYSTEM_PROMPT` | — | System prompt text |
| `LLM_CONTEXT_TOKENS` | `4096` | Triggers summarisation above 75% |
| `LLM_SUMMARY_KEEP_TURNS` | `6` | Verbatim turns kept after summarisation |
| `TTS_PROVIDER` | `say` | `say` or `kokoro` (requires `--features kokoro`) |
| `SAY_VOICE` | `Marisol (Enhanced)` | macOS voice name; list with `say -v ?` |
| `KOKORO_MODEL` | — | Path to `kokoro-v1.0.onnx` |
| `KOKORO_VOICES` | — | Path to `voices-v1.0.bin` |
| `KOKORO_VOICE` | `af_bella` | Voice style name |
| `KOKORO_LANGUAGE` | `en-us` | BCP-47 language for espeak-ng |
| `DB_PATH` | `voicebot.db` | SQLite database file |
| `SHELL_ENABLED` | `0` | `1` to enable the `run_shell` tool |
| `VISION_URL` | — | Enable screenshot tool; base URL of vision model |
| `AGENT_COMMAND` | — | External agent CLI command (enables `run_agent_async`) |
| `DAEMON_ENABLED` | `0` | `1` to enable background inference daemon |
| `SPEAKER_MODEL` | — | Path to speaker embedding ONNX model |

---

## Adding a New Provider — Checklist

### New TTS backend

- [ ] `src/tts/<name>.rs` — struct with `synthesize(&self, text: &str) -> Result<Vec<f32>>` and `sample_rate()`
- [ ] Add variant to `TtsEngine` enum in `src/tts/mod.rs`; add its two `match` arms
- [ ] Add branch to `match config.tts_provider.as_str()` in `main.rs`
- [ ] Add `TTS_PROVIDER=<name>` and any new vars to `readme.md`

### New STT backend

- [ ] `src/stt/<name>.rs` — struct with `transcribe(&self, audio: &[f32]) -> Result<String>` (input: 16kHz mono f32)
- [ ] Introduce `SttTranscriber` trait in `src/stt/mod.rs` if not already present
- [ ] Update `SttStream::new` to accept `Arc<dyn SttTranscriber>`
- [ ] Update `main.rs` construction block; add `STT_PROVIDER` env var

### New LLM backend (OpenAI-compatible)

- [ ] Just set `LLM_URL`, `LLM_MODEL`, `LLM_PROVIDER` in `.env` — `LlamaClient` handles it
- [ ] If the server needs extra request fields, add them in the `else` branch of `LlamaClient::stream`

### New LLM backend (non-OpenAI)

- [ ] Extract `LlmProvider` trait from `LlamaClient` (see the trait sketch in the LLM section above)
- [ ] `src/llm/<name>.rs` — implement `stream` and `complete`
- [ ] Change `run_pipeline` / `run_text_pipeline` / `daemon.rs` signatures to `Arc<dyn LlmProvider>`

---

## Log Targets

All logging uses [`tracing`](https://docs.rs/tracing). Filter targets independently with `RUST_LOG`.

| Target | Level(s) used | What it covers |
|--------|--------------|----------------|
| `voicebot` | info | Startup, config summary, shutdown |
| `audio` | info, debug, trace | Device selection, CPAL stream events, resampling, VAD threshold |
| `pipeline` | info, debug | Per-utterance lifecycle: SpeechEnd → STT → LLM → TTS → commit; barge-in; tool calls; cancellation; `[pipe=N]` run IDs |
| `performance` | info, debug | End-to-end latency milestones: `[+Xms] SpeechStart`, `STT wait`, `STT ready`, `LLM request`, `LLM first token`, `TTS start`, `TTS end`, `LLM TG` (total generation) |
| `stt` | info, debug | Whisper model load, per-submission timing (`seq=N`, duration, transcript) |
| `llm` | info, debug | LLM endpoint, streaming errors, speculative prefill, summarisation |
| `tts` | info, debug | Per-sentence synthesis (`[pipe=N] TTS sentence K: "…"`), `play_blocking start/done` with sample counts, WAV header debug |
| `db` | info, warn | Session restore/create, message save failures, summary persistence |
| `speaker` | info, debug, warn | Enrollment, similarity scores, auto-enroll |
| `daemon` | info, debug, warn | Background inference daemon ticks, proactive messages |
| `profile` | info, debug, warn | Profile fact extraction and injection |

### Useful `RUST_LOG` recipes

```bash
# Default — only info and above for all targets
RUST_LOG=info cargo run

# Full pipeline debug without noisy audio/VAD trace
RUST_LOG=info,pipeline=debug,llm=debug,tts=debug cargo run

# Latency profiling only
RUST_LOG=warn,performance=info cargo run

# TTS sentence trace (see every sentence sent to synthesizer and play_blocking timings)
RUST_LOG=info,tts=debug cargo run

# STT timing (see per-submission Whisper timings)
RUST_LOG=info,stt=debug cargo run

# Everything — very verbose
RUST_LOG=debug cargo run

# Silence whisper.cpp internal logs (always on via install_logging_hooks())
# — no env-var needed; handled in code
```

---

## Testing

```bash
cargo test                          # all unit + integration tests
cargo test -p voicebot <name>       # single test
RUST_LOG=debug cargo run            # verbose pipeline logs
RUST_LOG=info,tts=debug cargo run   # TTS-only debug (sentence + play_blocking trace)
```

Key test patterns:
- **VAD / buffer**: synthetic sine waves + silence
- **SentenceSplitter**: pure unit tests, no I/O
- **LLM client**: `wiremock` mock server for SSE response tests
- **Pipeline (e2e)**: `SttStream::mock(transcript)` + `TtsEngine::Mock` → no real audio hardware needed
- **Whisper tests**: require model file; tagged `#[ignore]` in CI
