# Voicebot Architecture

## Overview

Voicebot is a mono-user voice AI assistant in Rust. It runs as a single binary using a
**streaming STT → LLM → TTS pipeline** where every stage is connected by `tokio` channels.
There is no inter-service communication; everything runs in-process.

```
Microphone → AudioCapture (CPAL)
           → SttProvider (trait: WhisperSttProvider or ParakeetSttProvider)
             → Silero VAD (shared across providers)
             → Whisper transcription (whisper-cpp-plus) OR Parakeet TDT (ONNX)
           → LLM (OpenAI-compatible SSE streaming)
           → SentenceSplitter    ← fires per sentence as tokens arrive
           → TTS synthesizer     ← AvSpeech or Kokoro; sentence N overlaps playback of N-1
           → AudioOutput (CPAL)
```

All GPU/CPU-heavy work runs in `tokio::task::spawn_blocking` threads so the async event
loop stays unblocked.

---

## Project Structure

```
src/
├── main.rs                    # Pipeline orchestration, pipeline tasks, VAD loop
├── lib.rs                     # Library exports
├── config.rs                  # All config (env-var driven)
├── daemon.rs                  # InferenceDaemon — proactive suggestions
├── eyes.rs                    # EYES visual awareness daemon
├── e2e_tests.rs               # End-to-end tests
│
├── audio/
│   ├── mod.rs
│   ├── audio_capture.rs       # CPAL mic input
│   ├── audio_transform.rs     # Rubato resampling
│   ├── buffer.rs              # Circular VecDeque buffer
│   ├── output.rs              # CPAL speaker playback
│   ├── ambient_buffer.rs      # Ambient speech context buffer
│   └── speaker.rs             # Speaker verification (ONNX)
│
├── stt/
│   ├── mod.rs                 # SpeechEvent enum, module re-exports
│   ├── provider.rs            # SttProvider trait + create_provider() factory
│   ├── whisper.rs             # WhisperSttProvider (whisper-cpp-plus)
│   └── parakeet.rs            # ParakeetSttProvider (ONNX, --features parakeet)
│
├── llm/
│   ├── mod.rs                 # OpenAIClient, LlmSession, Message
│   ├── client.rs              # OpenAIClient — OpenAI-compatible streaming SSE client
│   ├── manager.rs             # LLM client management
│   └── session.rs             # LlmSession — message history
│
├── tts/
│   ├── mod.rs                 # TtsEngine enum (AvSpeech | Kokoro | Mock)
│   ├── avspeech.rs            # AvSpeechTts — macOS AVSpeechSynthesizer (native)
│   ├── kokoro.rs              # KokoroTts — ONNX model (--features kokoro)
│   ├── piper.rs               # PiperTts — reference only, not integrated
│   └── sentence.rs            # SentenceSplitter — buffers tokens; emits on punctuation
│
├── pipeline/
│   ├── mod.rs
│   ├── fsm.rs                 # PipelineState FSM (Idle/Listening/Thinking/Speaking/Paused)
│   ├── state.rs               # PipelineEvents channel types
│   ├── frames.rs              # PipelineFrame — utterance data carrier
│   ├── llm_task.rs            # LLM streaming task (per utterance)
│   ├── sen_task.rs            # Sentence splitting task
│   ├── tts_task.rs            # TTS synthesis + playback task
│   └── consolidation.rs       # Context consolidation (summarization)
│
├── tools/                     # Tool registry + individual tool implementations
│   ├── mod.rs                 # ToolRegistry, Tool trait
│   ├── current_time.rs
│   ├── clipboard.rs           # ReadClipboardTool, SetClipboardTool
│   ├── read_file.rs
│   ├── open_app.rs
│   ├── run_shell.rs           # SHELL_ENABLED=1
│   ├── take_screenshot.rs     # SECONDARY_LLM_URL for vision
│   ├── run_agent.rs           # Agent delegation (ACP/CLI)
│   ├── web_search.rs          # SearXNG web search
│   ├── conversation_mode.rs   # Active/Ambient mode switching
│   └── mcp_tool.rs            # MCP tool proxy
│
├── control/                   # HTTP + SSE Control API
│   ├── mod.rs
│   ├── api.rs                 # axum routes: state, events, history, mute, barge_in, input
│   ├── state.rs               # ControlState — shared mutable state for the API
│   └── broadcast.rs           # ControlEvent enum + broadcast channel
│
├── db/
│   ├── mod.rs
│   └── database.rs            # SQLite: sessions, messages, memories, user_profile
│
├── memory/                    # Memory extraction from conversations
├── profile/                   # User profile fact extraction
├── agents/                    # ACP protocol agent delegation
├── mcp/                       # MCP stdio protocol
├── analysis/                  # Identity analyzer, ContextLens
├── remote/                    # WebSocket server
├── tui/                       # Terminal UI (ratatui)
└── bin/acp_agent_chat.rs      # Debug ACP agent chat
```

---

## Provider Interfaces

The pipeline uses three pluggable provider layers: **STT**, **LLM**, and **TTS**.
Each is designed so the rest of the pipeline is completely backend-agnostic.

### TTS — `TtsEngine` (enum dispatch)

Location: `src/tts/mod.rs`

```rust
pub enum TtsEngine {
    #[cfg(feature = "avspeech")]
    AvSpeech(AvSpeechTts),         // macOS AVSpeechSynthesizer — default
    #[cfg(feature = "kokoro")]
    Kokoro(KokoroTts),             // ONNX model
    #[cfg(test)]
    Mock(MockTts),                 // test only
}

impl TtsEngine {
    pub fn synthesize(&self, text: &str) -> Result<Vec<f32>>;
    pub fn sample_rate(&self) -> u32;
}
```

TTS is selected at startup via `TTS_PROVIDER` env var. AvSpeech is the default on
macOS and requires `--features avspeech`. Kokoro requires `--features kokoro` plus
`espeak-ng` installed.

`piper.rs` exists as a Piper subprocess wrapper but is NOT integrated into `TtsEngine`.
All its public methods are marked `#[allow(dead_code)]`. It is kept for reference.

| Backend | Provider key | Feature flag | Notes |
|---------|-------------|--------------|-------|
| AvSpeech | `avspeech` | `avspeech` | default; macOS native AVSpeechSynthesizer |
| Kokoro ONNX | `kokoro` | `kokoro` | offline, high quality |
| Piper | — | — | `piper.rs` exists but is not integrated |

---

### STT — Provider Architecture

Location: `src/stt/`

The STT layer uses a provider trait with two implementations. Both share the same Silero VAD
state machine; only the transcription backend differs.

```rust
#[async_trait]
pub trait SttProvider: Send {
    fn provider_name(&self) -> &'static str;
    async fn process_audio(&mut self, audio: &[f32], tx: &mpsc::Sender<SpeechEvent>) -> Result<()>;
    fn transcribe_complete(&self, audio: &[f32]) -> Result<String>;
}

pub enum SpeechEvent {
    SpeechStart,
    Speech(String),
    SpeechEnd(String),
}
```

**Providers:**
- `WhisperSttProvider` (`src/stt/whisper.rs`) — whisper-cpp-plus, 99 languages
- `ParakeetSttProvider` (`src/stt/parakeet.rs`) — ParakeetTDT ONNX, 25 languages, `--features parakeet`

Factory: `create_provider(config: &Config) -> Result<Box<dyn SttProvider>>`

#### `WhisperSttProvider`

Built on whisper-cpp-plus. Processes audio chunks via an async tokio channel.

```rust
pub struct WhisperSttProvider {
    ctx: Arc<WhisperContext>,
    vad: WhisperVadProcessor,       // Silero VAD from whisper-cpp-plus
    // ... state machine fields
}
```

`WhisperSTTVADConfig` specifies whisper model path, VAD model path (`ggml-silero-vad.bin`),
language, and silence threshold in milliseconds. Defaults to 500ms silence.

The VAD runs Silero on 200ms probe windows with a configurable probability threshold
(0.5). A 300ms pre-roll buffer prevents clipping the first phoneme. 20s hard cap on
any single speech segment.

`src/stt/whisper.rs` contains legacy `WhisperStt` based on whisper-rs. It is deprecated
and maintained for backwards compatibility only.

---

### LLM — `OpenAIClient`

Location: `src/llm/client.rs`

```rust
pub struct OpenAIClient { ... }

impl OpenAIClient {
    // Streaming completion — yields StreamToken::Content or StreamToken::ToolCall
    pub async fn stream(
        &self,
        messages: &[serde_json::Value],
        tools: &[serde_json::Value],
    ) -> Result<mpsc::Receiver<StreamToken>>;

    // One-shot completion — used for memory extraction, profile, daemon
    pub async fn complete(&self, messages: &[Message]) -> Result<String>;

    // Lightweight one-shot — daemon and short prompts (no KV-cache pressure)
    pub async fn complete_short(&self, messages: &[Message]) -> Result<String>;
}
```

`OpenAIClient` targets **any OpenAI-compatible endpoint** (mlx-lm, oMLX, OpenAI, Ollama).
Extra sampling params (`repetition_penalty`, `top_k`, `min_p`) are sent unconditionally
on every streaming request for best output quality with mlx-lm and oMLX.

The `ThinkFilter` strips `<antThinking>...</antThinking>` blocks from reasoning model
output before tokens reach TTS or tool detection.

`src/llm/session.rs` contains `LlmSession` for message history management.
`src/llm/manager.rs` handles LLM client lifecycle.

---

## Pipeline FSM

Location: `src/pipeline/`

The pipeline is implemented as a finite state machine with per-utterance tasks.
Each utterance gets a unique `utterance_id` tracked through all states.

```rust
pub enum PipelineState {
    Idle,
    Listening { utterance_id: u64 },
    Thinking { utterance_id: u64 },
    Speaking { utterance_id: u64 },
    Paused { reason: PauseReason },
}
```

State is held in a `watch::Sender<PipelineState>`. Each task that owns a transition
writes directly with no central coordinator on the hot path. Observers (TUI, Control API)
subscribe via `watch::Receiver::changed()`.

Per-utterance tasks:
- `llm_task` — streams tokens from `OpenAIClient`, detects tools, feeds SentenceSplitter
- `sen_task` — token buffering and sentence boundary detection
- `tts_task` — synthesis and playback via `TtsEngine` + `AudioOutput`
- `consolidation_task` — context summarization when history grows too large

`PipelineFrame` carries utterance data (transcript, LLM response, tools used) between tasks.
`PipelineEvents` is the channel type for inter-task communication.

---

## Daemons

### InferenceDaemon (`src/daemon.rs`)

Periodic background loop that asks the LLM if there is something worth telling the user.
Calls `complete_short()` every `interval_secs`. If the LLM returns a non-`NOTHING`
response, pushes a `ProactiveEvent::InferenceDaemon` through the proactive channel.

### EyesDaemon (`src/eyes.rs`)

Periodic visual awareness. Takes a screenshot every `interval_secs`, sends it to a
secondary vision LLM (configured via `SECONDARY_LLM_URL`), and asks whether anything
on screen warrants a user notification. Responses follow a structured format:

```
warn_user: true|false
message: <optional natural-language sentence>
```

When `warn_user` is true, an `AgentResult` proactive event is pushed for the main
assistant LLM to reformulate and vocalize.

---

## Control API

Location: `src/control/`

An HTTP + SSE server (built on axum) for external management of the voicebot.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/control/events` | GET | SSE stream of `ControlEvent` updates |
| `/control/state` | GET | Current `PipelineState`, utterance ID, mute status |
| `/control/history` | GET | Conversation history |
| `/control/mute` | POST | Toggle TTS mute |
| `/control/barge_in` | POST | Trigger barge-in (cancel current pipeline) |
| `/control/input` | POST | Send text input to the pipeline |

`ControlEvent` types include: `StateChanged`, `Transcript`, `LlmToken`, `LlmDone`,
`TtsStart`, `ToolCall`, `MuteChanged`, `Error`.

---

## Data Flow

### Normal turn

```
SpeechStart  (from WhisperSTTVAD.process_audio)
  ├─ pre-roll flushed into speech_buffer
  └─ barge-in: if pipeline running → cancel=true, abort handle

SpeechEnd(transcript)
  ├─ spawn llm_task(transcript, ...)
  │   ├─ OpenAIClient.stream(messages, tools)
  │   ├─ sen_task buffers tokens, emits on punctuation
  │   └─ tts_task.synthesize(sentence) → AudioOutput.play_blocking()
  ├─ db.save_message(User + Assistant)
  └─ spawn maybe_consolidate()
```

### Barge-in

When `SpeechStart` fires while a pipeline is playing:
1. `events.barge_in_tx.send(utterance_id)` — broadcast to all pipeline tasks via `tokio::sync::broadcast`
2. Each task's `cancel_rx.recv()` triggers inside its `tokio::select!` loop
3. **LLM task**: aborts the HTTP stream handle, drops the token receiver, breaks the tool loop
4. **SEN task**: drains remaining tokens from `llm_rx`, resets the `SentenceSplitter`, flushes buffered state
5. **TTS task**: sets `play_cancel.store(true, SeqCst)` so the CPAL callback sees it and stops writing audio, then awaits the `play_handle` until the playback task actually exits before calling `handle_barge_in` to drain queued sentences and reset `play_cancel` back to `false`

> Note: `play_cancel` must stay `true` until the CPAL callback has had time to see it. The
> TTS task now uses a looped `select!` that keeps awaiting the `JoinHandle` after setting
> `play_cancel`, rather than dropping the handle when `cancel_rx` fires. This prevents a
> race where the handle was abandoned but the CPAL callback continued producing audio.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Single binary, tokio channels | No IPC overhead; easy to reason about cancellation |
| Integrated VAD+STT (`WhisperSTTVAD`) | Single struct, shared Whisper context; avoids double-buffering between separate VAD and STT stages |
| Apple MLX backends | mlx-lm and oMLX maintain KV-cache implicitly; substantially faster than llama.cpp on Apple Silicon |
| Sentence-by-sentence TTS | First word heard in <1s; synthesis of sentence N overlaps playback of N-1 |
| `TtsEngine` enum | Zero-cost dispatch; each variant owns its state; easy to add variants |
| Pipeline FSM | Clear state tracking per utterance; enables TUI, Control API, and external observers |
| SQLite for history | Survives restarts; restored to `LlmSession` on startup |
| `play_cancel: Arc<AtomicBool>` | Single barge-in flag shared between async tasks and CPAL blocking playback thread |

---

## Configuration (env vars)

All config is loaded from environment variables (`.env` file supported via `dotenvy`).

| Variable | Default | Description |
|----------|---------|-------------|
| `AUDIO_SAMPLE_RATE` | `16000` | Microphone sample rate |
| `AUDIO_DEVICE` | — | Substring match for input device name |
| `AUDIO_OUTPUT_DEVICE` | — | Substring match for output device name |
| `VOICEBOT_LANGUAGE` | `es` | `es` or `en` — Whisper language + TTS voice |
| `WHISPER_MODEL` | `models/ggml-large-v3-turbo.bin` | Path to GGML model |
| `LLM_URL` | `http://localhost:8000` | OpenAI-compatible endpoint (mlx-lm: 8000; oMLX: 8001) |
| `LLM_MODEL` | — | Model name sent in API requests |
| `LLM_MAX_TOKENS` | `1024` | Max tokens per response |
| `LLM_TEMPERATURE` | `0.7` | Sampling temperature |
| `LLM_SYSTEM_PROMPT` | — | System prompt text |
| `LLM_CONTEXT_TOKENS` | `4096` | Triggers consolidation above 75% |
| `LLM_SUMMARY_KEEP_TURNS` | `6` | Verbatim turns kept after consolidation |
| `TTS_PROVIDER` | `avspeech` | `avspeech` or `kokoro` |
| `AVSPEECH_VOICE` | `Marisol (Enhanced)` | macOS voice name |
| `AVSPEECH_RATE` | `1.0` | Speech rate multiplier |
| `KOKORO_MODEL` | — | Path to `kokoro-v1.0.onnx` |
| `KOKORO_VOICES` | — | Path to `voices-v1.0.bin` |
| `KOKORO_VOICE` | `af_bella` | Voice style name |
| `KOKORO_LANGUAGE` | `en-us` | BCP-47 language for espeak-ng |
| `DB_PATH` | `voicebot.db` | SQLite database file |
| `SHELL_ENABLED` | `0` | `1` to enable the `run_shell` tool |
| `AGENT_COMMAND` | — | External agent CLI command (enables `run_agent_async`) |
| `DAEMON_ENABLED` | `0` | `1` to enable InferenceDaemon |
| `SPEAKER_MODEL` | — | Path to speaker embedding ONNX model |

---

## Adding a New Provider — Checklist

### New TTS backend

- [ ] `src/tts/<name>.rs` — struct with `synthesize(&self, text: &str) -> Result<Vec<f32>>` and `sample_rate()`
- [ ] Add variant to `TtsEngine` enum in `src/tts/mod.rs`; add its two `match` arms
- [ ] Add branch to `match config.tts_provider.as_str()` in `main.rs`
- [ ] Add `TTS_PROVIDER=<name>` and any new vars to `readme.md`

### New STT backend

- [ ] `src/stt/<name>.rs` — struct that implements `process_audio` with VAD+transcription integration
- [ ] Replace `WhisperSTTVAD` in `main.rs` construction block
- [ ] Update config for new model paths and params

### New LLM backend (OpenAI-compatible)

- [ ] Just set `LLM_URL`, `LLM_MODEL`, `LLM_PROVIDER` in `.env` — `OpenAIClient` handles it
- [ ] If the server needs extra request fields, add them in `OpenAIClient::stream`

### New LLM backend (non-OpenAI)

- [ ] Extract `LlmProvider` trait from `OpenAIClient`
- [ ] `src/llm/<name>.rs` — implement `stream` and `complete`
- [ ] Change pipeline/daemon signatures to `Arc<dyn LlmProvider>`

---

## Log Targets

All logging uses [`tracing`](https://docs.rs/tracing). Filter targets independently with `RUST_LOG`.

| Target | Level(s) used | What it covers |
|--------|--------------|----------------|
| `voicebot` | info | Startup, config summary, shutdown |
| `audio` | info, debug, trace | Device selection, CPAL stream events, resampling |
| `pipeline` | info, debug | Per-utterance lifecycle; barge-in; tool calls; cancellation; `[pipe=N]` run IDs |
| `performance` | info, debug | End-to-end latency milestones |
| `sttvad` | info, debug, trace | WhisperSTTVAD events: SpeechStart/SpeechEnd, VAD threshold, transcribe timing |
| `llm` | info, debug | LLM endpoint, streaming errors, speculative prefill, consolidation |
| `tts` | info, debug | Per-sentence synthesis, `play_blocking` start/done, WAV header debug |
| `db` | info, warn | Session restore/create, message save, memory persistence |
| `speaker` | info, debug, warn | Enrollment, similarity scores, auto-enroll |
| `daemon` | info, debug, warn | Inference daemon ticks, proactive messages |
| `eyes` | info, debug, warn | EYES ticks, screenshot failures, vision LLM calls |
| `profile` | info, debug, warn | Profile fact extraction and injection |
| `control` | info | Control API startup, endpoint hits |

### Useful `RUST_LOG` recipes

```bash
# Default — only info and above for all targets
RUST_LOG=info cargo run

# Full pipeline debug without noisy audio trace
RUST_LOG=info,pipeline=debug,llm=debug,tts=debug cargo run

# Latency profiling only
RUST_LOG=warn,performance=info cargo run

# TTS sentence trace
RUST_LOG=info,tts=debug cargo run

# STT+VAD timing
RUST_LOG=info,sttvad=debug cargo run

# Everything — very verbose
RUST_LOG=debug cargo run
```

---

## Testing

```bash
cargo test                          # all unit + integration tests
cargo test -p voicebot <name>       # single test
RUST_LOG=debug cargo run            # verbose pipeline logs
RUST_LOG=info,tts=debug cargo run   # TTS-only debug
```

Key test patterns:
- **VAD / buffer**: synthetic sine waves + silence
- **SentenceSplitter**: pure unit tests, no I/O
- **LLM client**: `wiremock` mock server for SSE response tests
- **Pipeline (e2e)**: `TtsEngine::Mock` with mock transcripts — no real audio needed
- **Whisper tests**: require model file; tagged `#[ignore]` in CI
