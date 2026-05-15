# Voicebot: async architecture

This file explains the runtime architecture of the Voicebot application.
It describes the async task model, the per-utterance pipeline flow, the state
machine, and how tasks communicate via tokio channels.

## Runtime Model

The entire application runs on a **single tokio async runtime**. There are no
dedicated threads for pipeline stages. Each stage is a `tokio::spawn`
task that communicates through typed channels (`mpsc`, `broadcast`, `watch`).

On startup, `main()` initializes all subsystems, then spawns the following:
- **Audio capture task** on a dedicated worker thread (CPAL requires blocking I/O).
- **Per-utterance pipeline tasks**: `llm_task`, `sen_task`, `tts_task`,
  `consolidation_task` — spawned once and run for the lifetime of the process.
- **InferenceDaemon** (`src/daemon.rs`) — background "is there anything worth
  saying?" loop.
- **EyesDaemon** (`src/eyes.rs`) — periodic visual awareness via screenshots
  and a secondary vision LLM.
- **Control API** (`src/control/`) — HTTP+SSE server via axum for remote
  monitoring and control.

## The Per-Utterance Pipeline Flow

```
CPAL audio → stt_task (WhisperSTTVAD)
  on SpeechEnd
  ↓
  (PipelineFrame::TranscriptReady)
  ↓
llm_task → OpenAIClient::stream()
  emits tokens (PipelineFrame::LLMToken)
  ↓
sen_task → SentenceSplitter
  emits sentences (PipelineFrame::SentenceReady)
  ↓
tts_task → TtsEngine.synthesize() → AudioOutput
  emits (PipelineFrame::PlaybackDone)
```

Each actor sits in a `tokio::select!` loop listening for frames on an
`mpsc::Receiver` and a `broadcast::Receiver` for barge-in cancellation.

### PipelineFrame

All inter-stage messages flow through `PipelineFrame` (defined in
`src/pipeline/frames.rs`), a typed enum that carries an `utterance_id` for
log correlation and latency tracking:

| Variant | From → To | Meaning |
|---------|-----------|---------|
| `TranscriptReady` | stt → llm_task | STT finished, transcript ready |
| `TextInput` | TUI → llm_task | User typed input |
| `SystemNotification` | background → llm_task | Proactive/injected system turn |
| `LLMToken` | llm_task → sen_task | One streamed LLM token |
| `LLMResponseDone` | llm_task → sen_task | LLM stream finished |
| `SentenceReady` | sen_task → tts_task | Complete sentence for TTS |
| `PlaybackDone` | tts_task → pipeline | Last sentence for a turn finished |

## FSM: PipelineState

The pipeline state machine lives in `src/pipeline/fsm.rs` and is shared
through a `tokio::sync::watch` channel. Observers (TUI, logger, control API)
subscribe to the receiver.

| State | Meaning |
|---------|---------|
| `Idle` | No active utterance |
| `Listening { utterance_id }` | VAD detected speech; STT accumulating |
| `Thinking { utterance_id }` | Transcript ready; LLM generating response |
| `Speaking { utterance_id }` | LLM done; TTS playing response |
| `Paused { reason }` | Temporarily paused (e.g., consolidation running) |

Each actor writes its own state transition directly. No central coordinator
sits on the hot path.

## Actor Descriptions

### STT (WhisperSTTVAD)

File: `src/stt/mod.rs`

This is a unified STT+VAD component powered by `whisper-cpp-plus`. It
receives raw 16kHz f32 mono audio chunks from the CPAL capture task, runs
Silero VAD on 200ms probe windows, and manages a speech/silence state machine.

- On `SpeechStart`: fires `SpeechEvent::SpeechStart`, begins accumulating
  audio buffer (with 300ms pre-roll).
- On `SpeechEnd`: transcribes the accumulated buffer via Whisper and emits
  `SpeechEvent::SpeechEnd(transcript)`.
- On `Silence`: resets internal buffers.
- Hard cap of 20 seconds per speech segment before forcing a cut.

Events are forwarded to the pipeline through `mpsc::Sender<SpeechEvent>`.
The main loop converts `SpeechEnd` into `PipelineFrame::TranscriptReady`.

### llm_task

File: `src/pipeline/llm_task.rs`

Manages the LLM conversation loop. Receives transcript frames and typed/text
input, sends the full conversation history to the LLM, handles streaming,
tool calls, and response persistence.

- Blocks on `transcript_rx` until a transcript, text input, or system
  notification arrives.
- Sends `PipelineState::Thinking`.
- Calls `OpenAIClient::stream()` to get a `StreamToken` channel.
- Forwards each text token to `sen_task` via `llm_tx`.
- On `ToolCall`: executes the tool synchronously or as a background task,
  appends the result to conversation, and re-invokes the LLM (up to
  `MAX_TOOL_ITERATIONS = 5` sequential tool calls per turn).
- On stream finish: saves response to DB, sends `LLMResponseDone` to
  `sen_task`, notifies consolidation task via `llm_post_finished`,
  transitions to `PipelineState::Speaking`.
- Cancels all work on barge-in (user detected speaking) via `barge_in_tx`.

### sen_task

File: `src/pipeline/sen_task.rs`

Sentence splitter. Receives LLM tokens, buffers them internally until a
complete sentence boundary is detected (`.` `!` `?` `;` `:`), then forwards
the complete sentence to `tts_task`.

- On `LLMResponseDone`: flushes any remaining buffered text as a final
  sentence.
- Cancels on barge-in.

### tts_task

File: `src/pipeline/tts_task.rs`

Receives sentences, synthesizes audio, plays it through the audio output.
Handles concurrent play-and-generate: while sentence N plays, sentence N+1
may already be in the queue.

- Uses `AudioOutput` for playback (CPAL-based).
- Forwards audio packets to remote WebSocket clients if `remote` feature is
  enabled.
- Broadcasts TTS state to control API if `control` feature is enabled.
- Sends `PipelineFrame::PlaybackDone` for the last sentence, which causes
  the pipeline to transition to `Idle`.
- Cancels active playback on barge-in.

### consolidation_task

File: `src/pipeline/consolidation.rs`

Background context management task. Runs after each LLM response completes
when context fills beyond `LLM_CONSOLIDATION_THRESHOLD_PCT` of
`LLM_CONTEXT_TOKENS`.

- Extracts user profile facts from conversation history.
- Extracts persistent memories.
- Summarizes old conversation turns while keeping the most recent
  `LLM_SUMMARY_KEEP_TURNS` turns verbatim.
- Rebuilds the system prompt with `[USER PROFILE]` and `[MEMORIES]` blocks.
- Pauses the pipeline briefly during consolidation.

Additionally, idle consolidation runs on a configurable timer
(`LLM_IDLE_CONSOLIDATION_SECS`) to proactively manage context while the
user is inactive.

## Signals and Cancellation

### Barge-in

The barge-in mechanism uses `tokio::sync::broadcast`. When VAD detects
`SpeechStart`, it sends the new `utterance_id` through `barge_in_tx`. All
active tasks receive this and:
- **llm_task**: cancels the LLM HTTP stream.
- **sen_task**: clears buffered text.
- **tts_task**: stops active audio playback.

After cancellation, the pipeline returns to `Idle`, enters `Listening` for
the new utterance, and spawns a fresh pipeline flow.

### PipelineEvents

Defined in `src/pipeline/state.rs`:

| Signal | Type | Purpose |
|--------|------|---------|
| `barge_in_tx` | `broadcast::Sender<u64>` | VAD SpeechStart — cancel all active work |
| `llm_post_finished` | `Arc<Notify>` | LLM stream ended — used by consolidation task |

## Background Daemons

### InferenceDaemon

File: `src/daemon.rs`

Every `DAEMON_INTERVAL_SECS` (default 300s) asks the LLM via
`complete_short()` whether there is anything worth telling the user.
If the response is not `NOTHING`, pushes a `ProactiveEvent::InferenceDaemon`
to the proactive channel so the pipeline can vocalize it.
Runs as a `tokio::spawn` background task.

### EyesDaemon

File: `src/eyes.rs`

Every `EYES_INTERVAL_SECS` captures a screenshot, sends it to the secondary
vision LLM, and asks whether anything on screen warrants notifying the user.
The vision LLM responds with `warn_user: true|false` and an optional message.
Positive hits become `ProactiveEvent::AgentResult` events.
Runs as a `tokio::spawn` background task.

## OpenAIClient

File: `src/llm/client.rs`

HTTP client for OpenAI-compatible endpoints (`/v1/chat/completions`).

| Method | Purpose |
|--------|---------|
| `stream()` | SSE streaming completion. Returns `(Receiver<StreamToken>, JoinHandle)`. |
| `complete()` | Non-streaming completion. Returns full text. |
| `complete_short()` | Lightweight non-streaming call for background tasks (daemon, eyes). |

Also handles stripping `<think>...</think>` blocks from reasoning model
output and supports optional API key authentication.

## TTS Backends

File: `src/tts/mod.rs`

TTS is unified behind the `TtsEngine` enum. Backend selected via
`TTS_PROVIDER` environment variable.

| Backend | Feature | Config | Details |
|---------|---------|--------|---------|
| `avspeech` | `--features avspeech` | `AVSPEECH_VOICE`, `AVSPEECH_RATE` | Native macOS `AVSpeechSynthesizer` (default) |
| `kokoro` | `--features kokoro` | `KOKORO_MODEL`, `KOKORO_VOICE`, `KOKORO_LANGUAGE` | Kokoro ONNX model |

All variants expose the same `synthesize(&str) -> Result<Vec<f32>>` interface
so the pipeline is backend-agnostic.

## Tools

File: `src/tools/mod.rs`

Tools implement the `Tool` trait (`name()`, `description()`, `parameters()`,
`run(args: &str)`) and are registered in a `ToolRegistry`. The registry
exposes OpenAI function-calling schema for the system prompt.

| Tool name | Module | Description |
|-----------|--------|-------------|
| `current_time` | `current_time.rs` | Current date and time |
| `read_file` | `read_file.rs` | Read contents of a local file |
| `read_clipboard` | `clipboard.rs` | Read clipboard contents |
| `set_clipboard` | `clipboard.rs` | Write to clipboard |
| `open_app` | `open_app.rs` | Open an application by name |
| `run_shell` | `run_shell.rs` | Execute a shell command |
| `run_agent` | `run_agent.rs` | Delegate a task to a secondary agent (Hermes) |
| `take_screenshot` | `take_screenshot.rs` | Capture and save a screenshot |
| `web_search` | `web_search.rs` | Search the web via SearXNG |
| `set_conversation_mode` | `conversation_mode.rs` | Switch conversation mode (casual, developer, creative) |
| `mcp_tool` | `mcp_tool.rs` | Proxy call to external MCP tools |

Tools can be marked `is_background()` to run asynchronously without blocking
the LLM turn. Results are delivered via `ProactiveEvent`.

## Control API

File: `src/control/`

Optional HTTP+SSE server (enabled via `control` feature) built on axum.
Provides real-time pipeline state, transcript streaming, and remote control
commands. Broadcasts events through `ControlBroadcast`.

## Database

File: `src/db/`

SQLite persistence via `sqlx`. Stores sessions, messages, user profile facts,
and memories. Migration-first schema in `src/db/migrations/`.

## Module Map

| Directory | Purpose |
|-----------|---------|
| `src/main.rs` | Entry point, initialization, task spawning |
| `src/lib.rs` | Library root, module re-exports |
| `src/config.rs` | Environment-based configuration (`Config::from_env()`) |
| `src/pipeline/` | FSM, per-utterance tasks, frames, consolidation |
| `src/audio/` | CPAL capture, audio output, resampling |
| `src/stt/` | WhisperSTTVAD (whisper-cpp-plus + Silero VAD) |
| `src/llm/` | OpenAIClient, LlmSession, manager |
| `src/tts/` | TtsEngine (avspeech, kokoro), SentenceSplitter |
| `src/tools/` | Tool implementations and registry |
| `src/agents/` | ACP protocol for agent delegation |
| `src/db/` | SQLite database layer (sessions, messages, profile, memories) |
| `src/memory/` | Memory extraction from conversation |
| `src/profile/` | User profile fact extraction |
| `src/daemon.rs` | InferenceDaemon background loop |
| `src/eyes.rs` | EyesDaemon visual awareness loop |
| `src/control/` | HTTP+SSE control API (axum) |
| `src/remote/` | WebSocket server for remote audio streaming |
| `src/tui/` | Terminal UI (ratatui) |
| `src/analysis/` | Context analysis utilities |
| `src/mcp/` | MCP (Model Context Protocol) client integration |
| `src/e2e_tests.rs` | End-to-end integration tests |
