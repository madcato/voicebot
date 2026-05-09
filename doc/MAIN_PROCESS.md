# Main Process — Complete Flow

This document describes the full lifecycle of `async_main()`: how the process starts up, which tasks are spawned, and how audio flows through the pipeline from microphone to speaker.

---

## Startup Sequence

```
async_main()
  │
  ├─ Init tracing (to voicebot.log if TUI enabled)
  ├─ Load .env (dotenvy)
  ├─ Load config (Config::from_env)
  ├─ Device/voice listing shortcuts (--list-devices, --list-voices)
  ├─ Create proactive event channel
  ├─ Init secondary LLM client (OpenAIClient — optional, vision/summarization)
  ├─ Register tools (CurrentTime, ReadClipboard, SetClipboard, OpenApp,
  │   SetConversationMode, RunShell [conditional],
  │   TakeScreenshot [requires secondary LLM],
  │   WebSearch [conditional + optional secondary LLM synthesis],
  │   RunAgent [ACP mode or agent_command set],
  │   MCP tools [if mcp_command configured])
  ├─ ACP pre-warm (optional — spawns Hermes agent, runs session/new handshake)
  ├─ MCP server init (optional — spawns, discovers tool definitions, registers McpToolProxy)
  ├─ Init DB (SQLite) — load session, history, profile facts, memories
  ├─ Build system prompt (base + [USER PROFILE] + [MEMORIES] + tools)
  ├─ Init LLM session (LlmSession::from_history)
  ├─ Self-managed LLM process (optional — start + supervise if llm_self_managed)
  ├─ Init primary LLM client (OpenAIClient → /v1/chat/completions)
  ├─ InferenceDaemon (optional — background reasoning at fixed interval)
  ├─ EyesDaemon (optional — screenshots → secondary LLM at fixed interval)
  ├─ Init WhisperSTTVAD (unified STT + VAD via whisper-cpp-plus, Silero VAD)
  ├─ ContextLens + IdentityAnalyzer (speaker verification via sherpa_onnx)
  ├─ AmbientBuffer (ambient context storage)
  ├─ Init TTS (TtsEngine: avspeech/AVSpeechSynthesizer or kokoro/ONNX)
  ├─ Init AudioOutput (CPAL)
  ├─ Init AudioCapture (CPAL, bounded channel 200 slots)
  ├─ Pipeline FSM state (watch channel: PipelineState)
  ├─ Spawn permanent pipeline tasks: llm_task, sen_task, tts_task, consolidation_task
  ├─ Spawn FSM observer (logs state transitions + control broadcast)
  ├─ Spawn TUI (feature flag)
  ├─ Spawn Remote WebSocket server (feature flag)
  ├─ Spawn Control API HTTP+SSE server (feature flag)
  ├─ Startup consolidation (if context already exceeds idle threshold)
  └─ Startup greeting → transcript_tx(SystemNotification) → triggers LLM → speech out
```

---

## Spawned Tasks

| Task | Module | Trigger to start | Blocked on |
|------|--------|-----------------|------------|
| `llm_task` | `pipeline/llm_task.rs` | spawned at boot | `transcript_rx` (mpsc channel) |
| `sen_task` | `pipeline/sen_task.rs` | spawned at boot | `llm_tx` (mpsc channel from llm_task) |
| `tts_task` | `pipeline/tts_task.rs` | spawned at boot | `sentences_rx` (mpsc channel from sen_task/llm_task) |
| `consolidation_task` | `pipeline/consolidation.rs` | spawned at boot | timer + LLM completion detection via FSM |
| `InferenceDaemon` | `daemon.rs` | spawned at boot (optional) | interval timer |
| `EyesDaemon` | `eyes.rs` | spawned at boot (optional) | interval timer |
| `TUI` | `tui/` | spawned at boot (feature flag) | TUI events |
| `Remote WS server` | `remote/` | spawned at boot (feature flag) | WebSocket connections |
| `Control API server` | `control/` | spawned at boot (feature flag) | HTTP+SSE connections |
| `ACP pre-warm` | inline `tokio::spawn` | if agent_mode="acp" | Hermes spawn + init |
| FSM observer | inline `tokio::spawn` | spawned at boot | `pipeline_state_rx` (watch channel) |

---

## Pipeline Architecture

The pipeline uses a **channel-based architecture** with four permanent tasks connected by async mpsc channels for per-utterance flow. An FSM (`PipelineState` via watch channel) tracks pipeline-wide state transitions.

```
transcript_tx ──► llm_task ──► llm_tx ──► sen_task ──► sentences_tx ──► tts_task ──► speakers
(transcript_rx)     (llm_rx)                   (sentences_rx)
```

Communication primitives:
- **mpsc channels**: transcript → llm → sen → tts (per-utterance data)
- **watch channel**: `PipelineState` (FSM, shared across tasks)
- **broadcast channel**: `events.cancel_tx` (barge-in cancellation, 16 slots)
- **AtomicBool**: `play_cancel` (TTS playback interruption), `tts_muted`

---

## Main Audio Loop

The audio loop runs in `async_main()` inside a `tokio::select!`. CPAL delivers audio chunks to a bounded `async_channel` (200 slots). The loop processes each chunk through the unified `WhisperSTTVAD` (STT + VAD).

```
AudioCapture (CPAL)
  └─ bounded async_channel (200 slots) — AudioChunk (arbitrary sample rate, i16 → f32)
       └─ main loop: recv AudioChunk
            └─ AudioBuffer.write()
            └─ WhisperSTTVAD.process_chunk()
                 ├─ Internal Silero VAD (WhisperVadProcessor)
                 ├─ SpeechStart → emits SpeechEvent::SpeechStart
                 ├─ accumulates speech samples
                 ├─ SpeechEnd → transcribes via whisper-cpp-plus
                 │                emits SpeechEvent::SpeechEnd(transcript)
                 └─ Silence   → emits SpeechEvent::Silence

SpeechEvent handling in main loop:
  SpeechStart:
    ├─ barge-in: events.barge_in_tx.send() → cancels active pipeline tasks
    └─ start speech buffer + timer

  SpeechEnd("transcript"):
    ├─ IdentityAnalyzer.verify() — speaker verification + identity context update
    │    └─ updates ContextLens + conversation mode state
    │
    ├─ Non-main speaker:
    │    ├─ SpeechEvent::SpeechEnd(transcript) → stt_tx (handled externally)
    │    ├─ transcript pushed to AmbientBuffer
    │    └─ skip LLM pipeline
    │
    └─ Main speaker:
         ├─ AudioBuffer.write(changed) with speech samples
         └─ transcript_tx.send(PipelineFrame::UserTranscript) → wakes llm_task

  Silence:
    └─ update last_speech_at timer (ambient mode timeout tracking)
```

WhisperSTTVAD internals (`src/stt/mod.rs`):
- Silero VAD runs on 200ms probe windows (threshold 0.5)
- 300ms pre-roll retained before VAD onset
- whisper-cpp-plus (`WhisperContext`) transcribes on SpeechEnd
- Max segment: 20s before forced cut
- Config: `WHISPER_MODEL`, `VAD_MODEL`, `VOICEBOT_LANGUAGE`, `VAD_SILENCE_MS`

---

## Per-Utterance STT → LLM → TTS Pipeline

```mermaid
sequenceDiagram
    participant MIC as Microphone (CPAL)
    participant AUD as AudioCapture
    participant WHISPER as WhisperSTTVAD
    participant ID as IdentityAnalyzer
    participant LLM as llm_task
    participant SEN as sen_task
    participant TTS as tts_task
    participant SPK_OUT as AudioOutput (CPAL)
    participant DB as SQLite DB

    MIC->>AUD: AudioChunk
    AUD->>WHISPER: process_chunk()

    WHISPER->>WHISPER: Internal Silero VAD (200ms probes)
    WHISPER->>AUD: SpeechEvent::SpeechStart
    AUD-->>LLM: barge_in (broadcast cancel)
    AUD-->>SEN: barge_in (broadcast cancel)
    AUD-->>TTS: barge_in (broadcast cancel)
    Note over LLM,TTS: All tasks abort current work immediately

    WHISPER->>WHISPER: SpeechEnd → transcribe via whisper-cpp-plus
    WHISPER->>AUD: SpeechEvent::SpeechEnd(transcript)
    AUD->>ID: verify() — speaker identity
    ID-->>AUD: SpeakerIdentity

    alt Non-main speaker
        AUD->>AUD: push to AmbientBuffer, skip LLM
    else Main speaker
        AUD->>LLM: transcript_tx(UserTranscript)
    end

    LLM->>LLM: wake on transcript_rx
    LLM->>DB: persist user message
    LLM->>LLM: build messages from LlmSession + ContextLens
    LLM->>LLM: run tool calls (if any) — CurrentTime, WebSearch, RunAgent…
    LLM->>LLM: POST /v1/chat/completions (streaming SSE)

    loop token stream
        LLM->>SEN: llm_tx(PipelineFrame::Token)
    end

    SEN->>SEN: wake on llm_rx
    SEN->>SEN: scan for sentence boundary (. ! ? ; :)

    loop sentence boundaries
        SEN->>TTS: sentences_tx(PipelineFrame::Sentence)
    end

    TTS->>TTS: wake on sentences_rx
    TTS->>TTS: synthesize sentence (avspeech / kokoro)
    TTS->>SPK_OUT: play PCM audio
    TTS->>DB: persist assistant text (before playback)

    Note over TTS,SPK_OUT: While sentence N plays, LLM streams sentence N+1
```

---

## Conversation Mode State Machine

The `ConversationMode` is shared between the main audio loop and `SetConversationModeTool`.

```
Active  ──────────────────────────────────────────────────►  Active
  │  (user speaks)                                           ▲
  │                                                          │
  │  silence > ambient_clear_secs                           user speaks
  │  OR n consecutive non-main-speaker segments              │
  ▼                                                          │
Ambient ──── user explicitly sets "ambient locked" ──────► AmbientLocked
  │                                                          │
  └── any speech from main user ──────────────────────────►  returns to Active
                                                              (only wake-word in locked)
```

---

## Cancellation (Barge-in)

When `SpeechStart` fires (via `SpeechEvent::SpeechStart`), barge-in is triggered:

1. **`llm_task`** — detects pipeline state change via FSM, abandons in-progress turn
2. **`sen_task`** — returns to blocked state
3. **`tts_task`** — `play_cancel` AtomicBool set; `AudioOutput::play_blocking` polls it and exits early; clears pending sentences
4. **`consolidation_task`** — aborts if consolidation in progress

Barge-in delivered via `events.barge_in_tx` (mpsc) and broadcast cancel channel. All pipeline tasks monitor `PipelineState` via the watch channel to react to state transitions.

---

## Context Consolidation

`consolidation_task` runs after each completed LLM response or when the context exceeds `llm_idle_min_context_pct` on a timer.

```
LLM turn completes (detected via FSM state transitions)
  └─ consolidation_task wakes
       ├─ check LlmSession::needs_consolidation(context_tokens, threshold_pct)
       │    No → sleep until next trigger
       │    Yes ↓
       ├─ set pipeline state → Consolidating
       ├─ extract_memories()   — background LLM call (secondary or primary client)
       ├─ extract_facts()      — background LLM call (profile extraction)
       ├─ summarize history    — keep last N turns, write summary to DB
       ├─ rebuild LlmSession with new summary
       └─ set pipeline state → Idle
```

Idle consolidation also runs periodically when:
- Context exceeds `llm_idle_min_context_pct` of `LLM_CONTEXT_TOKENS`
- LLM has been idle for `LLM_IDLE_CONSOLIDATION_SECS`

---

## Persistent Memory Flow

```
DB (SQLite)
  ├─ sessions     — session id, summary text, cutoff message id
  ├─ messages     — all user/assistant turns
  ├─ user_profile — key/value/confidence facts extracted from conversation
  └─ memories     — free-form persistent notes

On startup:
  DB → load history, summary, profile, memories
     → build_system_prompt() → LlmSession

After each turn:
  persist user transcript + assistant response → DB

After consolidation:
  persist summary + pruned history + new memories + updated profile → DB
```

---

## Optional Daemons

| Daemon | Trigger | Purpose |
|--------|---------|---------|
| `InferenceDaemon` | fixed interval (`DAEMON_INTERVAL_SECS`) | proactive reasoning / background tasks |
| `EyesDaemon` | fixed interval (`EYES_INTERVAL_SECS`) | screenshot → secondary LLM → proactive context |

Both daemons emit `ProactiveEvent` via `proactive_tx`. The main loop drains `proactive_rx` inside the `tokio::select!`:
- `AgentResult` with `tool_call_id` injected as tool response
- `AgentResult` without `tool_call_id` queued for next idle window
- `AgentQuestion` handled when LLM is idle (triggers barge-in if busy)

---

## Pipeline FSM

The `PipelineState` machine (via `tokio::sync::watch` channel) tracks pipeline state:

```
Idle → Stt → Llm → Responding → (back to Idle)
                ↓
         Consolidating → Idle
```

FSM observer logs every transition. Under the `control` feature, state changes are broadcast as SSE events to connected clients.

---

## Control API (HTTP + SSE)

Available when compiled with `--features control`. Served on port from `CONTROL_PORT`.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/events` | GET | SSE stream of pipeline events (state changes, speech detected, etc.) |
| `/barge-in` | POST | Trigger barge-in from external client |
| `/tts/mute` | POST | Toggle TTS mute state |
| `/transcript` | POST | Submit a text transcript to the pipeline |
| `/history` | GET | Retrieve conversation history |

---

## TTS Backends

Two backends available via `TtsEngine` enum:

| Backend | Feature | Config | Description |
|---------|---------|--------|-------------|
| `avspeech` | `avspeech` | `AVSPEECH_VOICE`, `AVSPEECH_RATE` | macOS native AVSpeechSynthesizer (requires main thread CFRunLoop) |
| `kokoro` | `kokoro` | `KOKORO_MODEL`, `KOKORO_VOICE`, `KOKORO_LANGUAGE` | Kokoro ONNX TTS (cross-platform) |

Selected via `TTS_PROVIDER` env var (default: `avspeech` on macOS).

## Tools

Registered tools (tool names as seen by the LLM):

| Tool | Module | Enabled | Description |
|------|--------|---------|-------------|
| `current_time` | `tools/current_time.rs` | always | Returns current date/time |
| `read_clipboard` | `tools/clipboard.rs` | always | Reads system clipboard |
| `set_clipboard` | `tools/clipboard.rs` | always | Writes system clipboard |
| `open_app` | `tools/open_app.rs` | always | Opens macOS application by name |
| `set_conversation_mode` | `tools/conversation_mode.rs` | always | Switches Active/Ambient/AmbientLocked |
| `run_shell` | `tools/run_shell.rs` | if `SHELL_ENABLED=true` | Runs shell command with timeout |
| `take_screenshot` | `tools/take_screenshot.rs` | if `SECONDARY_LLM_URL` set | Screenshots + vision analysis via secondary LLM |
| `web_search` | `tools/web_search.rs` | if `SEARXNG_URL` set | SearXNG-backed search (optional LLM synthesis) |
| `run_agent` | `tools/run_agent.rs` | if `AGENT_MODE=acp` or `AGENT_COMMAND` set | ACP agent delegation via stdio JSON-RPC |
| `mcp_tool` | `tools/mcp_tool.rs` | if `MCP_COMMAND` set | Dynamically registered from MCP server |
