# Voicebot

A voice AI assistant built in Rust — real-time speech-to-speech pipeline with barge-in, persistent memory, and extensible tool/agent integration.

---

## Current State

The core pipeline is fully operational:

```
Microphone → VAD → AudioBuffer → Whisper STT → llama.cpp LLM → SentenceSplitter → macOS say TTS → Speaker
```

**Implemented features:**
- Real-time voice capture (CPAL), Silero VAD with pre-roll buffer
- Whisper.cpp STT (Metal GPU, state cached across utterances)
- Streaming LLM via llama.cpp HTTP (`cache_prompt=true`, KV-cache reuse across turns)
- Sentence-by-sentence TTS playback — first sentence plays while next is being generated
- **Barge-in**: user speech cancels active LLM/TTS pipeline instantly via `Arc<AtomicBool>`
- Persistent SQLite conversation history — restored on startup
- LLM session rollback on barge-in interruption
- **Tool use**: `<tool_call>` XML detection mid-stream; LLM loops back after tool result; `current_time` built-in
- **Context summarization**: auto-triggers at 75% of context window; keeps last N turns verbatim; summary persisted in DB and restored on restart
- **User profile**: background LLM extraction of user facts after every turn; stored in `user_profile` SQLite table; injected into system prompt on startup

---

## Architecture

### Current pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  VAD loop (tokio, non-blocking)                                 │
│                                                                  │
│  Mic → VAD ──SpeechEnd──► spawn pipeline task ──────────────►  │
│              │                                                   │
│              └──SpeechStart──► cancel flag + abort task         │
└─────────────────────────────────────────────────────────────────┘
                                  │
                         pipeline task (async)
                                  │
                    ┌─────────────▼─────────────┐
                    │  STT (spawn_blocking)      │
                    │  Whisper.cpp + Metal GPU   │
                    └─────────────┬─────────────┘
                                  │  transcript
                    ┌─────────────▼─────────────┐
                    │  LLM (streaming HTTP SSE)  │
                    │  llama.cpp, cache_prompt   │
                    └──────┬──────────┬──────────┘
                    text   │    <tool_call>?
                           │          │
                           │   ┌──────▼──────────┐
                           │   │  ToolRegistry   │
                           │   │  execute tool   │
                           │   └──────┬──────────┘
                           │   re-call LLM with result
                           │          │
                    ┌──────▼──────────▼──────────┐
                    │  TTS (macOS say)            │
                    │  sentence-by-sentence       │
                    └─────────────┬──────────────┘
                                  │  f32 PCM
                    ┌─────────────▼─────────────┐
                    │  AudioOutput (CPAL)        │
                    │  resample + play_blocking  │
                    └───────────────────────────┘
                                  │  (turn complete)
                    ┌─────────────▼─────────────┐
                    │  maybe_summarize()         │  ← if prompt > 75% context
                    │  background LLM call       │
                    └─────────────┬─────────────┘
                    ┌─────────────▼─────────────┐
                    │  extract_facts() [spawn]   │  ← fire-and-forget
                    │  update user_profile DB    │
                    └───────────────────────────┘
```

### Key modules

| Module | File | Description |
|--------|------|-------------|
| Audio capture | `src/audio/audio_capture.rs` | CPAL mic input, normalizes to f32 |
| VAD | `src/audio/vad.rs` | Silero energy VAD, pre-roll buffer |
| Audio buffer | `src/audio/buffer.rs` | Accumulates speech chunks |
| Audio output | `src/audio/output.rs` | CPAL playback, resample, cancel support |
| STT | `src/stt/whisper.rs` | whisper-rs, cached WhisperState (no Metal re-init) |
| LLM client | `src/llm/client.rs` | Streaming SSE + one-shot completion for llama.cpp |
| LLM session | `src/llm/session.rs` | Accumulated prompt, turn tracking, summarization |
| TTS | `src/tts/say.rs` | macOS `say` subprocess, WAV parsing |
| Sentence splitter | `src/tts/sentence.rs` | Buffers tokens, emits complete sentences |
| Tools | `src/tools/` | `Tool` trait + `ToolRegistry`; `current_time` built-in |
| Profile | `src/profile/mod.rs` | User fact extraction, JSON parsing, context builder |
| Database | `src/db/database.rs` | SQLite: sessions, messages, summary, user_profile |
| Config | `src/config.rs` | Environment-based configuration |
| Main loop | `src/main.rs` | VAD loop + barge-in + pipeline + summarization + profile |

---

## Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `VOICEBOT_LANGUAGE` | `es` | Language for STT and TTS voice selection |
| `SAY_VOICE` | `Marisol (Enhanced)` | macOS voice name |
| `WHISPER_MODEL` | — | Path to `.bin` Whisper model |
| `LLM_URL` | `http://localhost:8080` | llama.cpp base URL |
| `LLM_SYSTEM_PROMPT` | — | System prompt for the LLM |
| `LLM_MAX_TOKENS` | `400` | Max generation tokens per response |
| `LLM_TEMPERATURE` | `0.7` | LLM sampling temperature |
| `LLM_CONTEXT_TOKENS` | `4096` | Model context window size; triggers summarization at 75% |
| `LLM_SUMMARY_KEEP_TURNS` | `6` | Recent (role, content) turns kept verbatim after summarization |
| `AUDIO_DEVICE` | system default | Input device name substring |
| `AUDIO_OUTPUT_DEVICE` | system default | Output device name substring |
| `DB_PATH` | `data/voicebot.db` | SQLite database file path |
| `LIST_AUDIO_DEVICES` | `0` | Print devices and exit |

---

## Commands

```bash
cargo build --release
cargo run --release
cargo test
cargo fmt
cargo clippy
cargo run -- --list-devices
```

---

## Roadmap — Feature Analysis

This section documents implemented features and analyzes pending ones.

---

### 1. Tool Use (Function Calling) ✅ Implemented

**Goal:** The LLM can call tools and receive results before generating its spoken response.

**Implementation:**

Tool calls use a prompt-engineering approach with XML markers: `<tool_call>tool_name</tool_call>`. This works with any LLM without requiring native function-calling support.

```
STT → LLM streams tokens
         │
         ├── regular text → SentenceSplitter → TTS (streaming)
         └── <tool_call>name</tool_call> detected at end-of-stream
                  │
                  ▼ (TTS suppressed for tool call text)
             ToolRegistry.execute(name)
                  │
             add_tool_result() → session prompt updated
                  │
             LLM re-called → streams spoken response → TTS
```

**Key design:** Tool call XML contains no punctuation, so `SentenceSplitter` never emits it mid-stream. Detection happens safely at end-of-stream before the final `flush()`.

**Adding a new tool:** implement `trait Tool { fn name(); fn description(); fn run() -> String; }` and call `registry.register(MyTool)` in `main()`.

**Built-in tools:** `current_time` — returns local date and time.

---

### 2. MCP (Model Context Protocol) Integration

**Goal:** Connect to MCP servers to expose a broad ecosystem of tools (filesystem, browser, GitHub, Slack, databases, etc.) without implementing each tool manually.

**Approach:**

MCP uses a JSON-RPC protocol over stdio (subprocess) or SSE (HTTP). The voicebot acts as an MCP client:

```
LLM tool_call
     │
     ▼
ToolRouter
     │
     ├── built-in tools (Rust functions)
     ├── MCP servers (subprocess/HTTP JSON-RPC)
     └── agents (see feature 3)
```

**Implementation steps:**
1. Restore `src/tools/` and `src/mcp/` modules (currently gutted from the codebase)
2. `McpClient`: spawns/connects to MCP servers, implements `initialize` + `tools/list` + `tools/call` JSON-RPC methods
3. `ToolRouter`: on each tool call from the LLM, tries built-ins first, then queries registered MCP servers
4. Tool schemas from MCP (`tools/list`) are translated to llama.cpp-compatible JSON Schema and injected into the LLM payload

**Config:** MCP servers configured via a JSON/TOML file, e.g.:
```toml
[[mcp_servers]]
name = "filesystem"
command = ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/home/user"]

[[mcp_servers]]
name = "brave-search"
command = ["npx", "-y", "@modelcontextprotocol/server-brave-search"]
env = { BRAVE_API_KEY = "..." }
```

**Key challenge:** MCP servers are typically Node.js/Python subprocesses. Need async stdin/stdout communication without blocking the tokio event loop — use `tokio::process::Command` with async I/O.

---

### 3. Agent Delegation

**Goal:** The LLM can delegate complex tasks (deep research, code generation, long-running automation) to specialized agents. Results are routed back through the LLM, which summarizes them into a voice response.

**Two delegation modes:**

**Synchronous (simple tasks, < 5s):**
```
LLM calls "run_agent" tool → agent executes → result → LLM → TTS
```
Identical to tool use. The agent is just a long-running tool.

**Asynchronous (long tasks — research, coding, etc.):**
```
User: "Research X and tell me the summary"
LLM → TTS: "Lo investigo, te aviso en unos minutos"
              ↓
         agent runs in background (tokio::spawn)
              ↓ (minutes later)
         agent completes → pushes to proactive_tx channel
              ↓
         LLM synthesizes result → TTS plays proactively
```

**Implementation steps:**
1. `AgentManager` in `src/agents/`: registry of available agents with their capabilities and API contracts
2. `run_agent` tool definition: `{ name, task_description, async: bool }`
3. For async mode: `tokio::spawn` the agent task; on completion, push a `ProactiveEvent::AgentResult { agent, result }` to a channel that the main VAD loop listens to (see Feature 5)
4. The LLM is given agent descriptions at startup so it knows when to delegate

**Agent protocol options:**
- HTTP API (OpenAI-compatible agents, OpenClaw, etc.)
- Claude SDK / Anthropic API (for sub-agents using Claude)
- Local subprocess with structured I/O

---

### 4. Proactive Conversations (Bot-Initiated Speech)

**Goal:** The bot can speak without being prompted by the user — to deliver agent results, reminders, greetings, or contextual observations.

**Approach:**

The main `tokio::select!` loop is extended with a proactive events channel:

```rust
enum ProactiveEvent {
    AgentResult { task: String, result: String },
    Reminder { message: String },
    Scheduled { prompt: String },
}

tokio::select! {
    chunk = audio_rx.recv()    => { /* VAD processing */ }
    event = proactive_rx.recv() => { run_proactive_pipeline(event, ...).await }
    _ = ctrl_c()               => { /* shutdown */ }
}
```

**Event sources:**
- **Agent completion**: async agent task pushes to `proactive_tx` when done
- **Scheduler**: background task fires at configured times (reminders, daily briefings)
- **Idle trigger**: after N minutes of silence, LLM generates a contextual observation or question (configurable, off by default)
- **External trigger**: Unix socket or local HTTP endpoint that external processes can POST to

**Voice UX:**
- Play a subtle audio cue before speaking proactively (so the user isn't startled)
- Respect barge-in — user can interrupt proactive speech exactly like regular responses
- Check if the user appears to be in the middle of speaking before interrupting with a proactive message

**Key challenge:** Idle detection in the current VAD loop. The VAD only sees `Silence`/`Speech` events. Need a timer that resets on any `Speech` or `SpeechEnd` event, and fires a proactive event after a configurable idle timeout.

---

### 5. Voicebot as Agent Intermediary

**Goal:** The voicebot acts as a voice interface to any existing text-based agent or service — translating user voice to the agent's input format and the agent's text output to speech.

**Approach:**

This is a generalization of agent delegation. The voicebot becomes a transparent voice proxy:

```
User voice → STT → voicebot LLM (optional) → agent API → response text → TTS
```

Two proxy modes:

**Transparent proxy** (voicebot just relays, no LLM involved):
```
STT transcript → agent API → response → TTS
```
Useful when the agent is a full conversational AI (e.g., another Claude instance, a specialized chatbot). Latency is minimized.

**Mediated proxy** (voicebot LLM reformulates):
```
STT transcript → local LLM reformulates/enriches → agent API → response → local LLM summarizes → TTS
```
Useful when the agent's API expects specific input formats, or when responses are too long/technical for direct speech.

**Implementation:**
- `AgentProxy` struct: wraps an agent's API client, configured with input/output transformers
- The voicebot's system prompt can include: "You are the voice interface for [Agent X]. Translate user requests into queries for it and summarize its responses conversationally."
- Configurable per-agent: whether to use the mediated or transparent mode

**Integration with current architecture:** The `run_pipeline` function gets an optional `agent_proxy` parameter. If set, after STT the transcript goes to the proxy instead of (or alongside) the local LLM.

---

### 6. Context Summarization ✅ Implemented

**Goal:** Prevent the LLM context window from filling up during long conversations.

**Implementation:**

Triggered automatically at the end of each pipeline turn, after the assistant response is saved.

- **Detection:** `chars / 3.5 > context_tokens * 75%` — rough token estimate; tunable via `LLM_CONTEXT_TOKENS`
- **Summarization:** one-shot LLM call (`slot_id: -1`, `temperature: 0.3`) asking to summarize the old turns in the same language as the conversation
- **Compaction:** `LlmSession::apply_summary()` rebuilds `accumulated_prompt` as:
  ```
  <|im_start|>system
  {original system prompt}

  [CONVERSATION SUMMARY]
  {summary text}
  <|im_end|>
  {last N turns verbatim}
  ```
- **Persistence:** summary text and cutoff message ID saved in `sessions.summary` / `sessions.summary_through_id`; on next startup only messages after the cutoff are loaded

**Config:** `LLM_CONTEXT_TOKENS=4096`, `LLM_SUMMARY_KEEP_TURNS=6`

---

### 7. User Profile Extraction and Injection ✅ Implemented

**Goal:** Learn facts about the user from conversation and inject them into every system prompt.

**Implementation:**

**Extraction** — `tokio::spawn` fire-and-forget after every completed turn:
- One-shot LLM call (`slot_id: -1`, `temperature: 0.1`, `n_predict: 256`)
- Prompt instructs the model to return a JSON array: `[{"key": "name", "value": "Daniel", "confidence": 0.95}]`
- Keys are normalized to `lowercase_underscores`; markdown code fences stripped before parsing
- Standard key names suggested in prompt: `name`, `age`, `city`, `job`, `hobby_N`, `skill`, `communication_style`, `personality_trait`, etc.

**Storage** (`user_profile` table):
```sql
CREATE TABLE user_profile (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    updated_at TEXT NOT NULL
);
-- Upsert: only overwrites if new confidence is strictly higher
```

**Injection** — on startup, facts with `confidence >= 0.5` are formatted and prepended to the system prompt:
```
[USER PROFILE]
name: Daniel
city: Madrid
job: software engineer
hobby_1: Rust
communication_style: direct, technical
```

**Privacy:** all data stays local in SQLite. Nothing leaves the machine.

---

## Implementation Status

| Feature | Status | Notes |
|---------|--------|-------|
| Tool use (`current_time`) | ✅ Done | XML-based detection, extensible `Tool` trait |
| Context summarization | ✅ Done | Auto-trigger at 75% context; persisted in DB |
| User profile extraction + injection | ✅ Done | Background LLM extraction; `user_profile` table |
| MCP integration | Planned | `src/mcp/` module; JSON-RPC over stdio/HTTP |
| Agent delegation | Planned | Depends on tool use + proactive events |
| Proactive conversations | Planned | `tokio::select!` proactive channel + idle timer |
| Agent intermediary | Planned | Voice proxy for existing text agents |

Pending features form a dependency chain: MCP → agent delegation → proactive conversations → intermediary.

---

## S2S Model Reference

Available open-source Speech-to-Speech models (alternative to the current STT+LLM+TTS cascade):

| Model | Params | Notes |
|-------|--------|-------|
| [LFM2.5-Audio](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B) | 1.5B | llama.cpp GGUF compatible, best option for local S2S |
| [LLaMA-Omni 2](https://arxiv.org/abs/2505.02625) | 0.5B–14B | Qwen2.5 base, streaming synthesis, sub-second latency |
| [Moshi](https://github.com/kyutai-labs/moshi) | — | Full-duplex (listen + respond simultaneously) |
| [Ultravox](https://github.com/fixie-ai/ultravox) | — | Whisper + LLaMA hybrid |

The current cascade (Whisper + llama.cpp + say) is preferred for now because it supports streaming sentence-by-sentence TTS while the LLM is still generating — true S2S models don't stream output token-by-token in a way that maps to sentence-level TTS.

---

## License

Private project — all rights reserved.
