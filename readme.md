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
                    └─────────────┬─────────────┘
                                  │  tokens → sentences
                    ┌─────────────▼─────────────┐
                    │  TTS (macOS say)           │
                    │  sentence-by-sentence      │
                    └─────────────┬─────────────┘
                                  │  f32 PCM
                    ┌─────────────▼─────────────┐
                    │  AudioOutput (CPAL)        │
                    │  resample + play_blocking  │
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
| LLM client | `src/llm/client.rs` | Streaming SSE client for llama.cpp |
| LLM session | `src/llm/session.rs` | Accumulated prompt + KV-cache state |
| TTS | `src/tts/say.rs` | macOS `say` subprocess, WAV parsing |
| Sentence splitter | `src/tts/sentence.rs` | Buffers tokens, emits complete sentences |
| Database | `src/db/database.rs` | SQLite via sqlx, sessions + messages |
| Config | `src/config.rs` | Environment-based configuration |
| Main loop | `src/main.rs` | VAD loop + barge-in + pipeline orchestration |

---

## Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `VOICEBOT_LANGUAGE` | `es` | Language for STT and TTS voice selection |
| `SAY_VOICE` | `Marisol (Enhanced)` | macOS voice name |
| `WHISPER_MODEL` | — | Path to `.bin` Whisper model |
| `LLM_URL` | — | llama.cpp base URL |
| `LLM_SYSTEM_PROMPT` | — | System prompt for the LLM |
| `LLM_MAX_TOKENS` | — | Max generation tokens |
| `LLM_TEMPERATURE` | — | LLM temperature |
| `AUDIO_DEVICE` | system default | Input device name substring |
| `AUDIO_OUTPUT_DEVICE` | system default | Output device name substring |
| `DB_PATH` | — | SQLite database file path |
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

This section analyzes each planned feature: design approach, implementation challenges, and integration points with the current pipeline.

---

### 1. Tool Use (Function Calling)

**Goal:** The LLM can call tools (web search, calendar, file read, shell commands, etc.) and receive results before generating its spoken response.

**Approach:**

llama.cpp supports OpenAI-compatible function calling when the model has been trained for it (e.g., Qwen, Mistral, LLaMA-3.1+). The pipeline becomes a loop instead of a straight pass:

```
STT → LLM call →  text response?   → TTS → Speaker
                  tool_call?  → execute tool → LLM call (with tool result) → ...
```

**Implementation steps:**
1. Add `tools: Vec<ToolDefinition>` to `LlmSession`; include them in the llama.cpp payload as `"tools": [...]`
2. Parse the LLM response for tool calls (JSON in the token stream, or a `tool_calls` field in the final SSE message)
3. Route the call to the appropriate executor (built-in, MCP, agent — see below)
4. Append the tool result to the accumulated prompt as a `tool` turn
5. Re-call the LLM; repeat until a plain text response is returned
6. Feed final text to TTS as normal

**Voice UX consideration:** Tool calls can add 1–10 seconds of latency. Options:
- Play a short "thinking" audio clip while waiting
- The LLM can be instructed to acknowledge out loud before calling a tool ("Déjame buscar eso..." → TTS plays → tool executes → LLM continues)

**Key challenge:** Streaming SSE + tool call detection. Tool call JSON is emitted mid-stream; the current sentence splitter needs to be extended to detect and suppress tool-call tokens from TTS output.

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

### 6. Context Summarization

**Goal:** Prevent the LLM context window from filling up during long conversations by automatically summarizing old turns.

**Approach:**

The `LlmSession` accumulated prompt grows without bound. When it approaches the model's context limit:

1. **Detect:** Track approximate token count (chars / 3.5 is a reasonable estimate). Trigger at ~75% of the configured context window.
2. **Summarize:** Send the old conversation to the LLM with a summarization prompt:
   ```
   Summarize this conversation concisely, preserving all important facts, decisions, and context:
   [old turns]
   ```
3. **Replace:** Reconstruct `accumulated_prompt` as:
   ```
   <|im_start|>system
   {original system prompt}

   [CONVERSATION SUMMARY]
   {summary}
   <|im_end|>
   <|im_start|>user
   {current turn}...
   ```
4. **Reset KV-cache:** Set `cache_prompt: false` for the first call after summarization (the prompt has changed structurally, so the old KV-cache is invalid). Then resume `cache_prompt: true`.

**Strategy:** Keep the last N turns verbatim (e.g., last 5) and summarize everything before them. This preserves recent context fully while compressing the distant past.

**Implementation in `LlmSession`:**
```rust
impl LlmSession {
    pub fn needs_summarization(&self, context_limit_tokens: usize) -> bool { ... }
    pub fn apply_summary(&mut self, summary: &str, keep_recent_turns: usize) { ... }
}
```

Summarization is triggered asynchronously between pipeline runs (not during active speech) to avoid adding latency.

**DB persistence:** The summary is stored in the database so that on restart the conversation context is still compact and available.

---

### 7. User Profile Extraction and Injection

**Goal:** Automatically learn facts about the user (name, location, job, hobbies, preferences, personality) from conversation, store them persistently, and inject them into every LLM system prompt so the assistant always has personal context.

**Approach:**

**Extraction (background, after each turn):**

After the pipeline completes a turn, spawn a background task that sends the last exchange to the LLM with an extraction prompt:

```
From the following conversation excerpt, extract any new facts about the user.
Return ONLY a JSON array: [{"key": "name", "value": "Daniel", "confidence": 0.9}]
If no new facts, return [].

[User]: {transcript}
[Assistant]: {response}
```

Facts are stored in a `user_profile` SQLite table:
```sql
CREATE TABLE user_profile (
    key       TEXT PRIMARY KEY,   -- "name", "city", "job", "hobby_1", etc.
    value     TEXT NOT NULL,
    confidence REAL NOT NULL,
    source    TEXT,               -- which conversation turn revealed it
    updated_at TEXT NOT NULL
);
```

**Profile categories to extract:**
- **Identity**: name, age, gender, nationality
- **Location**: city, country, timezone
- **Professional**: job title, company, field, skills
- **Personal**: hobbies, interests, family situation, pets
- **Preferences**: communication style, topics of interest, things they dislike
- **Psychological**: personality traits inferred from conversation patterns (cautious, curious, direct, etc.)

**Injection into system prompt:**

On startup, load the profile and build a `{{user_context}}` block that is appended to the system prompt:

```
[USER PROFILE]
Name: Daniel | City: Madrid | Job: Software Engineer
Interests: Rust, AI, voice interfaces
Communication style: direct, technical, prefers Spanish
```

Low-confidence facts (< 0.5) are omitted or marked as uncertain. Facts are updated if contradicted by new information.

**Privacy note:** All profile data stays local in SQLite. No data leaves the machine.

---

## Implementation Priority

| Priority | Feature | Effort | Impact |
|----------|---------|--------|--------|
| 1 | User profile extraction + injection | Medium | High — improves every conversation immediately |
| 2 | Context summarization | Medium | High — essential for long-running sessions |
| 3 | Tool use (function calling) | High | High — unlocks real-world utility |
| 4 | MCP integration | Medium | High — huge tool ecosystem for free |
| 5 | Proactive conversations | Medium | Medium — needed for async agents |
| 6 | Agent delegation | High | Medium — depends on tool use + proactive |
| 7 | Agent intermediary | Low | Medium — relatively simple once agents work |

Features 1 and 2 are self-contained improvements to the existing pipeline and can be implemented without touching the tool/agent layer. Features 3–7 form a dependency chain: tools → MCP → agents → proactive → intermediary.

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
