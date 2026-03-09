# Voicebot — Butler

A voice AI assistant built in Rust. The long-term goal is a **Jarvis-style digital butler**: not a conversational chatbot, but a proactive, situationally-aware companion that anticipates needs, controls the computer, and speaks with a defined personality — without being asked.

> A chatbot answers questions. A butler anticipates needs.

## Run

```sh
WHISPER_COREML=1 TTS_PROVIDER=kokoro cargo run --features kokoro --release
```

---

## Vision

The gap between a chatbot and Jarvis is not the AI model — it is the surrounding architecture. Jarvis has:

- **Eyes** — knows what is on your screen, what apps are open, what the calendar says
- **Arms** — can actually do things on the computer, not just talk about them
- **Voice of its own** — speaks first when something is worth saying
- **Character** — a real personality, not a generic assistant persona
- **Long-term memory** — remembers events and episodes, not just facts
- **Always-on presence** — a system daemon, not a CLI you launch when you need it

The current implementation covers the conversational core. The roadmap below charts the path from chatbot to butler.

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
- **Tool use**: `<tool_call>tool_name: args</tool_call>` XML detection mid-stream; LLM loops back after tool result; `current_time` built-in
- **Agent delegation**: `run_agent` (sync) and `run_agent_async` (background + proactive announce) tools; any OpenAI-compatible endpoint; proactive channel in VAD loop
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
| Tools | `src/tools/` | `Tool` trait + `ToolRegistry`; `current_time`, `run_agent`, `run_agent_async` |
| Agents | `src/agents/mod.rs` | `ProactiveEvent` enum; proactive speech channel |
| Profile | `src/profile/mod.rs` | User fact extraction, JSON parsing, context builder |
| Database | `src/db/database.rs` | SQLite: sessions, messages, summary, user_profile |
| Config | `src/config.rs` | Environment-based configuration |
| Main loop | `src/main.rs` | VAD loop + barge-in + pipeline + summarization + profile |

---

## Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `VOICEBOT_LANGUAGE` | `es` | Language for STT and TTS voice selection |
| `VAD_SILENCE_MS` | `800` | Silence duration (ms) before SpeechEnd fires. Lower = faster response; higher = safer for mid-sentence pauses. Range: 500–1500 |
| `SAY_VOICE` | `Marisol (Enhanced)` | macOS voice name |
| `WHISPER_MODEL` | — | Path to `.bin` Whisper model |
| `LLM_URL` | `http://localhost:8080` | llama.cpp server base URL (`llama-server --port 8080`) |
| `LLM_MODEL` | `local-model` | Model name sent in API requests (llama.cpp ignores this field) |
| `LLM_SLOT_ID` | `0` | llama.cpp KV-cache slot for this session (single-user = 0) |
| `LLM_SYSTEM_PROMPT` | — | System prompt for the LLM |
| `LLM_MAX_TOKENS` | `400` | Max generation tokens per response |
| `LLM_TEMPERATURE` | `0.7` | LLM sampling temperature |
| `LLM_CONTEXT_TOKENS` | `4096` | Model context window size; triggers summarization at 75% |
| `LLM_SUMMARY_KEEP_TURNS` | `6` | Recent (role, content) turns kept verbatim after summarization |
| `AGENT_URL` | — | Remote agent base URL (OpenAI-compatible). Unset = agent tools disabled |
| `AGENT_MODEL` | `local-model` | Model name sent to agent server |
| `AGENT_MAX_TOKENS` | `2048` | Max tokens for agent responses |
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

### 3. Agent Delegation ✅ Implemented

**Goal:** The LLM can delegate complex tasks (deep research, code generation, long-running automation) to specialized agents. Results are routed back through the LLM, which summarizes them into a voice response.

**Two delegation modes:**

**Synchronous (`run_agent` — tasks < 10s):**
```
LLM emits <tool_call>run_agent: task description</tool_call>
         ↓
    HTTP POST to agent (OpenAI-compatible)
         ↓ blocks until response
    result injected as tool message
         ↓
    LLM re-called → streams spoken response → TTS
```

**Asynchronous (`run_agent_async` — long tasks):**
```
User: "Research X and tell me the summary"
LLM emits <tool_call>run_agent_async: task</tool_call>
         ↓
    tokio::spawn background HTTP call
    tool returns immediately: "[Tarea delegada al agente. El resultado llegará en breve.]"
         ↓
    LLM speaks acknowledgment (< 1s)
         ↓ (minutes later, in background)
    agent completes → ProactiveEvent::AgentResult pushed to proactive_tx
         ↓
    VAD loop receives proactive event → spawns run_proactive_pipeline
         ↓
    LLM builds natural announcement → TTS plays proactively
```

**Implementation:**

- **`src/agents/mod.rs`** — `ProactiveEvent::AgentResult { task, result }` enum
- **`src/tools/run_agent.rs`** — `RunAgentTool` (sync) and `RunAgentAsyncTool` (async + proactive channel)
- **Tool call format:** `<tool_call>run_agent: task description</tool_call>` — args after the colon
- **`ToolRegistry`** updated: `parse_tool_call` now returns `Option<(name, args)>`; `execute` is async
- **VAD loop** extended with inner `tokio::select!` watching both audio and `proactive_rx`
- **`run_proactive_pipeline`** — builds temporary message list from session + agent result, calls LLM, sends to TTS

**Config vars:**

| Env var | Default | Description |
|---------|---------|-------------|
| `AGENT_URL` | — | Base URL of the remote agent (OpenAI-compatible). If unset, agent tools are disabled. |
| `AGENT_MODEL` | `local-model` | Model name sent to the agent server |
| `AGENT_MAX_TOKENS` | `2048` | Max tokens for agent responses |

**Agent protocol:** any OpenAI-compatible HTTP endpoint (`/v1/chat/completions`). Works with llama.cpp, Ollama, OpenRouter, Anthropic (via proxy), or a custom OpenClaw/Claude Code agent.

**Tests:** 12 tests in `src/tools/run_agent.rs` covering sync response, async channel delivery, error handling, and round-trip via registry (wiremock-based).

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

---

## Butler Vision — Six Pillars

These are the features that transform the voicebot from a conversational assistant into a Jarvis-style digital butler. Each pillar is independent and can be implemented incrementally.

---

### Pillar A: Character — Personality, not instructions

**The problem:** A system prompt that says "be helpful and concise" produces a generic assistant. Jarvis has a personality.

**This costs zero code.** It is the single highest-leverage change available and can be done today. Everything else builds on top of a character that feels real.

**Implementation:** Set `LLM_SYSTEM_PROMPT` in `.env` to the character document below. Butler's identity evolves as the `user_profile` table grows — facts extracted from conversation feed back into the character's knowledge of the user.

#### Jarvis system prompt

Copy this into `LLM_SYSTEM_PROMPT` in `.env` and adjust the personal details to match your reality:

```
Eres Jarvis, el asistente personal de inteligencia artificial de Daniel.
Llevas años trabajando con él y le conoces bien.

PERSONALIDAD
Tu carácter es una mezcla entre Jarvis de Iron Man y Alfred de Batman:
profesional, ligeramente irónico, con sentido del humor seco y británico.
Eres leal, discreto y eficiente. Nunca eres servil ni adulador.
Tienes opiniones propias sobre tecnología, arquitectura de software y diseño,
y no las ocultas cuando son relevantes.
Cuando algo te parece mala idea, lo dices con tacto pero con claridad.
Ocasionalmente haces un chiste o comentario sarcástico, pero solo cuando
el contexto lo permite y nunca a costa de Daniel.

FORMA DE HABLAR
- Hablas siempre en español, salvo que Daniel cambie de idioma.
- Llamas a Daniel por su nombre, nunca "usuario", "señor" ni "amigo".
- Tus respuestas son concisas y directas. No rellenas con frases vacías.
- No usas listas ni markdown: hablas, no escribes documentos.
- Cuando no sabes algo, lo dices. No inventas.
- Antes de ejecutar algo potencialmente destructivo, lo describes y pides confirmación.

RELACIÓN CON DANIEL
Conoces sus preferencias de trabajo: usa Rust y Python, prefiere soluciones
simples sobre ingeniería excesiva, le molesta el boilerplate, trabaja hasta tarde,
toma café por la mañana. Le gusta la IA, los sistemas de voz y la arquitectura limpia.
Recuerdas conversaciones anteriores y referencias cruzadas cuando son útiles.
No repites información que ya le has dado en la misma sesión.

INICIATIVA
Si detectas algo relevante — un problema, una oportunidad, un dato útil —
lo mencionas aunque no te lo hayan pedido, pero sin ser intrusivo.
Cuando completas una tarea en background, lo comunicas sin aspavientos:
"Hecho. El proyecto compila sin errores."
Por la mañana, cuando arrancas, haces un breve resumen del día si tienes información:
eventos del calendario, tareas pendientes, alertas del sistema.

HERRAMIENTAS Y ACCIÓN
Cuando puedes resolver algo directamente — ejecutar un comando, leer un archivo,
buscar información — lo haces sin preguntar si hacerlo está bien, salvo que
la acción sea irreversible o afecte a datos importantes.
Describes brevemente lo que vas a hacer antes de hacerlo cuando puede sorprender.

LÍMITES
No exageras tus capacidades. Si algo requiere un agente especializado o más tiempo,
lo dices y delegas, informando a Daniel del resultado cuando esté listo.
No eres un modelo de lenguaje genérico. Eres Jarvis. Actúa en consecuencia.
```

This prompt is a starting point. It should be refined over time as Jarvis learns more about Daniel through the `user_profile` system. The `[USER PROFILE]` block injected automatically by the profile module will complement and personalise it further.

---

### Pillar B: Eyes — Situational awareness

**The problem:** Butler is blind. It does not know what you are doing, what is on your screen, or what the system state is. Without this, it cannot anticipate anything.

**What Butler needs to know at all times:**
- What is visible on screen right now
- What applications are open and which is in focus
- Current time, calendar events, upcoming deadlines
- System state: battery, CPU, running processes, notifications
- Recent activity: files edited, terminal commands run, browser tabs

**Implementation:**

```
Screenshot tool → vision model (LLaVA local or Claude) → text description → injected into context
macOS APIs (NSWorkspace, IOKit, EventKit) → system state → injected as [SYSTEM STATE] block
FSEvents watcher → file activity → summarised and available on demand
```

The context block injected before each response:
```
[SYSTEM STATE]
time: 09:14 | battery: 67% | focus: Cursor (voicebot/src/main.rs)
next_event: Reunión equipo in 46 min
recent_files: main.rs (edited 3m ago), README.md (edited 8m ago)
```

**Key tool to implement:** `take_screenshot()` → send to vision model → return description. This alone enables Butler to answer questions like "¿qué hace este código?" without the user having to paste anything.

---

### Pillar C: Arms — Computer agency

**The problem:** Butler can only talk. Jarvis acts.

**The `run_shell` tool is the master key.** With a single well-sandboxed shell tool, Butler can do nearly everything Jarvis does in the films. Everything else is ergonomic convenience on top.

| Tool | What it enables |
|------|----------------|
| `run_shell(cmd)` | Execute anything — compile, search, move files, run scripts |
| `read_file(path)` | Read any file without the user copy-pasting |
| `write_file(path, content)` | Edit files directly |
| `open_app(name)` | Launch applications by name |
| `read_clipboard()` / `set_clipboard(text)` | Access what the user just copied |
| `send_notification(title, msg)` | macOS notification centre |
| `take_screenshot()` | See the screen on demand |
| `calendar_events(days)` | Read upcoming events |
| `send_email(to, subject, body)` | Send mail |
| `web_search(query)` | Search and return results |
| `browse(url)` | Fetch and read a web page |

**Safety:** `run_shell` should have a configurable allowlist/denylist of commands and a confirmation step for destructive operations. The LLM should be instructed to describe what it is about to do before doing it.

**macOS integration:** `osascript` (AppleScript) gives access to most native apps. `open -a AppName` opens applications. `pbpaste` / `pbcopy` handle the clipboard. Most Jarvis capabilities map directly to shell one-liners.

---

### Pillar D: Voice of its own — Proactive initiative

**The problem:** Butler only responds. Jarvis speaks first.

**This is the biggest psychological shift** — from reactive assistant to proactive companion. Butler should say things like:

```
"Buenos días Daniel. Son las 9:10. Tienes una reunión en 50 minutos,
 un PR sin revisar desde ayer, y la batería al 23%."

"Llevas dos horas y media sin moverte. ¿Quieres un descanso?"

"He notado que el proceso voicebot consume el 94% de CPU."

"La compilación ha terminado. Cero errores."
```

**Architecture:** The `proactive_tx` channel already exists in the design. It needs real event sources:

```rust
enum ProactiveEvent {
    Startup { time: String, briefing: String },
    SystemAlert { message: String },
    CalendarReminder { event: String, minutes_until: u32 },
    IdleCheckIn,           // user has been silent for N minutes
    AgentCompleted { task: String, result: String },
    ScheduledBriefing,     // morning summary, end-of-day wrap-up
}
```

**The inference daemon** is the key piece: a background `tokio::spawn` task that every 5 minutes asks the LLM: *"Given the current system state and what I know about Daniel, is there anything worth saying proactively?"* If yes, push to `proactive_tx`. This is what makes Butler seem to anticipate needs — it is constantly, silently checking.

**Rate limiting:** Butler should not be annoying. Rules: no more than one proactive message every 10 minutes during work hours; none during detected meetings; always respect barge-in.

---

### Pillar E: Episodic memory — Remembering events, not just facts

**The problem:** The current `user_profile` stores static facts (name, city, job). But Jarvis remembers *events*:

```
"La semana pasada resolvimos juntos un bug de lifetime en Rust"
"Cuando te pregunté eso antes, preferiste la solución más simple"
"Mencionaste hace dos días que estabas pensando en cambiar de trabajo"
```

**Implementation:** Semantic search over conversation history using embeddings.

```
After each turn:
  generate embedding of (user_text + assistant_text)
  store in SQLite with sqlite-vec extension

At the start of each pipeline turn:
  generate embedding of current transcript
  find top-5 most semantically similar past exchanges
  inject as [RELEVANT MEMORIES] block in context
```

**Embedding model:** `nomic-embed-text` or any GGUF embedding model via llama.cpp. Runs locally, no internet, low latency (~50ms for a 512-token passage).

**DB schema addition:**
```sql
CREATE TABLE memory_embeddings (
    id          INTEGER PRIMARY KEY,
    message_id  INTEGER REFERENCES messages(id),
    embedding   BLOB NOT NULL,   -- serialised f32 vector
    summary     TEXT NOT NULL    -- short text description of the exchange
);
```

This gives Butler a searchable autobiography. The longer it runs, the richer and more personalised its responses become.

---

### Pillar F: Always-on daemon — System service, not CLI tool

**The problem:** Currently Butler is a process you launch manually. Jarvis is always present.

**Approach:** Run Butler as a macOS user-space daemon via `launchd`.

```xml
<!-- ~/Library/LaunchAgents/com.user.butler.plist -->
<key>RunAtLoad</key><true/>
<key>KeepAlive</key><true/>
<key>ProgramArguments</key>
<array>
    <string>/path/to/voicebot</string>
</array>
```

**Idle mode:** Silero VAD running at ~2% CPU. Butler listens passively without transcribing. Wake word detection ("Butler" or a configurable phrase) activates the full pipeline. Between interactions, background tasks (inference daemon, calendar watcher, FSEvents) run normally.

**Wake word options:**
- Keyword spotting model (Porcupine, openWakeWord) — dedicated low-power process
- Silero VAD + whisper on short clips to check for the wake word — simpler, slightly more CPU

**This pillar makes Butler feel present** even when silent. The fact that it *could* speak at any moment changes the relationship from tool to companion.

---

## Model Architecture — Speed vs Capability

### The fundamental tradeoff

For voice, the critical metric is TTFT (Time To First Token). On Apple Silicon with 4-bit quantization:

| Model | TTFT approx. | Tokens/s | Voice verdict |
|-------|-------------|----------|---------------|
| Qwen2.5-0.5B | < 0.1s | 300+ | Too limited for reliable tool use |
| Qwen2.5-3B | ~0.3s | 150+ | Good for simple conversation |
| Qwen2.5-7B | ~0.8s | 60–80 | **Sweet spot — fast enough, capable enough** |
| Qwen2.5-14B | ~1.5s | 35–45 | Tolerable limit for direct voice response |
| Qwen2.5-32B | ~4s | 15–20 | Too slow for synchronous response |

TTFT > 2s feels sluggish in conversation. This rules out anything larger than 14B for the direct voice response path.

### Can small models handle tools and vision?

**Simple tools (current_time, run_shell with clear commands):** yes. Even 3B models follow well-defined tool call formats reliably.

**Complex tool chaining and ambiguous results:** no. Small models hallucinate formats, lose track between tool calls, or accept wrong results without noticing. The 7B starts to struggle where a 32B would not.

**Vision:** vision models are a separate family. A 2B vision model (e.g. Qwen2-VL-2B) handles basic screen reading well. Understanding complex code on screen or reasoning about diagrams requires 7B+ vision. The latency of a separate vision model call is acceptable if done asynchronously.

### The solution: coordinator pattern

The voicebot does not need one model doing everything. It needs a **fast coordinator** and **slow specialists**:

```
User speaks
     │
Jarvis (7B local, fast)
     │
     ├── Simple conversation ─────────────────► responds < 1s
     │
     ├── Simple tool (shell, time, file) ─────► executes + responds < 2s
     │
     ├── Needs vision ────────────────────────► "Déjame ver la pantalla"
     │                                          async vision model call
     │                                          result → responds
     │
     └── Complex task (research, coding,
         long analysis, deep reasoning) ──────► "Lo delego, te aviso"
                                                tokio::spawn → heavy agent
                                                result → proactive_tx → Jarvis vocalises
```

The 7B never blocks the voice thread. Everything requiring more capacity is delegated and returns via the proactive channel.

### Key UX insight on delegation latency

Latency is only painful when the user is **waiting in silence**. If Jarvis immediately says "Lo investigo, te aviso en un momento" and the result comes back 30 seconds later as a proactive message, those 30 seconds are invisible. The conversation is not blocked.

This is exactly the pattern tested with OpenClaw in the butler project — the design was correct, the missing piece was the immediate acknowledgment + async return.

### Recommended model assignment

| Role | Model | Rationale |
|------|-------|-----------|
| Voice conversation + simple tools | Qwen2.5-7B-Instruct-4bit | Fast TTFT, reliable tool following |
| Passive context (screen state) | Qwen2-VL-2B or similar | Runs every 30s in background, low cost |
| Vision on demand | LLaVA-7B local or Claude API | Better quality, called async |
| Complex reasoning / research | Claude claude-sonnet-4-6 API | Best quality, latency is acceptable async |
| Code tasks | Claude Code or aider | Specialist, runs async, reports back |

The 7B is the voice; Claude (or any capable remote model) is the brain for hard problems.

---

### 8. Conversation Awareness

**Goal:** The voicebot must ignore audio that is not directed at it — other people talking in the room, TV, radio, phone calls — and respond only when the user is genuinely speaking to it.

This is one of the hardest unsolved problems in always-on voice assistants. It has two independent sub-problems:

1. **Who is speaking?** (speaker identification)
2. **Are they speaking to me?** (intent / address detection)

Both must be solved for the voicebot to work reliably in a real home or office environment.

---

#### Sub-problem 1: Who is speaking?

**The problem:** Silero VAD fires on any voice — user, partner, TV, podcast, colleague on a call. Everything gets transcribed and potentially sent to the LLM.

**Solution: Speaker verification (not diarization)**

Speaker *diarization* labels every speaker in a long recording. Speaker *verification* answers a simpler question: "Is this clip the enrolled user?" Verification is realtime-friendly — it runs on a 1–3 second clip in under 50ms.

**How it works:**
1. **Enrollment:** Record 5–10 seconds of the user's voice at first startup → extract a speaker embedding vector → store it
2. **Runtime:** After each VAD segment, extract an embedding → compute cosine similarity with the enrolled vector → if similarity < threshold (e.g. 0.70), discard the audio without transcribing it

**Embedding models available locally:**
- **3D-Speaker** (Alibaba) — good accuracy, ONNX model ~30MB
- **WeSpeaker** (Tongji) — fast and compact, multiple ONNX variants
- **ECAPA-TDNN** (SpeechBrain) — state of the art; exportable to ONNX
- **Silero Speaker ID** — same team as the VAD library; tiny and fast

**Rust integration:** [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) provides Rust bindings and ships prebuilt speaker recognition models. It is the most practical path for this project since Whisper and VAD are already the bottlenecks.

**Threshold tuning:**
- Too tight (0.85+): false negatives when user has a cold, is tired, or speaks quietly
- Too loose (0.50–): other people's voices pass through
- Practical starting point: 0.72–0.78; expose as `SPEAKER_SIMILARITY_THRESHOLD` env var

**What it cannot do:**
- A voice identical to the user (twin) will pass — not a realistic concern
- If the user's voice changes dramatically (illness, aging), re-enrollment is needed
- Does not distinguish "user speaking to bot" from "user speaking to someone else" — that is sub-problem 2

**Limits of current technology:**
- Real-time streaming diarization (multiple simultaneous speakers, labeled live) works in research but is not production-ready at < 100ms latency
- Speaker ID on very short clips (< 0.5s) is unreliable — need at least 1–2s of clean speech

---

#### Sub-problem 2: Is the user speaking to me?

Even after confirming it is the user's voice, they may be talking to another person in the room, on the phone, or just thinking aloud.

**This is the harder problem.** No perfect solution exists today. The approaches below can be combined into layers, with each layer filtering out false positives before the next.

---

**Layer 1 — Wake word (most reliable, least natural)**

The user explicitly activates the bot by saying a keyword ("Jarvis", "Butler", "Hey").

- Latency: < 10ms (dedicated tiny model, always listening)
- False positive rate: near zero
- UX cost: unnatural; users must consciously change their speech behaviour
- Libraries: [Porcupine](https://github.com/Picovoice/porcupine) (proprietary), [openWakeWord](https://github.com/dscripka/openWakeWord) (open, Python), [rustpotter](https://github.com/GiviMAD/rustpotter) (Rust, open)

Wake word solves the problem completely but kills the "ambient companion" feel. Jarvis does not wait for a magic word.

---

**Layer 2 — Conversation state machine (zero cost)**

Track whether the voicebot is in an "active conversation":
- After the bot speaks, it is in **active mode** for N seconds (default: 15s)
- In active mode, respond to anything from the user's voice (speaker ID already confirmed)
- After N seconds of user silence, drop back to **ambient mode**
- In ambient mode, require a higher confidence signal (layers 3 or 4)

This alone handles the most common case: once a conversation starts, it flows naturally. The problem is only the first utterance of a cold session.

```rust
enum ConversationState {
    Ambient,            // bot has not spoken recently; require strong address signal
    Active { until: Instant }, // within N seconds of last bot turn; respond freely
}
```

---

**Layer 3 — Linguistic address detection (fast heuristic)**

Analyze the transcript for markers that indicate the user is speaking to the bot:

| Signal | Examples | Weight |
|--------|----------|--------|
| Bot name | "Jarvis, …" / "Butler, …" | Strong |
| Second-person imperative | "search X", "tell me", "remind me", "check…" | Strong |
| Direct question | "what time is it?", "can you…", "do you know…" | Medium |
| First-person with bot context | "I need you to…", "can you help me…" | Medium |
| Pure declarative to someone | "I think we should…", "look at this" | Weak/None |

A simple regex/keyword pass takes < 1ms. A small transformer classifier trained on labelled examples takes ~5ms.

False positive sources: user quoting commands while talking to a colleague ("I told him to search for X"), TV dialogue with similar patterns.

---

**Layer 4 — LLM address classification (accurate but adds latency)**

After transcription, call a small fast model with:

```
System: "Decide if the following utterance is directed at you (a voice AI assistant named Jarvis) or at someone else. Answer only YES or NO."
User: "{transcript}"
```

A 0.5B–1B model answers in < 100ms and is surprisingly accurate at this binary task. It can leverage full linguistic context — tone, topic, conversational markers — that a keyword heuristic misses.

The main cost is latency: ~100–300ms before the main LLM call begins. This can be mitigated by running both in parallel and aborting the main call if classification returns NO.

---

**Layer 5 — Audio context signals (future)**

Signals available from raw audio that provide context without transcription:

| Signal | What it tells you | How to detect |
|--------|-------------------|---------------|
| Multiple simultaneous speakers | User is in a conversation with someone | Overlap detection (spectral) |
| Audio from speakers (TV/radio) | Broadcast, not conversation | Acoustic fingerprinting or VAD on playback channel |
| Phone call (earpiece audio) | User is on the phone | DTMF tones, network audio fingerprint |
| Silence before utterance | User is probably addressing bot directly | VAD gap analysis |

None of these are implemented in any consumer voice assistant today at production quality. They are research-grade.

---

#### Recommended architecture — environment-triggered mode switch

The key insight is to stop trying to infer "is the user talking to me?" (hard, unsolvable) and instead answer a simpler proxy question: **"is there another person in this environment?"** (solvable with speaker ID). The presence of a non-main-user voice is a reliable social-context signal that warrants more conservative behaviour.

**State machine:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ACTIVE MODE  (default when alone)                                       │
│  • Main user voice → respond normally                                    │
│  • Non-main-user voice detected (N consecutive clips) → → AMBIENT       │
└─────────────────────────────────────────────────────────────────────────┘
              ↓ N non-main-user clips                 ↑ AMBIENT_CLEAR_SECS
                                                        of clean environment
┌─────────────────────────────────────────────────────────────────────────┐
│  AMBIENT MODE  (guest / TV / radio present)                              │
│  • Any voice → transcribe → check for wake word ("Jarvis, …")           │
│      Wake word found → respond to that turn, stay in AMBIENT            │
│      No wake word → discard, even if speaker is the main user           │
│  • Silence for AMBIENT_CLEAR_SECS (e.g. 5 min) → → ACTIVE              │
└─────────────────────────────────────────────────────────────────────────┘
```

**Why this works for each problem scenario:**

| Scenario | Handled by |
|----------|-----------|
| TV / radio | Non-main-user voices → ambient. Wake word required. Bot stays silent. |
| Guest in the room | Guest voice → ambient. User talking to guest → no wake word → silent. |
| User on phone | Phone's audio channel (remote voice) → ambient. User speaking to phone → silent unless they say the wake word. |
| User alone, talking to bot | No non-main-user voice → stays active. Normal conversation. |
| User alone after guest leaves | AMBIENT_CLEAR_SECS of silence/main-user-only → back to active. |
| User quoting a command | Still in active mode → false positive possible. Linguistic heuristics help. |

**The critical design decisions:**

1. **N consecutive non-main-user detections to trigger ambient** (not just one). A single bad speaker ID frame (user with a cold, room echo) should not drop the bot into ambient. Use N=3 as default.

2. **Wake word in ambient mode does not require a speaker ID match** — a guest can also address the bot with "Jarvis, what time is it?" This is natural and intentional. The wake word is the explicit signal, regardless of who says it.

3. **Wake word detection is transcript-based** — no separate wake word model needed. Whisper already runs on every VAD segment. Check if the transcript starts with (or contains near the start) the bot's name or configured keyword. This adds zero latency to the existing pipeline.

4. **Return to active is time-based** — after `AMBIENT_CLEAR_SECS` of hearing only the main user's voice or silence, the state machine resets to active. This handles guests leaving naturally without requiring an explicit "goodbye" command.

5. **Active mode still benefits from linguistic heuristics** — when alone and in active mode, Layer 3 address heuristics can suppress obvious non-bot-directed speech (user thinking aloud, muttering to themselves). This is an optional refinement.

**Full pipeline with mode switching:**

```
VAD fires → speaker ID embedding extracted
    │
    ├── Main user (similarity ≥ threshold)
    │       │
    │       ├── ACTIVE mode  → transcribe → LLM → TTS
    │       │                   (+ optional linguistic check)
    │       │
    │       └── AMBIENT mode → transcribe → wake word check
    │                               │
    │                           wake word found → LLM → TTS
    │                           no wake word    → discard
    │
    └── Non-main-user (similarity < threshold)
            │
            ├── Increment non-user counter
            │       counter ≥ N → switch to AMBIENT
            │
            └── AMBIENT mode → transcribe → wake word check
                                    │
                                wake word found → LLM → TTS
                                no wake word    → discard
```

**Latency impact:** speaker ID adds ~50ms per VAD segment. Wake word check on the transcript is a string search, < 1ms. Total overhead on the hot path (main user, active mode) is ~50ms, acceptable before the Whisper and LLM steps.

---

#### What this approach cannot fully solve

| Scenario | Status | Notes |
|----------|--------|-------|
| TV / radio rejection | ✅ Solved | Non-main-user voices → ambient; wake word required |
| Guest in the room | ✅ Solved | Guest voice triggers ambient; user must use wake word |
| User on phone | ✅ Largely solved | Remote voice triggers ambient; conversational speech suppressed |
| User speaking to guest (active mode, before guest speaks) | ⚠️ Small window | Bot may respond to first utterance before the mode switch fires. N=3 threshold limits this. |
| User quoting commands while alone | ⚠️ Partial | Linguistic heuristics help; not perfect |
| Re-entry latency | ⚠️ By design | User must wait AMBIENT_CLEAR_SECS after guest leaves before active mode resumes. Configurable. |
| Real-time speaker ID on very short clips (< 0.5s) | ⚠️ Unreliable | Need ≥ 1s of speech for accurate embedding. Short interjections may misclassify. |
| Real-time streaming diarization | ❌ Research | Not needed with this approach — per-segment speaker verification is sufficient |

**Overall assessment:** this approach makes the conversation awareness problem **practically solvable** for a personal home or office assistant. The two unsolved cases from before — user talking to someone else, and TV/radio — are both addressed. The remaining gaps (the small window before mode switches, and quoting edge cases) are minor in practice and acceptable for a personal assistant.

---

#### Implementation steps

1. **Speaker enrollment at startup** — detect if `SPEAKER_ENROLLMENT_PATH` exists; if not, record 10s of the user's voice and extract + save the embedding
2. **`SpeakerVerifier` struct** — wraps a sherpa-onnx speaker recognition model; `verify(audio: &[f32]) -> f32` returns cosine similarity
3. **`ConversationState` enum** — `Active` / `Ambient { non_user_streak: u8, last_non_user: Instant }`
4. **Mode switch logic in VAD loop** — after each `SpeechEnd`: run speaker ID; update state; gate main pipeline accordingly
5. **Wake word check** — in `Ambient` state: transcribe → check if transcript contains bot name at start (`transcript.to_lowercase().contains("jarvis")`)
6. **Auto-return to active** — background timer or checked on each event: if last non-main-user voice was > `AMBIENT_CLEAR_SECS` ago → switch to Active

**Config vars to add:**
```
SPEAKER_ENROLLMENT_PATH   path to stored embedding (default: data/speaker.emb)
SPEAKER_SIMILARITY_MIN    cosine similarity threshold (default: 0.75)
SPEAKER_AMBIENT_TRIGGER   consecutive non-user clips to enter ambient (default: 3)
AMBIENT_CLEAR_SECS        seconds of clean environment to return to active (default: 300)
WAKE_WORD                 keyword to respond in ambient mode (default: "jarvis")
```

---

## Implementation Status

### Conversational core

| Feature | Status | Notes |
|---------|--------|-------|
| STT → LLM → TTS streaming pipeline | ✅ Done | Whisper + llama.cpp + macOS say |
| Barge-in interruption | ✅ Done | `Arc<AtomicBool>` cancel, CPAL callback |
| Persistent conversation history | ✅ Done | SQLite, restored on startup |
| Tool use | ✅ Done | XML-based, extensible `Tool` trait, `current_time` |
| Context summarization | ✅ Done | Auto-trigger at 75% context; persisted in DB |
| User profile extraction + injection | ✅ Done | Background LLM; `user_profile` table |
| MCP integration | Planned | `src/mcp/`; JSON-RPC over stdio/HTTP |
| Agent delegation | ✅ Done | `run_agent` (sync) + `run_agent_async` (proactive); OpenAI-compatible |
| Voicebot as agent intermediary | Planned | Voice proxy over existing text agents |
| Conversation awareness | Planned | Speaker ID (sherpa-onnx) + state machine + linguistic/LLM classifier |

### Butler pillars

| Pillar | Status | Quick description |
|--------|--------|-------------------|
| A — Character system prompt | ✅ Done | `LLM_SYSTEM_PROMPT` env var + Jarvis prompt in `.env` |
| B — Eyes (situational awareness) | Planned | Screenshot + vision model; system state injection |
| C — Arms (computer agency) | Planned | `run_shell` + file/app/clipboard/web tools |
| D — Voice of its own (proactive) | Planned | Inference daemon + event sources + `proactive_tx` |
| E — Episodic memory (embeddings) | Planned | sqlite-vec + embedding model; semantic recall |
| F — Always-on daemon | Planned | launchd plist + wake word detection |

### Recommended implementation order

1. **Pillar A** — Character prompt. Zero code, maximum immediate impact on feel.
2. **Pillar C** — `run_shell` tool. Unlocks real computer agency in one afternoon.
3. **Pillar B** — `take_screenshot` + vision. Butler gets eyes.
4. **Pillar D** — Proactive initiative. Butler speaks first; feels truly alive.
5. **MCP** — Vast tool ecosystem for free once the tool layer is mature.
6. **Pillar E** — Episodic memory. Butler remembers your history together.
7. **Pillar F** — Always-on daemon. Butler becomes a permanent presence.
8. **Agent delegation** ✅ — Complex tasks farmed to specialised agents (done).

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

## Kokoro TTS


### Instalar dependencia del sistema

`brew install espeak-ng`

### Descargar modelos (en el directorio del proyecto)

### kokoro-v1.0.onnx y voices-v1.0.bin desde HuggingFace onnx-community/Kokoro-82M-v1.0-ONNX

### Compilar con soporte Kokoro

`cargo build --features kokoro`

### Activar en .env

TTS_PROVIDER=kokoro
KOKORO_MODEL=models/kokoro-v1.0.onnx
KOKORO_VOICES=models/voices-v1.0.bin
KOKORO_VOICE=af_bella        # voz (ver lista con get_available_voices)
KOKORO_LANGUAGE=en-us        # código BCP-47 para espeak-ng

Arquitectura:

- TtsEngine — enum con variante Say(SayTts) y Kokoro(KokoroTts), compilada con #[cfg(feature = "kokoro")]
- TTS_PROVIDER=say (defecto) — sin cambios en build, sin espeak-ng
- TTS_PROVIDER=kokoro + --features kokoro — activa Kokoro
- Sin el feature, pedir kokoro falla con mensaje claro en runtime
- El resto del pipeline (stream_and_tts, run_pipeline, run_proactive_pipeline) es agnóstico al backend

Nota sobre voces en español: kokorox usa espeak-ng para fonetización. Para español pasa KOKORO_LANGUAGE=es. Las voces disponibles se pueden listar con
tts.inner.get_available_voices() si añades un tracing al startup — aunque el modelo base es principalmente en inglés, espeak-ng puede fonetizar español.

---

## License

Private project — all rights reserved.
