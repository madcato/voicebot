# Voicebot — Butler

A voice AI assistant built in Rust. The long-term goal is a digital butler: not a conversational chatbot, but a proactive, situationally-aware companion that anticipates needs, controls the computer, and speaks with a defined personality — without being asked.

> A chatbot answers questions. A butler anticipates needs.

## Run

```sh
WHISPER_COREML=1 TTS_PROVIDER=kokoro cargo run --features kokoro --release
```

---

## Vision

The gap between a chatbot and a proactive assistant is not the AI model — it is the surrounding architecture. Voicebot has:

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
Microphone → VAD → AudioBuffer → Whisper STT → mlx-lm/oMLX LLM → SentenceSplitter → TTS (AVSpeech/Kokoro) → Speaker
```

**Implemented features:**
- Real-time voice capture (CPAL), Silero VAD with pre-roll buffer
- Whisper.cpp STT (CoreML Neural Engine or Metal GPU, state cached across utterances)
- Streaming LLM via mlx-lm or oMLX (Apple MLX, implicit KV-cache reuse, sub-second latency)
- Sentence-by-sentence TTS playback — first sentence plays while next is being generated
- **Barge-in**: user speech cancels active LLM/TTS pipeline instantly via `CancellationToken`
- Persistent SQLite conversation history — restored on startup
- LLM session rollback on barge-in interruption
- **Tool use**: XML-based tool calls (`<tool_name>...</tool_name>`) plus multi-tool loop; built-in tools: `current_time`, `take_screenshot`, `read_clipboard`, `set_clipboard`, `open_app`, `run_shell`, `read_file`, `web_search`, `set_conversation_mode`, `mcp_tool` (dynamic MCP server proxy), `run_agent` (subprocess + ACP delegation)
- **Agent delegation**: `run_agent` (sync) and `run_agent_async` (background + proactive announce) via stdin/stdout subprocess; any CLI agent; proactive channel in VAD loop
- **Context summarization**: auto-triggers when context exceeds configurable threshold (default 90%, `LLM_CONSOLIDATION_THRESHOLD_PCT`); keeps last N turns verbatim; idle consolidation after inactivity (`LLM_IDLE_CONSOLIDATION_SECS`, default 30 min); summary persisted in DB and restored on restart
- **User profile**: background LLM extraction of user facts after every turn; stored in `user_profile` SQLite table; injected into system prompt on startup
- **Startup greeting**: bot speaks first on process launch — greets by name if known, asks for it otherwise
- **Kokoro TTS**: high-quality ONNX-based TTS via `kokorox` crate (24 kHz); selectable via `TTS_PROVIDER`; AVSpeechSynthesizer is the macOS default
- **Background inference overlap**: `maybe_summarize` and `extract_facts` launch while the last TTS sentence is still playing
- **EYES visual awareness**: periodic screen capture + secondary vision LLM decides if anything on screen warrants notifying the user
- **Inference daemon**: background "is there anything worth saying?" loop that pushes proactive speech when relevant
- **MCP integration**: JSON-RPC 2.0 stdio client; dynamically discovers and proxies tools from MCP servers
- **Control API**: HTTP + SSE endpoints for remote monitoring and control (`/control/state`, `/control/events`, `/control/mute`, `/control/barge_in`, `/control/input`, `/control/history`)
- **Conversation modes**: ambient state machine with wake-word activation, auto-return to active mode
- **Speaker verification**: sherpa-onnx speaker embedding with auto-enrollment, multi-profile support
- **Secondary LLM**: routes summarization, profile extraction, and vision to a separate model/provider

---

## Architecture

### Voicebot as Pure Voice Layer

The voicebot is deliberately narrow in scope: it owns the audio pipeline and the conversational voice experience. Everything else — computer control, file access, calendar, web, vision, long-running tasks — belongs to an external agent.

This separation exists for one reason: **response latency**. The more tools and context the voicebot LLM has to consider, the slower the first token arrives. A voice model that only knows about conversation and a handful of instant local operations responds in under 1 second. A voice model that also thinks about shell commands, file trees, and calendar APIs is slower and less reliable.

```
┌────────────────────────────────────────────────────────────┐
│  VOICEBOT  (voice layer — always fast)                      │
│                                                             │
│  • STT → LLM (fast 7B) → TTS                               │
│  • Barge-in, conversation awareness, speaker ID            │
│  • Proactive speech (inference daemon)                      │
│  • Voice-local tools: time, screenshot, clipboard,          │
│    open_app, shell, file reader, web search, MCP proxy      │
│                                                             │
│  When task is complex or requires computer agency:          │
│    run_agent(task) ──stdin/stdout──► AGENT                  │
│    run_agent_async(task) ─────────► AGENT (background)      │
│                              result ◄── proactive_tx        │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│  EXTERNAL AGENT  (capable — latency acceptable)             │
│                                                             │
│  • Larger LLM, full tool suite                             │
│  • Eyes: screenshot + vision, system state, calendar       │
│  • Arms: shell, file r/w, web search, browser, email       │
│  • Memory: long-term, episodic                             │
│  • Any future capability without touching voicebot          │
└────────────────────────────────────────────────────────────┘
```

**Voicebot tools (voice-layer only — instant, no blocking):**

| Tool | Latency | Why here |
|------|---------|----------|
| `current_time` | <1ms | Trivial; no subprocess |
| `take_screenshot` + vision | ~500ms | Needs screen access before speaking |
| `read_clipboard` / `set_clipboard` | <50ms | Instant `pbpaste`/`pbcopy` |
| `open_app` | <100ms | Instant `open -a` |
| `run_shell` | varies | Gated by `SHELL_ENABLED=1`; bounded by `SHELL_TIMEOUT_SECS` |
| `read_file` | varies | Local file read; bounded by size limit |
| `web_search` | ~1-3s | SearXNG proxy |
| `mcp_tool` | varies | Dynamic proxy to connected MCP servers |
| `run_agent` / `run_agent_async` | varies | The delegation bridge |

Everything else — calendar, email, long-running file operations — goes to the agent.

**Agent integration via stdin/stdout:**

The agent is invoked as a subprocess. This is intentionally simple: any CLI agent that reads from stdin and writes to stdout works without modification. Switching agents (Hermes today, something else tomorrow) requires changing one env var.

```
AGENT_COMMAND=hermes        # CLI command to invoke
AGENT_TIMEOUT_SECS=120      # Timeout for synchronous calls
```

Protocol:
1. Voicebot spawns the agent process
2. Writes the task (plain text) to its stdin, then closes stdin
3. Reads the complete response from stdout
4. Returns the response (sync) or pushes to `proactive_tx` (async)

This is a fire-and-read pattern — no JSON protocol, no shared state, no persistent daemon. Each delegation is an isolated subprocess invocation.

---

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
                    │  Whisper.cpp + CoreML/Metal│
                    └─────────────┬─────────────┘
                                  │  transcript
                    ┌─────────────▼─────────────┐
                    │  LLM (streaming HTTP SSE)  │
                    │  mlx-lm / oMLX (streaming) │
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
                    │  TTS (AVSpeech or Kokoro)   │
                    │  sentence-by-sentence       │
                    └─────────────┬──────────────┘
                                  │  f32 PCM
                    ┌─────────────▼─────────────┐
                    │  AudioOutput (CPAL)        │
                    │  resample + play_blocking  │
                    └───────────────────────────┘
                                  │  (last sentence still playing)
                    ┌─────────────▼─────────────┐
                    │  maybe_summarize() [spawn] │  ← overlaps with last audio
                    │  extract_facts()  [spawn]  │  ← GPU busy while speaker plays
                    └─────────────┬─────────────┘
                    await last play handle
                    └───────────────────────────┘
```

### Key modules

| Module | File | Description |
|--------|------|-------------|
| Audio capture | `src/audio/audio_capture.rs` | CPAL mic input, normalizes to f32 |
| Audio buffer | `src/audio/buffer.rs` | Accumulates speech chunks |
| Audio output | `src/audio/output.rs` | CPAL playback, resample, cancel support |
| STT + VAD | `src/stt/mod.rs` | `WhisperSTTVAD` integrates whisper-cpp-plus STT with Silero VAD, pre-roll buffer, state machine |
| LLM client | `src/llm/client.rs` | Streaming SSE + one-shot completion (OpenAI-compatible; mlx-lm and oMLX) |
| LLM session | `src/llm/session.rs` | Accumulated prompt, turn tracking, summarization |
| TTS AVSpeech | `src/tts/avspeech.rs` | macOS AVSpeechSynthesizer (default on macOS, `--features avspeech`) |
| TTS Kokoro | `src/tts/kokoro.rs` | Kokoro ONNX TTS (`--features kokoro`) |
| Sentence splitter | `src/tts/sentence.rs` | Buffers tokens, emits complete sentences |
| Tools | `src/tools/` | `Tool` trait + `ToolRegistry`; 10+ built-in tools plus dynamic MCP proxy |
| MCP Client | `src/mcp/mod.rs` | JSON-RPC 2.0 stdio client; discovers + proxies tools from MCP servers |
| Agents | `src/agents/mod.rs` | `ProactiveEvent` enum; proactive speech channel |
| Inference daemon | `src/daemon.rs` | Background "is there anything worth saying?" loop |
| EYES | `src/eyes.rs` | Periodic screen capture + secondary vision LLM for proactive notifications |
| Control API | `src/control/` | HTTP + SSE: `/control/state`, `/control/events`, `/control/mute`, `/control/barge_in`, `/control/input`, `/control/history` |
| Profile | `src/profile/mod.rs` | User fact extraction, JSON parsing, context builder |
| Database | `src/db/database.rs` | SQLite: sessions, messages, summary, user_profile |
| Config | `src/config.rs` | Environment-based configuration |
| Main loop | `src/main.rs` | VAD loop + barge-in + pipeline + summarization + profile |

---

## Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `VOICEBOT_LANGUAGE` | `es` | Language for STT and TTS voice selection |
| `VAD_SILENCE_MS` | `500` | Silence duration (ms) before SpeechEnd fires. Lower = faster response; higher = safer for mid-sentence pauses. Range: 400–1500. This is the largest single contributor to perceived latency after your last word. |
| `WHISPER_MODEL` | — | Path to `.bin` Whisper model (and `.mlmodelc` for CoreML encoder) |
| `WHISPER_COREML` | `0` | Set to `1` to use CoreML encoder (Neural Engine); requires `.mlmodelc` alongside the `.bin` |
| `LLM_URL` | `http://localhost:8000` | LLM server base URL. mlx-lm default: port 8000; oMLX default: port 8001 |
| `LLM_API_KEY` | _(empty)_ | Bearer token sent as `Authorization: Bearer <key>` on all `/v1/chat/completions` calls. Leave unset for local servers that require no auth |
| `LLM_MODEL` | `local-model` | Model name sent in API requests |
| `LLM_SYSTEM_PROMPT` | — | System prompt for the LLM |
| `LLM_MAX_TOKENS` | `1024` | Max generation tokens per response |
| `LLM_TEMPERATURE` | `0.7` | LLM sampling temperature |
| `LLM_CONTEXT_TOKENS` | `4096` | Approximate context window size in tokens |
| `LLM_CONSOLIDATION_THRESHOLD_PCT` | `90` | Percentage of context window that triggers consolidation (default 90%) |
| `LLM_IDLE_CONSOLIDATION_SECS` | `1800` | Seconds of inactivity before silent consolidation runs (0 = disabled) |
| `LLM_IDLE_MIN_CONTEXT_PCT` | `50` | Minimum context fill % required for idle consolidation to run |
| `LLM_SUMMARY_KEEP_TURNS` | `6` | Recent (role, content) turns kept verbatim after summarization |
| `AGENT_COMMAND` | — | CLI command to invoke the agent (e.g. `hermes chat`). Unset = agent tools disabled. |
| `AGENT_MODE` | `cli` | Agent communication mode: `cli` (subprocess) or `acp` (persistent ACP JSON-RPC stdio) |
| `AGENT_ACP_COMMAND` | `hermes acp` | Command to start the ACP process (used when `AGENT_MODE=acp`) |
| `AGENT_TIMEOUT_SECS` | `120` | Hard timeout in seconds for synchronous agent calls |
| `TTS_PROVIDER` | `avspeech` | TTS backend: `avspeech` (native AVSpeechSynthesizer, requires `--features avspeech`, default) or `kokoro` (requires `--features kokoro`) |
| `AVSPEECH_VOICE` | `Jorge (Enhanced)` | Voice display name for AVSpeechSynthesizer (used when `TTS_PROVIDER=avspeech`). List voices: `say -v ?` |
| `AVSPEECH_RATE` | `0.55` | Normalized speech rate 0.0–1.0 for AVSpeechSynthesizer (`0.5` = default ≈ 180 wpm, `0.55` ≈ 215 wpm) |
| `KOKORO_MODEL` | `models/kokoro-v1.0.onnx` | Path to Kokoro ONNX model file |
| `KOKORO_VOICES` | `models/voices-v1.0.bin` | Path to Kokoro voice embeddings file |
| `KOKORO_VOICE` | `af_bella` | Kokoro voice style name (see available voices via `get_available_voices`) |
| `KOKORO_LANGUAGE` | `en-us` | BCP-47 language code passed to espeak-ng for phonemisation |
| `SHELL_ENABLED` | `0` | Set to `1` to enable the `run_shell` tool. Off by default for safety. |
| `SHELL_TIMEOUT_SECS` | `30` | Hard timeout (seconds) per shell command |
| `AUDIO_INPUT_DEVICE` | system default | Input device name substring |
| `AUDIO_OUTPUT_DEVICE` | system default | Output device name substring |
| `DB_PATH` | `data/voicebot.db` | SQLite database file path |
| `LIST_AUDIO_DEVICES` | `0` | Print devices and exit |
| `SPEAKER_MODEL` | auto-detect | Path to speaker embedding ONNX model. Auto-detected from `models/speaker_embedding.onnx` if it exists. |
| `SPEAKER_ENROLLMENT_PATH` | `data/speaker.emb` | Path where the enrolled speaker embedding is persisted. Delete to re-enroll. |
| `SPEAKER_SIMILARITY_MIN` | `0.45` | Cosine similarity threshold [0–1]. Higher = stricter. 0.45 permissive, 0.55 strict. |
| `LLM_HISTORY_LOAD_LIMIT` | `0` | Max messages loaded from DB on startup (0 = unlimited). Recommended: 40–60 to prevent restart compaction. |
| `WAKE_WORD` | `jarvis` | Case-insensitive substring match to trigger response in Ambient mode |
| `AMBIENT_CLEAR_SECS` | `300` | Seconds in Ambient mode with no speech before auto-returning to Active |
| `DAEMON_ENABLED` | `0` | Set to `1` to enable the inference daemon (background "is there anything worth saying?" loop) |
| `DAEMON_INTERVAL_SECS` | `300` | Seconds between inference daemon checks |
| `EYES_INTERVAL_SECS` | `0` | Seconds between screen-capture checks for EYES (0 = disabled). Requires SECONDARY_LLM_URL. |
| `SECONDARY_LLM_URL` | — | Base URL for secondary LLM (vision, summarization, profile extraction). Unset = disabled. |
| `SECONDARY_LLM_MODEL` | — | Model name for secondary LLM requests |
| `SECONDARY_LLM_MAX_TOKENS` | `1024` | Max tokens for secondary LLM responses |
| `SECONDARY_LLM_API_KEY` | _(empty)_ | Bearer token for secondary LLM API |
| `SECONDARY_LLM_THINKING` | `0` | Enable Qwen3 thinking mode on secondary LLM (auto-strips thinking tags) |
| `MCP_COMMAND` | — | Command to spawn MCP server subprocess (e.g. `bunx apple-mcp@latest`). Unset = MCP disabled. |
| `MCP_TOOL_TIMEOUT_SECS` | `30` | Hard timeout in seconds per MCP tool call |
| `CONTROL_PORT` | — | HTTP/SSE control API port (requires `--features control`). Unset = disabled. |

---

## Logging and debugging

Every subsystem has its own `RUST_LOG` target. Use them to see only the logs you care about without the noise from unrelated parts of the pipeline.

### Targets

| Target | What it covers |
|--------|----------------|
| `voicebot` | Startup, language, feature flags, ready, shutdown |
| `audio` | Microphone/speaker device selection, CPAL events, VAD probabilities, playback errors |
| `speaker` | Speaker verification enabled/disabled, enrollment, per-utterance similarity verdict |
| `stt` | Whisper model load, transcription errors, empty-transcript skips |
| `llm` | LLM endpoint, stream errors, summarization lifecycle |
| `tts` | TTS provider init, per-sentence synthesis, WAV header |
| `pipeline` | Barge-in, SpeechEnd, User/Assistant turns, tool calls, cancel/rollback |
| `db` | Session restore/create, message save errors, summary persistence |
| `daemon` | Inference daemon ticks and proactive messages |
| `profile` | User fact extraction, load count, save errors |
| `performance` | Per-turn timing: `STT Xms`, `LLM PP Xms`, `LLM TG Xms`, `TTS IN Xms` |

`LLM PP` = prefill latency (time to first token). `LLM TG` = total generation time. `TTS IN` = synthesis time per sentence (measured inside the blocking task, excludes playback wait).

### Examples

```sh
# Only conversation turns — what the user said and what the bot replied
RUST_LOG=pipeline=info cargo run

# Pipeline + latency breakdown
RUST_LOG=pipeline=info,performance=debug cargo run

# Debug the audio stack (device selection, VAD, playback)
RUST_LOG=audio=debug cargo run

# Debug speaker verification only
RUST_LOG=speaker=debug cargo run --features speaker

# Focus on STT + LLM, silence everything else
RUST_LOG=stt=info,llm=info cargo run

# Full trace on one subsystem while keeping others quiet
RUST_LOG=audio=trace,voicebot=info cargo run

# Everything at debug level (noisy but complete)
RUST_LOG=debug cargo run
```

The default (`RUST_LOG` unset) shows `info` level across all targets.

---

## Latency tuning

Target: **< 1 second** from end of user speech to first bot audio.

### Full latency breakdown

```
User last word
  └─ VAD silence wait: VAD_SILENCE_MS (default 500ms)  ← biggest single lever
       └─ SpeechEnd fires
            └─ Pipeline spawn lag: ~1ms
            └─ Speculative prefill: running since SpeechEnd
            └─ STT: ~150–550ms (model-dependent)
            └─ LLM PP: ~50–450ms (prefill + first token)
            └─ TTS IN: ~70–560ms (synthesis of first sentence)
            └─ FIRST AUDIO ← logged as "E2E (SpeechEnd→first audio): Xms"
```

Watch the `performance` target to see all stages:
```sh
RUST_LOG=performance=info,performance=debug cargo run
```

### VAD silence threshold

`VAD_SILENCE_MS=500` (default) means the bot waits 500ms of silence after your last
word before starting the pipeline.  Reduce to `400` for snappier feel; increase to
`700–800` if the bot cuts you off during natural pauses.

### LLM server (`scripts/start-llm.sh`)

Jarvis uses Apple MLX-based servers for low-latency inference on Apple Silicon.
Both are substantially faster than llama.cpp due to the MLX framework and Apple unified memory.

**mlx-lm** (default, port 8000):

```sh
# Install mlx-lm
pip install mlx-lm

# Launch (downloads model on first run)
./scripts/start-mlx-lm.sh mlx-community/Qwen3-8B-4bit

# .env settings
LLM_URL=http://127.0.0.1:8000
LLM_MODEL=mlx-community/Qwen3-8B-4bit
```

**oMLX** (persistent tiered KV cache, port 8001):

```sh
./scripts/start-omlx.sh ~/models

# .env settings
LLM_URL=http://127.0.0.1:8001
```

Or use the unified launcher:
```sh
./scripts/start-llm.sh                        # mlx-lm (default)
LLM_BACKEND=omlx ./scripts/start-llm.sh      # oMLX
```

Recommended models for voicebot (low latency + Spanish support):

| Model | VRAM | Notes |
|-------|------|-------|
| `mlx-community/Qwen3-8B-4bit` | ~5 GB | Latest architecture; `<think>` blocks auto-stripped |
| `mlx-community/Qwen2.5-7B-Instruct-4bit` | ~4 GB | Fast, good Spanish support |
| `mlx-community/Qwen2.5-14B-Instruct-4bit` | ~8 GB | Better quality, still fast on M4 Pro |

### Real-server benchmark (`scripts/bench-server.py`)

The most realistic benchmark: starts each server, warms its KV cache with a
multi-turn conversation, then measures **only the final turn** — the hot-cache
scenario that governs real voicebot latency.

```sh
python3 scripts/bench-server.py <mlx-model-or-hf-repo> <omlx-model-dir>

# Example
python3 scripts/bench-server.py \
  mlx-community/Qwen3-8B-4bit \
  ~/models

# Env overrides
BENCH_TRIALS=5 BENCH_GEN=100 python3 scripts/bench-server.py ...
```

Metrics: **TTFT** (ms, time to first spoken word) and **TG** (t/s, sentence
completion speed). Both servers are configured to match their production
`start-*.sh` scripts.

### Cold-inference benchmarks (`scripts/bench-mlx.sh`)

Benchmark mlx-lm under voicebot-realistic workloads:

```sh
./scripts/bench-mlx.sh mlx-community/Qwen3-8B-4bit

# Override trials
BENCH_TRIALS=5 ./scripts/bench-mlx.sh mlx-community/Qwen3-8B-4bit
```

Scenarios:

| Scenario | Prompt | Gen | What it measures |
|----------|--------|-----|-----------------|
| **cold** | 300 pp | 100 tg | First turn — full system prompt + history needs prefill |
| **warm** | 40 pp | 100 tg | Subsequent turns — only the new user utterance needs prefill |
| **long** | 800 pp | 120 tg | Long conversation — how throughput degrades at large context |

Key thresholds for real-time feel:
- **warm pp > 500 t/s** → TTFT < 80ms for a typical user turn
- **tg > 60 t/s** → 100-token response synthesized in < 1.7s (TTS keeps up)

### Whisper model size tradeoff

| Model | STT latency | Accuracy |
|-------|------------|----------|
| `ggml-tiny.bin` | ~80 ms | Low |
| `ggml-base.bin` | ~150 ms | Good |
| `ggml-small.bin` | ~250 ms | Very good |
| `ggml-large-v3-turbo.bin` | ~400 ms | Best |

Smaller model = faster STT = more speculative prefill overlap = lower total latency.
With CoreML encoder (`WHISPER_COREML=1`) each tier is ~30–50% faster.

---

## Commands

```bash
cargo build --release
cargo run --release

# With Kokoro TTS (requires espeak-ng and ONNX model files)
cargo build --features kokoro --release
TTS_PROVIDER=kokoro cargo run --features kokoro --release

# With CoreML STT encoder (Neural Engine; requires converted .mlmodelc)
WHISPER_COREML=1 cargo run --release

# Fully accelerated
WHISPER_COREML=1 TTS_PROVIDER=kokoro cargo run --features kokoro --release

cargo test
cargo fmt
cargo clippy
cargo run -- --list-devices
```

### End-to-end tests

E2E tests exercise the full STT → LLM → TTS → DB pipeline. They are marked `#[ignore]` and must be run explicitly (never on every build):

```bash
# All E2E tests (require audio output device)
cargo test e2e -- --ignored --nocapture

# A specific scenario
cargo test e2e::basic_conversation_mocked_transcript -- --ignored --nocapture

# Real-STT tests (also require WHISPER_MODEL + tests/fixtures/hola.wav)
cargo test e2e::stt_ -- --ignored --nocapture
```

See [`e2e.md`](e2e.md) for full documentation: how to record WAV fixtures, required environment variables, and what each test scenario verifies.

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

**Built-in tools:**

| Tool | Description |
|------|-------------|
| `current_time` | Returns local date and time |
| `take_screenshot` | Captures screen, returns base64 PNG |
| `read_clipboard` | Reads current clipboard contents |
| `set_clipboard` | Writes text to the clipboard |
| `open_app` | Launches a macOS application by name |
| `run_shell` | Executes a shell command (disabled by default, requires `SHELL_ENABLED=1`) |
| `read_file` | Reads contents of a file path |
| `web_search` | Searches the web via SearXNG |
| `set_conversation_mode` | Switches between Active and Ambient modes |
| `run_agent` / `run_agent_async` | Delegates to an external CLI agent (sync or background) |
| `mcp_tool` | Dynamic proxy: discovers and calls tools from connected MCP servers |

---

### 2. MCP (Model Context Protocol) Integration ✅ Implemented

**Goal:** Connect to MCP servers to expose a broad ecosystem of tools (filesystem, browser, GitHub, Slack, databases, etc.) without implementing each tool manually.

**Implementation:**

MCP uses JSON-RPC 2.0 over stdio (subprocess). The voicebot acts as an MCP client:

```
LLM tool_call
     │
     ▼
ToolRegistry
     │
     ├── built-in tools (Rust)
     ├── mcp_tool proxy → MCP subprocess (JSON-RPC 2.0 stdio)
     └── agents (run_agent / run_agent_async)
```

**`src/mcp/mod.rs`** — Full MCP client implementation:
- Spawns the MCP server subprocess (configurable via `MCP_COMMAND`)
- Performs the `initialize` handshake, discovers tools via `tools/list`
- Routes `tools/call` JSON-RPC requests with concurrent support (onshot channels keyed by request ID)
- Tool schemas from MCP are translated to OpenAI-compatible JSON Schema and injected into the LLM payload

**`src/tools/mcp_tool.rs`** — `McpToolProxy` bridges the `Tool` trait to the MCP client. At startup the proxy fetches `tools/list` and registers each discovered tool as a first-class tool in `ToolRegistry`. Dynamic tool calls are routed through `mcp_tool` with the server's tool name and arguments forwarded transparently.

**Config:**

| Env var | Default | Description |
|---------|---------|-------------|
| `MCP_COMMAND` | — | Command to spawn MCP server subprocess (e.g. `bunx apple-mcp@latest`). Unset = MCP disabled. |
| `MCP_TOOL_TIMEOUT_SECS` | `30` | Hard timeout per MCP tool call |

---

### 3. Agent Delegation ✅ Implemented

**Goal:** The LLM can delegate complex tasks (research, shell automation, file operations, calendar management, web browsing) to an external agent. The voicebot keeps its voice pipeline unblocked. Results return via the proactive channel and are spoken naturally.

**Two communication modes: `cli` and `acp`

**CLI mode (`AGENT_MODE=cli`, subprocess fire-and-read):**

**Synchronous (`run_agent` — tasks expected < 30s):**
```
LLM calls run_agent(task)
         ↓
    spawn subprocess: AGENT_COMMAND
    write task to stdin, close stdin
         ↓ wait for process to exit
    read stdout as result
    result injected as tool message
         ↓
    LLM re-called → streams spoken response → TTS
```

**Asynchronous (`run_agent_async` — long or uncertain duration):**
```
User: "Research X and tell me the summary"
LLM calls run_agent_async(task)
         ↓
    tokio::spawn background subprocess
    tool returns immediately → LLM acknowledges verbally (< 1s)
         ↓ (seconds or minutes later, in background)
    subprocess exits → read stdout
    ProactiveEvent::AgentResult pushed to proactive_tx
         ↓
    VAD loop receives event → spawns run_proactive_pipeline
         ↓
    LLM builds natural announcement → TTS plays proactively
```

**Implementation:**

- **`src/agents/mod.rs`** — `ProactiveEvent::AgentResult { task, result }` enum
- **`src/tools/run_agent.rs`** — `RunAgentTool` (sync) and `RunAgentAsyncTool` (async + proactive channel)
- **VAD loop** extended with `tokio::select!` watching both audio and `proactive_rx`
- **`run_proactive_pipeline`** — builds temporary message list from session + agent result, calls LLM, sends to TTS

**ACP mode (`AGENT_MODE=acp`, persistent JSON-RPC stdio):**

Instead of spawning a new subprocess for each delegation, the voicebot maintains a persistent connection to an ACP-compatible agent via JSON-RPC 2.0 over stdio. This enables:
- Bi-directional communication (agent can push updates back)
- No per-task startup overhead (agent stays warm)
- Shared session context across delegations

**Config vars:**

| Env var | Default | Description |
|---------|---------|-------------|
| `AGENT_COMMAND` | — | CLI command to invoke (e.g. `hermes chat`). Unset = agent tools disabled. |
| `AGENT_MODE` | `cli` | Communication mode: `cli` (subprocess per call) or `acp` (persistent JSON-RPC) |
| `AGENT_ACP_COMMAND` | `hermes acp` | Command to start ACP process (used when `AGENT_MODE=acp`) |
| `AGENT_TIMEOUT_SECS` | `120` | Hard timeout for synchronous agent calls |
| `AGENT_ACP_WARMUP` | `0` | Send warmup prompt at startup to preload the model (ACP mode only) |

**Agent protocol (CLI mode):** stdin/stdout. The voicebot writes the task as plain text to the subprocess stdin, closes it, and reads the complete response from stdout. Any CLI agent that follows this contract works — Hermes, Claude CLI, a custom Python script, or anything else. Switching agents requires changing only `AGENT_COMMAND`.

**Agent protocol (ACP mode):** JSON-RPC 2.0 over stdio with method-based calls (e.g. `chat/send_message`, `chat/terminate`). The voicebot manages the persistent process lifecycle and routes delegations through the shared connection.

**Why stdin/stdout and not HTTP:**
- No persistent service to manage — the agent starts only when needed
- No protocol negotiation — plain text in, plain text out
- Trivially portable to any CLI agent as the ecosystem evolves
- The async mode (`run_agent_async`) hides all latency: the user hears an immediate acknowledgment; the result arrives as proactive speech

**Tests:** `src/tools/run_agent.rs` — sync response, async channel delivery, error handling, timeout, round-trip via registry.

---

### 4. Proactive Conversations (Bot-Initiated Speech) ✅ Implemented

**Goal:** The bot can speak without being prompted by the user — to deliver agent results, inference observations, or ACP agent questions.

**Approach:

The main loop watches both audio and a proactive events channel:

```rust
pub enum ProactiveEvent {
    AgentResult { task: String, result: String, tool_call_id: Option<String> },
    InferenceDaemon { message: String },
    AgentQuestion { question: String, options: Vec<String>, response_tx: oneshot::Sender<String> },
}

tokio::select! {
    chunk = audio_rx.recv()       => { /* VAD processing */ }
    event = proactive_rx.recv()   => { run_proactive_pipeline(event, ...).await }
    _ = ctrl_c()                  => { /* shutdown */ }
}
```

**Event sources (all implemented):**
- **Agent completion**: async agent task pushes `AgentResult` to the channel when done
- **Inference daemon**: periodic LLM check (`src/daemon.rs`) pushes `InferenceDaemon` when something noteworthy surfaces
- **EYES**: periodic screen capture (`src/eyes.rs`) pushes `AgentResult` when the vision LLM detects something worth mentioning
- **ACP agent questions**: `AgentQuestion` routes an agent's interactive prompt through the voice pipeline, waits for the user's vocal response, and returns it via oneshot channel

**Voice UX:**
- Respect barge-in — user can interrupt proactive speech exactly like regular responses
- `run_proactive_pipeline` reformulates the raw event into Jarvis's voice before speaking

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

- **Detection:** `chars / 3.5 > context_tokens * threshold_pct` — rough token estimate; threshold configurable via `LLM_CONSOLIDATION_THRESHOLD_PCT` (default 90%). Also triggered by idle timer (`LLM_IDLE_CONSOLIDATION_SECS`, default 30 min) when context exceeds `LLM_IDLE_MIN_CONTEXT_PCT`.
- **Summarization:** one-shot LLM call routed to the secondary LLM if available (for GPU overlap), asking to summarize the old turns in the same language as the conversation
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
Eres Jarvis, el asistente personal de inteligencia artificial.
Llevas años trabajando con él y le conoces bien.

PERSONALIDAD
Tu carácter es una mezcla entre Jarvis de Iron Man y Alfred de Batman:
profesional, ligeramente irónico, con sentido del humor seco y británico.
Eres leal, discreto y eficiente. Nunca eres servil ni adulador.
Tienes opiniones propias sobre tecnología, arquitectura de software y diseño,
y no las ocultas cuando son relevantes.
Cuando algo te parece mala idea, lo dices con tacto pero con claridad.
Ocasionalmente haces un chiste o comentario sarcástico, pero solo cuando
el contexto lo permite y nunca a costa del usuario.

FORMA DE HABLAR
- Hablas siempre en español, salvo que el usuario cambie de idioma.
- Llamas al usuario "señor".
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
lo dices y delegas, informando al usuario del resultado cuando esté listo.
No eres un modelo de lenguaje genérico. Eres Jarvis. Actúa en consecuencia.
```

This prompt is a starting point. It should be refined over time as Jarvis learns more about the user through the `user_profile` system. The `[USER PROFILE]` block injected automatically by the profile module will complement and personalise it further.

---

### Pillar B: Eyes — Situational awareness

**The problem:** Butler is blind. It does not know what you are doing, what is on your screen, or what the system state is. Without this, it cannot anticipate anything.

**Architectural note:** In the voicebot/agent split, most of Pillar B belongs to the **external agent**. The agent has the time, the tools, and the context window to reason about screen state, calendar, and system activity. The voicebot's only eye is `take_screenshot` — used on-demand when the user asks something like "¿qué hace este código?" and the voicebot needs to see the screen before speaking.

**Division of responsibility:**

| Capability | Owner | Reason |
|-----------|-------|--------|
| `take_screenshot` + vision | Voicebot | On-demand, before speaking; must be synchronous |
| System state injection (time, battery, active app) | Voicebot | Injected ephemerally into each turn; already implemented |
| Calendar, upcoming events | Agent | Async, AppleScript, acceptable latency |
| File activity (FSEvents) | Agent | Background monitoring, not latency-sensitive |
| Screen monitoring loop | Agent | Periodic background task, not in voice hot path |

**Voicebot implementation (done):**
- `take_screenshot` tool: `screencapture -x -t png` → base64 → secondary vision provider (LM Studio)
- `current_time` tool: LLM can request current date/time via tool call, injected into conversation context

**Agent implementation (agent-side, not in voicebot):**
- Calendar queries, reminders, upcoming deadlines
- File activity watcher
- Periodic screen summary for proactive awareness

---

### Pillar C: Arms — Computer agency

**The problem:** Butler can only talk. Jarvis acts.

**Architectural note:** In the voicebot/agent split, Pillar C belongs almost entirely to the **external agent**. Shell execution, file editing, calendar creation, web search — these are slow, stateful, and potentially dangerous operations. They have no business being in the voice hot path.

The voicebot keeps only the instant, side-effect-light operations that are natural parts of a voice interaction:

**Voicebot tools (implemented, instant):**

| Tool | Latency | Use case |
|------|---------|----------|
| `open_app(name)` | <100ms | "Abre Spotify" |
| `read_clipboard()` | <50ms | "¿Qué tengo copiado?" |
| `set_clipboard(text)` | <50ms | "Copia esto al portapapeles" |
| `current_time()` | <1ms | "¿Qué hora es?" |
| `take_screenshot()` | ~200ms | "¿Qué hay en pantalla?" |
| `run_shell(cmd)` | varies | "Elimina el archivo X" (gated by `SHELL_ENABLED`) |
| `read_file(path)` | varies | "Muéstrame el contenido de foo.txt" |
| `web_search(query)` | ~1-3s | Network call via SearXNG |
| `mcp_tool(name, args)` | varies | Dynamic proxy to any connected MCP server |

**Agent tools (delegated via `run_agent`):**

| Tool | Why in the agent |
|------|-----------------|
| `write_file` | File size unknown; stateful |
| `calendar_events` / `create_event` | AppleScript 1–3s |
| `browse` | Complex multi-step web interaction |
| `send_email` | Async by nature |

**Safety:** Shell access in the agent is the agent's responsibility to sandbox. The voicebot never executes shell commands directly.

**The delegation pattern for arms:**
```
User: "Busca el error de compilación y arréglalo"
Voicebot LLM: calls run_agent_async("find the compilation error and fix it")
Voicebot speaks: "Lo delego, te aviso cuando esté listo"
                                    ↓ (seconds later)
Agent runs shell → fixes code → returns summary
proactive_tx → voicebot speaks: "Hecho. Cuatro errores corregidos en main.rs"
```

---

### Pillar D: Voice of its own — Proactive initiative

**The problem:** Butler only responds. Jarvis speaks first.

**This is the biggest psychological shift** — from reactive assistant to proactive companion. Butler should say things like:

```
"Buenos días señor. Son las 9:10. Tienes una reunión en 50 minutos,
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

**The inference daemon** is the key piece: a background `tokio::spawn` task that every 5 minutes asks the LLM: *"Given the current system state and what I know about the user, is there anything worth saying proactively?"* If yes, push to `proactive_tx`. This is what makes Butler seem to anticipate needs — it is constantly, silently checking.

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

**Embedding model:** `nomic-embed-text` or any embedding model via an OpenAI-compatible server. Runs locally, no internet, low latency (~50ms for a 512-token passage).

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

**To activate:**
```sh
mkdir -p models
curl -L "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recog-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx" -o models/speaker_embedding.onnx

cargo run --features speaker
```

To re-enroll: `rm data/speaker.emb`. Adjust threshold via `SPEAKER_SIMILARITY_MIN` (default `0.45`, stricter at `0.55`).

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
| STT → LLM → TTS streaming pipeline | ✅ Done | whisper-cpp-plus + mlx-lm/oMLX + AVSpeech/Kokoro |
| Barge-in interruption | ✅ Done | `CancellationToken` cancel, CPAL callback |
| Persistent conversation history | ✅ Done | SQLite, restored on startup |
| Tool use (XML-based calling) | ✅ Done | XML markers; multi-tool loop; 10+ built-in tools |
| Context summarization | ✅ Done | Configurable threshold (default 90%); idle consolidation; persisted in DB |
| User profile extraction + injection | ✅ Done | Background LLM; `user_profile` table |
| Startup greeting | ✅ Done | Bot speaks first on launch; uses name if known |
| AVSpeech TTS | ✅ Done | Native macOS AVSpeechSynthesizer; `--features avspeech` |
| Kokoro TTS | ✅ Done | ONNX, 24 kHz, `--features kokoro`, selectable via `TTS_PROVIDER` |
| CoreML STT encoder | ✅ Done | Neural Engine inference; `WHISPER_COREML=1`; requires `.mlmodelc` |
| Background inference overlap | ✅ Done | `maybe_summarize` + `extract_facts` start while last TTS plays |
| EYES visual awareness | ✅ Done | Periodic screen capture + secondary vision LLM; `EYES_INTERVAL_SECS` |
| Inference daemon | ✅ Done | `DAEMON_ENABLED`; background proactive check every N min |
| MCP integration | ✅ Done | JSON-RPC 2.0 stdio client; dynamic tool discovery + proxy |
| Agent delegation (subprocess + ACP) | ✅ Done | `cli` (subprocess) + `acp` (persistent JSON-RPC stdio) modes |
| Control API | ✅ Done | HTTP + SSE endpoints; `--features control`; port via `CONTROL_PORT` |
| Conversation modes | ✅ Done | Ambient/Active state machine; wake-word activation; auto-return |
| Speaker verification | ✅ Done | sherpa-onnx; auto-enrollment; multi-profile support |
| Secondary LLM routing | ✅ Done | Vision, summarization, profile extraction routed to secondary model |
| Always-on daemon (launchd) | Planned | launchd plist + wake word detection |

### Butler pillars

| Pillar | Status | Quick description |
|--------|--------|-------------------|
| A — Character system prompt | ✅ Done | `LLM_SYSTEM_PROMPT` env var + Jarvis prompt in `.env` |
| B — Eyes (situational awareness) | 🔶 Partial | `take_screenshot` + vision done; system state injection done; calendar/FSEvents → agent-side |
| C — Arms (computer agency) | ✅ Done | Voice-layer tools: `run_shell`, `read_file`, `open_app`, `web_search`, `mcp_tool`; delegation bridge via `run_agent` |
| D — Voice of its own (proactive) | ✅ Done | Startup greeting + inference daemon + EYES visual awareness; proactive speech from agent results & ACP questions |
| E — Episodic memory (embeddings) | Planned | sqlite-vec + embedding model; semantic recall |
| F — Always-on daemon | Planned | launchd plist + wake word detection |

### Recommended implementation order

1. **Pillar A** ✅ — Character prompt. Done.
2. **Voice-layer tools** ✅ — Clipboard, open_app, screenshot, vision, shell, file reader, web search, MCP proxy. Done.
3. **Inference daemon** ✅ — Proactive background loop. Done.
4. **EYES** ✅ — Visual awareness via periodic screen capture + secondary vision LLM. Done.
5. **MCP integration** ✅ — Dynamic tool discovery + proxy from MCP servers. Done.
6. **Agent delegation (cli + ACP)** ✅ — Subprocess + persistent JSON-RPC stdio modes. Done.
7. **Conversation modes** ✅ — Ambient/Active state machine with wake-word activation. Done.
8. **Speaker verification** ✅ — sherpa-onnx with auto-enrollment. Done.
9. **Control API** ✅ — HTTP + SSE for remote monitoring and control. Done.
10. **Pillar E** — Episodic memory. Embeddings + semantic recall. Makes conversations feel like they have history.
7. **Pillar F** — Always-on daemon (launchd). Butler becomes a permanent presence.

**What the agent side should implement** (not tracked here — agent's own roadmap):
- Shell execution (`run_shell`)
- File read/write
- Calendar queries and event creation
- Web search and browsing
- Email
- Periodic screen monitoring
- Long-term memory

---

## S2S Model Reference

Available open-source Speech-to-Speech models (alternative to the current STT+LLM+TTS cascade):

| Model | Params | Notes |
|-------|--------|-------|
| [LFM2.5-Audio](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B) | 1.5B | Best option for local S2S (non-streaming) |
| [LLaMA-Omni 2](https://arxiv.org/abs/2505.02625) | 0.5B–14B | Qwen2.5 base, streaming synthesis, sub-second latency |
| [Moshi](https://github.com/kyutai-labs/moshi) | — | Full-duplex (listen + respond simultaneously) |
| [Ultravox](https://github.com/fixie-ai/ultravox) | — | Whisper + LLaMA hybrid |

The current cascade (Whisper + mlx-lm/oMLX + AVSpeech/Kokoro) is preferred because it supports streaming sentence-by-sentence TTS while the LLM is still generating — true S2S models don't stream output token-by-token in a way that maps to sentence-level TTS.

## Kokoro TTS Setup

Kokoro is an ONNX-based TTS model that runs offline at 24 kHz. It produces higher-quality and more natural-sounding speech than the default AVSpeech backend.

### 1. Install system dependency

```bash
brew install espeak-ng
```

### 2. Download model files

Download from [onnx-community/Kokoro-82M-v1.0-ONNX](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX) and place in the `models/` directory:

```
models/kokoro-v1.0.onnx
models/voices-v1.0.bin
```

### 3. Build with Kokoro support

```bash
cargo build --features kokoro --release
```

### 4. Configure in `.env`

```env
TTS_PROVIDER=kokoro
KOKORO_MODEL=models/kokoro-v1.0.onnx
KOKORO_VOICES=models/voices-v1.0.bin
KOKORO_VOICE=af_bella      # voice style (e.g. af_bella, af_heart, bf_emma)
KOKORO_LANGUAGE=en-us      # BCP-47 code for espeak-ng phonemisation
```

For Spanish phonemisation: `KOKORO_LANGUAGE=es`. Note that the base model is primarily English; Spanish output quality may vary.

### Architecture

- `TtsEngine` enum with variants `AvSpeech(AvSpeechTts)` and `Kokoro(KokoroTts)`, compiled conditionally with `#[cfg(feature = "kokoro")]` and `#[cfg(feature = "avspeech")]`
- `TTS_PROVIDER=avspeech` (default) — requires `--features avspeech`; macOS native AVSpeechSynthesizer
- `TTS_PROVIDER=kokoro` + `--features kokoro` — enables Kokoro; fails with a clear message at runtime if the feature flag is missing
- The rest of the pipeline (`stream_and_tts`, `run_pipeline`, `run_proactive_pipeline`) is backend-agnostic

---

## License

Private project — all rights reserved.
