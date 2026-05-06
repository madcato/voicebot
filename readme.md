# Jarvis Voicebot 🤖

<div align="center">

**An open-source voice-first AI butler built in Rust for macOS.**

Real-time voice interaction with natural conversation flow, proactive assistance, and computer automation.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org)
[![Platform](https://img.shields.io/badge/platform-macOS-red.svg)](https://www.apple.com/macos)

> **⚠️ Legal Notice:** Jarvis® is a trademark of Marvel Studios/Disney. This is an independent fan project with no commercial intent. See [LICENSE-VOICEBOT.md](LICENSE-VOICEBOT.md) for full details.

</div>

---

## Overview

Jarvis Voicebot is a **voice-first AI assistant** designed for natural, real-time conversation with your computer. Unlike traditional chatbots that you type to, Jarvis listens and speaks — it runs as an always-on background daemon that responds instantly when you talk.

### What makes Jarvis different?

A chatbot answers questions. A butler anticipates needs.

Jarvis is built from the ground up for voice interaction:
- **Always-listening** with automatic speech detection (no push-to-talk)
- **Real-time responses** under 3 second latency
- **Natural conversation flow** with context awareness and personality
- **Barge-in support** — interrupt it mid-speech instantly
- **Computer control** via delegated agent for complex tasks

---

## Features

✅ **Implemented today** | 🚧 **In progress / Planned**

### Core Voice Pipeline ✅

- 🔊 Real-time voice capture (CPAL) with VAD (Silero) and pre-roll buffer
- 🎤 Whisper STT (CoreML Neural Engine or Metal GPU, cached across utterances)
- 🤖 Streaming LLM via mlx-lm or oMLX (Apple MLX, KV-cache reuse, sub-second latency)
- 🔊 Sentence-by-sentence TTS playback — speaks while generating next sentence
- ⚡ **Barge-in** — user speech cancels active pipeline instantly
- 💾 Persistent SQLite conversation history with session restoration

### Advanced Features ✅

- 🧠 Context consolidation with persistent memory (Claude-like context management)
- 👤 User profile extraction from conversations (injects into system prompt)
- 🎭 Startup greeting with name recognition
- 🛠️ Tool calling system (`current_time`, `take_screenshot`, `send_notification`, `read/set_clipboard`, `open_app`, `web_search`)
- 🔍 Web search via SearXNG with multiturn agent support (talk, search, analyze, search, respond)
- 🆔 Multi-speaker registry (auto-enrolls up to N speakers, ONNX-based embeddings)
- 🎙️ Ambient context buffer — transcribes all ambient speech (TV, others) for contextual responses
- 💬 Two conversation modes: **Active** (respond freely to main user) / **Ambient** (wake-word only, with full context)

### Integration Options ✅

- **TTS backends**: AVSpeechSynthesizer (default), macOS `say`, Kokoro ONNX 
- **LLM providers**: mlx-lm, oMLX (Apple MLX framework — substantially faster on Apple Silicon)
- **Agent delegation**: `run_agent` / `run_agent_async` for complex tasks

### Terminal UI (TUI) ✅

- 💻 Full terminal interface with scrollable conversation view
- ⌨️ Type queries alongside voice — both input modes work simultaneously
- 🔊 Toggle TTS on/off with `Ctrl+T`
- 🔧 Tool call display inline in conversation
- 📊 Pipeline status indicator (Idle/Listening/Transcribing/Thinking/Speaking)
- 📝 Logs redirected to `voicebot.log` when TUI is active

Enable with: `cargo run --features tui`

### Remote Device Connectivity ✅

- 📱 WebSocket server for remote audio streaming (phone, watch, other computers)
- 🔊 Full STT→LLM→TTS pipeline runs on the server — remote device only sends/receives audio
- 📡 Wire protocol: PCM i16 LE mono 16kHz (binary frames) + JSON control messages (text frames)
- ⚡ Barge-in works over WebSocket — both audio-based (server VAD) and explicit signal
- 🍎 Apple Watch client guide included (see `doc/APPLE_WATCH_CLIENT.md`)

Enable with: `WS_PORT=9090 cargo run --features remote --release`

### Control API (HTTP + SSE) ✅

Exposes pipeline state, streaming LLM tokens, and control commands to any external process — a custom TUI, a web dashboard, or a shell script.

- **`GET /control/events`** — Server-Sent Events stream of pipeline events (JSON):
  - `state_changed` — FSM state transitions (Idle/Listening/Thinking/Speaking/Paused)
  - `transcript` — finalized user utterance text
  - `llm_token` — each streaming LLM token as it arrives
  - `llm_done` — full assistant response for the turn
  - `tts_start` — sentence sent to TTS
  - `tool_call` — tool name and result
  - `mute_changed` — TTS mute state change
- **`GET /control/state`** — JSON snapshot: current FSM state, utterance ID, mute flag
- **`GET /control/history`** — JSON: full conversation message history
- **`POST /control/mute`** — body `{"muted": true|false}` — mute/unmute TTS
- **`POST /control/barge_in`** — interrupt current TTS playback
- **`POST /control/input`** — body `{"text": "..."}` — inject text as user input

Enable with: `CONTROL_PORT=9001 cargo run --features control`

```bash
# Stream all pipeline events
curl -N http://127.0.0.1:9001/control/events

# Current state snapshot
curl http://127.0.0.1:9001/control/state

# Mute TTS
curl -X POST http://127.0.0.1:9001/control/mute \
  -H 'Content-Type: application/json' -d '{"muted":true}'

# Barge in
curl -X POST http://127.0.0.1:9001/control/barge_in

# Send text input
curl -X POST http://127.0.0.1:9001/control/input \
  -H 'Content-Type: application/json' -d '{"text":"hola"}'
```

### Roadmap 🚧

- Calendar, email, file system access
- Vision capabilities (screen awareness)
- Proactive suggestions based on context

---

## Requirements

### System

- **macOS** 12.0+ (Big Sur or later)
- Apple Silicon (M-series) recommended for optimal performance

### Dependencies

```bash
# Rust toolchain
rustup install stable

# Optional: Kokoro TTS requires espeak-ng
brew install espeak-ng

# Optional: Node.js for MCP servers (future)
brew install node
```

### Models Required 📦

You'll need to download the following models:

#### Whisper STT Model
```bash
# Download whisper.cpp model (choose size: tiny, small, base, medium, large-v3-turbo)
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin -O ./models/ggml-small.bin

# Optional: CoreML encoder for faster STT (requires conversion)
# See CONTRIBUTING.md for CoreML conversion instructions
```

#### LLM Model
```bash
# Download a GGUF model (Qwen2.5-7B recommended)
wget https://hgpu.space/file/hjz3n4QwZbU/Qwen2.5-7B-Instruct-Q4_K_M.gguf -O ./models/Qwen2.5-7B-Instruct-Q4_K_M.gguf

# Alternative: mlx-lm format (auto-downloads from HuggingFace)
# No manual download needed for mlx-lm
```

#### Optional: Kokoro TTS Models
```bash
# Download Kokoro ONNX model and voices
wget https://github.com/leloykun/kokoro/releases/download/v1.0/kokoro-v1.0.onnx -O ./models/kokoro-v1.0.onnx
wget https://github.com/leloykun/kokoro/releases/download/v1.0/voices-v1.0.bin -O ./models/voices-v1.0.bin
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Hive-Vote/voicebot.git
cd voicebot
```

### 2. Configure environment variables

Copy the example config and adjust:

```bash
cp .env.example .env
nano .env
```

**Minimum required configuration:**

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | — | Path to Whisper `.bin` model (e.g., `./models/ggml-small.bin`) |
| `LLM_URL` | `http://127.0.0.1:8080` | LLM server URL |
| `LLM_MODEL` | `local-model` | Model name/path for the LLM provider |

**Example `.env`:**

```env
WHISPER_MODEL=./models/ggml-small.bin
WHISPER_COREML=0
LLM_URL=http://127.0.0.1:8000
LLM_MODEL=mlx-community/Qwen3-8B-4bit
TTS_PROVIDER=say
SAY_VOICE="Jorge (Enhanced)"
VOICEBOT_LANGUAGE=es
```

### 3. Start the LLM server

Jarvis uses Apple MLX-based servers for low-latency inference on Apple Silicon.

**Using mlx-lm (recommended):**

```bash
./scripts/start-mlx-lm.sh mlx-community/Qwen3-8B-4bit
# Set in .env: LLM_URL=http://127.0.0.1:8000
```

Or manually:

```bash
mlx_lm.server \
  --model mlx-community/Qwen3-8B-4bit \
  --host 127.0.0.1 --port 8000 \
  --prompt-cache-size 1 \
  --chat-template-args '{"enable_thinking": false}'
```

**Using oMLX (alternative — persistent tiered KV cache):**

```bash
./scripts/start-omlx.sh ~/models
# Set in .env: LLM_URL=http://127.0.0.1:8001
```

### 4. Build and run Hive Voicebot

**Standard build (macOS `say` TTS - default):**

```bash
cargo build --release
cargo run --release
```

**With Kokoro TTS (high-quality, ONNX-based):**

```bash
cargo build --features kokoro --release
TTS_PROVIDER=kokoro cargo run --features kokoro --release
```

**With CoreML STT acceleration:**

```bash
WHISPER_COREML=1 cargo run --release
```

**List available voices for the active TTS provider:**

```bash
cargo run -- --list-voices
# or
LIST_VOICES=1 cargo run
```

The output depends on the `TTS_PROVIDER` setting:
- `say` — lists all macOS system voices (name, language, sample text)
- `avspeech` — lists all AVSpeechSynthesizer voices (name, language, quality, gender, identifier)
- `kokoro` — lists all Kokoro ONNX voice styles (voice ID, language, gender)

---

## Architecture

Jarvis Voicebot is intentionally **narrow in scope**: it owns the audio pipeline and conversational experience. Complex tasks are delegated to an external agent via stdin/stdout protocol.

### Why this separation?

**Response latency matters.** A voice bot that only handles conversation responds in under 1 second. Adding shell commands, file access, and calendar operations slows it down significantly.

```
┌─────────────────────────────────────────────┐
│         JARVIS VOICEBOT (fast layer)        │
│                                             │
│  • STT → LLM (7B) → TTS                    │
│  • Barge-in, conversation awareness        │
│  • Proactive suggestions                   │
│  • Voice-local tools                       │
│                                             │
│  Complex tasks → delegate to AGENT         │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│           EXTERNAL AGENT (power layer)      │
│                                             │
│  • Full tool suite                         │
│  • File system, calendar, web, email       │
│  • Long-running tasks                      │
└─────────────────────────────────────────────┘
```

See [doc/ARCHITECTURE.md](doc/ARCHITECTURE.md) for detailed architectural docs. Also [doc/doc.md](doc/doc.md) for additional info.

### Context Window & Memory Consolidation

Jarvis uses the full context window provided by the LLM (`LLM_CONTEXT_TOKENS`, default 8192). When the conversation approaches the configured threshold (`LLM_CONSOLIDATION_THRESHOLD_PCT`, default 80%), a consolidation cycle runs automatically.

There are two consolidation modes:

**Active (mid-conversation):** Triggered when the context threshold is reached after a turn.
1. **Announce offline** — Jarvis tells the user it needs a few minutes to reorganize its memory
2. **Extract profile facts** — Structured facts (name, city, preferences) are extracted and persisted in the `user_profile` DB table
3. **Extract memories** — Free-form persistent notes (projects, decisions, technical context) are extracted into the `memories` DB table
4. **Summarize** — Old conversation turns are summarized into a compact text
5. **Rebuild system prompt** — The system prompt is rebuilt with updated `[USER PROFILE]`, `[MEMORIES]`, and `[CONVERSATION SUMMARY]` sections
6. **Announce back online** — Jarvis announces it's available again and tells the user the current time

**Silent (idle):** Triggered when the user hasn't spoken for `LLM_IDLE_CONSOLIDATION_SECS` (default 15 minutes). Uses `LLM_IDLE_MIN_CONTEXT_PCT` (default 20%) as its threshold — lower than the hard limit — so the context is kept well below `LLM_CONSOLIDATION_THRESHOLD_PCT` while the user is away. Runs transparently, without any voice announcements.

Memories and profile facts persist across sessions via SQLite. On startup, they are loaded and injected into the system prompt so the LLM has full context from previous conversations.

---

## Configuration

### Environment Variables

Most configuration is done via environment variables (or `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| **Voice & Language** || |
| `VOICEBOT_LANGUAGE` | `es` | Language for STT and TTS |
| `VAD_SILENCE_MS` | `500` | Silence threshold (ms) before processing speech |
| **STT (Whisper)** || |
| `WHISPER_MODEL` | _required_ | Path to Whisper `.bin` model |
| `WHISPER_THREADS` | `0` (auto) | CPU threads for Whisper decoding. Set to physical core count for best throughput. |
| `WHISPER_COREML` | `0` | Use CoreML encoder (Neural Engine) |
| `STT_MIN_SUBMIT_MS` | `1000` | Minimum ms of accumulated speech before the first speculative Whisper submit. Avoids garbage transcriptions from tiny clips. |
| `STT_SUBMIT_INTERVAL_MS` | `500` | Interval (ms of new speech) between speculative Whisper submits during speech. Whisper runs in the background so the transcript is ready by SpeechEnd. Set to `0` to disable. |
| **LLM** || |
| `LLM_URL` | `http://127.0.0.1:8000` | LLM server URL (mlx-lm default; use IP not `localhost` to avoid DNS latency) |
| `LLM_SELF_MANAGED` | `0` | If `1`, voicebot launches and supervises the LLM server process automatically. Requires `LLM_COMMAND`. On crash, restarts up to 3 times before logging a fatal error. |
| `LLM_COMMAND` | — | Full shell command to launch the LLM server. Required when `LLM_SELF_MANAGED=1`. Example: `mlx_lm.server --model mlx-community/Qwen3-8B-4bit --host 0.0.0.0 --port 8080 --max-tokens 32768` |
| `LLM_MODEL` | `local-model` | Model name or path |
| `LLM_SYSTEM_PROMPT` | — | System prompt for the LLM |
| `LLM_MAX_TOKENS` | `200` | Max response tokens |
| `LLM_TEMPERATURE` | `0.3` | Sampling temperature |
| `LLM_CONTEXT_TOKENS` | `8192` | Context window size in tokens. Set to match your model's context length. |
| `LLM_CONSOLIDATION_THRESHOLD_PCT` | `80` | Percentage of context window that triggers memory consolidation (see below). |
| `LLM_IDLE_CONSOLIDATION_SECS` | `1800` | Seconds of user inactivity before a silent consolidation runs (0 = disabled). |
| `LLM_IDLE_MIN_CONTEXT_PCT` | `50` | Context fill % threshold used by idle-triggered consolidation. Consolidates proactively while idle to stay below the hard limit (0 = disabled). |
| **TTS** || |
| `TTS_PROVIDER` | `avspeech` | Provider: `avspeech`, `say`, or `kokoro` |
| `SAY_VOICE` | `Jorge (Enhanced)` | macOS voice name |
| `SAY_RATE` | `215` | Words per minute |
| `KOKORO_MODEL` | `./models/kokoro-v1.0.onnx` | Kokoro ONNX model path |
| **Agent Integration** || |
| `AGENT_COMMAND` | `hermes chat` | CLI command for agent subprocess (CLI mode) |
| `AGENT_TIMEOUT_SECS` | `120` | Timeout for synchronous CLI agent calls |
| `AGENT_MODE` | `cli` | `cli` = fire-and-forget subprocess; `acp` = persistent ACP bidirectional mode |
| `AGENT_ACP_COMMAND` | `hermes acp` | Command to start the ACP process (ACP mode only) |
| `AGENT_ACP_WARMUP` | `0` | Pre-warm the ACP session at startup (`1` = enabled). Spawns and handshakes the ACP process in the background so the first `run_agent` call skips cold-start. Set `AGENT_ACP_WARMUP=1` to also send a warmup prompt, forcing model load before the user's first request. Requires `AGENT_MODE=acp`. |
| **Secondary LLM** || |
| `SECONDARY_LLM_URL` | — | Base URL of secondary LLM. Enables `take_screenshot` tool, EYES visual awareness, and routes summarization + profile extraction to this model. |
| `SECONDARY_LLM_MODEL` | `local-model` | Model name for secondary LLM requests. |
| `SECONDARY_LLM_MAX_TOKENS` | `512` | Max tokens for secondary LLM responses (vision). |
| `SECONDARY_LLM_API_KEY` | — | Bearer token for secondary LLM API. |
| `SECONDARY_LLM_PROVIDER` | `mlx` | Backend for secondary LLM (mlx-lm or omlx). |
| **EYES (visual awareness)** || |
| `EYES_INTERVAL_SECS` | `0` (disabled) | Seconds between automatic screen captures. Set to e.g. `15` to enable. Requires `SECONDARY_LLM_URL` (vision model). Jarvis speaks when something important is detected on screen. |
| **Web Search (SearXNG)** || |
| `SEARXNG_URL` | — (disabled) | Base URL of SearXNG instance (e.g. `http://tesla.local:8080`). Enables the `web_search` tool. |
| `SEARXNG_SECRET` | (empty) | Bearer token for SearXNG API authentication. |
| **MCP (Model Context Protocol)** || |
| `MCP_COMMAND` | — (disabled) | Command to spawn an MCP stdio server (e.g. `bunx apple-mcp@latest`). All tools advertised by the server via `tools/list` are registered dynamically. Calls run in background — Jarvis acknowledges and speaks the result when ready. Compatible with any MCP server using stdio transport. |
| `MCP_TOOL_TIMEOUT_SECS` | `30` | Hard timeout per MCP tool call in seconds. |
| **Speaker Verification** || |
| `SPEAKER_MODEL` | auto-detect | Path to sherpa-onnx speaker embedding ONNX model. Auto-detected at `models/speaker_embedding.onnx`; disabled if absent. |
| `SPEAKER_ENROLLMENT_PATH` | `data/speaker.emb` | Base path for speaker profiles. Profiles saved as `speaker_0.emb`, `speaker_1.emb`, etc. in the same directory. |
| `SPEAKER_SIMILARITY_MIN` | `0.45` | Cosine similarity threshold [0..1] for speaker matching. |
| `SPEAKER_MAX_PROFILES` | `5` | Maximum number of speaker profiles to auto-enroll. The first speaker (id=0) is always the main user. |
| `SPEAKER_AMBIENT_TRIGGER` | `1` | Consecutive non-main-user segments before auto-switching to Ambient mode. |
| **Conversation Modes** || |
| `WAKE_WORD` | `jarvis` | Case-insensitive substring match triggering a response in Ambient mode. |
| `AMBIENT_CLEAR_SECS` | `300` | Seconds of silence before auto-switching from Active to Ambient mode. |
| **Ambient Context Buffer** || |
| `AMBIENT_BUFFER_MINUTES` | `3` | Rolling window duration for the ambient context buffer. |
| `AMBIENT_BUFFER_MAX_ENTRIES` | `30` | Maximum buffered utterances. Oldest are evicted when full. |
| **Remote Device (WebSocket)** || |
| `WS_PORT` | — (disabled) | WebSocket server port. Set to e.g. `9090` to enable remote device connectivity. Requires `--features remote`. |
| **Control API (HTTP + SSE)** || |
| `CONTROL_PORT` | — (disabled) | HTTP control/SSE API port. Set to e.g. `9001` to enable. Requires `--features control`. Binds to `127.0.0.1` only. |

See [.env.example](.env.example) for complete environment variable reference.

---

## Development

### Build commands

```bash
# Standard build
cargo build --release

# Build with TUI (terminal user interface)
cargo build --release --features tui

# Build with remote device support (WebSocket server)
cargo build --release --features remote

# Build with HTTP control API + SSE
cargo build --release --features control

# Run with debug
cargo run

# Run with TUI
cargo run --features tui

# Run tests (unit tests only)
cargo test

# E2E tests (require audio device + env vars set)
cargo test e2e -- --ignored --nocapture
```

### Logging

Debug different subsystems using `RUST_LOG`:

```bash
# Conversation flow only
RUST_LOG=pipeline=info cargo run

# Full debugging with performance metrics
RUST_LOG=performance=debug,voicebot=info cargo run

# TTS and audio debug
RUST_LOG=tts=debug,audio=debug cargo run
```

See [Documentation](doc/LOGGING.md) for all log targets.

When running with `--features tui`, all logs are redirected to `voicebot.log` in the working directory.

### TUI Key Bindings

| Key | Action |
|-----|--------|
| `Enter` | Send typed message |
| `Ctrl+T` | Toggle TTS on/off |
| `PageUp/PageDown` | Scroll conversation |
| `Esc` / `Ctrl+C` | Quit |

Voice input and text input work simultaneously — speak or type at any time.

### Benchmarks

Compare LLM server performance:

```bash
# mlx-lm benchmark
./scripts/bench-mlx.sh mlx-community/Qwen3-8B-4bit

# mlx-lm vs oMLX comparison
./scripts/bench-omlx.sh mlx-community/Qwen3-8B-4bit ~/models

# Full pipeline benchmark (STT → LLM → TTS) using WAV fixtures
# Requires: Whisper model + running mlx-lm or omlx server
RUST_LOG=performance=debug cargo run --bin bench_pipeline
```

#### VAD Latency Tuning

`VAD_SILENCE_MS` controls how long silence must persist before the pipeline starts (default: 250ms). Lower values feel more responsive but risk cutting speakers mid-pause. The speech buffer accumulates across pauses, so no audio is lost if the user resumes speaking.

```bash
# More responsive (may cut mid-pause)
VAD_SILENCE_MS=200 cargo run

# More conservative (waits longer for pauses)
VAD_SILENCE_MS=500 cargo run
```

---

## Troubleshooting

### "No audio device found"

Run `cargo run -- --list-devices` to see available devices, then set:

```bash
AUDIO_INPUT_DEVICE="Microphone"
AUDIO_OUTPUT_DEVICE="Speaker"
```

If a device appears multiple times (e.g. a headset with both USB and Bluetooth connections), the
code automatically picks the first candidate whose configuration is valid. To force a specific
match, append `#N` (0-based index) to the device name:

```bash
AUDIO_INPUT_DEVICE="Poly Sync 20-M#0"   # first match (USB)
AUDIO_INPUT_DEVICE="Poly Sync 20-M#1"   # second match (Bluetooth)
```

### TTS not working

- macOS `say`: Check voice is installed: `say -v ?`
- Kokoro: Ensure models exist in `./models/` directory
- Check feature flag: `--features kokoro` required for Kokoro

### High latency

1. Reduce `VAD_SILENCE_MS` to 400ms
2. Use CoreML STT (`WHISPER_COREML=1`)
3. Verify LLM server has Metal acceleration: `-ngl 99 --flash-attn on`
4. Check performance logs: `RUST_LOG=performance=debug`

<!-- See [doc/TROUBLESHOOTING.md](doc/TROUBLESHOOTING.md) for more issues. -->

---

## Roadmap & Contributing

### Priority Features

- Calendar sync
- Vision capabilities (screen awareness, OCR)
- Mobile companion app
- Multi-platform support (Linux/Windows)

<!-- See [ROADMAP.md](doc/ROADMAP.md) for full feature track. -->

### How to Contribute

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests: `cargo test`
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

This project is released under the **MIT License** with **commercialization restrictions**.

⚠️ **Jarvis® is a trademark of Marvel Studios/Disney.** This is an independent fan project.

See [LICENSE-VOICEBOT.md](LICENSE-VOICEBOT.md) for full legal details and license terms.

---

## Acknowledgments

Built with:
- **Rust** — Systems programming language
- **whisper-rs** — Whisper.cpp bindings for Rust
- **mlx-lm / oMLX** — Local LLM inference (Apple MLX framework)
- **CPAL** — Cross-platform audio I/O
- **Tokio** — Asynchronous runtime

---

<div align="center">

**Built with ❤️ by Daniel and the Jarvis Team**

*Voice is the future of computing.*

</div>

## Building with CoreML Support

To use Apple's Neural Engine (ANE) via CoreML for faster encoding:

```bash
# Clean previous build
cargo clean -p whisper-cpp-plus-sys

# Build with CoreML enabled  
CARGO_CFG_WHISPER_USE_COREML=1 WHISPER_USE_COREML=1 \
  cargo build --release --features metal
```

**Requirements:**
- You must have `<model>-encoder.mlmodelc` in your models directory
- For `ggml-large-v3-turbo.bin`, you need `ggml-large-v3-turbo-encoder.mlmodelc`
- CoreML provides ANE acceleration (faster than GPU for encoding)

Your current setup already has the CoreML encoder files in `models/`.
