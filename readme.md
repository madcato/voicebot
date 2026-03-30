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
- 🤖 Streaming LLM via llama.cpp or mlx-lm (KV-cache reuse, sub-second latency)
- 🔊 Sentence-by-sentence TTS playback — speaks while generating next sentence
- ⚡ **Barge-in** — user speech cancels active pipeline instantly
- 💾 Persistent SQLite conversation history with session restoration

### Advanced Features ✅

- 🧠 Context-aware summarization (keeps recent turns verbatim)
- 👤 User profile extraction from conversations (injects into system prompt)
- 🎭 Startup greeting with name recognition
- 🛠️ Tool calling system (`current_time`, `take_screenshot`, `send_notification`, `read/set_clipboard`, `open_app`)
- 🆔 Multi-speaker registry (auto-enrolls up to N speakers, ONNX-based embeddings)
- 🎙️ Ambient context buffer — transcribes all ambient speech (TV, others) for contextual responses
- 💬 Two conversation modes: **Active** (respond freely to main user) / **Ambient** (wake-word only, with full context)

### Integration Options ✅

- **TTS backends**: AVSpeechSynthesizer (default), macOS `say`, Kokoro ONNX 
- **LLM providers**: llama.cpp (local GGUF), mlx-lm (Apple MLX framework)
- **Agent delegation**: `run_agent` / `run_agent_async` for complex tasks

### Terminal UI (TUI) ✅

- 💻 Full terminal interface with scrollable conversation view
- ⌨️ Type queries alongside voice — both input modes work simultaneously
- 🔊 Toggle TTS on/off with `Ctrl+T`
- 🔧 Tool call display inline in conversation
- 📊 Pipeline status indicator (Idle/Listening/Transcribing/Thinking/Speaking)
- 📝 Logs redirected to `voicebot.log` when TUI is active

Enable with: `cargo run --features tui`

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
LLM_PROVIDER=llama
LLM_URL=http://127.0.0.1:8080
LLM_MODEL=./models/Qwen2.5-7B-Instruct-Q4_K_M.gguf
TTS_PROVIDER=say
SAY_VOICE="Jorge (Enhanced)"
VOICEBOT_LANGUAGE=es
```

### 3. Start the LLM server

**Using llama.cpp:**

```bash
# First build llama.cpp if you don't have it
# Then start the server with your model:
./scripts/start-llm.sh ./models/Qwen2.5-7B-Instruct-Q4_K_M.gguf
```

**Recommended llama-server flags for low latency:**

```bash
llama-server \
  --model ./models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8080 \
  --flash-attn \                     # GPU-accelerated attention (faster prefill)
  --cache-type-k q8_0 \             # Quantized KV cache (less VRAM, same quality)
  --cache-type-v q8_0 \
  --ctx-size 4096 \                  # Match LLM_CONTEXT_TOKENS (smaller = faster)
  --threads $(sysctl -n hw.perflevel0.physicalcpu) \  # Physical perf cores only
  --parallel 2 \                     # 2 slots: main conversation + background tasks
  --cont-batching \                  # Continuous batching for parallel slots
  --slots                            # Enable slot management
```

Key flags explained:
- `--flash-attn`: Enables flash attention for faster prefill, especially on longer contexts.
- `--cache-type-k/v q8_0`: Quantized KV cache reduces memory ~50% with negligible quality loss.
- `--ctx-size`: Keep small (4096-8192) — larger contexts increase prefill latency linearly.
- `--threads`: Use physical **performance** core count (not hyperthreads). On Apple Silicon: `sysctl -n hw.perflevel0.physicalcpu`.
- `--parallel 2`: Separate slots for conversation (slot 0) and background tasks (slot 1). Set `LLM_BACKGROUND_SLOT_ID=1` in `.env`.

**Using mlx-lm (alternative):**

```bash
./scripts/start-mlx-lm.sh mlx-community/Qwen2.5-7B-Instruct-4bit
# Set in .env: LLM_PROVIDER=mlx, LLM_URL=http://127.0.0.1:8000
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
| **LLM** || |
| `LLM_PROVIDER` | `llama` | Backend: `llama` or `mlx` |
| `LLM_URL` | `http://127.0.0.1:8080` | LLM server URL (use IP, not `localhost`, to avoid DNS latency) |
| `LLM_MODEL` | `local-model` | Model name or path |
| `LLM_SYSTEM_PROMPT` | — | System prompt for the LLM |
| `LLM_MAX_TOKENS` | `200` | Max response tokens |
| `LLM_TEMPERATURE` | `0.3` | Sampling temperature |
| `LLM_CONTEXT_TOKENS` | `4096` | Context window size |
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
| **Secondary LLM** || |
| `SECONDARY_LLM_URL` | — | Base URL of secondary LLM. Enables `take_screenshot` tool and routes summarization + profile extraction to this model. |
| `SECONDARY_LLM_MODEL` | `local-model` | Model name for secondary LLM requests. |
| `SECONDARY_LLM_MAX_TOKENS` | `512` | Max tokens for secondary LLM responses (vision). |
| `SECONDARY_LLM_API_KEY` | — | Bearer token for secondary LLM API. |
| `SECONDARY_LLM_PROVIDER` | `llama` | Backend for secondary LLM: `llama` or `mlx`. |
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

See [.env.example](.env.example) for complete environment variable reference.

---

## Development

### Build commands

```bash
# Standard build
cargo build --release

# Build with TUI (terminal user interface)
cargo build --release --features tui

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
# llama.cpp benchmark
./scripts/bench-llama.sh ./models/Qwen2.5-7B-Instruct-Q4_K_M.gguf

# mlx-lm benchmark
./scripts/bench-mlx.sh mlx-community/Qwen2.5-7B-Instruct-4bit

# Real-server KV-cache comparison
python3 scripts/bench-server.py <llama-model> <mlx-model>

# Full pipeline benchmark (STT → LLM → TTS) using WAV fixtures
# Requires: Whisper model + running llama-server
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
- **llama.cpp / mlx-lm**— Local LLM inference
- **CPAL** — Cross-platform audio I/O
- **Tokio** — Asynchronous runtime

---

<div align="center">

**Built with ❤️ by Daniel and the Jarvis Team**

*Voice is the future of computing.*

</div>
