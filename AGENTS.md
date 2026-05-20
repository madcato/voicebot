# Voicebot — Agent Instructions

## ⚠️ Legal & Naming

- **Jarvis®** is a trademark of Marvel Studios/Disney. This is an independent fan project.
- Refer to this project as **"Voicebot"**. Never "Jarvis", "Hive" or just "Voicebot".
- See `LICENSE-VOICEBOT.md` for full details.

---

## Commands (exact)

```bash
cargo build --release      # Release build
cargo run                  # Dev run (reads .env)
cargo run --release        # Production run
cargo test                 # All tests
cargo fmt                  # Format
cargo clippy               # Lint

# Feature flags
cargo run --features kokoro     # Kokoro TTS backend
cargo run --features tui        # Terminal UI
cargo run --features remote     # WebSocket server
cargo run --features speaker    # Speaker verification

# List devices/voices
cargo run -- --list-devices     # Or LIST_AUDIO_DEVICES=1 cargo run
cargo run -- --list-voices      # Or LIST_VOICES=1 cargo run

# Single test
cargo test <test_name>
cargo test -p voicebot <test_name>
```

---

## Architecture Overview

Mono-user voice AI chatbot in Rust. Streaming STT→LLM→TTS pipeline, single process, tokio channels.

```
Microphone → VAD (Silero) → STT (whisper-cpp-plus) → LLM (mlx-lm/oMLX SSE) → SentenceSplitter → TTS (AVSpeech/Kokoro) → AudioOutput
```

### Key Design Decisions

- **STT→LLM latency trick**: Accumulate partial Whisper transcripts; send full text when VAD signals end-of-speech. LLM server maintains KV-cache implicitly across requests.
- **LLM→TTS streaming**: Buffer tokens until punctuation (`. ! ? ; :`), synthesize immediately. While sentence N plays, sentence N+1 is being generated.
- **Language**: Spanish default (`VOICEBOT_LANGUAGE=es`), English supported. Affects Whisper hint and TTS voice.
- **Barge-in**: User speech cancels active pipeline via `CancellationToken` (tokio-util).

---

## Module Boundaries

| Directory | Purpose | Public exports (lib.rs) |
|-----------|---------|------------------------|
| `src/audio/` | CPAL capture, Silero VAD, resampling (rubato), playback | `AudioBuffer`, `AudioOutput`, `VoiceActivityDetector` |
| `src/stt/` | whisper-cpp-plus wrapper, 16kHz f32 mono, language detection | `WhisperStt` (alias for `WhisperSttPlus`) |
| `src/llm/` | HTTP client to `/v1/chat/completions`, session management | `OpenAIClient`, `LlmSession` |
| `src/tts/` | `avspeech.rs` (macOS AVSpeechSynthesizer), `sentence.rs` (boundary splitting), `kokoro.rs` (ONNX) | `SentenceSplitter` |
| `src/db/` | SQLite persistence: sessions, messages, user_profile, memories | `Database` |
| `src/config.rs` | Environment-based config (`Config::from_env()`) | `Config` |
| `src/memory/` | Extract persistent notes from conversation, archive outdated | Injects `[MEMORIES]` block into system prompt |
| `src/profile/` | User profile facts extraction | Startup greeting, name recognition |
| `src/tools/` | Tool implementations: time, screenshot, notifications, clipboard, open_app, web_search | SearXNG-backed web search |
| `src/agents/` | Agent delegation for complex tasks | ACP protocol via stdio |
| `src/remote/` | WebSocket server for remote audio streaming | Binary PCM i16 LE 16kHz + JSON control |
| `src/tui/` | Terminal UI (ratatui) | Enable with `--features tui` |
| `src/bin/acp_agent_chat.rs` | Debug/test TUI chat with ACP agent via JSON-RPC 2.0 over stdio | Run: `cargo run --bin acp_agent_chat` |
| `src/bin/test_stt_plus.rs` | Test binary for whisper-cpp-plus streaming functionality | Run: `cargo run --bin test_stt_plus --release` |

### Legacy Modules (do not extend)

- `src/websocket_client.rs` — No longer needed
- `provider/` — Python LFM2.5-Audio server (not used)
- `src/stt/whisper.rs` — Deprecated; replaced by `whisper_plus.rs`

**Do not extend legacy modules.** If you find code there, flag it for removal.

---

## Environment Variables (critical)

Read from `.env` (dotenvy loads automatically):

```bash
# Audio
AUDIO_SAMPLE_RATE=16000
AUDIO_CHANNELS=1
VOICEBOT_LANGUAGE=es          # es (default) or en
WHISPER_MODEL=models/ggml-large-v3-turbo.bin
WHISPER_THREADS=0             # auto

# LLM
LLM_URL=http://127.0.0.1:8000 # mlx-lm default; oMLX is 8001
LLM_MAX_TOKENS=1024
LLM_CONTEXT_TOKENS=8192
LLM_CONSOLIDATION_THRESHOLD_PCT=80
LLM_SUMMARY_KEEP_TURNS=6

# Web search
SEARXNG_URL=https://searxng.example.com
SEARXNG_SECRET=<bearer_token>

# Remote
WS_PORT=9090                    # Enable WebSocket server
```

---

## Testing Quirks

- **VAD/audio tests**: Use synthetic sine waves / silence (see `src/audio/` tests).
- **STT tests**: Skip if model file missing (`#[ignore]`). Uses `whisper-cpp-plus`.
- **TTS tests**: macOS requires voices installed; kokoro for Linux CI.
- **Parallel tests**: Use `temp-env` crate to safely override env vars.
- **Mock LLM**: Use `wiremock` crate for HTTP client tests.

Run specific test:
```bash
cargo test <test_name> -- --nocapture
```

### Debugging Binary

The `test_stt_plus` binary provides standalone STT testing without full pipeline:
```bash
cargo run --bin test_stt_plus --release
```

---

## Build Features & Dependencies

| Feature | Enables | Extra deps | Requirements |
|---------|---------|------------|--------------|
| (none) | Core pipeline | whisper-cpp-plus, reqwest, sqlx | — |
| `kokoro` | Kokoro ONNX TTS | kokorox | `brew install espeak-ng` |
| `tui` | Terminal UI | ratatui, crossterm | — |
| `remote` | WebSocket server | axum, tower | — |
| `speaker` | Speaker verification | sherpa-rs | `models/speaker_embedding.onnx` |
| `avspeech` | macOS AVSpeechSynthesizer | objc2*, block2 | macOS only |

**On macOS**: whisper-cpp-plus uses Metal by default (faster STT via whisper-cpp-plus metal feature). Model files: `models/ggml-large-v3-turbo.bin` + `models/*-encoder.mlmodelc` for CoreML encoder fallback.

---

## Code Style & Patterns

- **Error handling**: `anyhow::Result` with context strings; `thiserror` for custom types.
- **Logging**: `tracing` throughout (no println!); logs → `voicebot.log` when TUI active.
- **Async**: tokio runtime + channels (`mpsc`, `broadcast`) for inter-stage comms.
- **Serialization**: serde + serde_json.

### When Adding Tools

1. Define tool schema in `src/tools/mod.rs` or dedicated module.
2. Implement handler returning `Result<String, Error>`.
3. Register in main pipeline's tool map.
4. Add doc comment explaining use case and limitations.

### Database Migrations

Use sqlx migrations:
```bash
sqlx migrate add <migration_name>
sqlx migrate run
```

Migrations live in `src/db/migrations/`.

---

## Common Workflows

### Running Development

```bash
# 1. Ensure .env exists (cp .env.example .env if not)
# 2. Start external LLM server (mlx-lm or oMLX)
# 3. Run voicebot
cargo run --features tui --release
```

### Adding a New Feature

1. Read relevant section of `CLAUDE.md` first (architecture guidance).
2. Check existing tools/agents to avoid duplication.
3. If feature affects multiple modules, create integration test in `e2e_tests.rs`.
4. Update this file if you discover new conventions.

### Debugging Pipeline Latency

Check these stages:
- VAD sensitivity: `src/audio/vad.rs` (frame thresholds).
- Whisper decoding: `src/stt/whisper.rs` (thread count, model size).
- LLM response time: External server config, context window size.
- TTS synthesis: `say` vs Kokoro vs AVSpeech backend choice.

Log with `RUST_LOG=trace cargo run` for detailed timing.

---

## References

- `CLAUDE.md`: Detailed architecture (keep synced with this file).
- `readme.md`: User-facing documentation.
- `CONTRIBUTING.md`: Contributor guidelines.
- `secondary-agent.md`: Secondary LLM orchestration design (Spanish).
