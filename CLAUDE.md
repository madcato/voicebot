# CLAUDE.md

This file provides guidance to AI coding agents when working with code in this repository.

## Legal & Naming

- **Jarvis®** is a trademark of Marvel Studios/Disney. This is an independent fan project.
- Refer to this project as **"Voicebot"**. Never "Jarvis" or "Hive".
- See `LICENSE-VOICEBOT.md` for full details.

## Commands

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

## Architecture

Voicebot is a mono-user voice AI chatbot in Rust. Streaming STT→LLM→TTS pipeline, single process, tokio channels.

### Data flow

```
Microphone → AudioCapture (CPAL) → WhisperSTTVAD (whisper-cpp-plus + Silero VAD)
      partial transcripts accumulated in-memory
→ LLM client (OpenAI-compatible /v1/chat/completions, streaming SSE)
      tokens streamed as they arrive
→ SentenceSplitter (buffer until punctuation boundary)
→ TTS (macOS AVSpeechSynthesizer or Kokoro ONNX)
      synthesizes sentence by sentence
→ AudioOutput (CPAL speaker)
```

### Key Design Decisions

- **Single binary**: no inter-service communication; all stages connected by `tokio::sync` channels
- **STT→LLM latency trick**: partial Whisper transcripts are accumulated in a `String`; when VAD signals end-of-speech the full transcript is sent to the LLM server. The server maintains its own KV-cache implicitly across requests within a session.
- **LLM→TTS streaming**: LLM tokens arrive via SSE and are buffered until a sentence boundary (`.`, `!`, `?`, `;`, `:`) — then that sentence is synthesized immediately. While sentence N plays, sentence N+1 is being generated and synthesized.
- **Language**: Spanish by default (`VOICEBOT_LANGUAGE=es`), English supported. Affects Whisper hint and TTS voice.
- **Barge-in**: Implemented via `CancellationToken` (tokio-util). User speech cancels the active pipeline.
- **Agent delegation**: Complex tasks can be delegated to external AI agents via the ACP protocol (Agent Communication Protocol) over stdio.

## Key Modules

| Directory | Purpose | Key Files |
|-----------|---------|-----------|
| `src/audio/` | Audio pipeline: capture, VAD, resampling, playback | `audio_capture.rs`, `vad.rs`, `buffer.rs`, `audio_transform.rs`, `output.rs`, `speaker.rs`, `ambient_buffer.rs` |
| `src/stt/` | Speech-to-Text via whisper-cpp-plus | `mod.rs` (WhisperSTTVAD), `whisper.rs` (DEPRECATED) |
| `src/llm/` | LLM client, session management, context consolidation | `client.rs` (OpenAIClient), `session.rs` (LlmSession), `manager.rs` (LLM manager) |
| `src/tts/` | Text-to-Speech backends | `avspeech.rs` (macOS AVSpeech), `kokoro.rs` (ONNX TTS), `sentence.rs` (SentenceSplitter), `piper.rs` (reference only) |
| `src/pipeline/` | Pipeline orchestration with FSM | `mod.rs`, `fsm.rs` (PipelineState), `llm_task.rs`, `tts_task.rs`, `sen_task.rs`, `frames.rs`, `state.rs`, `consolidation.rs` |
| `src/daemon.rs` | InferenceDaemon — main inference lifecycle | Loops: listen VAD → STT → LLM → TTS |
| `src/eyes.rs` | EyesDaemon — visual/status monitoring | Periodically observes system state |
| `src/control/` | Control API (HTTP/WebSocket) | `api.rs`, `state.rs`, `broadcast.rs`, `client.rs`, `mod.rs` |
| `src/mcp/` | Model Context Protocol integration | `mod.rs` (McpClient, McpToolDef, call_tool) |
| `src/tools/` | LLM-invocable tools | `clipboard.rs`, `conversation_mode.rs`, `current_time.rs`, `mcp_tool.rs`, `open_app.rs`, `read_file.rs`, `run_agent.rs`, `run_shell.rs`, `take_screenshot.rs`, `web_search.rs` |
| `src/agents/` | Agent delegation via ACP protocol | `mod.rs`, `config.rs`, `session_manager.rs`, `session_events.rs` |
| `src/remote/` | WebSocket server for remote audio streaming | `mod.rs`, `server.rs`, `protocol.rs` |
| `src/analysis/` | Identity analysis | `mod.rs`, `identity.rs` |
| `src/profile/` | User profile facts extraction | `mod.rs` |
| `src/tui/` | Terminal UI (ratatui) | `app.rs`, `events.rs`, `input.rs`, `mod.rs`, `ui.rs` |
| `src/memory/` | Persistent memory system | `mod.rs` (extract_memories, build_memory_context) |
| `src/db/` | SQLite persistence | `database.rs`, `mod.rs` (sessions, messages, user_profile, memories tables) |
| `src/config.rs` | Environment-based configuration | `Config::from_env()` |
| `src/i18n.rs` | Internationalization support | Language-specific strings |
| `src/bin/` | Standalone binaries | `acp_agent_chat.rs` (ACP debug/test) |
| `src/e2e_tests.rs` | End-to-end pipeline tests | Integration tests |

### Legacy Modules (do not extend)

- `src/stt/whisper.rs` — **DEPRECATED** — legacy whisper-rs wrapper; replaced by `whisper-cpp-plus` in `src/stt/mod.rs`
- `src/websocket_client.rs` — No longer needed
- `provider/` — Python LFM2.5-Audio server (not used)
- `src/tts/piper.rs` — Piper subprocess wrapper (kept for reference, not active)

**Do not extend legacy modules.** If you find code there, flag it for removal.

## Environment Variables

Read from `.env` (dotenvy loads automatically):

| Variable | Default | Description |
|----------|---------|-------------|
| `AUDIO_SAMPLE_RATE` | `16000` | Audio sample rate |
| `AUDIO_CHANNELS` | `1` | Audio channels |
| `VOICEBOT_LANGUAGE` | `es` | Language (`es` or `en`) |
| `WHISPER_MODEL` | `models/ggml-large-v3-turbo.bin` | Whisper GGML model path |
| `WHISPER_THREADS` | `0` | CPU threads (0 = auto) |
| `LLM_URL` | `http://127.0.0.1:8000` | LLM server URL (mlx-lm/oMLX) |
| `LLM_MAX_TOKENS` | `1024` | Max tokens per response |
| `LLM_CONTEXT_TOKENS` | `8192` | Context window size |
| `LLM_CONSOLIDATION_THRESHOLD_PCT` | `80` | % threshold for consolidation |
| `LLM_SUMMARY_KEEP_TURNS` | `6` | Recent turns to keep after consolidation |
| `AVSPEECH_VOICE` | `"Jorge (Enhanced)"` | macOS AVSpeech voice name |
| `AVSPEECH_RATE` | `0.55` | Speech rate (0.0–1.0) |
| `SEARXNG_URL` | — | SearXNG base URL (enables web_search) |
| `SEARXNG_SECRET` | — | SearXNG bearer token |
| `WS_PORT` | `9090` | WebSocket server port |

## Build Features

| Feature | Enables | Dependencies | Requirements |
|---------|---------|-------------|--------------|
| (none) | Core pipeline | whisper-cpp-plus, reqwest, sqlx | — |
| `kokoro` | Kokoro ONNX TTS | kokorox | `brew install espeak-ng` |
| `tui` | Terminal UI | ratatui, crossterm | — |
| `remote` | WebSocket server | axum, tower | — |
| `speaker` | Speaker verification | sherpa-rs | `models/speaker_embedding.onnx` |
| `avspeech` | macOS AVSpeechSynthesizer | objc2*, block2 | macOS only |

## Design Patterns

- **Error handling**: `anyhow::Result` with context strings; `thiserror` for custom error types
- **Logging**: `tracing` throughout (no println!); logs → `voicebot.log` when TUI active
- **Async**: tokio runtime + channels (`mpsc`, `broadcast`) for inter-stage communication
- **Cancellation**: `CancellationToken` (tokio-util) for barge-in support
- **Serialization**: serde + serde_json
- **Tool calling**: LLM uses `⟨tool_name: args⟩` syntax; parsed by ToolRegistry

## Testing Approach

- **VAD/audio tests**: Use synthetic sine waves / silence
- **STT tests**: Skip if model file missing (`#[ignore]`)
- **TTS tests**: macOS requires voices installed; kokoro for Linux CI
- **Mock LLM**: Use `wiremock` crate for HTTP client tests
- **Parallel tests**: Use `temp-env` crate to safely override env vars

Run specific test:
```bash
cargo test <test_name> -- --nocapture
```

## References

- `AGENTS.md`: Agent work instructions (git workflow, issue management, agent authority)
- `LICENSE-VOICEBOT.md`: Trademark information
- `README.md`: User-facing documentation
