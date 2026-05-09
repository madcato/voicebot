# Doc Review - Learnings

## Verified architecture facts (as of 2026-05-09)

### STT
- Uses `whisper-cpp-plus` crate (NOT `whisper-rs`)
- VAD (Silero) is integrated into `WhisperSTTVAD` in `src/stt/mod.rs` — NOT a separate module
- No separate `VoiceActivityDetector` struct exists

### TTS
- `avspeech` feature → `AvSpeechTts` (AVSpeechSynthesizer) — NOT `say`
- `kokoro` feature → `KokoroTts` (ONNX)
- Config uses `AVSPEECH_VOICE` / `AVSPEECH_RATE` — NOT `SAY_VOICE`
- No `say.rs` file exists

### LLM
- `OpenAIClient` in `src/llm/client.rs` — NOT `LlamaClient` / `LlmClient`
- Client-side sampling: `top_p: 0.90`, `top_k: 40`, `min_p: 0.05`, `repetition_penalty: 1.1`
- `enable_thinking` sent via `chat_template_kwargs`

### Tools
- No `send_notification` tool exists
- Actual tools: `CurrentTimeTool`, `ReadClipboardTool`, `SetClipboardTool`, `OpenAppTool`, `TakeScreenshotTool`, `WebSearchTool`, `RunShellTool`, `ReadFileTool`, `SetConversationModeTool`, `RunAgentTool`
- Tool trait: `name()`, `description()`, `parameters()`, `run(args) -> String`

### Non-existent modules
- No `src/s2s/` directory (was mentioned in testing_strategy.md)
- No `src/session/` directory (session logic is in `src/llm/session.rs` and `src/db/`)
- No `src/tools/builtin/` subdirectory (all tools are flat in `src/tools/`)
- No `ToolRouter` — tools are in `ToolRegistry`
- No `SessionManager` — sessions are managed in `src/llm/session.rs` (`LlmSession`)

### E2E tests
- Live in `src/e2e_tests.rs` as `#[cfg(test)]` module of `main.rs`
- Use `E2eHarness` with direct transcript injection — NOT `SttStream::mock()`
- All tests are `#[ignore]`
- Uses `TtsEngine::Mock`, `Wiremock`, real SQLite in `TempDir`
- `AudioOutput::null()` for audio (no hardware required)

### Agents
- Hermes integration exists via `AGENT_COMMAND` env var and `src/tools/run_agent.rs`
- ACP protocol via stdio (JSON-RPC 2.0)
- `src/bin/acp_agent_chat.rs` for standalone testing
- `src/agents/mod.rs` for proactive events
