# Contributing to Hive Voicebot

Thank you for your interest in contributing to **Hive Voicebot**! This document provides guidelines and information for contributors.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Project Structure](#project-structure)
- [Coding Style](#coding-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Feature Requests & Bug Reports](#feature-requests--bug-reports)

---

## Code of Conduct

Hive Voicebot follows a simple code of conduct:

- Be respectful and inclusive in all interactions
- Provide constructive feedback, not criticism
- Focus on what's best for the community and project
- If you see something violating this, report it to maintainers

---

## Getting Started

### Prerequisites

- **Rust**: Version 1.75+ (stable channel)
- **macOS**: 12.0+ (Big Sur or later) with Apple Silicon recommended
- **Git**: For version control
- **Terminal**: Comfortable with command-line tools

### Setup Your Development Environment

```bash
# Install Rust if you haven't already
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone the repository
git clone https://github.com/Hive-Vote/voicebot.git
cd voicebot

# Set up your environment variables
cp .env.example .env
# Edit .env with your configuration (see README.md)

# Download required models
# See README.md "Models Required" section for instructions
```

### Build & Run

```bash
# Verify the build works
cargo build --release

# Run with default configuration
cargo run --release

# Run with Kokoro TTS
cargo run --features kokoro --release

# Run unit tests
cargo test
```

---

## Development Workflow

### 1. Fork the Repository

Click the "Fork" button on GitHub to create your own copy of the repository.

### 2. Create a Feature Branch

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/voicebot.git
cd voicebot

# Create and switch to a new branch
git checkout -b feature/amazing-feature-name

# OR for bug fixes
git checkout -b fix/bug-description
```

Branch naming conventions:
- `feature/<name>` - New features
- `fix/<name>` - Bug fixes
- `docs/<name>` - Documentation changes
- `refactor/<name>` - Code refactoring
- `test/<name>` - Test improvements
- `chore/<name>` - Maintenance tasks

### 3. Make Your Changes

Follow the coding style guidelines below. Make small, focused commits with descriptive messages.

### 4. Run Tests & Linting

```bash
# Format code
cargo fmt

# Check for issues
cargo clippy --all-targets --all-features

# Run all tests
cargo test --all-features

# Run E2E tests (requires audio device + env vars)
cargo test e2e -- --ignored --nocapture
```

### 5. Commit Your Changes

Write clear, conventional commit messages:

```bash
git commit -m "feat: add speaker verification with ONNX model"
```

Commit message conventions:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `refactor:` - Code refactoring
- `test:` - Test additions/changes
- `chore:` - Maintenance/build changes

See [Conventional Commits](https://www.conventionalcommits.org/) for details.

### 6. Push & Open PR

```bash
git push origin feature/amazing-feature-name
```

Then open a Pull Request on GitHub with:
- Clear title describing the change
- Description of what and why (not how - that's in the code)
- Links to related issues
- Testing instructions if applicable

---

## Project Structure

```
voicebot/
├── src/
│   ├── main.rs              # Entry point, VAD loop, conversation pipeline
│   ├── config.rs            # Environment-based configuration
│   ├── lib.rs               # Library exports
│   ├── daemon.rs            # Background process handling
│   ├── e2e_tests.rs         # End-to-end test suite
│   │
│   ├── audio/               # Audio processing layer
│   │   ├── mod.rs
│   │   ├── audio_capture.rs # CPAL microphone input
│   │   ├── buffer.rs        # Audio buffering
│   │   ├── output.rs        # CPAL speaker playback
│   │   └── vad.rs           # Voice Activity Detection (Silero)
│   │
│   ├── stt/                 # Speech-to-Text
│   │   ├── mod.rs
│   │   └── whisper.rs       # Whisper.cpp integration
│   │
│   ├── llm/                 # Large Language Model interaction
│   │   ├── mod.rs
│   │   ├── client.rs        # HTTP client for LLM server (SSE streaming)
│   │   └── session.rs       # Conversation state, context summaries
│   │
│   ├── tts/                 # Text-to-Speech
│   │   ├── mod.rs
│   │   ├── say.rs           # macOS `say` command
│   │   ├── kokoro.rs        # Kokoro ONNX TTS
│   │   └── sentence.rs      # Sentence splitter for streaming TTS
│   │
│   ├── tools/               # Tool calling system
│   │   ├── mod.rs
│   │   ├── registry.rs      # Tool registry and execution
│   │   ├── current_time.rs  # Get current date/time
│   │   ├── screenshot.rs    # Take screenshots
│   │   ├── clipboard.rs     # Read/write clipboard
│   │   └── notification.rs  # Send desktop notifications
│   │
│   ├── agents/              # External agent delegation
│   │   └── mod.rs           # run_agent / run_agent_async
│   │
│   ├── profile/             # User profile extraction
│   │   ├── mod.rs
│   │   └── extractor.rs     # Extract user facts from conversations
│   │
│   └── db/                  # SQLite database layer
│       ├── mod.rs
│       └── database.rs      # Sessions, messages, summaries storage
│
├── models/                  # Downloaded ML models
├── scripts/                 # Build, test, and benchmark scripts
├── data/                    # Runtime data (database, embeddings)
├── tests/                   # Integration/E2E test fixtures
└── doc/                     # Technical documentation
```

### Key Modules to Understand Before Contributing

| Module | Purpose | If You're Adding... |
|--------|---------|---------------------|
| `audio/vad.rs` | Detects speech start/end | New VAD models, better barge-in |
| `stt/whisper.rs` | Transcribes audio to text | Alternative STT backends |
| `llm/client.rs` | Talks to LLM server | New LLM providers, auth |
| `tts/say.rs`, `kokoro.rs` | Synthesizes speech | New TTS backend |
| `tools/registry.rs` | Executes tool calls | New voice-local tools |
| `profile/extractor.rs` | Learns user facts | Better profile extraction |
| `config.rs` | Reads environment vars | New configuration options |

---

## Coding Style

### Rust Standards

- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use snake_case for functions/variables, PascalCase for types
- Prefer `Result` over `Option` when failure has meaning
- Document public APIs with `///` doc comments

```rust
/// Checks if the audio buffer contains enough samples for processing.
///
/// # Arguments
/// * `min_samples` - The minimum number of f32 samples required
///
/// # Returns
/// `true` if the buffer has at least `min_samples`, `false` otherwise
pub fn has_minimal_data(&self, min_samples: usize) -> bool {
    self.samples.len() >= min_samples
}
```

### Formatting & Linting

Run before every commit:

```bash
# Auto-format code (required)
cargo fmt

# Lint with Clippy (fix all auto-fixable issues first)
cargo clippy --all-features || true
cargo fix --allow-dirty --all-targets --all-features -e "warn"
cargo fmt  # Re-format after fixes
```

### Error Handling

- Use `anyhow::Result` for internal functions
- Use custom `thiserror` types for library boundaries
- Prefer descriptive error messages with context

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SttError {
    #[error("Whisper model not found: {0}")]
    ModelNotFound(String),
    
    #[error("Failed to transcribe audio: {0}")]
    TranscriptionFailed(#[source] anyhow::Error),
}
```

### Logging

Use `tracing` for all logging. See [doc/LOGGING.md](doc/LOGGING.md) for target guidelines.

```rust
use tracing::{debug, info, warn, error};

info!("User said: {}", transcript);
warn!("VAD confidence low ({:.2}), may skip transcription", probability);
error!(?err, "Failed to synthesize TTS");
```

### Async Patterns

- All I/O (HTTP, file system, audio) must be async-aware
- Use `spawn_blocking` for CPU-heavy tasks (Whisper STT, Kokoro TTS)
- Never block the tokio runtime in `.await` paths

```rust
// ✅ Correct: CPU work offloaded to thread pool
let transcript = tokio::task::spawn_blocking(move || {
    stt.transcribe(samples)?
}).await??;

// ❌ Wrong: blocks event loop
let result = heavy_computation();  // Don't do this!
```

---

## Testing

### Test Types

1. **Unit Tests** (`cargo test`): Fast, isolated tests for individual functions
2. **Integration Tests**: Tests across module boundaries
3. **E2E Tests** (`cargo test e2e -- --ignored`): Full pipeline with audio I/O

### Writing Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentence_splitter_single() {
        let mut splitter = SentenceSplitter::new();
        assert!(!splitter.process_token("Hello").is_some());
        assert_eq!(splitter.process_token("!"), Some("Hello!".to_string()));
    }

    #[tokio::test]
    async fn test_llm_client_connect() {
        let client = LlamaClientBuilder::new("http://localhost:8080")
            .model("test")
            .build();
        
        let res = client.health_check().await;
        assert!(res.is_ok());
    }
}
```

### Running Tests

```bash
# All unit tests
cargo test

# With specific feature
cargo test --features kokoro

# Verbose output
cargo test -- --nocapture

# Single test
cargo test test_sentence_splitter_single

# E2E pipeline tests (marked #[ignore])
cargo test e2e -- --ignored --nocapture
```

### E2E Test Prerequisites

E2E tests require:
- Audio output device configured
- `WHISPER_MODEL` and `LLM_URL` set in `.env`
- WAV audio fixtures in `tests/fixtures/` (see [doc/E2E.md](doc/E2E.md))

---

## Documentation

### When to Document

**Always document:**
- Public APIs (`pub fn`, `pub struct`)
- Non-obvious implementation decisions
- Module-level architecture

**Don't over-document:**
- Private helper functions (let code speak)
- Obvious operations (e.g., `if x > y` doesn't need comments)

### Doc Comment Style

Use triple-slash (`///`) for items that will appear in `cargo doc`:

```rust
/// Transcribes audio samples to text using Whisper model.
///
/// # Errors
/// Returns `SttError::ModelNotFound` if the model path is invalid.
///
/// # Panics
/// Will panic if the model fails to initialize (this should be caught at startup).
pub fn transcribe(&self, samples: Vec<f32>) -> Result<String, SttError> { ... }
```

### Keep Docs Updated

When adding features:
- Update `README.md` in "Features" and "Configuration" sections
- Add to `doc/REFERENCE.md` if it affects environment variables
- Write inline docs for public APIs

---

## Submitting Changes

### Before Opening a PR

1. ✅ All tests pass: `cargo test --all-features`
2. ✅ Code is formatted: `cargo fmt`
3. ✅ No Clippy warnings: `cargo clippy --all-targets --all-features`
4. ✅ Docs updated (README, inline docs if needed)
5. ✅ Branch synced with main: `git pull origin main`

### PR Checklist

- [ ] Clear title summarizing the change
- [ ] Description explains **what** and **why** (not how)
- [ ] Links to related issues (GH-123)
- [ ] Testing instructions for reviewers
- [ ] Screenshots/GIFs if UI-affected (rare, but nice)

### PR Review Process

1. Maintainers will test your changes locally
2. May request modifications or ask clarifying questions
3. Once approved, squash-and-merge into main
4. Don't be discouraged by review feedback — it ensures quality!

---

## Feature Requests & Bug Reports

### Before Opening an Issue

**Search existing issues first!** Your question or bug may already be tracked.

### How to Report a Bug

Include:
- **Environment**: OS version, Rust version, hardware (M1/M2/etc.)
- **Steps to reproduce**: Precise steps leading to the bug
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens (with logs)
- **Logs**: Run with `RUST_LOG=voicebot=debug` and attach output

Example:
```bash
# Capture debug logs
RUST_LOG=voicebot=debug,vad=debug cargo run 2>&1 | tee bug.log
```

### How to Request a Feature

Include:
- **Use case**: Why is this feature important?
- **Proposed solution**: How should it work?
- **Alternatives considered**: Other approaches you thought of
- **Examples**: Code snippets, config changes, or pseudo-code if applicable

---

## Areas Needing Help

Here are some ways to contribute without deep code knowledge:

- 📝 **Documentation improvements** (README, examples, guides)
- 🧪 **Test coverage** (unit tests for untested functions)
- 🔍 **Bug reports** with detailed reproduction steps
- 🌐 **Translations** of system prompt templates
- 💡 **Feature proposals** with clear use cases

### Good First Issues

Check the GitHub issue tracker for issues labeled `good first issue` or `help wanted`.

---

## Questions?

If you have questions that aren't covered here:

1. Check existing issues and discussions on GitHub
2. Create a new discussion if it's a general question
3. Reach out in GitHub Discussions (preferred) or email maintainers

**No question is too small!** We'd rather answer than block your contribution.

---

## Acknowledgments

- Inspired by [mlx-lm](https://github.com/ml-explore/mlx-lm), [whisper-rs](https://github.com/robertodober/whisper-rs), and the Rust community
- Built with ❤️ by Daniel and the Hive Team

Happy coding! 🚀
