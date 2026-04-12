#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment variables
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    set -a
    # shellcheck source=.env
    source "$SCRIPT_DIR/.env"
    set +a
fi

# WHISPER_SILENCE=1 WHISPER_COREML=1 RUST_LOG=info exec cargo run --release --bin voicebot --features avspeech,tui -- "$@" 2>/dev/null

## STT Performance
# WHISPER_SILENCE=1 WHISPER_COREML=1 RUST_LOG=performance=debug exec cargo run --release --bin voicebot --features avspeech,tui -- "$@" 2>/dev/null

## Performance
WHISPER_SILENCE=1 WHISPER_COREML=1 RUST_LOG=performance=debug exec cargo run --release --bin voicebot --features avspeech,tui -- "$@" 2> >(grep -vE "^(whisper_|ggml_)" >&2)

## Tools and agent debugging
# WHISPER_SILENCE=1 WHISPER_COREML=1 RUST_LOG=pipeline=debug,llm=debug,tools=debug,agent=debug exec cargo run --release --bin voicebot --features avspeech,tui -- "$@" 2>/dev/null

