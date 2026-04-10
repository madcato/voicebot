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

# WHISPER_COREML=1 RUST_LOG=info exec cargo run --release --bin voicebot --features avspeech,tui -- "$@"

## STT Performance
# WHISPER_COREML=1 RUST_LOG=performance=debug exec cargo run --release --bin voicebot --features avspeech,tui -- "$@"

## Performance
# RUST_LOG=performance=info exec cargo run --release --bin voicebot --features avspeech,tui -- "$@"

## Tools and agent debugging
RUST_LOG=pipeline=debug,llm=debug,tools=debug,agent=debug exec cargo run --release --bin voicebot --features avspeech,tui -- "$@"
