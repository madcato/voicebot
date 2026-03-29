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

RUST_LOG=performance=debug exec cargo run --release --bin voicebot --features avspeech,tui -- "$@"
