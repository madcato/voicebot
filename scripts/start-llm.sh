#!/usr/bin/env bash
# Launch llama.cpp server for voicebot (single-user, Apple Silicon)
#
# Prerequisites:
#   - llama-server in PATH  (brew install llama.cpp  or build from source)
#   - A GGUF model file (see LLM_MODEL below)
#
# Usage:
#   ./scripts/start-llm.sh
#   LLM_MODEL=/path/to/model.gguf ./scripts/start-llm.sh
#
# Key llama.cpp options for voicebot:
#   --parallel 1        Single user — one KV-cache slot, no batching overhead
#   --cache-type-k q8_0 KV-cache quantisation: saves VRAM, minimal quality loss
#   --cache-type-v q8_0
#   --flash-attn on     Flash Attention — faster inference on Apple Silicon
#   -ngl 99             Offload all layers to Metal GPU
#   --mlock             Lock model weights in RAM to avoid swap latency
#   --ctx-size 32768    ~32k token context window (full conversation history)
#   --repeat-penalty    Reduce repetitive responses

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Configuration (override via env vars) ────────────────────────────────────

# Path to the GGUF model. Default: look in project models/ directory.
LLM_MODEL="${LLM_MODEL:-${ROOT_DIR}/models/Qwen3.5-4B-Q8_0.gguf}"

# llama-server bind address and port
LLM_HOST="${LLM_HOST:-0.0.0.0}"
LLM_PORT="${LLM_PORT:-8080}"

# Context size in tokens
LLM_CTX="${LLM_CTX:-32768}"

# Number of CPU threads (leave 2 free for audio pipeline)
LLM_THREADS="${LLM_THREADS:-$(( $(sysctl -n hw.logicalcpu 2>/dev/null || echo 8) - 2 ))}"

# ── Validation ────────────────────────────────────────────────────────────────

if ! command -v llama-server &>/dev/null; then
    echo "ERROR: llama-server not found in PATH."
    echo "Install with:  brew install llama.cpp"
    echo "Or build from: https://github.com/ggerganov/llama.cpp"
    exit 1
fi

if [[ ! -f "$LLM_MODEL" ]]; then
    echo "ERROR: Model file not found: $LLM_MODEL"
    echo ""
    echo "Download a model, e.g.:"
    echo "  huggingface-cli download bartowski/Qwen2.5-7B-Instruct-GGUF \\"
    echo "    Qwen2.5-7B-Instruct-Q8_0.gguf --local-dir ${ROOT_DIR}/models/"
    echo ""
    echo "Then set LLM_MODEL=/path/to/model.gguf or place it in models/"
    exit 1
fi

# ── Launch ────────────────────────────────────────────────────────────────────

echo "======================================================"
echo "  Voicebot LLM Server"
echo "======================================================"
echo "  Model:   $LLM_MODEL"
echo "  Endpoint: http://${LLM_HOST}:${LLM_PORT}"
echo "  Context:  ${LLM_CTX} tokens"
echo "  Threads:  ${LLM_THREADS}"
echo "======================================================"
echo ""

exec llama-server \
    --model "$LLM_MODEL" \
    --host "$LLM_HOST" \
    --port "$LLM_PORT" \
    --ctx-size "$LLM_CTX" \
    --threads "$LLM_THREADS" \
    --n-gpu-layers 99 \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --flash-attn on \
    --mlock \
    --parallel 1 \
    --repeat-penalty 1.1 \
    --temp 0.6 \
    --n-predict 120 \
    --reasoning-budget 0 \
    --verbose
