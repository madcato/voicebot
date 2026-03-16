#!/usr/bin/env bash
# start-omlx.sh — Launch omlx server optimized for single-user voicebot
#
# omlx is an Apple Silicon LLM server with persistent tiered KV caching
# (hot RAM + SSD), OpenAI-compatible API, and multi-model support.
# https://github.com/jundot/omlx
#
# Usage:
#   ./scripts/start-omlx.sh [model_dir]
#
# Examples:
#   ./scripts/start-omlx.sh ~/models
#   ./scripts/start-omlx.sh ~/models/Qwen2.5-7B-Instruct
#   OMLX_MODEL_DIR=~/models ./scripts/start-omlx.sh
#
# After launch, set in .env:
#   LLM_URL=http://127.0.0.1:8000
#   LLM_PROVIDER=mlx

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via env vars)
# ---------------------------------------------------------------------------

MODEL_DIR="${1:-${OMLX_MODEL_DIR:-${HOME}/.lmstudio/models}}"
HOST="${OMLX_HOST:-127.0.0.1}"
PORT="${OMLX_PORT:-8001}"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if ! command -v omlx &>/dev/null; then
  echo "Error: omlx not found in PATH."
  echo ""
  echo "Install from: https://github.com/jundot/omlx"
  exit 1
fi

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Error: model directory not found: $MODEL_DIR"
  echo ""
  echo "Usage:  $0 <model_dir>"
  echo "        OMLX_MODEL_DIR=<path> $0"
  echo ""
  echo "Example: $0 ~/models"
  exit 1
fi

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

echo "Starting omlx server"
echo "  Model dir:  $MODEL_DIR"
echo "  Endpoint:   http://$HOST:$PORT/v1"
echo ""

exec omlx serve \
  --model-dir "$MODEL_DIR" \
  --host      "$HOST" \
  --port      "$PORT"
