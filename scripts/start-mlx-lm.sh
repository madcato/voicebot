#!/usr/bin/env bash
# start-mlx-lm.sh — Launch mlx-lm server optimized for single-user voicebot
#
# Usage:
#   ./scripts/start-mlx-lm.sh [model_path_or_hf_repo]
#
# Examples:
#   ./scripts/start-mlx-lm.sh mlx-community/Qwen2.5-7B-Instruct-4bit
#   ./scripts/start-mlx-lm.sh ./models/my-mlx-model
#   MLX_MODEL=mlx-community/Qwen2.5-7B-Instruct-4bit ./scripts/start-mlx-lm.sh
#
# After launch, set in .env:
#   LLM_URL=http://127.0.0.1:8000
#   LLM_PROVIDER=mlx

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via env vars)
# ---------------------------------------------------------------------------

MODEL="${1:-${MLX_MODEL:-}}"
HOST="${MLX_HOST:-127.0.0.1}"
PORT="${MLX_PORT:-8000}"
MAX_TOKENS="${MLX_MAX_TOKENS:-600}"
TEMPERATURE="${MLX_TEMP:-0.7}"

# KV-cache: 1 slot (mono-user). Size in bytes — 3 GB is enough for long
# conversations at 8K–32K context with 7B–14B models. Raise if you see
# cache evictions in the logs.
CACHE_SIZE_BYTES="${MLX_CACHE_BYTES:-$((3 * 1024 * 1024 * 1024))}"  # 3 GB

# Prefill chunk size. 512 keeps TTFT low when the new user turn is short
# (typical utterance = 10–50 tokens, fits in a single chunk).
# Increase to 1024 or 2048 if you send very long prompts.
PREFILL_STEP="${MLX_PREFILL_STEP:-512}"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if [[ -z "$MODEL" ]]; then
  echo "Error: no model specified."
  echo ""
  echo "Usage:  $0 <model_path_or_hf_repo>"
  echo "        MLX_MODEL=<model> $0"
  echo ""
  echo "Example MLX models (mlx-community on HuggingFace):"
  echo "  mlx-community/Qwen2.5-7B-Instruct-4bit"
  echo "  mlx-community/Qwen2.5-14B-Instruct-4bit"
  echo "  mlx-community/Mistral-7B-Instruct-v0.3-4bit"
  echo "  mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
  exit 1
fi

# ---------------------------------------------------------------------------
# Detect mlx-lm runner
# ---------------------------------------------------------------------------

if command -v mlx_lm &>/dev/null; then
  RUNNER="mlx_lm"
elif python3 -c "import mlx_lm" &>/dev/null 2>&1; then
  RUNNER="python3 -m mlx_lm"
elif command -v uvx &>/dev/null; then
  RUNNER="uvx mlx_lm"
else
  echo "Error: mlx-lm not found. Install it with:"
  echo "  pip install mlx-lm"
  echo "  # or: brew install uv && uvx mlx_lm server --help"
  exit 1
fi

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

echo "Starting mlx-lm server"
echo "  Model:         $MODEL"
echo "  Endpoint:      http://$HOST:$PORT/v1"
echo "  KV-cache:      1 slot, $(( CACHE_SIZE_BYTES / 1024 / 1024 / 1024 )) GB max"
echo "  Prefill step:  $PREFILL_STEP tokens"
echo "  Max tokens:    $MAX_TOKENS"
echo "  Temperature:   $TEMPERATURE"
echo ""

exec $RUNNER server \
  --model          "$MODEL" \
  --host           "$HOST" \
  --port           "$PORT" \
  --prompt-cache-size  1 \
  --prompt-cache-bytes "$CACHE_SIZE_BYTES" \
  --prefill-step-size  "$PREFILL_STEP" \
  --max-tokens     "$MAX_TOKENS" \
  --temp           "$TEMPERATURE" \
  --chat-template-args '{"enable_thinking": false}' \
  --log-level      INFO

# exec $RUNNER benchmark \
#   --model          "$MODEL"