#!/usr/bin/env bash
# bench-omlx.sh — Three-way KV-cache benchmark: llama.cpp vs mlx-lm vs omlx
#
# Wrapper around bench-server.py that includes omlx as a third provider.
#
# Usage:
#   ./scripts/bench-omlx.sh [llama.gguf] [mlx-model] [omlx-model-dir]
#
# Examples:
#   ./scripts/bench-omlx.sh                          # use env var defaults
#   ./scripts/bench-omlx.sh \
#       ./models/Qwen3.5-2B-Q4_K_M.gguf \
#       mlx-community/Qwen2.5-7B-Instruct-4bit \
#       ~/models
#
# Env vars (used when positional args are not given):
#   LLAMA_MODEL   path to .gguf file       (default: models/Qwen3.5-2B-Q4_K_M.gguf)
#   MLX_MODEL     mlx model or HF repo     (default: mlx-community/Qwen2.5-7B-Instruct-4bit)
#   OMLX_DIR      omlx model directory     (default: ~/.lmstudio/models)
#
#   LLAMA_PORT    llama.cpp port           (default: 8080)
#   MLX_PORT      mlx-lm port              (default: 8000)
#   OMLX_PORT     omlx port                (default: 8001)
#   BENCH_TRIALS  measurement trials       (default: 3)
#   BENCH_GEN     tokens to generate       (default: 80)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

LLAMA_MODEL="${1:-${LLAMA_MODEL:-${ROOT_DIR}/models/Qwen3.5-2B-Q4_K_M.gguf}}"
MLX_MODEL="${2:-${MLX_MODEL:-mlx-community/Qwen2.5-7B-Instruct-4bit}}"
OMLX_DIR="${3:-${OMLX_DIR:-${HOME}/.lmstudio/models}}"

echo "Three-way LLM benchmark"
echo "  llama.cpp model : $(basename "$LLAMA_MODEL")"
echo "  mlx-lm model    : $MLX_MODEL"
echo "  omlx model-dir  : $OMLX_DIR"
echo ""

exec python3 "${SCRIPT_DIR}/bench-server.py" "$LLAMA_MODEL" "$MLX_MODEL" "$OMLX_DIR"

# sample
# ./bench-omlx.sh ./models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf mlx-community/qwen3.5-35b-a3b mlx-community/qwen3.5-35b-a3b