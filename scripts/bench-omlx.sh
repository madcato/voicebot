#!/usr/bin/env bash
# bench-omlx.sh — Two-way KV-cache benchmark: mlx-lm vs omlx
#
# Wrapper around bench-server.py that benchmarks both Apple MLX backends.
#
# Usage:
#   ./scripts/bench-omlx.sh [mlx-model] [omlx-model-dir]
#
# Examples:
#   ./scripts/bench-omlx.sh                          # use env var defaults
#   ./scripts/bench-omlx.sh \
#       mlx-community/Qwen3-8B-4bit \
#       ~/models
#
# Env vars (used when positional args are not given):
#   MLX_MODEL     mlx model or HF repo     (default: mlx-community/Qwen2.5-7B-Instruct-4bit)
#   OMLX_DIR      omlx model directory     (default: ~/.lmstudio/models)
#
#   MLX_PORT      mlx-lm port              (default: 8000)
#   OMLX_PORT     omlx port                (default: 8001)
#   BENCH_TRIALS  measurement trials       (default: 3)
#   BENCH_GEN     tokens to generate       (default: 80)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

MLX_MODEL="${1:-${MLX_MODEL:-mlx-community/Qwen2.5-7B-Instruct-4bit}}"
OMLX_DIR="${2:-${OMLX_DIR:-${HOME}/.lmstudio/models}}"

echo "Two-way LLM benchmark (Apple MLX)"
echo "  mlx-lm model    : $MLX_MODEL"
echo "  omlx model-dir  : $OMLX_DIR"
echo ""

exec python3 "${SCRIPT_DIR}/bench-server.py" "$MLX_MODEL" "$OMLX_DIR"

# sample
# ./bench-omlx.sh mlx-community/qwen3.5-35b-a3b ~/models
