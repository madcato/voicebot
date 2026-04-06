#!/usr/bin/env bash
# start-llm.sh — Launch an LLM server for voicebot (Apple Silicon)
#
# Delegates to the appropriate backend script based on LLM_BACKEND:
#
#   mlx-lm  (default) — mlx_lm.server, port 8000
#   omlx              — omlx serve,    port 8001
#
# Usage:
#   ./scripts/start-llm.sh
#   LLM_BACKEND=omlx ./scripts/start-llm.sh
#   LLM_BACKEND=mlx-lm MLX_MODEL=mlx-community/Qwen3-8B-4bit ./scripts/start-llm.sh
#
# After launch, set in .env:
#   LLM_URL=http://127.0.0.1:8000   # mlx-lm default
#   LLM_URL=http://127.0.0.1:8001   # omlx default

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BACKEND="${LLM_BACKEND:-mlx-lm}"

case "$BACKEND" in
  mlx-lm|mlx_lm|mlx)
    exec "$SCRIPT_DIR/start-mlx-lm.sh" "$@"
    ;;
  omlx)
    exec "$SCRIPT_DIR/start-omlx.sh" "$@"
    ;;
  *)
    echo "Error: unknown LLM_BACKEND=$BACKEND"
    echo ""
    echo "Supported backends:"
    echo "  mlx-lm  (default) — mlx_lm.server (pip install mlx-lm)"
    echo "  omlx              — omlx serve     (https://github.com/jundot/omlx)"
    echo ""
    echo "Usage: LLM_BACKEND=mlx-lm ./scripts/start-llm.sh [model]"
    exit 1
    ;;
esac
