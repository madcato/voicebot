#!/usr/bin/env bash
# bench-omlx-models.sh — omlx multi-model KV-cache benchmark
#
# Tests TTFT, PP, TG, and KV-cache health for a curated list of models
# served by omlx.  Starts omlx automatically if it is not already running,
# or connects to an existing instance on OMLX_PORT.
#
# Usage:
#   ./scripts/bench-omlx-models.sh [model-dir]
#
# Examples:
#   ./scripts/bench-omlx-models.sh
#   ./scripts/bench-omlx-models.sh ~/models
#   OMLX_PORT=8001 BENCH_TRIALS=5 ./scripts/bench-omlx-models.sh ~/models
#
# Env vars:
#   OMLX_DIR       omlx model directory  (default: ~/.lmstudio/models)
#   OMLX_PORT      omlx server port      (default: 8001)
#   BENCH_TRIALS   hot measurement trials (default: 3)
#   BENCH_GEN      tokens to generate    (default: 80)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OMLX_DIR="${1:-${OMLX_DIR:-${HOME}/.lmstudio/models}}"

export OMLX_DIR
export OMLX_PORT="${OMLX_PORT:-8000}"
export BENCH_TRIALS="${BENCH_TRIALS:-3}"
export BENCH_GEN="${BENCH_GEN:-80}"

echo "omlx Multi-Model Benchmark"
echo "  Model dir : $OMLX_DIR"
echo "  Port      : $OMLX_PORT"
echo "  Trials    : $BENCH_TRIALS"
echo "  Gen tokens: $BENCH_GEN"
echo ""

exec python3 "${SCRIPT_DIR}/bench-omlx-models.py" "$OMLX_DIR"
