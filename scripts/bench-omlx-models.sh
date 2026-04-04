#!/usr/bin/env bash
# bench-omlx-models.sh — multi-model KV-cache benchmark
#
# Tests TTFT, PP, TG, and KV-cache health for a curated list of models.
# Supports omlx, LM Studio, and any OpenAI-compatible server.
#
# Usage:
#   ./scripts/bench-omlx-models.sh [model-dir]
#
# Examples:
#   # omlx (auto-start if not running)
#   ./scripts/bench-omlx-models.sh ~/models
#
#   # omlx already running
#   BENCH_PORT=8000 ./scripts/bench-omlx-models.sh
#
#   # LM Studio (must be running with local server enabled)
#   BENCH_PORT=1234 BENCH_TOKEN="" BENCH_PROVIDER=lmstudio ./scripts/bench-omlx-models.sh
#
# Env vars:
#   BENCH_PORT       server port           (default: 8000)
#   BENCH_TOKEN      Bearer auth token     (default: "asdf" for omlx, set "" for LM Studio)
#   BENCH_PROVIDER   label in output       (default: "omlx")
#   BENCH_TRIALS     measurement trials    (default: 3)
#   BENCH_GEN        tokens to generate    (default: 80)
#   OMLX_DIR         model dir for omlx    (default: ~/.lmstudio/models)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OMLX_DIR="${1:-${OMLX_DIR:-${HOME}/.lmstudio/models}}"

export OMLX_DIR
export BENCH_PORT="${BENCH_PORT:-${OMLX_PORT:-8080}}"
export BENCH_TOKEN="${BENCH_TOKEN:-${OMLX_TOKEN:-asdf}}"
export BENCH_PROVIDER="${BENCH_PROVIDER:-omlx}"
export BENCH_TRIALS="${BENCH_TRIALS:-3}"
export BENCH_GEN="${BENCH_GEN:-80}"

echo "Multi-Model KV-Cache Benchmark"
echo "  Provider  : $BENCH_PROVIDER"
echo "  Port      : $BENCH_PORT"
echo "  Trials    : $BENCH_TRIALS"
echo "  Gen tokens: $BENCH_GEN"
echo ""

exec python3 "${SCRIPT_DIR}/bench-omlx-models.py" "$OMLX_DIR"
