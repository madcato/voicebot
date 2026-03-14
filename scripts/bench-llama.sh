#!/usr/bin/env bash
# bench-llama.sh — Benchmark llama.cpp for voicebot workloads
#
# Tests three realistic scenarios the voicebot encounters:
#
#   cold  (300pp, 100tg) — First turn: system prompt + short history + user turn.
#                          Typical when llama-server starts cold or slot is evicted.
#
#   warm  ( 40pp, 100tg) — Subsequent turns: only the new user utterance needs
#                          prefill. KV-cache holds the rest (cache_prompt=true).
#                          This is the most common case and the biggest llama.cpp
#                          latency advantage over stateless backends.
#
#   long  (800pp, 120tg) — Long conversation: many turns accumulated in context.
#                          Tests how well KV-cache quantization holds up.
#
# All flags mirror scripts/start-llm.sh so results reflect production behaviour.
#
# Usage:
#   ./scripts/bench-llama.sh <model.gguf>
#   BENCH_REPS=5 ./scripts/bench-llama.sh ./models/Qwen2.5-7B-Instruct-Q4_K_M.gguf

set -euo pipefail

MODEL="${1:-}"
if [[ -z "$MODEL" ]]; then
    echo "Usage: $0 <model.gguf>"
    echo ""
    echo "Examples:"
    echo "  $0 ./models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    echo "  $0 ./models/Qwen2.5-14B-Instruct-Q4_K_M.gguf"
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "Error: model file not found: $MODEL"
    exit 1
fi

if ! command -v llama-bench &>/dev/null; then
    echo "Error: llama-bench not found."
    echo "Install with: brew install llama.cpp"
    exit 1
fi

# ── Config (match start-llm.sh) ───────────────────────────────────────────────

THREADS="${LLM_THREADS:-$(( $(sysctl -n hw.logicalcpu 2>/dev/null || echo 8) - 2 ))}"
REPS="${BENCH_REPS:-3}"

# ── Header ────────────────────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          llama.cpp — Voicebot Benchmark                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Model:       $(basename "$MODEL")"
echo "  Threads:     $THREADS (hw.logicalcpu - 2)"
echo "  GPU layers:  99 (Metal, all layers)"
echo "  KV cache:    q4_0/q4_0 (matches start-llm.sh)"
echo "  Flash attn:  on"
echo "  Repetitions: $REPS"
echo ""
echo "  Scenarios:"
echo "    cold  300pp 100tg — Full prompt, slot cold (first turn or evicted)"
echo "    warm   40pp 100tg — New user turn only, KV cache hot (cache_prompt=true)"
echo "    long  800pp 120tg — Long conversation, many accumulated turns"
echo ""
echo "  Note: 'warm' simulates the key llama.cpp advantage. At the server level,"
echo "  cache_prompt=true means only ~40 new tokens need prefill per turn."
echo "  llama-bench measures each run independently; the warm number shows"
echo "  raw 40-token prefill speed, which approximates the cached-turn TTFT."
echo ""

# ── Run ───────────────────────────────────────────────────────────────────────

llama-bench \
    -m        "$MODEL" \
    -ngl      99 \
    -ctk      q4_0 \
    -ctv      q4_0 \
    -fa       1 \
    -t        "$THREADS" \
    -p        0 \
    -n        0 \
    -pg       300,100 \
    -pg       40,100 \
    -pg       800,120 \
    -r        "$REPS" \
    --progress \
    -o        md

echo ""
echo "  Metrics:"
echo "    pp   = prompt processing (prefill) — tokens/sec. Higher = lower TTFT."
echo "    tg   = token generation (decode)   — tokens/sec. Higher = faster speech."
echo ""
echo "  Target for voicebot (subjective real-time feel):"
echo "    warm pp > 500 t/s   → TTFT under 80ms for a 40-token user turn"
echo "    tg      > 60 t/s    → 100 tokens generated in < 1.7s (TTS can keep up)"
echo ""
