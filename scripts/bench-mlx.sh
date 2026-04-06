#!/usr/bin/env bash
# bench-mlx.sh — Benchmark mlx-lm for voicebot workloads
#
# Tests three realistic scenarios the voicebot encounters:
#
#   cold  (300pp, 100tg) — First turn: system prompt + short history + user turn.
#
#   warm  ( 40pp, 100tg) — Subsequent turns: new user utterance only.
#                          mlx-lm caches prompts server-side (--prompt-cache-size 1).
#                          This benchmark measures raw 40-token prefill speed, which
#                          approximates the cached-turn TTFT.
#
#   long  (800pp, 120tg) — Long conversation: many turns accumulated in context.
#
# Flags mirror scripts/start-mlx-lm.sh so results reflect production behaviour.
#
# Usage:
#   ./scripts/bench-mlx.sh <model-path-or-hf-repo>
#   BENCH_TRIALS=5 ./scripts/bench-mlx.sh mlx-community/Qwen2.5-7B-Instruct-4bit
#   ./scripts/bench-mlx.sh ./models/my-local-mlx-model

set -euo pipefail

MODEL="${1:-}"
if [[ -z "$MODEL" ]]; then
    echo "Usage: $0 <model-path-or-hf-repo>"
    echo ""
    echo "Examples:"
    echo "  $0 mlx-community/Qwen2.5-7B-Instruct-4bit"
    echo "  $0 mlx-community/Qwen2.5-14B-Instruct-4bit"
    echo "  $0 mlx-community/Qwen3-8B-4bit"
    exit 1
fi

# ── Detect runner ─────────────────────────────────────────────────────────────

if command -v mlx_lm &>/dev/null; then
    RUNNER="mlx_lm"
elif python3 -c "import mlx_lm" &>/dev/null 2>&1; then
    RUNNER="python3 -m mlx_lm"
elif command -v uvx &>/dev/null; then
    RUNNER="uvx mlx_lm"
else
    echo "Error: mlx-lm not found. Install with: pip install mlx-lm"
    exit 1
fi

# ── Config (match start-mlx-lm.sh) ────────────────────────────────────────────

TRIALS="${BENCH_TRIALS:-3}"
PREFILL_STEP="${MLX_PREFILL_STEP:-512}"   # matches start-mlx-lm.sh

# ── Header ────────────────────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          mlx-lm — Voicebot Benchmark                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Model:         $MODEL"
echo "  Runner:        $RUNNER"
echo "  Prefill step:  $PREFILL_STEP tokens (matches start-mlx-lm.sh)"
echo "  Trials:        $TRIALS per scenario"
echo ""
echo "  Scenarios:"
echo "    cold  300pp 100tg — Full prompt, server cold"
echo "    warm   40pp 100tg — New user turn only, prompt cache hot"
echo "    long  800pp 120tg — Long conversation context"
echo ""

# ── Helper ────────────────────────────────────────────────────────────────────

run_scenario() {
    local label="$1"
    local prompt_tokens="$2"
    local gen_tokens="$3"

    echo "────────────────────────────────────────────────────────────────"
    echo "  Scenario: $label  (${prompt_tokens}pp, ${gen_tokens}tg)"
    echo "────────────────────────────────────────────────────────────────"

    $RUNNER benchmark \
        --model               "$MODEL" \
        --prompt-tokens       "$prompt_tokens" \
        --generation-tokens   "$gen_tokens" \
        --prefill-step-size   "$PREFILL_STEP" \
        --num-trials          "$TRIALS"

    echo ""
}

# ── Run all scenarios ─────────────────────────────────────────────────────────

run_scenario "cold  (full prompt, first turn)"       300 100
run_scenario "warm  (new user turn, cache hot)"       40 100
run_scenario "long  (long conversation context)"     800 120

# ── Footer ────────────────────────────────────────────────────────────────────

echo "────────────────────────────────────────────────────────────────"
echo ""
echo "  Metrics:"
echo "    Prompt:     prefill throughput (tokens/sec). Higher = lower TTFT."
echo "    Generation: decode throughput  (tokens/sec). Higher = faster speech."
echo ""
echo "  Target for voicebot (subjective real-time feel):"
echo "    warm prompt > 500 t/s   → TTFT under 80ms for a 40-token user turn"
echo "    generation  >  60 t/s   → 100 tokens in < 1.7s (TTS can keep up)"
echo ""
echo "  Compare with: ./scripts/bench-omlx.sh <mlx-model> <omlx-model-dir>"
echo ""
