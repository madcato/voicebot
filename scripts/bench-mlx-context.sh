#!/usr/bin/env bash
# bench-mlx-context.sh — Context-window vs KV-cache prewarm benchmark
#
# Measures how context-window size and KV-cache prewarm depth affect TTFT and
# TG throughput. Starts a live mlx-lm server for each scenario pair, warms the
# KV cache with a synthetic conversation, then fires a short query and records
# wall-clock TTFT and generation t/s.
#
# mlx-lm controls the KV-cache budget via --prompt-cache-bytes (bytes).
# "Short context" = small byte budget (~32K tokens worth).
# "Max context"   = large byte budget (~128K tokens worth).
# Exact bytes-per-token depend on the model architecture; adjust SHORT_CTX_BYTES
# and MAX_CTX_BYTES if needed.
#
# Four scenarios:
#   S1: short-ctx KV budget  + short prewarm (~1K  tokens) → 64-token query
#   S2: short-ctx KV budget  + long  prewarm (~30K tokens) → 64-token query
#   S3: max-ctx   KV budget  + short prewarm (~1K  tokens) → 64-token query
#   S4: max-ctx   KV budget  + ~90% prewarm                → 64-token query
#
# Usage:
#   ./scripts/bench-mlx-context.sh <model>
#   BENCH_PORT=8099 MAX_CTX=131072 \
#     ./scripts/bench-mlx-context.sh mlx-community/Qwen3.5-4B-MLX-4bit
#
# Env vars:
#   BENCH_PORT        server port                           (default 8080)
#   BENCH_TRIALS      hot trials per scenario               (default 3)
#   SHORT_CTX_BYTES   KV-cache bytes for short-ctx server   (default 2 GB)
#   MAX_CTX_BYTES     KV-cache bytes for max-ctx server     (default 8 GB)
#   MAX_CTX           max context tokens (for 90% prewarm)  (default 131072)
#   PREWARM_SHORT     short prewarm tokens                  (default 1000)
#   PREWARM_LONG      long prewarm tokens (S2)              (default 30000)
#   GEN_TOKENS        tokens to generate per query          (default 80)
#   MLX_PREFILL_STEP  prefill chunk size                    (default 512)

set -euo pipefail

# ── Args ──────────────────────────────────────────────────────────────────────

MODEL="${1:-}"
if [[ -z "$MODEL" ]]; then
    echo "Usage: $0 <model-path-or-hf-repo>"
    echo ""
    echo "Examples:"
    echo "  $0 mlx-community/Qwen3.5-4B-MLX-4bit"
    echo "  $0 mlx-community/gemma-4-26b-a4b-it-4bit"
    exit 1
fi

# ── Config ────────────────────────────────────────────────────────────────────

PORT="${BENCH_PORT:-8080}"
TRIALS="${BENCH_TRIALS:-3}"
# KV-cache byte budgets.  2 GB ≈ 32K tokens; 8 GB ≈ 128K tokens for typical
# 7-14B models with GQA.  Adjust for your model's head/layer counts.
SHORT_CTX_BYTES="${SHORT_CTX_BYTES:-$(( 2 * 1024 * 1024 * 1024 ))}"   # 2 GB
MAX_CTX_BYTES="${MAX_CTX_BYTES:-$(( 8 * 1024 * 1024 * 1024 ))}"       # 8 GB
MAX_CTX="${MAX_CTX:-131072}"    # token count used only to compute 90% prewarm
PREWARM_SHORT="${PREWARM_SHORT:-1000}"
PREWARM_LONG="${PREWARM_LONG:-30000}"
GEN_TOKENS="${GEN_TOKENS:-80}"
PREFILL_STEP="${MLX_PREFILL_STEP:-512}"

HOST="127.0.0.1"

# 90% of max context for S4 (reserve room for system prompt + query)
PREWARM_MAX=$(( MAX_CTX * 90 / 100 ))

# ── Detect mlx-lm runner ──────────────────────────────────────────────────────

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

# ── Server lifecycle ──────────────────────────────────────────────────────────

SERVER_PID=""

start_server() {
    local cache_bytes="$1"
    local label="$2"
    local cache_gb=$(( cache_bytes / 1024 / 1024 / 1024 ))
    echo ""
    echo "  Starting mlx-lm (KV cache ${cache_gb} GB, $label) on port ${PORT} ..."

    $RUNNER server \
        --model                "$MODEL" \
        --host                 "$HOST" \
        --port                 "$PORT" \
        --prompt-cache-size    1 \
        --prompt-cache-bytes   "$cache_bytes" \
        --prefill-step-size    "$PREFILL_STEP" \
        --max-tokens           "$GEN_TOKENS" \
        --temp                 0.0 \
        --chat-template-args   '{"enable_thinking": false}' \
        --log-level            WARNING \
        >/tmp/bench-mlx-ctx-server.log 2>&1 &
    SERVER_PID=$!

    # Wait until /v1/models responds (up to 180s)
    local deadline=$(( $(date +%s) + 180 ))
    while (( $(date +%s) < deadline )); do
        if curl -sf "http://${HOST}:${PORT}/v1/models" -o /dev/null 2>/dev/null; then
            echo "  Server ready (pid $SERVER_PID)"
            return
        fi
        sleep 1
    done
    echo "  ERROR: server did not become ready within 120s"
    stop_server
    exit 1
}

stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
        echo "  Server stopped."
    fi
}

# ── Python measurement kernel ─────────────────────────────────────────────────
# Embedded inline so the script stays self-contained.

read -r -d '' PY_MEASURE <<'PYEOF' || true
"""
Measure TTFT (ms) and TG t/s for a single streaming /v1/chat/completions call.

argv: <host> <port> <max_tokens> <prewarm_chars> <query> <trials> <model>

Output (one JSON line per trial):
  {"trial": 1, "ttft_ms": 123.4, "tg_tps": 56.7, "tokens": 80}
"""
import sys, json, http.client, time, math

host          = sys.argv[1]
port          = int(sys.argv[2])
max_tokens    = int(sys.argv[3])
prewarm_chars = int(sys.argv[4])
query         = sys.argv[5]
trials        = int(sys.argv[6])
MODEL_ID      = sys.argv[7]

# ── Prewarm text (Spanish, ~4 chars/token) ────────────────────────────────────
# Repeated naturalistic sentences so the tokeniser produces a realistic
# distribution. Repetition is intentional — we only care about token count.
_UNIT = (
    "El sistema de inteligencia artificial procesa la información del usuario "
    "y genera una respuesta coherente en tiempo real. "
    "La latencia depende del tamaño del contexto y de la velocidad de prefill. "
    "Jarvis mantiene el historial de conversación para ofrecer respuestas relevantes. "
    "El caché KV almacena los estados de atención de tokens anteriores. "
)
_UNIT_LEN = len(_UNIT)

def make_prewarm_content(n_chars: int) -> str:
    if n_chars <= 0:
        return ""
    reps = math.ceil(n_chars / _UNIT_LEN)
    return (_UNIT * reps)[:n_chars]

prewarm_text = make_prewarm_content(prewarm_chars)

# ── Message builders ──────────────────────────────────────────────────────────

SYSTEM = (
    "Eres Jarvis, el asistente personal de IA. Llevas años trabajando con él "
    "y le conoces bien. Respuestas concisas, siempre en español."
)

def build_cold_messages():
    """System + prewarm block (as a single user message) — no assistant reply."""
    msgs = [{"role": "system", "content": SYSTEM}]
    if prewarm_text:
        msgs.append({"role": "user",      "content": prewarm_text})
        msgs.append({"role": "assistant", "content": "Entendido, señor. Estoy listo."})
    msgs.append({"role": "user", "content": query})
    return msgs

def build_hot_messages():
    """Same as cold — the server's prompt cache makes the difference."""
    return build_cold_messages()

# ── HTTP streaming helper ─────────────────────────────────────────────────────

def stream_request(messages):
    payload = json.dumps({
        "model":                MODEL_ID,
        "messages":             messages,
        "max_tokens":           max_tokens,
        "temperature":          0.0,
        "stream":               True,
        "enable_thinking":      False,
        "chat_template_kwargs": {"enable_thinking": False},
        "thinking":             {"type": "disabled"},
    }).encode()

    conn = http.client.HTTPConnection(host, port, timeout=300)
    conn.request("POST", "/v1/chat/completions", body=payload,
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:200].decode()}")

    t_start  = time.perf_counter()
    t_first  = None
    t_last   = None
    t_done   = None
    n_tokens = 0
    buf      = ""

    try:
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            buf += chunk.decode("utf-8", errors="replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.rstrip("\r")
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    t_done = time.perf_counter()
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = (obj.get("choices") or [{}])[0].get("delta", {})
                content = (
                    delta.get("content") or delta.get("reasoning_content") or
                    delta.get("reasoning") or delta.get("thinking") or ""
                )
                if content:
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    t_last   = now
                    n_tokens += len(content.split())
    finally:
        conn.close()

    if t_first is None:
        raise RuntimeError("no content tokens received")

    ttft_ms = (t_first - t_start) * 1000
    tg_end  = t_last if (t_last and t_last > t_first + 0.001) else t_done
    tg_secs = (tg_end - t_first) if tg_end and tg_end > t_first + 0.001 else None
    tg_tps  = n_tokens / tg_secs if tg_secs else float("nan")

    return ttft_ms, tg_tps, n_tokens

# ── Run ───────────────────────────────────────────────────────────────────────

# Prewarm: single cold request to populate KV cache (not timed)
sys.stderr.write(f"  Prewarm ({prewarm_chars:,} chars ≈ {prewarm_chars//4:,} tokens) ...\n")
sys.stderr.flush()
try:
    stream_request(build_cold_messages())
    sys.stderr.write("  Prewarm done.\n")
except Exception as e:
    sys.stderr.write(f"  Prewarm FAILED: {e}\n")
sys.stderr.flush()

# Hot trials
for i in range(1, trials + 1):
    sys.stderr.write(f"  Trial {i}/{trials} ... ")
    sys.stderr.flush()
    try:
        ttft, tg, n = stream_request(build_hot_messages())
        sys.stderr.write(f"TTFT {ttft:.0f} ms   TG {tg:.1f} t/s   ({n} tokens)\n")
        print(json.dumps({"trial": i, "ttft_ms": ttft, "tg_tps": tg, "tokens": n}))
    except Exception as e:
        sys.stderr.write(f"FAILED: {e}\n")
    sys.stderr.flush()
PYEOF

# Write the measurement kernel to a temp file once (avoids heredoc quoting issues)
PY_SCRIPT=$(mktemp /tmp/bench-mlx-ctx-XXXXXX.py)
printf '%s' "$PY_MEASURE" > "$PY_SCRIPT"
trap "stop_server; rm -f '$PY_SCRIPT' /tmp/bench-ctx-results-*.json /tmp/bench-mlx-ctx-server.log" EXIT INT TERM

# ── Scenario runner ───────────────────────────────────────────────────────────

declare -a SCENARIO_LABELS SCENARIO_PREWARM_CHARS SCENARIO_TTFTS SCENARIO_TGS

SCENARIO_LABELS=()
SCENARIO_PREWARM_CHARS=()
SCENARIO_TTFTS=()
SCENARIO_TGS=()

QUERY="¿Puedes explicar brevemente cómo funciona el caché KV en los transformers y por qué mejora la latencia?"

run_scenario() {
    local idx="$1"
    local label="$2"
    local prewarm_chars="$3"

    echo ""
    echo "  ── Scenario $idx: $label"
    echo "     prewarm ${prewarm_chars} chars (~$(( prewarm_chars / 4 )) tokens)"

    python3 "$PY_SCRIPT" \
        "$HOST" "$PORT" "$GEN_TOKENS" "$prewarm_chars" "$QUERY" "$TRIALS" "$MODEL" \
        >/tmp/bench-ctx-results-$idx.json

    # Parse results with Python (jq-free)
    local stats
    stats=$(python3 - /tmp/bench-ctx-results-$idx.json <<'STATS'
import json, sys, math
from statistics import mean, stdev
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
if not rows:
    print("0\t0\t0\t0")
    sys.exit()
ttfts = [r["ttft_ms"] for r in rows]
tgs   = [r["tg_tps"]  for r in rows if not math.isnan(r["tg_tps"])]
avg_ttft = mean(ttfts)
sd_ttft  = stdev(ttfts) if len(ttfts) > 1 else 0.0
avg_tg   = mean(tgs)   if tgs else 0.0
sd_tg    = stdev(tgs)  if len(tgs) > 1 else 0.0
print(f"{avg_ttft:.1f}\t{sd_ttft:.1f}\t{avg_tg:.1f}\t{sd_tg:.1f}")
STATS
)

    SCENARIO_LABELS+=("$label")
    SCENARIO_PREWARM_CHARS+=("$prewarm_chars")

    local avg_ttft="N/A" sd_ttft="0" avg_tg="N/A" sd_tg="0"
    if [[ -n "$stats" ]]; then
        IFS=$'\t' read -r avg_ttft sd_ttft avg_tg sd_tg <<< "$stats"
    fi
    SCENARIO_TTFTS+=("${avg_ttft}±${sd_ttft}")
    SCENARIO_TGS+=("${avg_tg}±${sd_tg}")

    echo "     → TTFT ${avg_ttft} ± ${sd_ttft} ms    TG ${avg_tg} ± ${sd_tg} t/s"
}

# ── Header ────────────────────────────────────────────────────────────────────

W=72
echo ""
printf '═%.0s' $(seq 1 $W); echo ""
echo "  mlx-lm — Context Window vs KV-Cache Prewarm Benchmark"
printf '═%.0s' $(seq 1 $W); echo ""
echo ""
SHORT_GB=$(( SHORT_CTX_BYTES / 1024 / 1024 / 1024 ))
MAX_GB=$(( MAX_CTX_BYTES / 1024 / 1024 / 1024 ))

echo "  Model      : $MODEL"
echo "  Runner     : $RUNNER"
echo "  Port       : $PORT"
echo "  Trials     : $TRIALS"
echo "  Gen tokens : $GEN_TOKENS"
echo "  Short-ctx  : ${SHORT_GB} GB KV cache  (~32K tokens for typical 7-14B model)"
echo "  Max-ctx    : ${MAX_GB} GB KV cache  (~128K tokens for typical 7-14B model)"
echo "  Query      : \"${QUERY:0:60}...\""
echo ""
echo "  Scenarios:"
echo "    S1: short-ctx (${SHORT_GB} GB)  + short prewarm (~${PREWARM_SHORT} tokens)"
echo "    S2: short-ctx (${SHORT_GB} GB)  + long  prewarm (~${PREWARM_LONG} tokens)"
echo "    S3: max-ctx   (${MAX_GB} GB)  + short prewarm (~${PREWARM_SHORT} tokens)"
echo "    S4: max-ctx   (${MAX_GB} GB)  + 90%  prewarm (~${PREWARM_MAX} tokens)"
echo ""

# ── Run S1 + S2 (short context server) ───────────────────────────────────────

printf '─%.0s' $(seq 1 $W); echo ""
echo "  [Server 1/2]  KV cache = ${SHORT_GB} GB  (short context)"
printf '─%.0s' $(seq 1 $W); echo ""

start_server "$SHORT_CTX_BYTES" "short"

run_scenario 1 \
    "short-ctx + short prewarm" \
    $(( PREWARM_SHORT * 4 ))   # chars ≈ tokens × 4

run_scenario 2 \
    "short-ctx + long  prewarm" \
    $(( PREWARM_LONG * 4 ))

stop_server

# ── Run S3 + S4 (max context server) ─────────────────────────────────────────

printf '─%.0s' $(seq 1 $W); echo ""
echo "  [Server 2/2]  KV cache = ${MAX_GB} GB  (max context)"
printf '─%.0s' $(seq 1 $W); echo ""

start_server "$MAX_CTX_BYTES" "max"

run_scenario 3 \
    "max-ctx   + short prewarm" \
    $(( PREWARM_SHORT * 4 ))

run_scenario 4 \
    "max-ctx   + 90%   prewarm" \
    $(( PREWARM_MAX * 4 ))

stop_server

# ── Results table ─────────────────────────────────────────────────────────────

echo ""
printf '═%.0s' $(seq 1 $W); echo ""
echo "  RESULTS"
printf '═%.0s' $(seq 1 $W); echo ""
echo ""

printf "  %-3s  %-30s  %18s  %16s\n" \
    "#" "Scenario" "TTFT warm (ms)" "TG (t/s)"
echo "  $(printf '─%.0s' $(seq 1 70))"

for i in 0 1 2 3; do
    printf "  S%-2d  %-30s  %18s  %16s\n" \
        $(( i + 1 )) \
        "${SCENARIO_LABELS[$i]}" \
        "${SCENARIO_TTFTS[$i]}" \
        "${SCENARIO_TGS[$i]}"
done

echo ""
echo "  Interpretation:"
echo "    TTFT (S2 vs S1): KV-cache fill cost — does a longer prewarm hurt TTFT?"
echo "    TTFT (S3 vs S1): Context-window overhead — does a larger ctx window hurt?"
echo "    TTFT (S4 vs S3): Combined effect of max ctx + near-full KV cache."
echo "    TG   (S* vs S*): TG t/s should be ~stable; drops hint memory pressure."
echo ""
printf '═%.0s' $(seq 1 $W); echo ""
echo ""
