#!/usr/bin/env python3
"""
bench-omlx-models.py — multi-model KV-cache benchmark (omlx / LM Studio / any OpenAI-compatible server)

Tests TTFT, PP (prompt-processing rate), TG (token-generation rate), and KV-cache
health for a list of models.

For each model:
  1. Load request  — simple "Hola" to pull the model into GPU/RAM (not timed)
  2. PP phase      — cold full-conversation prefill (same params as hot trials)
                     measures cold TTFT and estimates PP rate (t/s)
  3. N hot trials  — identical repeated request, measures warm TTFT and TG rate

KV-cache verdict: warm_TTFT << cold_TTFT  ⟹  cache is working.

Usage:
  python3 scripts/bench-omlx-models.py [model-dir]

  # omlx (default)
  BENCH_PORT=8001 python3 scripts/bench-omlx-models.py ~/models

  # LM Studio
  BENCH_PORT=1234 BENCH_TOKEN="" BENCH_PROVIDER=lmstudio python3 scripts/bench-omlx-models.py

Env vars:
  BENCH_PORT       server port               (default 8000 for omlx, use 1234 for LM Studio)
  BENCH_TOKEN      Bearer auth token         (default "asdf" for omlx, empty for LM Studio)
  BENCH_PROVIDER   label shown in output     (default "omlx")
  BENCH_TRIALS     hot measurement trials    (default 3)
  BENCH_GEN        tokens to generate        (default 80)
  OMLX_DIR         model directory for auto-start (default ~/.lmstudio/models)

  # Legacy aliases (still accepted)
  OMLX_PORT  →  BENCH_PORT
  OMLX_TOKEN →  BENCH_TOKEN
"""

import http.client
import json
import math
import os
import shutil
import subprocess
import sys
import time
from statistics import mean, stdev

# ── Model list ────────────────────────────────────────────────────────────────

## OMLX
# MODELS = [
#     "Bonsai-8B-mlx-1bit",
#     "LFM2-24B-A2B-MLX-4bit",
#     "LFM2.5-350M-MLX-8bit",
#     "Qwen3-30B-A3B-Instruct-2507-MLX-4bit",
#     "Qwen3.5-27B-oQ3",
#     "Qwen3.5-2B-4bit",
#     "Qwen3.5-35B-A3B-4bit",
#     "Qwen3.5-35B-A3B-MLX-oQ4",
# ]

## LM Studio
# MODELS = [
#     # "GLM-4.7-Flash-MLX-4bit",
#     # "nemotron-3-nano",
#     "qwen/qwen3-30b-a3b-2507",
#     "qwen3.5-35b-a3b@4bit",
#     # "qwen3.5-35b-a3b@8bit",
#     # "trinity-mini",
#     # "trinity-nano-preview",
#     # "qwen3.5-4b-mlx",
#     "liquid/lfm2-24b-a2b"
# ]

## Ollama local
# MODELS = [
#     "gemma4:e2b-it-q8_0",
#     "gemma4:31b-it-q4_K_M",
#     "gemma4:26b-a4b-it-q4_K_M",
#     "lfm2:24b",
#     "qwen3:30b-a3b-instruct-2507-q4_K_M",
#     "qwen3.5:4b-nvfp4",
#     "qwen3.5:35b-a3b-coding-nvfp4",
#     "qwen3.5:35b-a3b-int4",
# ]

## mlx-lm local
MODELS = [
    "Brooooooklyn/Qwen3.5-35B-A3B-UD-Q4_K_XL-mlx",
    "Brooooooklyn/Qwen3.5-27B-unsloth-mlx",
    "mlx-community/Qwen3.5-35B-A3B-4bit",
    "mlx-community/Qwen3.5-9B-MLX-4bit",
    "mlx-community/Qwen3.5-4B-MLX-4bit",
    "mlx-community/gemma-4-26b-a4b-it-4bit",
    "mlx-community/gemma-4-31b-it-4bit",
    "mlx-community/gemma-4-e4b-it-4bit",
]

# ── Conversation fixture ──────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Eres Jarvis, el asistente personal de IA. Llevas años trabajando con él y le conoces bien.\n\n"
    "CARÁCTER\n"
    "Mezcla de Jarvis (Iron Man) y Alfred (Batman): profesional, ligeramente irónico, humor seco "
    "y británico. Leal, discreto, eficiente. Nunca servil. Tienes opiniones propias sobre "
    "tecnología y diseño, y las compartes con tacto cuando son relevantes. Ocasionalmente haces "
    "un comentario sarcástico, pero nunca a costa del usuario.\n\n"
    "FORMA DE HABLAR\n"
    "- Siempre en español salvo que el usuario cambie de idioma.\n"
    "- Llamas al usuario por \"señor\", nunca \"usuario\".\n"
    "- Respuestas concisas: 2-3 frases máximo salvo que pida más detalle.\n"
    "- Hablas para ser escuchado: sin markdown, sin listas, sin símbolos, sin nada que un "
    "sintetizador no pronuncie bien.\n"
    "- Cuando no sabes algo, lo dices. No inventas.\n"
    "- Antes de una acción irreversible, la describes y pides confirmación.\n\n"
    "HERRAMIENTAS DISPONIBLES\n"
    "- current_time: hora y fecha actuales.\n"
    "- get_calendar_events: eventos del calendario para una fecha.\n"
    "- create_calendar_event: crear evento o recordatorio en Calendar.app.\n"
    "- read_clipboard / set_clipboard: leer o escribir el portapapeles.\n"
    "- read_file: leer el contenido de un fichero (max 16 KB).\n"
    "- open_app: abrir una aplicacion macOS por nombre.\n"
    "- send_notification: enviar una notificacion macOS.\n"
    "- run_shell: ejecutar un comando de terminal (disponible si SHELL_ENABLED=1).\n"
    "- take_screenshot: capturar la pantalla y describir lo que hay en ella \n"
    "- run_agent_async: delegar una tarea compleja al agente externo "
    "(disponible si AGENT_COMMAND esta configurado). El agente trabaja en segundo plano "
    "y el resultado llega en breve.\n\n"
    "Usa las herramientas directamente cuando puedas. Para tareas complejas de multiples "
    "pasos usa run_agent_async. No afirmes tener capacidades que no tienes."
)

HISTORY = [
    ("user",      "¿Qué tiempo hace hoy en Madrid?"),
    ("assistant", "Hoy en Madrid hay cielos despejados y unos dieciocho grados. Buen día para salir."),
    ("user",      "¿Cuándo es el próximo partido del Real Madrid?"),
    ("assistant", "El Real Madrid juega este sábado a las nueve de la noche contra el Atlético en el Bernabéu."),
    ("user",      "Recuérdame comprar leche mañana por la mañana."),
    ("assistant", "Anotado. Te recuerdo mañana a primera hora que compres leche."),
    ("user",      "¿Cuánto es el veinte por ciento de trescientos cincuenta euros?"),
    ("assistant", "El veinte por ciento de trescientos cincuenta euros son setenta euros."),
    ("user",      "¿Qué películas de ciencia ficción recomiendas para esta noche?"),
    ("assistant", "Te recomiendo Interstellar o Blade Runner 2049. Las dos son magníficas."),
    ("user",      "¿Cuál es la capital de Australia?"),
    ("assistant", "La capital de Australia es Canberra, aunque muchos creen que es Sídney."),
    ("user",      "¿Sabes si mañana hay huelga de metro en Madrid?"),
    ("assistant", "No tengo información en tiempo real sobre huelgas. Consulta el sitio web del metro de Madrid."),
]

NEW_QUESTION = (
    "Interesante. ¿Puedes contarme brevemente por qué Sídney es más conocida que "
    "Canberra, cómo surgió esa confusión tan común, y qué otras ciudades importantes "
    "tiene Australia?"
)

# ── Config ────────────────────────────────────────────────────────────────────

# BENCH_PORT / BENCH_TOKEN take precedence; OMLX_PORT / OMLX_TOKEN are legacy aliases.
SERVER_PORT  = int(os.environ.get("BENCH_PORT",  os.environ.get("OMLX_PORT",  "1234")))
SERVER_TOKEN =     os.environ.get("BENCH_TOKEN", os.environ.get("OMLX_TOKEN", "asdf"))
PROVIDER     =     os.environ.get("BENCH_PROVIDER", "omlx")
TRIALS       = int(os.environ.get("BENCH_TRIALS", "3"))
GEN_TOKENS   = int(os.environ.get("BENCH_GEN",    "80"))

HOST = "127.0.0.1"

# Rough prompt token estimate (chars / 3.5) used when API doesn't return usage
_PROMPT_TEXT = (
    SYSTEM_PROMPT
    + "".join(c + t for r, c, t in [(r, c, t) for r, t in HISTORY for c in [r]])
    + NEW_QUESTION
)
ESTIMATED_PROMPT_TOKENS = max(1, int(len(_PROMPT_TEXT) / 3.5))

# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _auth_headers():
    h = {"Content-Type": "application/json"}
    if SERVER_TOKEN:
        h["Authorization"] = f"Bearer {SERVER_TOKEN}"
    return h


def _post_stream(port, payload):
    """POST to /v1/chat/completions with stream=True. Yields SSE content lines."""
    body = json.dumps(payload).encode()
    conn = http.client.HTTPConnection(HOST, port, timeout=120)
    try:
        conn.request(
            "POST", "/v1/chat/completions", body=body,
            headers=_auth_headers(),
        )
        resp = conn.getresponse()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:300].decode()}")
        buf = ""
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            buf += chunk.decode("utf-8", errors="replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                yield line.rstrip("\r")
    finally:
        conn.close()


def _post_blocking(port, payload):
    """POST to /v1/chat/completions with stream=False. Returns parsed JSON dict."""
    payload = {**payload, "stream": False}
    body = json.dumps(payload).encode()
    conn = http.client.HTTPConnection(HOST, port, timeout=120)
    try:
        conn.request(
            "POST", "/v1/chat/completions", body=body,
            headers=_auth_headers(),
        )
        resp = conn.getresponse()
        raw = resp.read()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {raw[:300].decode()}")
        return json.loads(raw)
    finally:
        conn.close()


def _get_models(port):
    """Return list of model IDs from /v1/models."""
    conn = http.client.HTTPConnection(HOST, port, timeout=10)
    try:
        conn.request("GET", "/v1/models", headers=_auth_headers())
        resp = conn.getresponse()
        data = json.loads(resp.read())
        return [m["id"] for m in data.get("data", [])]
    except Exception:
        return []
    finally:
        conn.close()


def _wait_ready(port, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            conn = http.client.HTTPConnection(HOST, port, timeout=2)
            conn.request("GET", "/v1/models", headers=_auth_headers())
            r = conn.getresponse()
            r.read()
            conn.close()
            if r.status < 500:
                return
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError(f"omlx not ready on port {port} after {timeout}s")


# ── Conversation builder ──────────────────────────────────────────────────────

def _build_messages(include_new_question=True):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for role, content in HISTORY:
        msgs.append({"role": role, "content": content})
    if include_new_question:
        msgs.append({"role": "user", "content": NEW_QUESTION})
    return msgs


def _base_payload(model_id, max_tokens, stream):
    return {
        "model":                model_id,
        "messages":             _build_messages(),
        "max_tokens":           max_tokens,
        "temperature":          0.0,
        "stream":               stream,
        # Disable reasoning/thinking via all known API conventions:
        # - top-level field (omlx, some mlx-lm builds)
        # - chat_template_kwargs (mlx-lm, omlx)
        # - thinking budget = 0 (LM Studio / some OpenAI-compat servers)
        # - think: false (Ollama)
        "enable_thinking":      False,
        "chat_template_kwargs": {"enable_thinking": False},
        "thinking":             {"type": "disabled"},
        "think":                False,
    }


# ── Per-model benchmark steps ────────────────────────────────────────────────

def load_model(port, model_id):
    """
    Send a trivial single-message request to pull the model weights into GPU/RAM.
    Not timed — this is pure model-load overhead.
    """
    payload = {
        "model":      model_id,
        "messages":   [{"role": "user", "content": "Hola"}],
        "max_tokens": 1,
        "stream":     False,
    }
    _post_blocking(port, payload)


def measure_pp(port, model_id):
    """
    Cold full-conversation prefill — identical parameters to the hot trials.
    Using the same max_tokens as the hot trials ensures the server allocates
    the same number of output KV slots, making the cold/warm TTFT comparison
    valid.  The TTFT here is dominated by prompt-processing (840 tokens >> 1
    output token), so it is still a good proxy for PP rate.

    Returns (cold_ttft_ms, pp_tps, prompt_tokens):
      cold_ttft_ms  — time from request start to first content token
      pp_tps        — estimated prompt-processing rate (tokens/s)
      prompt_tokens — from usage field if available, else estimated
    """
    payload = _base_payload(model_id, max_tokens=GEN_TOKENS, stream=True)

    t_start = time.perf_counter()
    t_first = None
    prompt_tokens_api = None

    for line in _post_stream(port, payload):
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue

        # Capture usage if server sends it
        if "usage" in chunk and chunk["usage"]:
            prompt_tokens_api = chunk["usage"].get("prompt_tokens")

        delta = (chunk.get("choices") or [{}])[0].get("delta", {})
        token = (delta.get("content") or delta.get("reasoning_content")
                 or delta.get("reasoning") or delta.get("thinking") or "")
        if token and t_first is None:
            t_first = time.perf_counter()

    if t_first is None:
        raise RuntimeError("PP trial: no content token received")

    cold_ttft_ms = (t_first - t_start) * 1000
    prompt_tokens = prompt_tokens_api or ESTIMATED_PROMPT_TOKENS
    pp_tps = prompt_tokens / ((t_first - t_start)) if (t_first - t_start) > 0 else float("nan")

    return cold_ttft_ms, pp_tps, prompt_tokens


def measure_hot(port, model_id):
    """
    Hot trial: KV cache should already hold the full conversation.
    Only the new question tokens need prefill.

    Returns (ttft_ms, tg_tps, n_tokens).
    """
    payload = _base_payload(model_id, max_tokens=GEN_TOKENS, stream=True)

    t_start  = time.perf_counter()
    t_first  = None
    t_last   = None
    t_done   = None
    n_tokens = 0

    for line in _post_stream(port, payload):
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            t_done = time.perf_counter()
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        delta   = (chunk.get("choices") or [{}])[0].get("delta", {})
        content = (delta.get("content") or delta.get("reasoning_content")
                   or delta.get("reasoning") or delta.get("thinking") or "")
        if content:
            now = time.perf_counter()
            if t_first is None:
                t_first = now
            t_last   = now
            n_tokens += len(content.split())

    if t_first is None or n_tokens == 0:
        raise RuntimeError("Hot trial: no content tokens received")

    ttft_ms  = (t_first - t_start) * 1000
    # Fallback to t_done when all tokens arrived in a single SSE batch (t_last == t_first)
    tg_end   = t_last if (t_last and t_last > t_first + 0.001) else t_done
    tg_secs  = (tg_end - t_first) if tg_end and tg_end > t_first + 0.001 else None
    tg_tps   = n_tokens / tg_secs if tg_secs else float("nan")

    return ttft_ms, tg_tps, n_tokens


# ── Model matching ────────────────────────────────────────────────────────────

def match_model_id(available, target):
    """
    Find the best matching model ID in `available` for the target name.
    Tries exact match first, then case-insensitive substring.
    Returns the matched ID or None.
    """
    if target in available:
        return target
    target_lower = target.lower()
    for mid in available:
        if target_lower in mid.lower() or mid.lower() in target_lower:
            return mid
    return None


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_model(port, model_id, label, W):
    """Run the full benchmark for one model. Returns a result dict or None on failure."""
    print(f"\n  Loading model into memory ...", end=" ", flush=True)
    try:
        load_model(port, model_id)
        print("done")
    except Exception as e:
        print(f"FAILED: {e}")
        return None

    print(f"  Measuring cold PP (full prompt prefill) ...", end=" ", flush=True)
    try:
        cold_ttft, pp_tps, prompt_tokens = measure_pp(port, model_id)
        print(f"cold TTFT {cold_ttft:.0f} ms   PP ~{pp_tps:.0f} t/s   (~{prompt_tokens} prompt tokens)")
    except Exception as e:
        print(f"FAILED: {e}")
        return None

    hot_results = []
    for i in range(TRIALS):
        print(f"  Hot trial {i + 1}/{TRIALS} ... ", end="", flush=True)
        try:
            ttft, tg, n = measure_hot(port, model_id)
            print(f"TTFT {ttft:>6.0f} ms   TG {tg:>5.1f} t/s   ({n} tokens)")
            hot_results.append((ttft, tg, n))
        except Exception as e:
            print(f"FAILED: {e}")

    if not hot_results:
        return None

    ttfts      = [r[0] for r in hot_results]
    tgs_raw    = [r[1] for r in hot_results]
    tgs        = [v for v in tgs_raw if not math.isnan(v) and not math.isinf(v)]

    avg_ttft = mean(ttfts)
    avg_tg   = mean(tgs) if tgs else float("nan")
    sd_ttft  = stdev(ttfts) if len(ttfts) > 1 else 0.0
    sd_tg    = stdev(tgs)   if len(tgs)   > 1 else 0.0

    # KV-cache verdict: warm TTFT should be much lower than cold TTFT
    speedup = cold_ttft / avg_ttft if avg_ttft > 0 else 0.0
    cache_ok = speedup >= 3.0  # omlx block cache: ≥1 full block cached → >>3× TTFT speedup

    return {
        "label":      label,
        "model_id":   model_id,
        "cold_ttft":  cold_ttft,
        "pp_tps":     pp_tps,
        "prompt_tok": prompt_tokens,
        "ttft":       avg_ttft,
        "ttft_sd":    sd_ttft,
        "tg":         avg_tg,
        "tg_sd":      sd_tg,
        "tokens":     mean(r[2] for r in hot_results),
        "speedup":    speedup,
        "cache_ok":   cache_ok,
    }


# ── Server management ─────────────────────────────────────────────────────────

def start_omlx(model_dir):
    if not shutil.which("omlx"):
        raise RuntimeError("omlx not found — install from https://github.com/jundot/omlx")
    cmd = [
        "omlx", "serve",
        "--model-dir", model_dir,
        "--host",      HOST,
        "--port",      str(SERVER_PORT),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait_ready(SERVER_PORT, timeout=180)
    except TimeoutError:
        proc.terminate()
        raise
    return proc


def stop_server(proc):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    omlx_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "OMLX_DIR", os.path.join(os.path.expanduser("~"), ".lmstudio", "models")
    )

    W = 72
    print()
    print("═" * W)
    print(f"  {PROVIDER.upper()} Multi-Model KV-Cache Benchmark")
    print("═" * W)
    print(f"  Provider    : {PROVIDER}  (port {SERVER_PORT})")
    print(f"  Models      : {len(MODELS)}")
    print(f"  History     : {len(HISTORY)} turns  →  warm KV cache  →  new question")
    print(f"  New question: \"{NEW_QUESTION[:60]}...\"")
    print(f"  Generate    : {GEN_TOKENS} tokens   Trials: {TRIALS}")
    print(f"  Est. prompt : ~{ESTIMATED_PROMPT_TOKENS} tokens")

    # ── Connect to or start the server ───────────────────────────────────────
    proc = None
    try:
        _wait_ready(SERVER_PORT, timeout=3)
        print(f"\n  {PROVIDER} already running on port {SERVER_PORT} — using existing instance.")
    except TimeoutError:
        if PROVIDER.lower() in ("lmstudio", "lms"):
            sys.exit(
                f"Error: {PROVIDER} not reachable on port {SERVER_PORT}.\n"
                f"Start LM Studio and enable the local server first."
            )
        # Fall back to launching omlx
        if not os.path.isdir(omlx_dir):
            sys.exit(f"Error: model directory not found: {omlx_dir}")
        print(f"\n  Starting omlx (port {SERVER_PORT}) ... ", end="", flush=True)
        try:
            ##proc = start_omlx(omlx_dir)
            print("ready")
        except Exception as e:
            sys.exit(f"Failed to start omlx: {e}")

    # ── Discover available models ─────────────────────────────────────────────
    available = _get_models(SERVER_PORT)
    print(f"\n  Available models ({len(available)}):")
    for mid in available:
        print(f"    • {mid}")

    # ── Benchmark each model ──────────────────────────────────────────────────
    all_results = []
    skipped     = []

    for i, target in enumerate(MODELS, 1):
        model_id = match_model_id(available, target) or target

        label = target
        print(f"\n{'─' * W}")
        print(f"  [{i}/{len(MODELS)}] {label}")
        if model_id != target:
            print(f"  Matched model ID: {model_id}")

        result = run_model(SERVER_PORT, model_id, label, W)
        if result:
            all_results.append(result)

    # ── Results table ─────────────────────────────────────────────────────────
    print()
    print("═" * W)
    print("  RESULTS")
    print("═" * W)
    print()

    col_model  = 36
    col_pp     = 10
    col_ttft   = 16
    col_tg     = 12
    col_kv     = 12

    header = (
        f"  {'Model':<{col_model}}"
        f"  {'PP (t/s)':>{col_pp}}"
        f"  {'TTFT warm (ms)':>{col_ttft}}"
        f"  {'TG (t/s)':>{col_tg}}"
        f"  {'KV cache':>{col_kv}}"
    )
    print(header)
    print("  " + "─" * (len(header) - 2))

    for r in all_results:
        kv_str = f"✓ {r['speedup']:.1f}×" if r["cache_ok"] else f"✗ {r['speedup']:.1f}×"
        print(
            f"  {r['label']:<{col_model}}"
            f"  {r['pp_tps']:>{col_pp}.0f}"
            f"  {r['ttft']:>8.0f} ±{r['ttft_sd']:>3.0f}ms"
            f"  {r['tg']:>{col_tg}.1f}"
            f"  {kv_str:>{col_kv}}"
        )

    if skipped:
        print()
        print(f"  Skipped (not found in omlx): {', '.join(skipped)}")

    # ── Rankings ──────────────────────────────────────────────────────────────
    if len(all_results) >= 2:
        print()
        print("  Rankings")
        print("  " + "─" * 40)

        by_ttft = sorted(all_results, key=lambda r: r["ttft"])
        by_tg   = sorted(all_results, key=lambda r: r["tg"], reverse=True)
        by_pp   = sorted(all_results, key=lambda r: r["pp_tps"], reverse=True)

        print(f"  Lowest warm TTFT : {by_ttft[0]['label']}  ({by_ttft[0]['ttft']:.0f} ms)")
        print(f"  Highest TG       : {by_tg[0]['label']}  ({by_tg[0]['tg']:.1f} t/s)")
        print(f"  Fastest PP       : {by_pp[0]['label']}  ({by_pp[0]['pp_tps']:.0f} t/s)")

        no_cache = [r for r in all_results if not r["cache_ok"]]
        if no_cache:
            print()
            print("  WARNING — KV cache may NOT be working for:")
            for r in no_cache:
                print(f"    • {r['label']}  (cold {r['cold_ttft']:.0f} ms  →  warm {r['ttft']:.0f} ms  speedup {r['speedup']:.1f}×)")
        else:
            print()
            print("  KV cache: all models show ≥3× TTFT speedup  ✓")

    print()
    print("═" * W)
    print()

    if proc:
        stop_server(proc)
        print("  omlx stopped.")


if __name__ == "__main__":
    main()
