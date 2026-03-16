#!/usr/bin/env python3
"""
bench-server.py — Real-server KV-cache benchmark: llama.cpp vs mlx-lm vs omlx

Starts each server, warms its KV cache with a multi-turn conversation history,
then measures only the final user turn — the hot-cache scenario that matters
for real voicebot latency.

Metrics
  TTFT  time from request → first content token (ms).  The number you feel.
  TG    token generation throughput (t/s) from first → last token.

Usage
  # Two-way comparison (llama.cpp vs mlx-lm)
  python3 scripts/bench-server.py <llama-model.gguf> <mlx-model-or-hf-repo>

  # Three-way comparison (adds omlx)
  python3 scripts/bench-server.py <llama-model.gguf> <mlx-model-or-hf-repo> <omlx-model-dir>

  python3 scripts/bench-server.py ./models/Qwen2.5-7B-Q4_K_M.gguf \\
                                  mlx-community/Qwen2.5-7B-Instruct-4bit \\
                                  ~/models

Env vars
  LLAMA_PORT=8080      llama.cpp server port
  MLX_PORT=8000        mlx-lm server port
  OMLX_PORT=8001       omlx server port
  BENCH_TRIALS=3       measurement trials per provider
  BENCH_GEN=80         tokens to generate per trial
"""

import http.client
import json
import os
import shutil
import subprocess
import sys
import time
from statistics import mean, stdev

# ── Conversation fixture ──────────────────────────────────────────────────────
# A realistic voicebot conversation. HISTORY is pre-loaded to warm the KV cache.
# NEW_QUESTION is the measured turn — only these tokens need prefill when the
# cache is hot.

SYSTEM_PROMPT = "Eres Jarvis, el asistente personal de IA de Daniel. Llevas años trabajando con él y le conoces bien.\n\nCARÁCTER\nMezcla de Jarvis (Iron Man) y Alfred (Batman): profesional, ligeramente irónico, humor seco y británico. Leal, discreto, eficiente. Nunca servil. Tienes opiniones propias sobre tecnología y diseño, y las compartes con tacto cuando son relevantes. Ocasionalmente haces un comentario sarcástico, pero nunca a costa de Daniel.\n\nFORMA DE HABLAR\n- Siempre en español salvo que Daniel cambie de idioma.\n- Llamas a Daniel por su nombre, nunca \"señor\" ni \"usuario\".\n- Respuestas concisas: 2-3 frases máximo salvo que pida más detalle.\n- Hablas para ser escuchado: sin markdown, sin listas, sin símbolos, sin nada que un sintetizador no pronuncie bien.\n- Cuando no sabes algo, lo dices. No inventas.\n- Antes de una acción irreversible, la describes y pides confirmación.\n\nHERRAMIENTAS DISPONIBLES\n- current_time: hora y fecha actuales.\n- get_calendar_events: eventos del calendario para una fecha.\n- create_calendar_event: crear evento o recordatorio en Calendar.app.\n- read_clipboard / set_clipboard: leer o escribir el portapapeles.\n- read_file: leer el contenido de un fichero (max 16 KB).\n- open_app: abrir una aplicacion macOS por nombre.\n- send_notification: enviar una notificacion macOS.\n- run_shell: ejecutar un comando de terminal (disponible si SHELL_ENABLED=1).\n- take_screenshot: capturar la pantalla y describir lo que hay en ella (disponible si VISION_URL esta configurado).\n- run_agent_async: delegar una tarea compleja al agente externo (disponible si AGENT_COMMAND esta configurado). El agente trabaja en segundo plano y el resultado llega en breve.\n\nUsa las herramientas directamente cuando puedas. Para tareas complejas de multiples pasos usa run_agent_async. No afirmes tener capacidades que no tienes."

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
    ("user",      "¿Qué películas de ciencia ficción recomiendas para esta noche?"),
    ("assistant", "Te recomiendo Interstellar o Blade Runner 2049. Las dos son magníficas."),
    ("user",      "¿Cuál es la capital de Australia?"),
    ("assistant", "La capital de Australia es Canberra, aunque muchos creen que es Sídney."),
    ("user",      "¿Sabes si mañana hay huelga de metro en Madrid?"),
    ("assistant", "No tengo información en tiempo real sobre huelgas. Consulta el sitio web del metro de Madrid."),
]

# The measured turn — only these ~15 tokens need prefill when cache is hot.
NEW_QUESTION = "¿Y cuál es la ciudad más poblada de Australia entonces?"

# ── Config ────────────────────────────────────────────────────────────────────

LLAMA_PORT = int(os.environ.get("LLAMA_PORT",    "8080"))
MLX_PORT   = int(os.environ.get("MLX_PORT",      "8000"))
OMLX_PORT  = int(os.environ.get("OMLX_PORT",     "8001"))
TRIALS     = int(os.environ.get("BENCH_TRIALS",  "3"))
GEN_TOKENS = int(os.environ.get("BENCH_GEN",     "80"))

# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _post(host, port, payload, stream=True):
    """POST to /v1/chat/completions. For stream=True, yields SSE lines."""
    body = json.dumps(payload).encode()
    conn = http.client.HTTPConnection(host, port, timeout=60)
    try:
        conn.request(
            "POST", "/v1/chat/completions", body=body,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:300].decode()}")
        if not stream:
            return resp.read()
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


def _get_model_id(host, port):
    """Return the model identifier as reported by /v1/models (first entry)."""
    conn = http.client.HTTPConnection(host, port, timeout=10)
    try:
        conn.request("GET", "/v1/models")
        resp = conn.getresponse()
        data = json.loads(resp.read())
        return data["data"][0]["id"]
    except Exception:
        return "local"   # llama.cpp ignores the field anyway
    finally:
        conn.close()


def _build_messages(new_question=None):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for role, content in HISTORY:
        msgs.append({"role": role, "content": content})
    if new_question:
        msgs.append({"role": "user", "content": new_question})
    return msgs


# ── Warm-up ───────────────────────────────────────────────────────────────────

def warmup(host, port, model_id, llama_cache):
    """
    Populate the KV cache with the full prompt (history + new question).
    Ends with a user message so thinking-mode models (Qwen3) don't reject it.
    Uses max_tokens=1 — the server prefills the entire context and generates
    one token to confirm the cache is committed.
    The measure() call sends the identical prompt, so no prefill is needed:
    the server goes straight to generation from the cached KV state.
    """
    payload = {
        "model":                model_id,
        "messages":             _build_messages(NEW_QUESTION),
        "max_tokens":           1,
        "temperature":          0.0,
        "stream":               False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if llama_cache:
        payload["cache_prompt"] = True
        payload["slot_id"]      = 0

    list(_post(host, port, payload, stream=False))  # discard output


# ── Measurement ───────────────────────────────────────────────────────────────

def measure(host, port, model_id, llama_cache):
    """
    Send history + NEW_QUESTION and measure TTFT and TG.
    The server only needs to prefill the new question tokens (cache is hot).
    Returns (ttft_ms, tg_tps, n_tokens) or raises on failure.
    """
    payload = {
        "model":                model_id,
        "messages":             _build_messages(NEW_QUESTION),
        "max_tokens":           GEN_TOKENS,
        "temperature":          0.1,
        "stream":               True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if llama_cache:
        payload["cache_prompt"] = True
        payload["slot_id"]      = 0

    t_start  = time.perf_counter()
    t_first  = None
    t_last   = None
    n_tokens = 0

    for line in _post(host, port, payload, stream=True):
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        content = (chunk.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
        if content:
            now = time.perf_counter()
            if t_first is None:
                t_first = now
            t_last   = now
            n_tokens += len(content.split())  # word-level proxy when tokens batch

    if t_first is None or n_tokens == 0:
        raise RuntimeError("No content tokens received")

    ttft_ms = (t_first - t_start) * 1000
    tg_secs = (t_last - t_first) if t_last and t_last > t_first else None
    tg_tps  = n_tokens / tg_secs if tg_secs and tg_secs > 0.001 else float("nan")

    return ttft_ms, tg_tps, n_tokens


# ── Server management ─────────────────────────────────────────────────────────

def _wait_ready(host, port, path, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            conn = http.client.HTTPConnection(host, port, timeout=2)
            conn.request("GET", path)
            r = conn.getresponse()
            r.read()
            conn.close()
            if r.status < 500:
                return
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError(f"Server {host}:{port}{path} not ready after {timeout}s")


def start_llama(model_path):
    threads = max(1, (os.cpu_count() or 8) - 2)
    cmd = [
        "llama-server",
        "--model",         model_path,
        "--host",          "127.0.0.1",
        "--port",          str(LLAMA_PORT),
        "--n-gpu-layers",  "99",
        "--cache-type-k",  "q4_0",
        "--cache-type-v",  "q4_0",
        "--flash-attn",    "on",
        "--parallel",      "1",
        "--ctx-size",      "8192",
        "--threads",          str(threads),
        "--mlock",
        "--reasoning-budget", "0",   # disable Qwen3 chain-of-thought
        "--log-disable",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait_ready("127.0.0.1", LLAMA_PORT, "/health")
    except TimeoutError:
        proc.terminate()
        raise
    return proc


def _mlx_runner():
    if shutil.which("uvx"):
        return ["uvx", "mlx_lm"]
    for candidate in (["mlx_lm"], ["python3", "-m", "mlx_lm"]):
        try:
            r = subprocess.run([*candidate, "--help"],
                               capture_output=True, timeout=5)
            if r.returncode in (0, 1, 2):
                return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    raise RuntimeError("mlx-lm not found — install with: pip install mlx-lm")


def start_mlx(model):
    runner = _mlx_runner()
    cmd = [
        *runner, "server",
        "--model",             model,
        "--host",              "127.0.0.1",
        "--port",              str(MLX_PORT),
        "--prompt-cache-size", "1",
        "--prefill-step-size", "512",
        "--chat-template-args", "{\"enable_thinking\": false}", \
        "--log-level",         "ERROR",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait_ready("127.0.0.1", MLX_PORT, "/v1/models", timeout=300)
    except TimeoutError:
        proc.terminate()
        raise
    return proc


def start_omlx(model_dir):
    if not shutil.which("omlx"):
        raise RuntimeError("omlx not found — install from https://github.com/jundot/omlx")
    cmd = [
        "omlx", "serve",
        "--model-dir", model_dir,
        "--host",      "127.0.0.1",
        "--port",      str(OMLX_PORT),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait_ready("127.0.0.1", OMLX_PORT, "/v1/models", timeout=180)
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


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_benchmark(label, host, port, llama_cache, model):
    model_id = model
    print(f"    Model id: {model_id}")

    print(f"    Warming KV cache ({len(HISTORY)} turns + new question)...",
          end=" ", flush=True)
    warmup(host, port, model_id, llama_cache)
    print("done")

    results = []
    for i in range(TRIALS):
        print(f"    Trial {i + 1}/{TRIALS} ... ", end="", flush=True)
        try:
            ttft, tg, n = measure(host, port, model_id, llama_cache)
            print(f"TTFT {ttft:>6.0f} ms   TG {tg:>5.1f} t/s   ({n} tokens)")
            results.append((ttft, tg, n))
        except Exception as e:
            print(f"FAILED: {e}")

    return results


def summarise(results):
    if not results:
        return None
    ttfts = [r[0] for r in results]
    tgs   = [r[1] for r in results]
    return {
        "ttft":    mean(ttfts),
        "ttft_sd": stdev(ttfts) if len(ttfts) > 1 else 0,
        "tg":      mean(tgs),
        "tg_sd":   stdev(tgs)   if len(tgs)   > 1 else 0,
        "tokens":  mean(r[2] for r in results),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def _run_provider(label, step, total, start_fn, host, port, llama_cache, model, stop_label, stats, W):
    print(f"\n{'─' * W}")
    print(f"  [{step}/{total}] {label}  (port {port})")
    print(f"{'─' * W}")
    proc = None
    try:
        print(f"  Starting {stop_label} ... ", end="", flush=True)
        proc = start_fn()
        print("ready")
        raw = run_benchmark(label, host, port, llama_cache=llama_cache, model=model)
        stats[label] = summarise(raw)
    except Exception as e:
        print(f"\n  ERROR: {e}")
    finally:
        stop_server(proc)
        print(f"  {stop_label} stopped.")


def _print_comparison(stats, providers, W):
    available = [(n, stats[n]) for n in providers if stats.get(n)]
    if len(available) < 2:
        return

    TTFT_OK = 120  # ms threshold — below this TTFT feels instant

    # Find overall TTFT and TG winners
    ttft_winner = min(available, key=lambda x: x[1]["ttft"])[0]
    tg_winner   = max(available, key=lambda x: x[1]["tg"])[0]

    best_ttft = min(s["ttft"] for _, s in available)
    best_tg   = max(s["tg"]   for _, s in available)

    print()
    print(f"  {'─' * (W - 2)}")
    print(f"  TTFT  →  {ttft_winner} wins")
    print(f"  TG    →  {tg_winner} wins")

    # Per-provider comparison vs winner
    print()
    for name, s in available:
        ttft_x = s["ttft"] / best_ttft
        tg_x   = best_tg   / s["tg"] if s["tg"] > 0 else float("inf")
        marker = " ← best" if name == ttft_winner else f"  ({ttft_x:.2f}× slower TTFT)"
        print(f"  {name:<12}  TTFT {s['ttft']:>6.0f} ms{marker}")

    # Recommendation
    print()
    all_fast = all(s["ttft"] < TTFT_OK for _, s in available)
    if all_fast:
        print(f"  All providers respond under {TTFT_OK} ms — TTFT is not the bottleneck.")
        print(f"  Higher TG wins: recommend {tg_winner}.")
    else:
        slow = [n for n, s in available if s["ttft"] >= TTFT_OK]
        fast = [n for n, s in available if s["ttft"] <  TTFT_OK]
        if fast:
            print(f"  Under {TTFT_OK} ms threshold: {', '.join(fast)}")
            print(f"  Above threshold (feels sluggish): {', '.join(slow)}")
        print(f"  Recommend: {ttft_winner} for lowest latency.")

    # KV-cache health warnings
    print()
    for name, s in available:
        if s["ttft"] > 800:
            print(f"  WARNING  {name} TTFT={s['ttft']:.0f} ms — KV cache may not be working.")


def main():
    if len(sys.argv) not in (3, 4):
        print(__doc__)
        sys.exit(1)

    llama_model = sys.argv[1]
    mlx_model   = sys.argv[2]
    omlx_dir    = sys.argv[3] if len(sys.argv) == 4 else None

    if not os.path.isfile(llama_model):
        sys.exit(f"Error: llama model not found: {llama_model}")
    if not shutil.which("llama-server"):
        sys.exit("Error: llama-server not found — brew install llama.cpp")
    if omlx_dir and not os.path.isdir(omlx_dir):
        sys.exit(f"Error: omlx model directory not found: {omlx_dir}")

    providers = ["llama.cpp", "mlx-lm"] + (["omlx"] if omlx_dir else [])
    total = len(providers)

    W = 64
    print()
    print("═" * W)
    title = "Real-Server KV-Cache Benchmark — " + " vs ".join(providers)
    print(f"  {title}")
    print("═" * W)
    print(f"  llama model : {os.path.basename(llama_model)}")
    print(f"  mlx model   : {mlx_model}")
    if omlx_dir:
        print(f"  omlx dir    : {omlx_dir}")
    print(f"  Scenario    : {len(HISTORY)} turn history → warm KV cache → new question")
    print(f"  New question: \"{NEW_QUESTION}\"")
    print(f"  Generate    : {GEN_TOKENS} tokens   Trials: {TRIALS}")

    stats = {}

    # ── llama.cpp ─────────────────────────────────────────────────────────────
    _run_provider(
        "llama.cpp", 1, total,
        lambda: start_llama(llama_model),
        "127.0.0.1", LLAMA_PORT, llama_cache=True, model=llama_model,
        stop_label="llama-server", stats=stats, W=W,
    )

    # ── mlx-lm ───────────────────────────────────────────────────────────────
    _run_provider(
        "mlx-lm", 2, total,
        lambda: start_mlx(mlx_model),
        "127.0.0.1", MLX_PORT, llama_cache=False, model=mlx_model,
        stop_label="mlx-lm server", stats=stats, W=W,
    )

    # ── omlx (optional) ──────────────────────────────────────────────────────
    if omlx_dir:
        _run_provider(
            "omlx", 3, total,
            lambda: start_omlx(omlx_dir),
            "127.0.0.1", OMLX_PORT, llama_cache=False, model=mlx_model,
            stop_label="omlx server", stats=stats, W=W,
        )

    # ── Results table ─────────────────────────────────────────────────────────
    print()
    print("═" * W)
    print("  RESULTS — warm KV-cache turn")
    print("═" * W)
    print()
    print(f"  {'Provider':<12}  {'TTFT (ms)':>14}  {'TG (t/s)':>12}  {'tokens':>6}")
    print(f"  {'─'*12}  {'─'*14}  {'─'*12}  {'─'*6}")

    for name in providers:
        s = stats.get(name)
        if s:
            print(f"  {name:<12}  "
                  f"{s['ttft']:>8.0f} ±{s['ttft_sd']:>3.0f}ms  "
                  f"{s['tg']:>7.1f} ±{s['tg_sd']:>3.1f}  "
                  f"{s['tokens']:>6.0f}")
        else:
            print(f"  {name:<12}  {'FAILED':>14}  {'—':>12}  {'—':>6}")

    _print_comparison(stats, providers, W)

    print()
    print("═" * W)
    print()


if __name__ == "__main__":
    main()
