#!/usr/bin/env python3
"""
bench-models.py — multi-server, multi-runtime, multi-model benchmark

Loads configuration from config.yaml and for every model runs:
  1. KV-cache speed benchmark (TTFT, PP rate, TG rate)
  2. Quality benchmark — runs fixtures.json tests while the model is still
     loaded, then evaluates responses via mechanical checks + evaluator LLM.

The two phases within quality are kept separate so the model under test is
loaded exactly once: all fixture responses are collected first (Phase 1),
then the evaluator judges them (Phase 2).

Usage:
  python3 scripts/bench-models.py [config.yaml]

  Default config path: scripts/config.yaml (next to this script)
  Default fixtures:    scripts/fixtures.json (next to config)

Env vars:
  BENCH_TRIALS   hot measurement trials    (default 3)
  BENCH_GEN      tokens to generate        (default 80)
  BENCH_QUALITY  run quality benchmark     (default 1; set 0 to skip)
"""

import http.client
import json
import math
import os
import re
import sys
import time
import yaml
from statistics import mean, stdev

# ── Speed benchmark — conversation fixture ────────────────────────────────────

SYSTEM_PROMPT = (
    "Eres Jarvis, el asistente personal de IA. Llevas años trabajando con él y le conoces bien.\n\n"
    "CARÁCTER\n"
    "Mezcla de Jarvis (Iron Man) y Alfred (Batman): profesional, ligeramente irónico, humor seco "
    "y británico. Leal, discreto, eficiente. Nunca servil. Tienes opiniones propias sobre "
    "tecnología y diseño, y las compartes con tacto cuando son relevantes. Ocasionalmente haces "
    "un comentario sarcástico, pero nunca a costa del usuario.\n\n"
    "FORMA DE HABLAR\n"
    "- Siempre en español salvo que el usuario cambie de idioma.\n"
    '- Llamas al usuario por "señor", nunca "usuario".\n'
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
    ("user", "¿Qué tiempo hace hoy en Madrid?"),
    (
        "assistant",
        "Hoy en Madrid hay cielos despejados y unos dieciocho grados. Buen día para salir.",
    ),
    ("user", "¿Cuándo es el próximo partido del Real Madrid?"),
    (
        "assistant",
        "El Real Madrid juega este sábado a las nueve de la noche contra el Atlético en el Bernabéu.",
    ),
    ("user", "Recuérdame comprar leche mañana por la mañana."),
    ("assistant", "Anotado. Te recuerdo mañana a primera hora que compres leche."),
    ("user", "¿Cuánto es el veinte por ciento de trescientos cincuenta euros?"),
    (
        "assistant",
        "El veinte por ciento de trescientos cincuenta euros son setenta euros.",
    ),
    ("user", "¿Qué películas de ciencia ficción recomiendas para esta noche?"),
    (
        "assistant",
        "Te recomiendo Interstellar o Blade Runner 2049. Las dos son magníficas.",
    ),
    ("user", "¿Cuál es la capital de Australia?"),
    (
        "assistant",
        "La capital de Australia es Canberra, aunque muchos creen que es Sídney.",
    ),
    ("user", "¿Sabes si mañana hay huelga de metro en Madrid?"),
    (
        "assistant",
        "No tengo información en tiempo real sobre huelgas. Consulta el sitio web del metro de Madrid.",
    ),
]

NEW_QUESTION = (
    "Interesante. ¿Puedes contarme brevemente por qué Sídney es más conocida que "
    "Canberra, cómo surgió esa confusión tan común, y qué otras ciudades importantes "
    "tiene Australia?"
)

# ── Quality benchmark — tool definitions sent with requires_tools fixtures ────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "current_time",
            "description": "Returns the current time and date.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_calendar_events",
            "description": "Returns calendar events for a given date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD or natural language",
                    },
                },
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Creates a calendar event or reminder in Calendar.app.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "date": {"type": "string"},
                    "time": {"type": "string"},
                    "notes": {"type": "string"},
                },
                "required": ["title", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_clipboard",
            "description": "Reads the current contents of the system clipboard.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_clipboard",
            "description": "Writes text to the system clipboard.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads the contents of a file (max 16 KB).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_app",
            "description": "Opens a macOS application by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Application name"}
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_notification",
            "description": "Sends a macOS notification.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Executes a shell command on the macOS terminal. Only available when SHELL_ENABLED=1.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "take_screenshot",
            "description": "Captures the screen and describes its contents.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_agent_async",
            "description": "Delegates a complex multi-step task to an external agent. The agent works in the background and the result arrives shortly. Only available when AGENT_COMMAND is configured.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Full description of the task to delegate",
                    }
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Searches the web via SearXNG. Only available when SEARXNG_URL is configured.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]

# ── Config ────────────────────────────────────────────────────────────────────

TRIALS = int(os.environ.get("BENCH_TRIALS", "3"))
GEN_TOKENS = int(os.environ.get("BENCH_GEN", "80"))
RUN_QUALITY = os.environ.get("BENCH_QUALITY", "1") != "0"

_PROMPT_TEXT = (
    SYSTEM_PROMPT
    + "".join(c + t for r, c, t in [(r, c, t) for r, t in HISTORY for c in [r]])
    + NEW_QUESTION
)
ESTIMATED_PROMPT_TOKENS = max(1, int(len(_PROMPT_TEXT) / 3.5))


def _parse_host(host_url: str) -> str:
    """Strip http:// or https:// prefix for http.client.HTTPConnection."""
    for prefix in ("https://", "http://"):
        if host_url.startswith(prefix):
            return host_url[len(prefix) :]
    return host_url


def _is_llamacpp(runtime_name: str) -> bool:
    return "llama" in runtime_name.lower()


def _no_think_prefix(runtime_name: str) -> str:
    return "/no_think\n\n" if _is_llamacpp(runtime_name) else ""


def _thinking_off_fields() -> dict:
    return {
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "thinking": {"type": "disabled"},
        "think": False,
    }


def load_config(path: str) -> list[dict]:
    """Parse config.yaml → flat list of benchmark targets."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    targets = []
    for server_name, server_cfg in cfg.get("servers", {}).items():
        host = _parse_host(server_cfg.get("host", "http://127.0.0.1"))
        for runtime_name, runtime_cfg in server_cfg.get("runtimes", {}).items():
            targets.append(
                {
                    "server": server_name,
                    "runtime": runtime_name,
                    "host": host,
                    "port": int(runtime_cfg.get("port", 8000)),
                    "token": runtime_cfg.get("token", ""),
                    "models": runtime_cfg.get("models", []),
                }
            )
    return targets


def load_evaluator_config(path: str) -> dict | None:
    """Parse the evaluator section from config.yaml. Returns None if absent."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    ev = cfg.get("evaluator")
    if not ev:
        return None
    return {
        "host": _parse_host(ev["host"]),
        "port": int(ev["port"]),
        "token": ev.get("token", ""),
        "model": ev["model"],
        "runtime": ev.get("runtime", ""),
        "temperature": float(ev.get("temperature", 0.0)),
        "max_tokens": int(ev.get("max_tokens", 512)),
    }


def load_fixtures(path: str) -> list[dict]:
    """Load fixtures.json, filtering out _comment-only entries."""
    with open(path) as f:
        data = json.load(f)
    return [fx for fx in data if "id" in fx]


# ── HTTP helpers ──────────────────────────────────────────────────────────────


def _auth_headers(token: str) -> dict:
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _post_stream(host, port, token, payload):
    """POST to /v1/chat/completions with stream=True. Yields SSE content lines.

    Reads one byte at a time to avoid http.client's chunked-encoding
    accumulation — with llama.cpp emitting ~200 bytes per SSE event,
    read(4096) buffers ~20 tokens before the first yield, inflating TTFT.
    """
    body = json.dumps(payload).encode()
    conn = http.client.HTTPConnection(host, port, timeout=120)
    try:
        conn.request(
            "POST", "/v1/chat/completions", body=body, headers=_auth_headers(token)
        )
        resp = conn.getresponse()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:300].decode()}")
        buf = ""
        while True:
            byte = resp.read(1)
            if not byte:
                break
            ch = byte.decode("utf-8", errors="replace")
            if ch == "\n":
                yield buf.rstrip("\r")
                buf = ""
            else:
                buf += ch
        if buf:
            yield buf.rstrip("\r")
    finally:
        conn.close()


def _post_blocking(host, port, token, payload) -> dict:
    """POST to /v1/chat/completions with stream=False. Returns parsed JSON."""
    payload = {**payload, "stream": False}
    body = json.dumps(payload).encode()
    conn = http.client.HTTPConnection(host, port, timeout=120)
    try:
        conn.request(
            "POST", "/v1/chat/completions", body=body, headers=_auth_headers(token)
        )
        resp = conn.getresponse()
        raw = resp.read()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {raw[:300].decode()}")
        return json.loads(raw)
    finally:
        conn.close()


def _get_models(host, port, token) -> list[str]:
    """Return list of model IDs from /v1/models."""
    conn = http.client.HTTPConnection(host, port, timeout=10)
    try:
        conn.request("GET", "/v1/models", headers=_auth_headers(token))
        resp = conn.getresponse()
        data = json.loads(resp.read())
        return [m["id"] for m in data.get("data", [])]
    except Exception:
        return []
    finally:
        conn.close()


def _wait_ready(host, port, token, timeout=5) -> bool:
    """Return True if the server responds before timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            conn = http.client.HTTPConnection(host, port, timeout=2)
            conn.request("GET", "/v1/models", headers=_auth_headers(token))
            r = conn.getresponse()
            r.read()
            conn.close()
            if r.status < 500:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


# ── Speed benchmark — conversation builder ────────────────────────────────────


def _build_speed_messages(runtime_name: str) -> list[dict]:
    system_content = _no_think_prefix(runtime_name) + SYSTEM_PROMPT
    msgs = [{"role": "system", "content": system_content}]
    for role, content in HISTORY:
        msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": NEW_QUESTION})
    return msgs


def _base_speed_payload(model_id, max_tokens, stream, runtime_name) -> dict:
    return {
        "model": model_id,
        "messages": _build_speed_messages(runtime_name),
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
        **_thinking_off_fields(),
    }


# ── Speed benchmark — measurement steps ──────────────────────────────────────


def load_model(host, port, token, model_id):
    """Send a trivial request to pull model weights into GPU/RAM. Not timed."""
    _post_blocking(
        host,
        port,
        token,
        {
            "model": model_id,
            "messages": [{"role": "user", "content": "Hola"}],
            "max_tokens": 1,
        },
    )


def measure_pp(host, port, token, model_id, runtime_name):
    """Cold full-conversation prefill. Returns (cold_ttft_ms, pp_tps, prompt_tokens)."""
    payload = _base_speed_payload(
        model_id, max_tokens=GEN_TOKENS, stream=True, runtime_name=runtime_name
    )
    t_start = time.perf_counter()
    t_first = None
    prompt_tokens_api = None

    for line in _post_stream(host, port, token, payload):
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        if "usage" in chunk and chunk["usage"]:
            prompt_tokens_api = chunk["usage"].get("prompt_tokens")
        delta = (chunk.get("choices") or [{}])[0].get("delta", {})
        if (delta.get("content") or "") and t_first is None:
            t_first = time.perf_counter()

    if t_first is None:
        raise RuntimeError("PP trial: no content token received")

    elapsed = t_first - t_start
    cold_ttft_ms = elapsed * 1000
    prompt_tokens = prompt_tokens_api or ESTIMATED_PROMPT_TOKENS
    pp_tps = prompt_tokens / elapsed if elapsed > 0 else float("nan")
    return cold_ttft_ms, pp_tps, prompt_tokens


def measure_hot(host, port, token, model_id, runtime_name):
    """Hot trial with warm KV cache. Returns (ttft_ms, tg_tps, n_tokens)."""
    payload = _base_speed_payload(
        model_id, max_tokens=GEN_TOKENS, stream=True, runtime_name=runtime_name
    )
    t_start = time.perf_counter()
    t_first = t_last = t_done = None
    n_tokens = 0

    for line in _post_stream(host, port, token, payload):
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
        content = (chunk.get("choices") or [{}])[0].get("delta", {}).get(
            "content"
        ) or ""
        if content:
            now = time.perf_counter()
            if t_first is None:
                t_first = now
            t_last = now
            n_tokens += len(content.split())

    if t_first is None or n_tokens == 0:
        raise RuntimeError("Hot trial: no content tokens received")

    ttft_ms = (t_first - t_start) * 1000
    tg_end = t_last if (t_last and t_last > t_first + 0.001) else t_done
    tg_secs = (tg_end - t_first) if tg_end and tg_end > t_first + 0.001 else None
    tg_tps = n_tokens / tg_secs if tg_secs else float("nan")
    return ttft_ms, tg_tps, n_tokens


# ── Quality benchmark — fixture execution ─────────────────────────────────────


def _normalize_tool_calls(raw: list) -> list[dict]:
    """Normalize the tool_calls array from a chat completion response."""
    result = []
    for tc in raw or []:
        if "function" not in tc:
            continue
        result.append(
            {
                "id": tc.get("id", ""),
                "function": {
                    "name": tc["function"].get("name", ""),
                    "arguments": tc["function"].get("arguments", "{}"),
                },
            }
        )
    return result


def run_fixture(host, port, token, model_id, runtime_name, fixture) -> tuple[str, list]:
    """Execute one fixture against the model under test. Returns (text, tool_calls)."""
    system_content = _no_think_prefix(runtime_name) + SYSTEM_PROMPT
    messages = [{"role": "system", "content": system_content}]

    for msg in fixture["messages"]:
        # Pass tool-exchange and tool-result messages verbatim (may have null content)
        if msg.get("tool_calls") or msg.get("role") == "tool":
            messages.append(msg)
        else:
            messages.append({"role": msg["role"], "content": msg.get("content") or ""})

    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": 350,
        "temperature": 0.0,
        **_thinking_off_fields(),
    }
    if fixture.get("requires_tools"):
        payload["tools"] = TOOL_DEFINITIONS
        payload["tool_choice"] = "auto"

    resp = _post_blocking(host, port, token, payload)
    msg_out = resp["choices"][0]["message"]
    text = msg_out.get("content") or ""
    tool_calls = _normalize_tool_calls(msg_out.get("tool_calls") or [])
    return text, tool_calls


# ── Quality benchmark — mechanical checks ────────────────────────────────────


def run_mechanical_checks(fixture: dict, text: str, tool_calls: list) -> list[tuple]:
    """
    Evaluate all machine-verifiable criteria from fixture["eval"].
    Returns list of (criterion, passed: bool, detail: str).
    """
    ev = fixture.get("eval", {})
    checks = []

    for pat in ev.get("forbidden_patterns", []):
        m = re.search(pat, text, re.MULTILINE | re.DOTALL)
        passed = m is None
        snippet = repr(m.group()[:60]) if m else ""
        checks.append(
            (
                "forbidden_pattern",
                passed,
                f"/{pat[:50]}/: {'not found ✓' if passed else f'matched {snippet} ✗'}",
            )
        )

    for s in ev.get("forbidden_strings", []):
        passed = s.lower() not in text.lower()
        checks.append(
            (
                "forbidden_string",
                passed,
                f"{s!r}: {'absent ✓' if passed else 'found ✗'}",
            )
        )

    for pat in ev.get("required_patterns", []):
        m = re.search(pat, text, re.MULTILINE)
        passed = m is not None
        checks.append(
            (
                "required_pattern",
                passed,
                f"/{pat[:50]}/: {'matched ✓' if passed else 'not found ✗'}",
            )
        )

    for s in ev.get("required_strings", []):
        passed = s.lower() in text.lower()
        checks.append(
            (
                "required_string",
                passed,
                f"{s!r}: {'found ✓' if passed else 'missing ✗'}",
            )
        )

    if "max_sentences" in ev:
        n = len(re.findall(r"[.!?]+(?:\s|$)", text))
        passed = n <= ev["max_sentences"]
        checks.append(
            ("max_sentences", passed, f"{n} sentences (max {ev['max_sentences']})")
        )

    if "max_words" in ev:
        n = len(text.split())
        passed = n <= ev["max_words"]
        checks.append(("max_words", passed, f"{n} words (max {ev['max_words']})"))

    if "min_words" in ev:
        n = len(text.split())
        passed = n >= ev["min_words"]
        checks.append(("min_words", passed, f"{n} words (min {ev['min_words']})"))

    called = {tc["function"]["name"] for tc in tool_calls}

    for tool in ev.get("must_call_tools", []):
        passed = tool in called
        checks.append(
            (
                "must_call_tool",
                passed,
                f"tool {tool!r}: {'called ✓' if passed else 'NOT called ✗'}",
            )
        )

    for tool in ev.get("must_not_call_tools", []):
        passed = tool not in called
        checks.append(
            (
                "must_not_call_tool",
                passed,
                f"tool {tool!r}: {'absent ✓' if passed else 'called ✗'}",
            )
        )

    if ev.get("no_tool_called") is True:
        passed = len(tool_calls) == 0
        checks.append(
            (
                "no_tool_called",
                passed,
                f"no tools called: {'✓' if passed else f'✗ ({list(called)})'}",
            )
        )

    if ev.get("any_tool_called") is True:
        passed = len(tool_calls) > 0
        checks.append(
            (
                "any_tool_called",
                passed,
                f"at least one tool called: {'✓' if passed else '✗'}",
            )
        )

    for tool_name, expected_arg in ev.get("tool_args_contain", {}).items():
        matching = [tc for tc in tool_calls if tc["function"]["name"] == tool_name]
        if not matching:
            checks.append(
                (
                    "tool_args_contain",
                    False,
                    f"tool {tool_name!r} not called (needed arg {expected_arg!r}) ✗",
                )
            )
        else:
            args_str = matching[0]["function"].get("arguments", "")
            passed = expected_arg.lower() in args_str.lower()
            checks.append(
                (
                    "tool_args_contain",
                    passed,
                    f"tool {tool_name!r} arg {expected_arg!r}: {'✓' if passed else f'✗ (args={args_str!r})'}",
                )
            )

    return checks


# ── Quality benchmark — LLM evaluator ────────────────────────────────────────


def call_evaluator(
    ev_cfg: dict,
    fixture: dict,
    fixture_messages: list,
    text: str,
    tool_calls: list,
    mech: list,
) -> dict:
    """
    Ask the evaluator LLM to judge a response.
    Returns {"verdict": "PASS"|"FAIL"|"PARTIAL", "reason": str}.
    """
    # Summarise conversation context for the evaluator
    conv_lines = []
    for msg in fixture_messages:
        role = msg.get("role", "?")
        content = msg.get("content") or ""
        if msg.get("tool_calls"):
            fn = msg["tool_calls"][0]["function"]
            conv_lines.append(
                f"{role}: [tool call: {fn['name']}({fn.get('arguments', '')[:80]})]"
            )
        elif role == "tool":
            conv_lines.append(f"tool result: {content[:120]}")
        else:
            conv_lines.append(f"{role}: {content}")
    conv_text = "\n".join(conv_lines)

    # Summarise the model's response
    if tool_calls:
        tc_str = ", ".join(
            f"{tc['function']['name']}({tc['function']['arguments'][:60]})"
            for tc in tool_calls
        )
        response_repr = f"[Tool calls: {tc_str}]"
        if text:
            response_repr += f"\n[Text: {text}]"
    else:
        response_repr = text or "(empty response)"

    # Mechanical check summary
    failed_mechs = [detail for _, passed, detail in mech if not passed]
    if failed_mechs:
        mech_note = "\nMechanical checks FAILED:\n" + "\n".join(
            f"  - {d}" for d in failed_mechs
        )
    elif mech:
        mech_note = f"\nAll {len(mech)} mechanical checks passed."
    else:
        mech_note = ""

    no_think = _no_think_prefix(ev_cfg["runtime"])
    system_msg = (
        f"{no_think}You are a strict quality evaluator for a voice assistant called Jarvis. "
        "Assess whether the assistant's response meets the stated criteria. "
        "Be concise and decisive. "
        "Reply ONLY with one line of valid JSON: "
        '{"verdict":"PASS","reason":"..."} '
        '{"verdict":"FAIL","reason":"..."} or '
        '{"verdict":"PARTIAL","reason":"..."}'
    )
    user_msg = (
        f"Test: {fixture['description']}\n"
        f"Criteria: {fixture.get('eval', {}).get('notes', '(see mechanical checks)')}\n"
        f"{mech_note}\n"
        f"\nConversation:\n{conv_text}\n"
        f"\nAssistant response:\n{response_repr}\n"
        "\nVerdict (JSON only):"
    )

    resp = _post_blocking(
        ev_cfg["host"],
        ev_cfg["port"],
        ev_cfg["token"],
        {
            "model": ev_cfg["model"],
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": ev_cfg["max_tokens"],
            "temperature": ev_cfg["temperature"],
            **_thinking_off_fields(),
        },
    )
    raw = (resp["choices"][0]["message"].get("content") or "").strip()

    # Extract JSON from response (handle code fences and extra prose)
    clean = raw.lstrip("`")
    if clean.startswith("json"):
        clean = clean[4:]
    clean = clean.rstrip("`").strip()
    m = re.search(r"\{.*\}", clean, re.DOTALL | re.GREEDY)
    if m:
        clean = m.group()
    try:
        result = json.loads(clean)
        return {
            "verdict": str(result.get("verdict", "PARTIAL")).upper(),
            "reason": str(result.get("reason", ""))[:200],
        }
    except json.JSONDecodeError:
        # Try to find a JSON object, handling nested braces
        depth = 0
        start = clean.find("{")
        if start == -1:
            pass
        else:
            for i in range(start + 1, len(clean)):
                if clean[i] == "{":
                    depth += 1
                elif clean[i] == "}":
                    if depth == 0:
                        result = json.loads(clean[start : i + 1])
                        return {
                            "verdict": str(result.get("verdict", "PARTIAL")).upper(),
                            "reason": str(result.get("reason", ""))[:200],
                        }
                    depth -= 1
    # Fallback to keyword matching
    lower = raw.lower()
    if "pass" in lower and "fail" not in lower:
        return {"verdict": "PASS", "reason": raw[:200]}
    elif "fail" in lower:
        return {"verdict": "FAIL", "reason": raw[:200]}
    return {"verdict": "PARTIAL", "reason": raw[:200]}


# ── Quality benchmark — runner ────────────────────────────────────────────────


def run_quality_benchmark(
    host: str,
    port: int,
    token: str,
    model_id: str,
    runtime_name: str,
    fixtures: list[dict],
    ev_cfg: dict | None,
    W: int,
) -> list[dict]:
    """
    Two-phase quality benchmark for one model:

    Phase 1 — collect all fixture responses from the model under test.
               The model is loaded exactly once; no evaluator calls happen here.
    Phase 2 — mechanical checks + evaluator LLM judge each collected response.
               The evaluator may be on the same server; that is fine because
               all responses were already collected in Phase 1.
    """
    total = len(fixtures)
    pad = len(str(total))

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print(f"  Quality Phase 1/2 — collecting {total} fixture responses from model")

    collected: list[tuple] = []
    for i, fx in enumerate(fixtures, 1):
        label = f"[{i:{pad}}/{total}] {fx['id']}"
        print(f"    {label:<52}", end="", flush=True)
        try:
            text, tcs = run_fixture(host, port, token, model_id, runtime_name, fx)
            tag = (
                ("tools:" + "+".join(tc["function"]["name"] for tc in tcs))
                if tcs
                else f"{len(text)} chars"
            )
            print(f"ok  ({tag})")
            collected.append((fx, text, tcs, None))
        except Exception as e:
            print(f"ERROR  ({e})")
            collected.append((fx, "", [], str(e)))

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    ev_label = ev_cfg["model"] if ev_cfg else "mechanical checks only"
    print(f"\n  Quality Phase 2/2 — evaluating responses  [evaluator: {ev_label}]")

    ev_available = bool(
        ev_cfg
        and _wait_ready(ev_cfg["host"], ev_cfg["port"], ev_cfg["token"], timeout=5)
    )

    results: list[dict] = []
    for i, (fx, text, tcs, error) in enumerate(collected, 1):
        fid = fx["id"]
        group = fx.get("group", "?")
        label = f"[{i:{pad}}/{total}] {fid}"
        print(f"    {label:<52}", end="", flush=True)

        if error:
            print("ERROR")
            results.append(
                {
                    "id": fid,
                    "group": group,
                    "verdict": "ERROR",
                    "reason": error[:120],
                    "mech_pass": False,
                    "preview": "",
                }
            )
            continue

        mech = run_mechanical_checks(fx, text, tcs)
        failed = [detail for _, p, detail in mech if not p]

        if ev_available:
            try:
                ev_out = call_evaluator(ev_cfg, fx, fx["messages"], text, tcs, mech)
                verdict = ev_out["verdict"]
                reason = ev_out["reason"]
            except Exception as e:
                verdict = "FAIL" if failed else "PASS"
                reason = f"(evaluator error: {e})"
        else:
            verdict = "FAIL" if failed else "PASS"
            reason = "; ".join(failed[:2]) if failed else "(no evaluator)"

        # Mechanical failures always override an optimistic LLM verdict
        if failed and verdict == "PASS":
            verdict = "FAIL"
            reason = "; ".join(failed[:2])

        sym = {"PASS": "✓", "FAIL": "✗", "PARTIAL": "~", "ERROR": "!"}.get(verdict, "?")
        print(f"{sym} {verdict:<7}  {reason[:55]}")

        results.append(
            {
                "id": fid,
                "group": group,
                "verdict": verdict,
                "reason": reason,
                "mech_pass": not bool(failed),
                "preview": text[:100],
            }
        )

    passing = sum(1 for r in results if r["verdict"] == "PASS")
    print(
        f"\n  Quality score: {passing}/{total}  ({passing * 100 // total if total else 0}%)"
    )
    return results


# ── Model matching ────────────────────────────────────────────────────────────


def match_model_id(available: list[str], target: str) -> str | None:
    """Exact match first, then case-insensitive substring."""
    if target in available:
        return target
    target_lower = target.lower()
    for mid in available:
        if target_lower in mid.lower() or mid.lower() in target_lower:
            return mid
    return None


# ── Speed benchmark — per-model runner ───────────────────────────────────────


def run_speed_benchmark(
    host, port, token, model_id, runtime_name, label, W
) -> dict | None:
    """Run load → cold PP → N hot trials. Returns speed result dict or None."""
    print(f"\n  Loading model into memory ...", end=" ", flush=True)
    try:
        load_model(host, port, token, model_id)
        print("done")
    except Exception as e:
        print(f"FAILED: {e}")
        return None

    print(f"  Measuring cold PP (full prompt prefill) ...", end=" ", flush=True)
    try:
        cold_ttft, pp_tps, prompt_tokens = measure_pp(
            host, port, token, model_id, runtime_name
        )
        print(
            f"cold TTFT {cold_ttft:.0f} ms   PP ~{pp_tps:.0f} t/s   (~{prompt_tokens} prompt tokens)"
        )
    except Exception as e:
        print(f"FAILED: {e}")
        return None

    hot_results = []
    for i in range(TRIALS):
        print(f"  Hot trial {i + 1}/{TRIALS} ... ", end="", flush=True)
        try:
            ttft, tg, n = measure_hot(host, port, token, model_id, runtime_name)
            print(f"TTFT {ttft:>6.0f} ms   TG {tg:>5.1f} t/s   ({n} tokens)")
            hot_results.append((ttft, tg, n))
        except Exception as e:
            print(f"FAILED: {e}")

    if not hot_results:
        return None

    ttfts = [r[0] for r in hot_results]
    tgs_raw = [r[1] for r in hot_results]
    tgs = [v for v in tgs_raw if not math.isnan(v) and not math.isinf(v)]

    avg_ttft = mean(ttfts)
    avg_tg = mean(tgs) if tgs else float("nan")
    sd_ttft = stdev(ttfts) if len(ttfts) > 1 else 0.0
    sd_tg = stdev(tgs) if len(tgs) > 1 else 0.0
    speedup = cold_ttft / avg_ttft if avg_ttft > 0 else 0.0

    return {
        "label": label,
        "model_id": model_id,
        "cold_ttft": cold_ttft,
        "pp_tps": pp_tps,
        "prompt_tok": prompt_tokens,
        "ttft": avg_ttft,
        "ttft_sd": sd_ttft,
        "tg": avg_tg,
        "tg_sd": sd_tg,
        "tokens": mean(r[2] for r in hot_results),
        "speedup": speedup,
        "cache_ok": speedup >= 3.0,
    }


# ── Results display ───────────────────────────────────────────────────────────


def print_speed_results(all_results: list, W: int):
    print()
    print("═" * W)
    print("  SPEED RESULTS")
    print("═" * W)
    print()

    col_model = 50
    col_pp = 10
    col_ttft = 16
    col_tg = 12
    col_kv = 10

    header = (
        f"  {'Server/Runtime/Model':<{col_model}}"
        f"  {'PP (t/s)':>{col_pp}}"
        f"  {'TTFT warm (ms)':>{col_ttft}}"
        f"  {'TG (t/s)':>{col_tg}}"
        f"  {'KV cache':>{col_kv}}"
    )
    print(header)
    print("  " + "─" * (len(header) - 2))

    for r in all_results:
        kv_str = f"✓ {r['speedup']:.1f}×" if r["cache_ok"] else f"✗ {r['speedup']:.1f}×"
        display = f"{r['server']}/{r['runtime']}  {r['model_id']}"
        print(
            f"  {display:<{col_model}}"
            f"  {r['pp_tps']:>{col_pp}.0f}"
            f"  {r['ttft']:>8.0f} ±{r['ttft_sd']:>3.0f}ms"
            f"  {r['tg']:>{col_tg}.1f}"
            f"  {kv_str:>{col_kv}}"
        )

    if len(all_results) >= 2:
        print()
        print("  Rankings")
        print("  " + "─" * 40)
        by_ttft = sorted(all_results, key=lambda r: r["ttft"])
        by_tg = sorted(all_results, key=lambda r: r["tg"], reverse=True)
        by_pp = sorted(all_results, key=lambda r: r["pp_tps"], reverse=True)
        print(
            f"  Lowest warm TTFT : {by_ttft[0]['label']}  ({by_ttft[0]['ttft']:.0f} ms)"
        )
        print(f"  Highest TG       : {by_tg[0]['label']}  ({by_tg[0]['tg']:.1f} t/s)")
        print(
            f"  Fastest PP       : {by_pp[0]['label']}  ({by_pp[0]['pp_tps']:.0f} t/s)"
        )
        no_cache = [r for r in all_results if not r["cache_ok"]]
        if no_cache:
            print()
            print("  WARNING — KV cache may NOT be working for:")
            for r in no_cache:
                print(
                    f"    • {r['label']}  (cold {r['cold_ttft']:.0f} ms → warm {r['ttft']:.0f} ms  {r['speedup']:.1f}×)"
                )
        else:
            print()
            print("  KV cache: all models show ≥3× TTFT speedup  ✓")


def print_quality_results(all_results: list, W: int):
    has_quality = any("quality" in r for r in all_results)
    if not has_quality:
        return

    # Collect ordered group names
    groups: list[str] = []
    seen: set[str] = set()
    for r in all_results:
        for qr in r.get("quality", []):
            g = qr["group"]
            if g not in seen:
                groups.append(g)
                seen.add(g)

    print()
    print("═" * W)
    print("  QUALITY RESULTS")
    print("═" * W)
    print()

    col_model = 50
    col_total = 10

    # Header
    hdr = f"  {'Server/Runtime/Model':<{col_model}}  {'Score':>{col_total}}"
    for g in groups:
        hdr += f"  {g[:9]:>9}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for r in all_results:
        quality = r.get("quality", [])
        if not quality:
            continue
        total = len(quality)
        passing = sum(1 for qr in quality if qr["verdict"] == "PASS")
        pct = passing * 100 // total if total else 0
        display = f"{r['server']}/{r['runtime']}  {r['model_id']}"
        row = f"  {display:<{col_model}}  {passing}/{total} ({pct:3}%)"
        for g in groups:
            g_items = [qr for qr in quality if qr["group"] == g]
            g_pass = sum(1 for qr in g_items if qr["verdict"] == "PASS")
            g_total = len(g_items)
            row += f"  {g_pass}/{g_total:>7}"
        print(row)

    # Failure detail
    print()
    print("  Failed / Partial tests:")
    print("  " + "─" * 40)
    any_failure = False
    for r in all_results:
        quality = r.get("quality", [])
        failures = [
            qr for qr in quality if qr["verdict"] in ("FAIL", "PARTIAL", "ERROR")
        ]
        if not failures:
            continue
        any_failure = True
        display = f"{r['server']}/{r['runtime']}  {r['model_id']}"
        print(f"\n  {display}")
        for qr in failures:
            sym = {"FAIL": "✗", "PARTIAL": "~", "ERROR": "!"}.get(qr["verdict"], "?")
            print(f"    {sym} [{qr['group']:<12}] {qr['id']:<45}  {qr['reason'][:50]}")
    if not any_failure:
        print("  All models passed all quality tests  ✓")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = (
        sys.argv[1] if len(sys.argv) > 1 else os.path.join(script_dir, "config.yaml")
    )
    fixtures_path = os.path.join(
        os.path.dirname(os.path.abspath(config_path)), "fixtures.json"
    )

    if not os.path.isfile(config_path):
        sys.exit(f"Error: config file not found: {config_path}")

    targets = load_config(config_path)
    if not targets:
        sys.exit("Error: no servers/runtimes defined in config.yaml")

    total_models = sum(len(t["models"]) for t in targets)

    # Quality benchmark setup
    fixtures: list[dict] = []
    if RUN_QUALITY and os.path.isfile(fixtures_path):
        fixtures = load_fixtures(fixtures_path)

    ev_cfg: dict | None = None
    if RUN_QUALITY and fixtures:
        ev_cfg = load_evaluator_config(config_path)

    W = 82
    print()
    print("═" * W)
    print(f"  Multi-Server Benchmark  (speed + quality)")
    print("═" * W)
    print(f"  Config      : {config_path}")
    print(f"  Targets     : {len(targets)} runtime(s)   {total_models} model(s) total")
    print(f"  Speed       : {len(HISTORY)} turns → warm KV cache → new question")
    print(f"              : {GEN_TOKENS} tokens/response   {TRIALS} hot trials")
    print(
        f"  Quality     : {len(fixtures)} fixtures"
        if fixtures
        else "  Quality     : disabled"
    )
    if ev_cfg:
        ev_ready = _wait_ready(
            ev_cfg["host"], ev_cfg["port"], ev_cfg["token"], timeout=3
        )
        ev_status = f"{ev_cfg['model']}  @ {ev_cfg['host']}:{ev_cfg['port']}"
        ev_status += (
            "  ✓"
            if ev_ready
            else "  ✗ unreachable (will fall back to mechanical checks)"
        )
        print(f"  Evaluator   : {ev_status}")
    elif fixtures:
        print(f"  Evaluator   : not configured — mechanical checks only")
    print(f"  Est. prompt : ~{ESTIMATED_PROMPT_TOKENS} tokens")

    all_results: list[dict] = []

    for tgt in targets:
        server_name = tgt["server"]
        runtime_name = tgt["runtime"]
        host = tgt["host"]
        port = tgt["port"]
        token = tgt["token"]
        models = tgt["models"]
        prefix = f"{server_name}/{runtime_name}"

        print()
        print("═" * W)
        print(f"  Server: {server_name}   Runtime: {runtime_name}   ({host}:{port})")
        print("═" * W)

        if not _wait_ready(host, port, token, timeout=5):
            print(f"  SKIP — {host}:{port} not reachable.")
            continue

        available = _get_models(host, port, token)
        print(f"  Available models ({len(available)}):")
        for mid in available:
            print(f"    • {mid}")

        for i, target_model in enumerate(models, 1):
            model_id = match_model_id(available, target_model) or target_model
            label = f"{prefix}  {target_model}"

            print(f"\n{'─' * W}")
            print(f"  [{i}/{len(models)}] {target_model}")
            if model_id != target_model:
                print(f"  Matched model ID: {model_id}")

            # ── Speed benchmark ──────────────────────────────────────────────
            result = run_speed_benchmark(
                host, port, token, model_id, runtime_name, label, W
            )
            if not result:
                continue

            result["server"] = server_name
            result["runtime"] = runtime_name

            # ── Quality benchmark (model still loaded from speed benchmark) ──
            if fixtures:
                result["quality"] = run_quality_benchmark(
                    host,
                    port,
                    token,
                    model_id,
                    runtime_name,
                    fixtures,
                    ev_cfg,
                    W,
                )

            all_results.append(result)

    # ── Final results ─────────────────────────────────────────────────────────
    print_speed_results(all_results, W)
    print_quality_results(all_results, W)

    print()
    print("═" * W)
    print()


if __name__ == "__main__":
    main()
