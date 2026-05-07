#!/usr/bin/env python3
"""
eval_prompts.py — Prompt Evaluator for Jarvis Voicebot

Runs quality benchmark against multiple system prompts to find the best one.
Uses the same fixture system as bench-models.py but iterates over prompts
instead of models.

Usage:
    python3 scripts/eval_prompts.py [config.yaml]

    Default config path: scripts/config.yaml
    Default prompts:   scripts/prompts.json
    Default fixtures: scripts/fixtures.json

Env vars:
    EVAL_TRIALS    measurement trials (default 1)
    EVAL_PROMPT    single prompt ID to test (default: all)
"""

import http.client
import json
import os
import re
import ssl
import sys
import time
import yaml
from statistics import mean

# Default system prompt (fallback when no prompt provided)
DEFAULT_SYSTEM_PROMPT = (
    "Eres un asistente de voz útil y conciso. "
    "Responde siempre en el mismo idioma que el usuario. "
    "Habla de forma natural y directa, sin listas ni formato markdown. "
    "Empieza siempre con la respuesta directa, sin preámbulos. "
    "Por defecto, limita tus respuestas a 2-3 frases cortas."
)

# Tool definitions - should match voicebot tools
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
            "name": "run_agent",
            "description": "Delegates a complex multi-step task to an external agent (HERMES). The agent works in background and the result arrives shortly. Only available when AGENT_COMMAND is configured.",
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

TRIALS = int(os.environ.get("EVAL_TRIALS", "1"))
SINGLE_PROMPT = os.environ.get("EVAL_PROMPT", None)


def _parse_host(host_url: str) -> str:
    for prefix in ("https://", "http://"):
        if host_url.startswith(prefix):
            return host_url[len(prefix) :]
    return host_url


def _auth_headers(token: str) -> dict:
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _get_api_path(provider: str = "", host: str = "") -> str:
    """Get the API path for the given provider."""
    # Auto-detect from host if provider not specified
    if not provider:
        if "opencode.ai" in host or "zen" in host.lower():
            return "/zen/v1/chat/completions"
        return "/v1/chat/completions"
    if provider == "zen":
        return "/zen/v1/chat/completions"
    return "/v1/chat/completions"


def load_prompts(path: str) -> list[dict]:
    """Load prompts.json → list of prompt configs."""
    with open(path) as f:
        data = json.load(f)
    return [p for p in data if "id" in p and "system_prompt" in p]


def load_fixtures(path: str) -> list[dict]:
    """Load fixtures.json, filtering out _comment-only entries."""
    with open(path) as f:
        data = json.load(f)
    return [fx for fx in data if "id" in fx]


def load_evaluator_config(path: str) -> dict | None:
    """Parse the evaluator section from config.yaml."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    ev = cfg.get("evaluator")
    if not ev:
        return None
    runtime = ev.get("runtime", "")
    return {
        "host": _parse_host(ev["host"]),
        "port": int(ev["port"]),
        "token": ev.get("token", ""),
        "model": ev["model"],
        "runtime": runtime,
        "provider": ev.get("provider", ""),
        "temperature": float(ev.get("temperature", 0.0)),
        "max_tokens": int(ev.get("max_tokens", 512)),
    }


def load_model_config(path: str) -> dict | None:
    """Parse the first server/runtime/model from config.yaml."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    servers = cfg.get("servers", {})
    if not servers:
        return None
    first_server = list(servers.values())[0]
    host = _parse_host(first_server.get("host", "http://127.0.0.1"))
    runtimes = first_server.get("runtimes", {})
    if not runtimes:
        return None
    first_runtime = list(runtimes.values())[0]
    port = int(first_runtime.get("port", 8000))
    token = first_runtime.get("token", "")
    models = first_runtime.get("models", [])
    if not models:
        return None
    return {
        "host": host,
        "port": port,
        "token": token,
        "model": models[0],
        "runtime": list(runtimes.keys())[0],
    }


# ── HTTP helpers ──────────────────────────────────────────────────────────────


def _post_stream(host, port, token, payload):
    """POST to /v1/chat/completions with stream=True. Yields SSE content lines."""
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


def _post_blocking(host, port, token, payload, provider="", model="") -> dict:
    """POST to /v1/chat/completions with stream=False. Returns parsed JSON."""
    payload = {**payload, "stream": False}
    body = json.dumps(payload).encode()
    api_path = _get_api_path(provider, host)

    # Use HTTPS for OpenCode Zen (port 443)
    use_ssl = port == 443
    if use_ssl:
        context = ssl.create_default_context()
        conn = http.client.HTTPSConnection(host, port, timeout=120, context=context)
    else:
        conn = http.client.HTTPConnection(host, port, timeout=120)

    try:
        conn.request(
            "POST", api_path, body=body, headers=_auth_headers(token)
        )
        resp = conn.getresponse()
        raw = resp.read()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {raw[:300].decode()}")
        return json.loads(raw)
    finally:
        conn.close()


def _wait_ready(host, port, token, timeout=5) -> bool:
    """Return True if the server responds before timeout."""
    # For OpenCode Zen, skip the check (assume available if configured)
    if "opencode.ai" in host or port == 443:
        return True

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


def _no_think_prefix(runtime_name: str) -> str:
    return "/no_think\n\n" if "llama" in runtime_name.lower() else ""


def _thinking_off_fields() -> dict:
    return {
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "thinking": {"type": "disabled"},
        "think": False,
    }


# ── Quality benchmark — fixture execution ─────────────────────────────────────


def _normalize_tool_calls(raw: list) -> list[dict]:
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


def run_fixture(host, port, token, model_id, runtime_name, system_prompt, fixture) -> tuple[str, list]:
    """Execute one fixture against the model with given system prompt."""
    no_think = _no_think_prefix(runtime_name)
    system_content = no_think + system_prompt

    messages = [{"role": "system", "content": system_content}]

    for msg in fixture["messages"]:
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
                f"/{pat[:50]}/: {'not found' if passed else f'matched {snippet}'}",
            )
        )

    for s in ev.get("forbidden_strings", []):
        passed = s.lower() not in text.lower()
        checks.append(
            (
                "forbidden_string",
                passed,
                f"{s!r}: {'absent' if passed else 'found'}",
            )
        )

    for pat in ev.get("required_patterns", []):
        m = re.search(pat, text, re.MULTILINE)
        passed = m is not None
        checks.append(
            (
                "required_pattern",
                passed,
                f"/{pat[:50]}/: {'matched' if passed else 'not found'}",
            )
        )

    for s in ev.get("required_strings", []):
        passed = s.lower() in text.lower()
        checks.append(
            (
                "required_string",
                passed,
                f"{s!r}: {'found' if passed else 'missing'}",
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
                f"tool {tool!r}: {'called' if passed else 'NOT called'}",
            )
        )

    for tool in ev.get("must_not_call_tools", []):
        passed = tool not in called
        checks.append(
            (
                "must_not_call_tool",
                passed,
                f"tool {tool!r}: {'absent' if passed else 'called'}",
            )
        )

    if ev.get("no_tool_called") is True:
        passed = len(tool_calls) == 0
        checks.append(
            (
                "no_tool_called",
                passed,
                f"no tools called: {'ok' if passed else f'{list(called)}'}",
            )
        )

    if ev.get("any_tool_called") is True:
        passed = len(tool_calls) > 0
        checks.append(
            (
                "any_tool_called",
                passed,
                f"at least one tool called: {'ok' if passed else 'no'}",
            )
        )

    for tool_name, expected_arg in ev.get("tool_args_contain", {}).items():
        matching = [tc for tc in tool_calls if tc["function"]["name"] == tool_name]
        if not matching:
            checks.append(
                (
                    "tool_args_contain",
                    False,
                    f"tool {tool_name!r} not called (needed {expected_arg!r})",
                )
            )
        else:
            args_str = matching[0]["function"].get("arguments", "")
            passed = expected_arg.lower() in args_str.lower()
            checks.append(
                (
                    "tool_args_contain",
                    passed,
                    f"tool {tool_name!r} arg {expected_arg!r}: {'ok' if passed else f'args={args_str!r}'}",
                )
            )

    return checks


# ── Quality benchmark — LLM evaluator ────────────────────────────────────────


def _extract_content(resp: dict) -> str:
    """Extract content from response, handling different API formats."""
    # Standard OpenAI format
    if "choices" in resp:
        msg = resp.get("choices", [{}])[0].get("message", {})
        return msg.get("content", "") or ""
    # OpenCode Zen / alternative format
    if "output" in resp:
        return resp.get("output", "") or ""
    return ""


def call_evaluator(
    ev_cfg: dict,
    fixture: dict,
    fixture_messages: list,
    text: str,
    tool_calls: list,
    mech: list,
) -> dict:
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

    failed_mechs = [detail for _, passed, detail in mech if not passed]
    if failed_mechs:
        mech_note = "\nMechanical checks FAILED:\n" + "\n".join(
            f"  - {d}" for d in failed_mechs
        )
    elif mech:
        mech_note = f"\nAll {len(mech)} mechanical checks passed."
    else:
        mech_note = ""

    provider = ev_cfg.get("runtime", "")
    system_msg = (
        "You are a strict quality evaluator for a voice assistant called Jarvis. "
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

    # Build payload - Zen may not need "model" field
    payload = {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": ev_cfg["max_tokens"],
        "temperature": ev_cfg["temperature"],
    }
    if ev_cfg.get("model"):
        payload["model"] = ev_cfg["model"]

    resp = _post_blocking(
        ev_cfg["host"],
        ev_cfg["port"],
        ev_cfg["token"],
        payload,
        provider=provider,
    )

    raw = _extract_content(resp)

    clean = raw.lstrip("`")
    if clean.startswith("json"):
        clean = clean[4:]
    clean = clean.rstrip("`").strip()
    m = re.search(r"\{.*\}", clean, re.DOTALL)
    if m:
        clean = m.group()
    try:
        result = json.loads(clean)
        return {
            "verdict": str(result.get("verdict", "PARTIAL")).upper(),
            "reason": str(result.get("reason", ""))[:200],
        }
    except json.JSONDecodeError:
        pass

    lower = raw.lower()
    if "pass" in lower and "fail" not in lower:
        return {"verdict": "PASS", "reason": raw[:200]}
    elif "fail" in lower:
        return {"verdict": "FAIL", "reason": raw[:200]}
    return {"verdict": "PARTIAL", "reason": raw[:200]}


# ── Quality benchmark — runner ────────────────────────────────────────────


def run_quality_benchmark(
    host: str,
    port: int,
    token: str,
    model_id: str,
    runtime_name: str,
    system_prompt: str,
    fixtures: list[dict],
    ev_cfg: dict | None,
    W: int,
) -> list[dict]:
    total = len(fixtures)
    pad = len(str(total))

    print(f"\n{'─' * W}")
    print(f"  Quality Phase 1/2 — collecting {total} fixture responses")

    collected: list[tuple] = []
    phase1_start = time.perf_counter()
    for i, fx in enumerate(fixtures, 1):
        label = f"[{i:{pad}}/{total}] {fx['id']}"
        print(f"    {label:<52}", end="", flush=True)
        t0 = time.perf_counter()
        try:
            text, tcs = run_fixture(
                host, port, token, model_id, runtime_name, system_prompt, fx
            )
            lat_ms = (time.perf_counter() - t0) * 1000
            tag = (
                ("tools:" + "+".join(tc["function"]["name"] for tc in tcs))
                if tcs
                else f"{len(text)} chars"
            )
            print(f"ok  ({tag})  {lat_ms:.0f}ms")
            collected.append((fx, text, tcs, None, lat_ms))
        except Exception as e:
            lat_ms = (time.perf_counter() - t0) * 1000
            print(f"ERROR  ({e})")
            collected.append((fx, "", [], str(e), lat_ms))
    phase1_elapsed = time.perf_counter() - phase1_start
    print(f"\n  Phase 1 total: {phase1_elapsed:.1f}s   avg: {phase1_elapsed * 1000 / total:.0f}ms")

    ev_label = ev_cfg["model"] if ev_cfg else "mechanical only"
    print(f"\n  Quality Phase 2/2 — evaluating  [evaluator: {ev_label}]")

    ev_available = bool(
        ev_cfg
        and _wait_ready(ev_cfg["host"], ev_cfg["port"], ev_cfg["token"], timeout=5)
    )

    results: list[dict] = []
    for i, (fx, text, tcs, error, lat_ms) in enumerate(collected, 1):
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
                    "latency_ms": lat_ms,
                }
            )
            continue

        mech = run_mechanical_checks(fx, text, tcs)
        failed = [detail for _, p, detail in mech if not p]

        if ev_available:
            try:
                ev_out = call_evaluator(
                    ev_cfg, fx, fx["messages"], text, tcs, mech
                )
                verdict = ev_out["verdict"]
                reason = ev_out["reason"]
            except Exception as e:
                verdict = "FAIL" if failed else "PASS"
                reason = f"(evaluator error: {e})"
        else:
            verdict = "FAIL" if failed else "PASS"
            reason = "; ".join(failed[:2]) if failed else "(no evaluator)"

        if failed and verdict == "PASS":
            verdict = "FAIL"
            reason = "; ".join(failed[:2])

        sym = {"PASS": "OK", "FAIL": "XX", "PARTIAL": "--", "ERROR": "!!"}.get(
            verdict, "?"
        )
        print(f"{sym} {verdict:<7}  {reason[:55]}")

        results.append(
            {
                "id": fid,
                "group": group,
                "verdict": verdict,
                "reason": reason,
                "mech_pass": not bool(failed),
                "preview": text[:100],
                "latency_ms": lat_ms,
            }
        )

    passing = sum(1 for r in results if r["verdict"] == "PASS")
    lats = [r["latency_ms"] for r in results]
    avg_lat = mean(lats) if lats else 0.0
    print(
        f"\n  Quality score: {passing}/{total}  ({passing * 100 // total if total else 0}%)"
        f"   avg latency: {avg_lat:.0f}ms"
    )
    return results


# ── Results display ───────────────────────────────────────────────────────────


def print_prompt_results(all_results: list, W: int):
    print()
    print("═" * W)
    print("  PROMPT EVALUATION RESULTS")
    print("═" * W)
    print()

    col_prompt = 50
    col_score = 10
    col_lat = 10

    # Collect group names
    groups: list[str] = []
    seen = set()
    for r in all_results:
        for qr in r.get("quality", []):
            g = qr["group"]
            if g not in seen:
                groups.append(g)
                seen.add(g)

    hdr = f"  {'Prompt ID':<{col_prompt}}  {'Score':>{col_score}}  {'Avg lat':>{col_lat}}"
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
        lats = [qr["latency_ms"] for qr in quality if "latency_ms" in qr]
        avg_lat_ms = mean(lats) if lats else float("nan")
        display = r["prompt_id"]
        row = f"  {display:<{col_prompt}}  {passing}/{total} ({pct:3}%)  {avg_lat_ms:>{col_lat}.0f}ms"
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
        print(f"\n  Prompt: {r['prompt_id']}")
        for qr in failures:
            sym = {"FAIL": "XX", "PARTIAL": "--", "ERROR": "!!"}.get(qr["verdict"], "?")
            print(f"    {sym} [{qr['group']:<12}] {qr['id']:<45}  {qr['reason'][:50]}")

    if not any_failure:
        print("  All prompts passed all quality tests  OK")


# ── Main ───────────────────────────────────────────────────────────��─��────────


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = (
        sys.argv[1] if len(sys.argv) > 1 else os.path.join(script_dir, "config.yaml")
    )
    prompts_path = os.path.join(script_dir, "prompts.json")
    fixtures_path = os.path.join(script_dir, "fixtures.json")
    base_dir = os.path.dirname(script_dir)

    if not os.path.isfile(config_path):
        sys.exit(f"Error: config file not found: {config_path}")
    if not os.path.isfile(prompts_path):
        sys.exit(f"Error: prompts file not found: {prompts_path}")
    if not os.path.isfile(fixtures_path):
        sys.exit(f"Error: fixtures file not found: {fixtures_path}")

    prompts = load_prompts(prompts_path)
    if not prompts:
        sys.exit("Error: no prompts found in prompts.json")

    if SINGLE_PROMPT:
        prompts = [p for p in prompts if p["id"] == SINGLE_PROMPT]
        if not prompts:
            sys.exit(f"Error: prompt '{SINGLE_PROMPT}' not found")

    fixtures = load_fixtures(fixtures_path)
    if not fixtures:
        sys.exit("Error: no fixtures found in fixtures.json")

    model_cfg = load_model_config(config_path)
    if not model_cfg:
        sys.exit("Error: no server/runtime/model found in config.yaml")

    ev_cfg = load_evaluator_config(config_path)
    if not ev_cfg:
        print("Warning: evaluator not configured — using mechanical checks only")

    W = 82
    print()
    print("═" * W)
    print("  Prompt Evaluator for Jarvis Voicebot")
    print("═" * W)
    print(f"  Config     : {config_path}")
    print(f"  Prompts   : {len(prompts)} prompt(s)")
    print(f"  Fixtures : {len(fixtures)} test(s)")
    print(f"  Model    : {model_cfg['model']} @ {model_cfg['host']}:{model_cfg['port']}")
    if ev_cfg:
        ev_ready = _wait_ready(
            ev_cfg["host"], ev_cfg["port"], ev_cfg["token"], timeout=3
        )
        ev_status = f"{ev_cfg['model']} @ {ev_cfg['host']}:{ev_cfg['port']}"
        ev_status += " OK" if ev_ready else " (unreachable)"
        print(f"  Evaluator : {ev_status}")

    if not _wait_ready(
        model_cfg["host"], model_cfg["port"], model_cfg["token"], timeout=5
    ):
        sys.exit(f"Error: model server not reachable")

    all_results: list[dict] = []

    for prompt in prompts:
        prompt_id = prompt["id"]
        prompt_name = prompt.get("name", prompt_id)
        system_prompt = prompt.get("system_prompt", DEFAULT_SYSTEM_PROMPT)

        print()
        print("═" * W)
        print(f"  Prompt: {prompt_id}")
        if prompt_name != prompt_id:
            print(f"  Name:  {prompt_name}")
        print("═" * W)

        # Truncate prompt for display
        prompt_preview = system_prompt[:200].replace("\n", "\\n")
        print(f"  System: {prompt_preview}...")

        quality = run_quality_benchmark(
            model_cfg["host"],
            model_cfg["port"],
            model_cfg["token"],
            model_cfg["model"],
            model_cfg["runtime"],
            system_prompt,
            fixtures,
            ev_cfg,
            W,
        )

        all_results.append(
            {
                "prompt_id": prompt_id,
                "prompt_name": prompt_name,
                "system_prompt": system_prompt,
                "quality": quality,
            }
        )

    print_prompt_results(all_results, W)

    print()
    print("═" * W)
    print()


if __name__ == "__main__":
    main()