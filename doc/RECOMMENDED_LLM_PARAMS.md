# Recommended LLM Parameters for Voicebot (Jarvis)

Optimized for real-time voice conversation with Qwen3.5-35B-A3B (MoE, 3B active params).
The goal: the user feels like talking to Jarvis (Iron Man) — fast, natural, concise responses.

## Server-side parameters

| Parameter | mlx-lm | oMLX | Rationale |
|-----------|--------|------|-----------|
| CTX Window | 3GB cache (`--prompt-cache-bytes`) | server-managed | Voice turns are short. 8K holds ~50 turns + summary. Summarization triggers at 75%. |
| Max Tokens | `--max-tokens 300` | per-request | Jarvis replies in 2-3 sentences. 300 tokens ~ 200 words. Keeps TTS queue short. |
| Temperature | `--temp 0.5` | per-request | 0.3 is too robotic for Jarvis personality. 0.5 adds natural variety without hallucination. |
| Top P | 0.90 (client-side) | 0.90 (client-side) | Tighter nucleus than 0.95. Safety net — min_p does the heavy lifting. |
| Top K | 40 (client-side) | 40 (client-side) | Caps candidate tokens. Prevents obscure word choices in voice output. |
| Min P | 0.05 (client-side) | 0.05 (client-side) | Most impactful sampler for Qwen3. Prunes tokens below 5% of top probability. |
| Repetition Penalty | 1.1 (client-side) | 1.1 (client-side) | Prevents sentence loops. 1.1 is the sweet spot — higher values cause unnatural phrasing. |
| Presence Penalty | 0.0 (disabled) | 0.0 (disabled) | Not needed with repetition_penalty. Too aggressive for Spanish (articles/prepositions repeat naturally). |
| Thinking Mode | `--chat-template-args '{"enable_thinking": false}'` | per-request | Qwen3.5 wastes tokens on internal reasoning. Disabling makes tool calls reliable. |

## Client-side parameters (per-request)

All sampling params are sent per-request by the client (mlx-lm and oMLX require per-request configuration).

| Parameter | Streaming (conversation) | complete() (summarization) | complete_short() (extraction) |
|-----------|--------------------------|----------------------------|-------------------------------|
| temperature | 0.5 (from config) | 0.3 (hardcoded) | 0.1 (hardcoded) |
| max_tokens | 300 (from config) | 512 (hardcoded) | 256 (hardcoded) |
| top_p | 0.90 | not sent | not sent |
| top_k | 40 | not sent | not sent |
| min_p | 0.05 | not sent | not sent |
| repetition_penalty | 1.1 | not sent | not sent |
| enable_thinking | false | false | false |

## mlx-lm launch command

```bash
mlx_lm server \
    --model mlx-community/Qwen3.5-35B-A3B \
    --prompt-cache-size 1 \
    --prompt-cache-bytes $((3 * 1024 * 1024 * 1024)) \
    --prefill-step-size 512 \
    --max-tokens 300 \
    --temp 0.5 \
    --chat-template-args '{"enable_thinking": false}'
```

## oMLX launch command

```bash
omlx serve \
    --model-dir ~/models \
    --host 127.0.0.1 \
    --port 8001
```

## .env configuration

```
LLM_URL=http://127.0.0.1:8000
LLM_CONTEXT_TOKENS=8192
LLM_MAX_TOKENS=300
LLM_TEMPERATURE=0.5
LLM_SUMMARY_KEEP_TURNS=6
```

## Parameter priority (impact ranking)

1. **min_p = 0.05** — Primary quality control for Qwen3. Dynamically prunes low-probability tokens.
2. **temperature = 0.5** — Natural Jarvis personality without hallucination.
3. **max_tokens = 300** — Faster TTS pipeline, concise responses.
4. **ctx_size = 8192** — Saves VRAM, faster prefill. Summarization handles long conversations.
5. **top_k = 40** — Extra safety net, negligible performance cost.
6. **top_p = 0.90** — Minor tightening, less impactful with min_p active.
7. **Tool result truncation** — Prevent context blowout from large tool outputs (1024 token limit recommended).
