# Recommended LLM Parameters for Voicebot (Jarvis)

Optimized for real-time voice conversation with Qwen3.5-35B-A3B (MoE, 3B active params).
The goal: the user feels like talking to Jarvis (Iron Man) — fast, natural, concise responses.

## Server-side parameters

| Parameter | llama.cpp | mlx-lm | Rationale |
|-----------|-----------|--------|-----------|
| CTX Window | `--ctx-size 8192` | 3GB cache (~8K) | Voice turns are short. 8K holds ~50 turns + summary. Summarization triggers at 75%. |
| Max Tokens | 300 (via client) | `--max-tokens 300` | Jarvis replies in 2-3 sentences. 300 tokens ~ 200 words. Keeps TTS queue short. |
| Temperature | `--temp 0.5` | `--temp 0.5` | 0.3 is too robotic for Jarvis personality. 0.5 adds natural variety without hallucination. |
| Top P | `--top-p 0.90` | 0.90 (client-side) | Tighter nucleus than 0.95. Safety net — min_p does the heavy lifting. |
| Top K | `--top-k 40` | 40 (client-side) | Caps candidate tokens. Prevents obscure word choices in voice output. |
| Min P | `--min-p 0.05` | 0.05 (client-side) | Most impactful sampler for Qwen3. Prunes tokens below 5% of top probability. |
| Repetition Penalty | `--repeat-penalty 1.1` | 1.1 (client-side) | Prevents sentence loops. 1.1 is the sweet spot — higher values cause unnatural phrasing. |
| Presence Penalty | 0.0 (disabled) | 0.0 (disabled) | Not needed with repetition_penalty. Too aggressive for Spanish (articles/prepositions repeat naturally). |
| TTL | 0 (never expire) | N/A (persists) | Mono-user — keep KV-cache alive indefinitely. |
| Thinking Mode | `--reasoning-budget 0 --reasoning-format none` | `--chat-template-args '{"enable_thinking": false}'` | Qwen3.5 wastes tokens on internal reasoning. Disabling makes tool calls reliable. |

## Client-side parameters (per-request)

| Parameter | Streaming (conversation) | complete() (summarization) | complete_short() (extraction) |
|-----------|--------------------------|----------------------------|-------------------------------|
| temperature | 0.5 (from config) | 0.3 (hardcoded) | 0.1 (hardcoded) |
| max_tokens | 300 (from config) | 512 (hardcoded) | 256 (hardcoded) |
| top_p | 0.90 | not sent | not sent |
| top_k | 40 (mlx only) | not sent | not sent |
| min_p | 0.05 (mlx only) | not sent | not sent |
| repetition_penalty | 1.1 (mlx only) | not sent | not sent |
| cache_prompt | true (llama only) | false | false |
| enable_thinking | false | false | false |

Note: For llama.cpp, sampling params (temp, top_p, top_k, min_p, repeat_penalty) are set
server-side and apply to all requests. The client only sends them for mlx-lm which requires
per-request configuration.

## llama.cpp launch command

```bash
llama-server \
    --model models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
    --ctx-size 8192 \
    --n-gpu-layers 99 \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    --flash-attn on \
    --spec-type ngram-mod \
    --draft-max 12 \
    --mlock \
    -b 2048 \
    --ubatch-size 2048 \
    --parallel 1 \
    --repeat-penalty 1.1 \
    --temp 0.5 \
    --top-p 0.90 \
    --top-k 40 \
    --min-p 0.05 \
    --reasoning-budget 0 \
    --reasoning-format none
```

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

## .env configuration

```
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
5. **parallel = 1** — Frees VRAM for mono-user voicebot (llama.cpp only).
6. **top_k = 40** — Extra safety net, negligible performance cost.
7. **top_p = 0.90** — Minor tightening, less impactful with min_p active.
8. **Tool result truncation** — Prevent context blowout from large tool outputs (1024 token limit recommended).

## Sampler order

llama.cpp applies samplers in this order by default: `top_k -> tfs_z -> typ_p -> top_p -> min_p -> temperature`.
This means min_p prunes after top_p, which is the correct behavior — top_p provides a soft ceiling,
min_p does the fine-grained quality filtering.
