"""
LFM2.5-Audio — OpenAI-compatible audio chat completions server (mlx-audio backend).

Implements POST /v1/chat/completions accepting audio input_audio parts and
returning base64-encoded WAV audio in the response, matching the format expected
by the Rust voicebot's LFMModel client.

Usage:
    python provider/server.py

Environment variables:
    MODEL_ID         HuggingFace model ID or local path
                     (default: mlx-community/LFM2.5-Audio-1.5B-4bit)
    HOST             bind address  (default: 0.0.0.0)
    PORT             bind port     (default: 8000)
    API_KEY          optional bearer token to require (default: none)

    MAX_NEW_TOKENS   default generation limit (default: 2048)
"""

import asyncio
import base64
import io
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Literal, Optional, Union

import mlx.core as mx
import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from mlx_audio.sts.models.lfm_audio import (
    LFM2AudioModel,
    LFM2AudioProcessor,
    ChatState,
    LFMModality,
)
from mlx_audio.sts.models.lfm_audio.model import AUDIO_EOS_TOKEN
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = os.environ.get("MODEL_ID", "mlx-community/LFM2.5-Audio-1.5B-4bit")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
API_KEY = os.environ.get("API_KEY", "")

MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "2048"))

model: Optional[LFM2AudioModel] = None
processor: Optional[LFM2AudioProcessor] = None


# ── Startup / shutdown ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor
    logger.info("Loading %s …", MODEL_ID)
    model = LFM2AudioModel.from_pretrained(MODEL_ID)
    processor = LFM2AudioProcessor.from_pretrained(MODEL_ID)
    logger.info("LFM2.5-Audio ready  sample_rate=%d Hz", model.sample_rate)
    yield
    del model, processor
    logger.info("Model unloaded")


app = FastAPI(title="LFM2.5-Audio provider", lifespan=lifespan)


# ── OpenAI request / response schemas ─────────────────────────────────────────

class InputAudio(BaseModel):
    data: str          # base64-encoded audio file
    format: str = "wav"


class ContentPart(BaseModel):
    type: Literal["text", "input_audio"]
    text: Optional[str] = None
    input_audio: Optional[InputAudio] = None


class Message(BaseModel):
    role: str
    content: Union[str, List[ContentPart]]


class AudioOutputConfig(BaseModel):
    voice: str = "alloy"
    format: str = "wav"


class ChatCompletionRequest(BaseModel):
    model: str = "lfm-2.5-audio"
    messages: List[Message]
    modalities: List[str] = ["text", "audio"]
    audio: Optional[AudioOutputConfig] = None
    temperature: float = 0.7
    max_tokens: int = MAX_NEW_TOKENS


# ── Audio helpers ──────────────────────────────────────────────────────────────

def _read_audio_part(part: InputAudio) -> tuple[mx.array, int]:
    """Decode a base64 audio part → (mx.array float32 mono, sample_rate)."""
    raw = base64.b64decode(part.data)
    audio_np, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)  # mix down to mono
    return mx.array(audio_np), sr


def _encode_wav(audio_np: np.ndarray, sample_rate: int) -> str:
    """Encode a float32 numpy array as a base64 16-bit PCM WAV string."""
    buf = io.BytesIO()
    sf.write(buf, audio_np, sample_rate, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


# ── Inference (runs in a thread pool — MLX is not async) ──────────────────────

def _infer(
    messages: List[Message],
    temperature: float,
    max_tokens: int,
) -> tuple[np.ndarray, int, Optional[str]]:
    """
    Run LFM2.5-Audio speech-to-speech inference.

    Collects all audio tokens during generation then decodes them in one call
    with processor.decode_audio — the method available in the current mlx-audio
    release.  (The HuggingFace model card documents decode_with_detokenizer for
    chunked streaming, but that method is not yet shipped in the library.)

    Returns:
        waveform     — float32 numpy array (mono, model.sample_rate Hz)
        sample_rate  — model output sample rate (24 000 Hz)
        output_text  — assistant text response (may be None or empty)
    """
    chat = ChatState(processor)

    for msg in messages:
        chat.new_turn(msg.role)

        if isinstance(msg.content, str):
            chat.add_text(msg.content)
        else:
            for part in msg.content:
                if part.type == "text" and part.text:
                    chat.add_text(part.text)
                elif part.type == "input_audio" and part.input_audio:
                    audio_mx, sr = _read_audio_part(part.input_audio)
                    # add_audio handles resampling to the model's expected rate
                    chat.add_audio(audio_mx, sample_rate=sr)

        chat.end_turn()

    # Open the assistant turn before generation
    chat.new_turn("assistant")

    # ── Generation — collect all tokens ────────────────────────────────────────
    text_pieces: list[str] = []
    audio_tokens: list[mx.array] = []

    for token, modality in model.generate_interleaved(
        **dict(chat),
        max_new_tokens=max_tokens,
        temperature=temperature if temperature > 0 else None,
    ):
        mx.eval(token)

        if modality == LFMModality.TEXT:
            text_pieces.append(processor.decode_text(token[None]))

        elif modality == LFMModality.AUDIO_OUT:
            if token[0].item() == AUDIO_EOS_TOKEN:
                break
            audio_tokens.append(token)

    # ── Decode audio ────────────────────────────────────────────────────────────
    # Stack collected tokens → (1, 8, T) then decode to waveform.
    # Shape mirrors the TTS example from the model card:
    #   mx.stack(tokens, axis=0)[None, :].transpose(0, 2, 1) → (1, 8, T)
    if audio_tokens:
        codes = mx.stack(audio_tokens, axis=0)[None, :].transpose(0, 2, 1)  # (1, 8, T)
        waveform_mx = processor.decode_audio(codes)
        waveform = np.array(waveform_mx[0])
    else:
        waveform = np.zeros(model.sample_rate, dtype=np.float32)
    output_text = "".join(text_pieces) or None

    logger.info(
        "Generated %.2f s of audio, text=%r",
        len(waveform) / model.sample_rate,
        output_text,
    )
    return waveform, model.sample_rate, output_text


# ── Middleware ─────────────────────────────────────────────────────────────────

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if API_KEY and request.url.path.startswith("/v1"):
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {API_KEY}":
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return await call_next(request)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded yet")

    logger.info(
        "Request: %d messages, temp=%.2f, max_tokens=%d",
        len(req.messages), req.temperature, req.max_tokens,
    )

    try:
        waveform, sr, output_text = await asyncio.to_thread(
            _infer, req.messages, req.temperature, req.max_tokens
        )
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(500, str(exc))

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": output_text,
                    "audio": {
                        "id": f"audio-{uuid.uuid4().hex[:12]}",
                        "data": _encode_wav(waveform, sr),
                        "transcript": None,  # set by caller if ASR pass is run
                    },
                },
                "finish_reason": "stop",
            }
        ],
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "loaded": model is not None,
        "sample_rate": model.sample_rate if model else None,
    }


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
