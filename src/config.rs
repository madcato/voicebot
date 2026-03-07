use anyhow::{Context, Result};
use std::env;

#[derive(Debug, Clone)]
pub struct Config {
    // ── Audio input ──────────────────────────────────────────────────────────
    /// Microphone sample rate (default 16000 — required by Silero VAD)
    pub sample_rate: u32,
    pub channels: u16,
    pub chunk_ms: u32,
    pub audio_device: Option<String>,
    pub audio_output_device: Option<String>,
    pub list_devices: bool,

    // ── Language ─────────────────────────────────────────────────────────────
    /// "es" (default) or "en"
    pub language: String,

    // ── STT ──────────────────────────────────────────────────────────────────
    /// Path to whisper.cpp GGML model file (.bin)
    pub whisper_model: String,

    // ── LLM ──────────────────────────────────────────────────────────────────
    /// llama.cpp server base URL
    pub llm_url: String,
    /// KV-cache slot ID (0 for single-user)
    pub llm_slot_id: u8,
    /// Max tokens per response
    pub llm_max_tokens: u32,
    pub llm_system_prompt: String,
    pub llm_temperature: f32,

    // ── TTS ──────────────────────────────────────────────────────────────────
    /// Path to Piper ONNX config JSON for Spanish
    pub piper_model_es: String,
    /// Path to Piper ONNX config JSON for English
    pub piper_model_en: String,

    // ── Persistence ───────────────────────────────────────────────────────────
    pub db_path: String,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            // Audio
            sample_rate: env::var("AUDIO_SAMPLE_RATE")
                .unwrap_or_else(|_| "16000".to_string())
                .parse()
                .context("Invalid AUDIO_SAMPLE_RATE")?,
            channels: env::var("AUDIO_CHANNELS")
                .unwrap_or_else(|_| "1".to_string())
                .parse()
                .context("Invalid AUDIO_CHANNELS")?,
            chunk_ms: env::var("AUDIO_CHUNK_MS")
                .unwrap_or_else(|_| "100".to_string())
                .parse()
                .context("Invalid AUDIO_CHUNK_MS")?,
            audio_device: env::var("AUDIO_DEVICE").ok(),
            audio_output_device: env::var("AUDIO_OUTPUT_DEVICE").ok(),
            list_devices: env::var("LIST_AUDIO_DEVICES")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(false),

            // Language
            language: env::var("VOICEBOT_LANGUAGE").unwrap_or_else(|_| "es".to_string()),

            // STT
            whisper_model: env::var("WHISPER_MODEL")
                .unwrap_or_else(|_| "models/ggml-large-v3-turbo.bin".to_string()),

            // LLM
            llm_url: env::var("LLM_URL")
                .unwrap_or_else(|_| "http://localhost:8080".to_string()),
            llm_slot_id: env::var("LLM_SLOT_ID")
                .unwrap_or_else(|_| "0".to_string())
                .parse()
                .context("Invalid LLM_SLOT_ID")?,
            llm_max_tokens: env::var("LLM_MAX_TOKENS")
                .unwrap_or_else(|_| "400".to_string())
                .parse()
                .context("Invalid LLM_MAX_TOKENS")?,
            llm_system_prompt: env::var("LLM_SYSTEM_PROMPT").unwrap_or_else(|_| {
                "Eres un asistente de voz útil y conciso. \
                 Responde siempre en el mismo idioma que el usuario. \
                 Habla de forma natural y directa, sin listas ni formato markdown."
                    .to_string()
            }),
            llm_temperature: env::var("LLM_TEMPERATURE")
                .unwrap_or_else(|_| "0.7".to_string())
                .parse()
                .context("Invalid LLM_TEMPERATURE")?,

            // TTS
            piper_model_es: env::var("PIPER_MODEL_ES")
                .unwrap_or_else(|_| "models/es_ES-sharvard-medium.onnx.json".to_string()),
            piper_model_en: env::var("PIPER_MODEL_EN")
                .unwrap_or_else(|_| "models/en_US-lessac-medium.onnx.json".to_string()),

            // DB
            db_path: env::var("DB_PATH")
                .unwrap_or_else(|_| "data/voicebot.db".to_string()),
        })
    }

    pub fn samples_per_chunk(&self) -> usize {
        (self.sample_rate as usize * self.chunk_ms as usize) / 1000
    }

    /// Returns the Piper model config path for the configured language.
    pub fn piper_model_path(&self) -> &str {
        if self.language == "en" {
            &self.piper_model_en
        } else {
            &self.piper_model_es
        }
    }
}
