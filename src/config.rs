use anyhow::{Context, Result};
use std::env;

#[derive(Debug, Clone)]
pub struct Config {
    // ── Audio input ──────────────────────────────────────────────────────────
    /// Microphone sample rate (default 16000 — required by Silero VAD)
    pub sample_rate: u32,
    pub channels: u16,
    pub chunk_ms: u32,
    pub audio_input_device: Option<String>,
    pub audio_output_device: Option<String>,
    pub list_devices: bool,

    // ── VAD ───────────────────────────────────────────────────────────────────
    /// Milliseconds of continuous silence before SpeechEnd fires.
    /// Lower = faster response; higher = fewer false cuts mid-sentence.
    pub vad_silence_ms: u32,

    // ── Language ─────────────────────────────────────────────────────────────
    /// "es" (default) or "en"
    pub language: String,

    // ── STT ──────────────────────────────────────────────────────────────────
    /// Path to whisper.cpp GGML model file (.bin)
    pub whisper_model: String,

    // ── LLM ──────────────────────────────────────────────────────────────────
    /// LLM server base URL (OpenAI-compatible)
    pub llm_url: String,
    /// Model name sent in the `model` field of API requests
    pub llm_model: String,
    /// KV-cache slot ID (0 for single-user, llama.cpp only)
    pub llm_slot_id: u8,
    /// Max tokens per response
    pub llm_max_tokens: u32,
    pub llm_system_prompt: String,
    pub llm_temperature: f32,

    // ── TTS ──────────────────────────────────────────────────────────────────
    /// TTS backend: "say" (default, macOS) or "kokoro" (--features kokoro)
    pub tts_provider: String,
    /// macOS `say` voice name (SAY_VOICE). List with: say -v ?
    pub say_voice: String,
    /// Path to kokoro-v1.0.onnx model file (KOKORO_MODEL)
    pub kokoro_model: String,
    /// Path to voices-v1.0.bin embeddings file (KOKORO_VOICES)
    pub kokoro_voices: String,
    /// Kokoro voice style name, e.g. "af_bella" or "es_*" (KOKORO_VOICE)
    pub kokoro_voice: String,
    /// BCP-47 language code for espeak-ng, e.g. "en-us" or "es" (KOKORO_LANGUAGE)
    pub kokoro_language: String,

    // ── Context summarization ─────────────────────────────────────────────────
    /// Approximate context window of the LLM model in tokens.
    /// Summarization triggers when the prompt exceeds 75% of this limit.
    pub llm_context_tokens: usize,
    /// Number of most-recent (role, content) turns to keep verbatim after summarization.
    pub llm_summary_keep_turns: usize,

    // ── Agent delegation ──────────────────────────────────────────────────────
    /// CLI command used to invoke the agent (e.g. "hermes"). May include arguments.
    /// None = agent tools disabled. The voicebot writes the task to stdin and reads
    /// the result from stdout.
    pub agent_command: Option<String>,
    /// Hard timeout in seconds for synchronous agent calls (AGENT_TIMEOUT_SECS).
    pub agent_timeout_secs: u64,

    // ── Inference daemon ──────────────────────────────────────────────────────
    /// Enable the background "is there anything worth saying?" loop.
    pub daemon_enabled: bool,
    /// Seconds between daemon checks (DAEMON_INTERVAL_SECS, default 300).
    pub daemon_interval_secs: u64,

    // ── System state injection ────────────────────────────────────────────────
    /// Prepend `[SYSTEM STATE]` (time, active app, battery) to each user turn.
    /// Enabled via `INJECT_SYSTEM_DATA=true`.
    pub inject_system_data: bool,

    // ── Vision ────────────────────────────────────────────────────────────────
    /// Base URL of the vision model provider (VISION_URL). None = disabled.
    pub vision_url: Option<String>,
    /// Model name for vision requests (VISION_MODEL).
    pub vision_model: String,
    /// Max tokens for vision responses (VISION_MAX_TOKENS, default 512).
    pub vision_max_tokens: u32,

    // ── Shell tool ────────────────────────────────────────────────────────────
    /// Enable the `run_shell` tool (SHELL_ENABLED=1). Off by default.
    pub shell_enabled: bool,
    /// Hard timeout per shell command in seconds (SHELL_TIMEOUT_SECS).
    pub shell_timeout_secs: u64,

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
            audio_input_device: env::var("AUDIO_INPUT_DEVICE").ok(),
            audio_output_device: env::var("AUDIO_OUTPUT_DEVICE").ok(),
            list_devices: env::var("LIST_AUDIO_DEVICES")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(false),

            // VAD
            vad_silence_ms: env::var("VAD_SILENCE_MS")
                .unwrap_or_else(|_| "800".to_string())
                .parse()
                .context("Invalid VAD_SILENCE_MS")?,

            // Language
            language: env::var("VOICEBOT_LANGUAGE").unwrap_or_else(|_| "es".to_string()),

            // STT
            whisper_model: env::var("WHISPER_MODEL")
                .unwrap_or_else(|_| "models/ggml-large-v3-turbo.bin".to_string()),

            // LLM
            llm_url: env::var("LLM_URL")
                .unwrap_or_else(|_| "http://localhost:8080".to_string()),
            llm_model: env::var("LLM_MODEL")
                .unwrap_or_else(|_| "local-model".to_string()),
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
            tts_provider: env::var("TTS_PROVIDER")
                .unwrap_or_else(|_| "say".to_string()),
            say_voice: env::var("SAY_VOICE")
                .unwrap_or_else(|_| "Marisol (Enhanced)".to_string()),
            kokoro_model: env::var("KOKORO_MODEL")
                .unwrap_or_else(|_| "models/kokoro-v1.0.onnx".to_string()),
            kokoro_voices: env::var("KOKORO_VOICES")
                .unwrap_or_else(|_| "models/voices-v1.0.bin".to_string()),
            kokoro_voice: env::var("KOKORO_VOICE")
                .unwrap_or_else(|_| "af_bella".to_string()),
            kokoro_language: env::var("KOKORO_LANGUAGE")
                .unwrap_or_else(|_| "en-us".to_string()),

            // Context summarization
            llm_context_tokens: env::var("LLM_CONTEXT_TOKENS")
                .unwrap_or_else(|_| "4096".to_string())
                .parse()
                .context("Invalid LLM_CONTEXT_TOKENS")?,
            llm_summary_keep_turns: env::var("LLM_SUMMARY_KEEP_TURNS")
                .unwrap_or_else(|_| "6".to_string())
                .parse()
                .context("Invalid LLM_SUMMARY_KEEP_TURNS")?,

            // Agent delegation
            agent_command: env::var("AGENT_COMMAND").ok(),
            agent_timeout_secs: env::var("AGENT_TIMEOUT_SECS")
                .unwrap_or_else(|_| "120".to_string())
                .parse()
                .context("Invalid AGENT_TIMEOUT_SECS")?,

            // Inference daemon
            daemon_enabled: env::var("DAEMON_ENABLED")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(false),
            daemon_interval_secs: env::var("DAEMON_INTERVAL_SECS")
                .unwrap_or_else(|_| "300".to_string())
                .parse()
                .context("Invalid DAEMON_INTERVAL_SECS")?,

            // System state injection
            inject_system_data: env::var("INJECT_SYSTEM_DATA")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(false),

            // Vision
            vision_url: env::var("VISION_URL").ok(),
            vision_model: env::var("VISION_MODEL")
                .unwrap_or_else(|_| "local-model".to_string()),
            vision_max_tokens: env::var("VISION_MAX_TOKENS")
                .unwrap_or_else(|_| "512".to_string())
                .parse()
                .context("Invalid VISION_MAX_TOKENS")?,

            // Shell tool
            shell_enabled: env::var("SHELL_ENABLED")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(false),
            shell_timeout_secs: env::var("SHELL_TIMEOUT_SECS")
                .unwrap_or_else(|_| "30".to_string())
                .parse()
                .context("Invalid SHELL_TIMEOUT_SECS")?,

            // DB
            db_path: env::var("DB_PATH")
                .unwrap_or_else(|_| "data/voicebot.db".to_string()),
        })
    }

    pub fn samples_per_chunk(&self) -> usize {
        (self.sample_rate as usize * self.chunk_ms as usize) / 1000
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::LlmSession;

    // ── Config loading from env ───────────────────────────────────────────────

    #[test]
    fn system_prompt_loaded_from_env_var() {
        let prompt = "Eres Jarvis, el asistente personal de Daniel.";
        temp_env::with_var("LLM_SYSTEM_PROMPT", Some(prompt), || {
            let config = Config::from_env().unwrap();
            assert_eq!(config.llm_system_prompt, prompt);
        });
    }

    #[test]
    fn system_prompt_uses_default_when_env_var_absent() {
        temp_env::with_var("LLM_SYSTEM_PROMPT", None::<&str>, || {
            let config = Config::from_env().unwrap();
            assert!(!config.llm_system_prompt.is_empty(), "default must not be empty");
            // The default is the built-in Spanish assistant prompt.
            assert!(
                config.llm_system_prompt.contains("asistente"),
                "default should be the Spanish assistant prompt, got: {:?}",
                config.llm_system_prompt
            );
        });
    }

    #[test]
    fn system_prompt_can_be_multiline() {
        let prompt = "Eres Jarvis.\nHablas español.\nEres conciso.";
        temp_env::with_var("LLM_SYSTEM_PROMPT", Some(prompt), || {
            let config = Config::from_env().unwrap();
            assert_eq!(config.llm_system_prompt, prompt);
        });
    }

    // ── Session construction from config ──────────────────────────────────────

    #[test]
    fn system_prompt_from_config_becomes_first_message() {
        let prompt = "Eres Jarvis, el asistente personal de Daniel.";
        temp_env::with_var("LLM_SYSTEM_PROMPT", Some(prompt), || {
            let config = Config::from_env().unwrap();
            let session = LlmSession::new(&config.llm_system_prompt, config.llm_slot_id);
            let msgs = session.all_messages();

            assert_eq!(msgs[0].role, "system");
            assert_eq!(msgs[0].content, prompt);
        });
    }

    #[test]
    fn system_message_is_always_first_regardless_of_turns() {
        let prompt = "Eres Jarvis.";
        temp_env::with_var("LLM_SYSTEM_PROMPT", Some(prompt), || {
            let config = Config::from_env().unwrap();
            let mut session = LlmSession::new(&config.llm_system_prompt, config.llm_slot_id);
            session.add_user_turn("Hola");
            session.add_assistant_turn("Hola, Daniel.");
            session.add_user_turn("¿Qué hora es?");

            let msgs = session.all_messages();
            assert_eq!(msgs[0].role, "system", "system must always be first");
            assert_eq!(msgs[0].content, prompt);
            assert_eq!(msgs.len(), 1 + 3); // system + 3 conversation messages
        });
    }

    // ── Full chain: .env → Config → LlmSession → API payload ─────────────────

    #[test]
    fn full_chain_env_to_context() {
        // This test mirrors exactly what main.rs does when building the session.
        let prompt = "Eres Jarvis, el asistente personal de Daniel. Llevas años trabajando con él.";

        temp_env::with_var("LLM_SYSTEM_PROMPT", Some(prompt), || {
            // Step 1: load config (mirrors dotenvy::dotenv() + Config::from_env() in main)
            let config = Config::from_env().unwrap();
            assert_eq!(config.llm_system_prompt, prompt);

            // Step 2: build the composite system prompt (mirrors main.rs lines 89-94)
            // No profile facts or tools in this test — they are tested separately.
            let system_prompt = config.llm_system_prompt.clone();

            // Step 3: create session (mirrors main.rs line 95-100)
            let mut session =
                LlmSession::new(&system_prompt, config.llm_slot_id);
            session.add_user_turn("¿Qué hora es?");

            // Step 4: verify the payload that would be sent to the LLM
            let msgs = session.all_messages();
            assert_eq!(msgs[0].role, "system");
            assert_eq!(
                msgs[0].content, prompt,
                "the system prompt from .env must appear verbatim in the API payload"
            );
            assert_eq!(msgs[1].role, "user");
            assert_eq!(msgs[1].content, "¿Qué hora es?");
        });
    }

    #[test]
    fn system_prompt_preserved_after_multiple_turns() {
        let prompt = "Eres Jarvis.";
        temp_env::with_var("LLM_SYSTEM_PROMPT", Some(prompt), || {
            let config = Config::from_env().unwrap();
            let mut session = LlmSession::new(&config.llm_system_prompt, config.llm_slot_id);

            for i in 0..5 {
                session.add_user_turn(&format!("Mensaje {i}"));
                session.add_assistant_turn(&format!("Respuesta {i}"));
            }

            // System message must remain unchanged through all turns.
            let msgs = session.all_messages();
            assert_eq!(msgs[0].role, "system");
            assert_eq!(msgs[0].content, prompt);
            assert_eq!(msgs.len(), 1 + 10); // system + 10 conversation messages
        });
    }

    #[test]
    fn system_prompt_preserved_after_summarization() {
        let prompt = "Eres Jarvis, el asistente de Daniel.";
        temp_env::with_var("LLM_SYSTEM_PROMPT", Some(prompt), || {
            let config = Config::from_env().unwrap();
            let mut session = LlmSession::new(&config.llm_system_prompt, config.llm_slot_id);

            for i in 0..5 {
                session.add_user_turn(&format!("Pregunta {i}"));
                session.add_assistant_turn(&format!("Respuesta {i}"));
            }

            // Summarize — the original system prompt must survive compaction.
            session.apply_summary("Resumen de la conversación anterior.", 4);

            let msgs = session.all_messages();
            assert_eq!(msgs[0].role, "system");
            // Original prompt is still there, summary appended below it.
            assert!(
                msgs[0].content.starts_with(prompt),
                "original prompt must be preserved: {:?}",
                msgs[0].content
            );
            assert!(msgs[0].content.contains("[CONVERSATION SUMMARY]"));
            assert!(msgs[0].content.contains("Resumen de la conversación anterior."));
        });
    }
}
