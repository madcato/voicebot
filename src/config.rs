use anyhow::{Context, Result};
use std::env;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Config {
    // ── Audio input ──────────────────────────────────────────────────────────
    /// Microphone sample rate (default 16000 — required by Silero VAD)
    pub sample_rate: u32,
    pub channels: u16,
    pub chunk_ms: u32,
    pub audio_input_device: Option<String>,
    pub audio_output_device: Option<String>,
    pub list_devices: bool,
    pub list_voices: bool,

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
    /// Number of CPU threads for Whisper decoding (0 = auto).
    /// Set to physical core count for best throughput.
    pub whisper_threads: u32,

    // ── LLM ──────────────────────────────────────────────────────────────────
    /// LLM server base URL (OpenAI-compatible)
    pub llm_url: String,
    /// API key sent as `Authorization: Bearer <key>`. Empty = no auth header.
    pub llm_api_key: String,
    /// Model name sent in the `model` field of API requests
    pub llm_model: String,
    /// LLM backend: "llama" (default, llama.cpp) or "mlx" (mlx-lm).
    /// Controls whether llama.cpp-specific fields (cache_prompt, slot_id) are sent.
    pub llm_provider: String,
    /// KV-cache slot ID (0 for single-user, llama.cpp only)
    pub llm_slot_id: u8,
    /// Slot used for background calls (summarization, profile extraction).
    /// -1 = let llama.cpp pick any free slot (default).
    /// Set to 1 when running llama-server with --parallel 2.
    pub llm_background_slot_id: i32,
    /// Max tokens per response
    pub llm_max_tokens: u32,
    pub llm_system_prompt: String,
    pub llm_temperature: f32,

    // ── TTS ──────────────────────────────────────────────────────────────────
    /// TTS backend: "say" (default, macOS), "avspeech" (native AVSpeechSynthesizer,
    /// --features avspeech), or "kokoro" (--features kokoro).
    pub tts_provider: String,
    /// macOS `say` voice name (SAY_VOICE). List with: say -v ?
    pub say_voice: String,
    /// macOS `say` speaking rate in words per minute (SAY_RATE, default 215).
    pub say_rate: u32,
    /// AVSpeechSynthesizer voice display name (AVSPEECH_VOICE, default "Jorge (Enhanced)").
    pub avspeech_voice: String,
    /// AVSpeechSynthesizer normalized speech rate 0.0–1.0 (AVSPEECH_RATE, default 0.55).
    /// AVSpeechUtteranceDefaultSpeechRate (0.5) ≈ 180 wpm; 0.55 ≈ 215 wpm.
    pub avspeech_rate: f32,
    /// Path to kokoro-v1.0.onnx model file (KOKORO_MODEL)
    pub kokoro_model: String,
    /// Path to voices-v1.0.bin embeddings file (KOKORO_VOICES)
    pub kokoro_voices: String,
    /// Kokoro voice style name, e.g. "af_bella" or "es_*" (KOKORO_VOICE)
    pub kokoro_voice: String,
    /// BCP-47 language code for espeak-ng, e.g. "en-us" or "es" (KOKORO_LANGUAGE)
    pub kokoro_language: String,

    // ── Context consolidation ────────────────────────────────────────────────
    /// Approximate context window of the LLM model in tokens.
    /// Context consolidation triggers when the prompt exceeds the configured
    /// threshold percentage of this limit.
    pub llm_context_tokens: usize,
    /// Number of most-recent (role, content) turns to keep verbatim after consolidation.
    pub llm_summary_keep_turns: usize,
    /// Percentage of the context window that triggers consolidation (default 80).
    pub llm_consolidation_threshold_pct: usize,
    /// Seconds of user inactivity after which a silent consolidation is triggered
    /// (if context needs it). 0 = disabled. Default: 900 (15 minutes).
    pub llm_idle_consolidation_secs: u64,
    /// Minimum context fill percentage required for an idle-triggered consolidation to run.
    /// If the current context is below this threshold, idle consolidation is skipped.
    /// Default: 20. Set to 0 to disable the minimum check.
    pub llm_idle_min_context_pct: usize,
    /// Maximum number of messages loaded from the DB on startup (0 = unlimited).
    /// Older messages beyond this count are skipped — the session summary covers them.
    /// Default: 0. Recommended: 40–60 to prevent restart compaction. (LLM_HISTORY_LOAD_LIMIT)
    pub llm_history_load_limit: usize,

    // ── Agent delegation ──────────────────────────────────────────────────────
    /// CLI command used to invoke the agent (e.g. "hermes chat"). May include arguments.
    /// None = agent tools disabled. Used in "cli" mode only.
    pub agent_command: Option<String>,
    /// Hard timeout in seconds for synchronous agent calls (AGENT_TIMEOUT_SECS).
    pub agent_timeout_secs: u64,
    /// Agent communication mode: "cli" (default, fire-and-forget subprocess) or
    /// "acp" (persistent ACP JSON-RPC stdio process with bidirectional communication).
    pub agent_mode: String,
    /// Command to start the ACP process (AGENT_ACP_COMMAND, default "hermes acp").
    /// Only used when agent_mode = "acp".
    pub agent_acp_command: String,
    /// When true, send a warmup prompt to Hermes at startup to force model load.
    /// AGENT_ACP_WARMUP=1. Only applies when agent_mode = "acp".
    pub agent_acp_warmup: bool,

    // ── Inference daemon ──────────────────────────────────────────────────────
    /// Enable the background "is there anything worth saying?" loop.
    pub daemon_enabled: bool,
    /// Seconds between daemon checks (DAEMON_INTERVAL_SECS, default 300).
    pub daemon_interval_secs: u64,

    // ── EYES (visual awareness) ───────────────────────────────────────────────
    /// Seconds between screen-capture checks for EYES (EYES_INTERVAL_SECS).
    /// 0 = disabled (default). Requires SECONDARY_LLM_URL to be set.
    pub eyes_interval_secs: u64,

    // ── Secondary LLM (vision + background tasks) ────────────────────────────
    /// Base URL of the secondary LLM provider (SECONDARY_LLM_URL). None = disabled.
    /// When set, enables the vision tool and routes summarization + profile
    /// extraction to this model instead of the primary.
    pub secondary_llm_url: Option<String>,
    /// Model name for secondary LLM requests (SECONDARY_LLM_MODEL).
    pub secondary_llm_model: String,
    /// Max tokens for secondary LLM responses (SECONDARY_LLM_MAX_TOKENS, default 512).
    pub secondary_llm_max_tokens: u32,
    /// Bearer token for secondary LLM API (SECONDARY_LLM_API_KEY, default empty).
    pub secondary_llm_api_key: String,
    /// Backend for secondary LLM: "llama" or "mlx" (SECONDARY_LLM_PROVIDER, default "llama").
    pub secondary_llm_provider: String,
    /// Enable Qwen3 thinking mode on the secondary LLM (SECONDARY_LLM_THINKING, default false).
    /// When true, `chat_template_kwargs: {"enable_thinking": true}` is sent in requests and
    /// `<think>…</think>` blocks are stripped from the returned text.
    pub secondary_llm_thinking: bool,

    // ── Shell tool ────────────────────────────────────────────────────────────
    /// Enable the `run_shell` tool (SHELL_ENABLED=1). Off by default.
    pub shell_enabled: bool,
    /// Hard timeout per shell command in seconds (SHELL_TIMEOUT_SECS).
    pub shell_timeout_secs: u64,

    // ── Web Search (SearXNG) ─────────────────────────────────────────────────
    /// Base URL of the SearXNG instance (SEARXNG_URL). None = web_search tool disabled.
    pub searxng_url: Option<String>,
    /// Bearer token for SearXNG authentication (SEARXNG_SECRET).
    pub searxng_secret: String,
    /// Enable the web_search tool (WEB_SEARCH_ENABLED, default true).
    /// Set to 0 to disable without removing SEARXNG_URL.
    pub web_search_enabled: bool,

    // ── Speaker verification ──────────────────────────────────────────────────
    /// Path to sherpa-onnx speaker embedding ONNX model (SPEAKER_MODEL).
    /// None = auto-detect from models/speaker_embedding.onnx; disabled if absent.
    pub speaker_model: Option<String>,
    /// Path where the enrolled speaker embedding is persisted (SPEAKER_ENROLLMENT_PATH).
    pub speaker_enrollment_path: String,
    /// Cosine similarity threshold [0..1] (SPEAKER_SIMILARITY_MIN, default 0.45).
    pub speaker_similarity_min: f32,

    // ── Conversation mode (ambient state machine) ─────────────────────────────
    /// Wake word that triggers a response in Ambient mode (WAKE_WORD, default "jarvis").
    /// Case-insensitive substring match against the STT transcript.
    pub wake_word: String,
    /// Seconds in Ambient mode with no speech before auto-returning to Active
    /// (AMBIENT_CLEAR_SECS, default 300).
    pub ambient_clear_secs: u64,
    /// Consecutive non-enrolled-speaker VAD segments before auto-switching to
    /// Ambient mode (SPEAKER_AMBIENT_TRIGGER, default 3). Only applies when
    /// speaker verification is enabled.
    pub speaker_ambient_trigger: u8,

    // ── Ambient context buffer ────────────────────────────────────────────────
    /// Maximum number of speaker profiles to auto-enroll (SPEAKER_MAX_PROFILES, default 5).
    /// The first enrolled speaker is always the "main user" (id=0).
    pub speaker_max_profiles: u8,
    /// Rolling window duration for the ambient context buffer in minutes
    /// (AMBIENT_BUFFER_MINUTES, default 3).
    pub ambient_buffer_minutes: u64,
    /// Maximum number of utterances to keep in the ambient context buffer
    /// (AMBIENT_BUFFER_MAX_ENTRIES, default 30).
    pub ambient_buffer_max_entries: usize,

    // ── Remote device (WebSocket) ──────────────────────────────────────────────
    /// WebSocket server port. None = disabled (WS_PORT).
    pub ws_port: Option<u16>,

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
            list_voices: env::var("LIST_VOICES")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(false),

            // VAD
            vad_silence_ms: env::var("VAD_SILENCE_MS")
                .unwrap_or_else(|_| "250".to_string())
                .parse()
                .context("Invalid VAD_SILENCE_MS")?,

            // Language
            language: env::var("VOICEBOT_LANGUAGE").unwrap_or_else(|_| "es".to_string()),

            // STT
            whisper_model: env::var("WHISPER_MODEL")
                .unwrap_or_else(|_| "models/ggml-large-v3-turbo.bin".to_string()),
            whisper_threads: env::var("WHISPER_THREADS")
                .unwrap_or_else(|_| "0".to_string())
                .parse()
                .context("Invalid WHISPER_THREADS")?,

            // LLM
            llm_url: env::var("LLM_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:8080".to_string()),
            llm_api_key: env::var("LLM_API_KEY").unwrap_or_default(),
            llm_model: env::var("LLM_MODEL")
                .unwrap_or_else(|_| "local-model".to_string()),
            llm_provider: env::var("LLM_PROVIDER")
                .unwrap_or_else(|_| "llama".to_string()),
            llm_slot_id: env::var("LLM_SLOT_ID")
                .unwrap_or_else(|_| "0".to_string())
                .parse()
                .context("Invalid LLM_SLOT_ID")?,
            llm_background_slot_id: env::var("LLM_BACKGROUND_SLOT_ID")
                .unwrap_or_else(|_| "-1".to_string())
                .parse()
                .context("Invalid LLM_BACKGROUND_SLOT_ID")?,
            llm_max_tokens: env::var("LLM_MAX_TOKENS")
                .unwrap_or_else(|_| "200".to_string())
                .parse()
                .context("Invalid LLM_MAX_TOKENS")?,
            llm_system_prompt: env::var("LLM_SYSTEM_PROMPT").unwrap_or_else(|_| {
                "Eres un asistente de voz útil y conciso. \
                 Responde siempre en el mismo idioma que el usuario. \
                 Habla de forma natural y directa, sin listas ni formato markdown. \
                 Empieza siempre con la respuesta directa, sin preámbulos. \
                 Por defecto, limita tus respuestas a 2-3 frases cortas. \
                 Si el usuario pide expresamente más detalle, una explicación completa \
                 o un resumen extenso, responde con la profundidad necesaria."
                    .to_string()
            }),
            llm_temperature: env::var("LLM_TEMPERATURE")
                .unwrap_or_else(|_| "0.3".to_string())
                .parse()
                .context("Invalid LLM_TEMPERATURE")?,

            // TTS
            tts_provider: env::var("TTS_PROVIDER")
                .unwrap_or_else(|_| "say".to_string()),
            say_voice: env::var("SAY_VOICE")
                .unwrap_or_else(|_| "Jorge (Enhanced)".to_string()),
            say_rate: env::var("SAY_RATE")
                .unwrap_or_else(|_| "215".to_string())
                .parse()
                .context("Invalid SAY_RATE")?,
            avspeech_voice: env::var("AVSPEECH_VOICE")
                .unwrap_or_else(|_| "Jorge (Enhanced)".to_string()),
            avspeech_rate: env::var("AVSPEECH_RATE")
                .unwrap_or_else(|_| "0.55".to_string())
                .parse()
                .context("Invalid AVSPEECH_RATE")?,
            kokoro_model: env::var("KOKORO_MODEL")
                .unwrap_or_else(|_| "models/kokoro-v1.0.onnx".to_string()),
            kokoro_voices: env::var("KOKORO_VOICES")
                .unwrap_or_else(|_| "models/voices-v1.0.bin".to_string()),
            kokoro_voice: env::var("KOKORO_VOICE")
                .unwrap_or_else(|_| "af_bella".to_string()),
            kokoro_language: env::var("KOKORO_LANGUAGE")
                .unwrap_or_else(|_| "en-us".to_string()),

            // Context consolidation
            llm_context_tokens: env::var("LLM_CONTEXT_TOKENS")
                .unwrap_or_else(|_| "8192".to_string())
                .parse()
                .context("Invalid LLM_CONTEXT_TOKENS")?,
            llm_summary_keep_turns: env::var("LLM_SUMMARY_KEEP_TURNS")
                .unwrap_or_else(|_| "6".to_string())
                .parse()
                .context("Invalid LLM_SUMMARY_KEEP_TURNS")?,
            llm_consolidation_threshold_pct: env::var("LLM_CONSOLIDATION_THRESHOLD_PCT")
                .unwrap_or_else(|_| "80".to_string())
                .parse()
                .context("Invalid LLM_CONSOLIDATION_THRESHOLD_PCT")?,
            llm_idle_consolidation_secs: env::var("LLM_IDLE_CONSOLIDATION_SECS")
                .unwrap_or_else(|_| "900".to_string())
                .parse()
                .context("Invalid LLM_IDLE_CONSOLIDATION_SECS")?,
            llm_idle_min_context_pct: env::var("LLM_IDLE_MIN_CONTEXT_PCT")
                .unwrap_or_else(|_| "20".to_string())
                .parse()
                .context("Invalid LLM_IDLE_MIN_CONTEXT_PCT")?,
            llm_history_load_limit: env::var("LLM_HISTORY_LOAD_LIMIT")
                .unwrap_or_else(|_| "0".to_string())
                .parse()
                .context("Invalid LLM_HISTORY_LOAD_LIMIT")?,

            // Agent delegation
            agent_command: env::var("AGENT_COMMAND").ok(),
            agent_timeout_secs: env::var("AGENT_TIMEOUT_SECS")
                .unwrap_or_else(|_| "120".to_string())
                .parse()
                .context("Invalid AGENT_TIMEOUT_SECS")?,
            agent_mode: env::var("AGENT_MODE").unwrap_or_else(|_| "cli".to_string()),
            agent_acp_command: env::var("AGENT_ACP_COMMAND")
                .unwrap_or_else(|_| "hermes acp".to_string()),
            agent_acp_warmup: env::var("AGENT_ACP_WARMUP").as_deref() == Ok("1"),

            // Inference daemon
            daemon_enabled: env::var("DAEMON_ENABLED")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(false),
            daemon_interval_secs: env::var("DAEMON_INTERVAL_SECS")
                .unwrap_or_else(|_| "300".to_string())
                .parse()
                .context("Invalid DAEMON_INTERVAL_SECS")?,

            // EYES
            eyes_interval_secs: env::var("EYES_INTERVAL_SECS")
                .unwrap_or_else(|_| "0".to_string())
                .parse()
                .context("Invalid EYES_INTERVAL_SECS")?,

            // Secondary LLM
            secondary_llm_url: env::var("SECONDARY_LLM_URL").ok(),
            secondary_llm_model: env::var("SECONDARY_LLM_MODEL")
                .unwrap_or_else(|_| "local-model".to_string()),
            secondary_llm_max_tokens: env::var("SECONDARY_LLM_MAX_TOKENS")
                .unwrap_or_else(|_| "512".to_string())
                .parse()
                .context("Invalid SECONDARY_LLM_MAX_TOKENS")?,
            secondary_llm_api_key: env::var("SECONDARY_LLM_API_KEY").unwrap_or_default(),
            secondary_llm_provider: env::var("SECONDARY_LLM_PROVIDER")
                .unwrap_or_else(|_| "llama".to_string()),
            secondary_llm_thinking: env::var("SECONDARY_LLM_THINKING")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(false),

            // Shell tool
            shell_enabled: env::var("SHELL_ENABLED")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(false),
            shell_timeout_secs: env::var("SHELL_TIMEOUT_SECS")
                .unwrap_or_else(|_| "30".to_string())
                .parse()
                .context("Invalid SHELL_TIMEOUT_SECS")?,

            // Web Search (SearXNG)
            searxng_url: env::var("SEARXNG_URL").ok(),
            searxng_secret: env::var("SEARXNG_SECRET").unwrap_or_default(),
            web_search_enabled: env::var("WEB_SEARCH_ENABLED")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(true),

            // Speaker verification
            speaker_model: {
                let default = "models/speaker_embedding.onnx";
                match env::var("SPEAKER_MODEL") {
                    Ok(v) => Some(v),
                    Err(_) => {
                        if std::path::Path::new(default).exists() {
                            Some(default.into())
                        } else {
                            None
                        }
                    }
                }
            },
            speaker_enrollment_path: env::var("SPEAKER_ENROLLMENT_PATH")
                .unwrap_or_else(|_| "data/speaker.emb".to_string()),
            speaker_similarity_min: env::var("SPEAKER_SIMILARITY_MIN")
                .unwrap_or_else(|_| "0.45".to_string())
                .parse()
                .context("Invalid SPEAKER_SIMILARITY_MIN")?,

            // Conversation mode
            wake_word: env::var("WAKE_WORD")
                .unwrap_or_else(|_| "jarvis".to_string())
                .to_lowercase(),
            ambient_clear_secs: env::var("AMBIENT_CLEAR_SECS")
                .unwrap_or_else(|_| "300".to_string())
                .parse()
                .context("Invalid AMBIENT_CLEAR_SECS")?,
            speaker_ambient_trigger: env::var("SPEAKER_AMBIENT_TRIGGER")
                .unwrap_or_else(|_| "1".to_string())
                .parse()
                .context("Invalid SPEAKER_AMBIENT_TRIGGER")?,

            // Ambient context buffer
            speaker_max_profiles: env::var("SPEAKER_MAX_PROFILES")
                .unwrap_or_else(|_| "5".to_string())
                .parse()
                .context("Invalid SPEAKER_MAX_PROFILES")?,
            ambient_buffer_minutes: env::var("AMBIENT_BUFFER_MINUTES")
                .unwrap_or_else(|_| "3".to_string())
                .parse()
                .context("Invalid AMBIENT_BUFFER_MINUTES")?,
            ambient_buffer_max_entries: env::var("AMBIENT_BUFFER_MAX_ENTRIES")
                .unwrap_or_else(|_| "30".to_string())
                .parse()
                .context("Invalid AMBIENT_BUFFER_MAX_ENTRIES")?,

            // Remote device (WebSocket)
            ws_port: env::var("WS_PORT")
                .ok()
                .map(|v| v.parse::<u16>())
                .transpose()
                .context("Invalid WS_PORT")?,

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
        let prompt = "Eres Jarvis, el asistente personal.";
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
        let prompt = "Eres Jarvis, el asistente personal.";
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
            session.add_assistant_turn("Hola, señor.");
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
        let prompt = "Eres Jarvis, el asistente personal. Llevas años trabajando con él.";

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
        let prompt = "Eres Jarvis, el asistente.";
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
