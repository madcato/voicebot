#![allow(unreachable_code)]
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]

mod agents;
mod audio;
mod config;
mod daemon;
mod db;
mod eyes;
mod llm;
mod mcp;
mod memory;
mod profile;
mod stt;
mod tools;
#[cfg(feature = "tui")]
mod tui;
mod tts;
#[cfg(feature = "remote")]
mod remote;

use anyhow::Result;
use async_channel::bounded;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, mpsc, Notify};
use tracing::{debug, error, info, warn};

/// Monotonically increasing counter for tagging each pipeline run with a unique ID.
/// Lets us detect if two pipelines are active simultaneously in logs.
static PIPELINE_RUN_ID: AtomicU64 = AtomicU64::new(0);
use tracing_subscriber::EnvFilter;

/// Shared state between the pipeline tasks.
/// Properties documented in doc/PROCESS_ARCHITECTURE.md.
pub(crate) struct SharedSession {
    /// Latest text transcribed by STT. Replaced on each new STT result.
    pub(crate) transliterated_text: Mutex<String>,
    /// LLM token stream buffer — text not yet split into sentences.
    assistant_text: Mutex<String>,
    /// Sentences ready for TTS playback.
    pub(crate) sentences: Mutex<VecDeque<String>>,
    /// True when the current LLM streaming POST has finished.
    llm_post_finished: AtomicBool,
    /// True while the LLM task is actively processing a turn.
    llm_busy: AtomicBool,
    /// True when the pending transliterated_text came from TUI text input (not voice).
    pub(crate) text_input_pending: AtomicBool,
    /// Timestamp of the most recent VAD SpeechEnd — used for end-to-end latency logging.
    pub(crate) t_vad_end: Mutex<Option<Instant>>,
    /// Timestamp of the POST sent to LLM - used to calculate TTFT
    pub(crate) t_llm_post_send: Mutex<Option<Instant>>,
    /// Track if first sentence playback has been logged
    first_speech_played: AtomicBool,
    /// True while the consolidation task is running. The LLM task will not
    /// process new input until this flag is cleared.
    pub(crate) consolidation_active: AtomicBool,
   /// True when an STT result has been obtained but not yet fully processed
    /// by the LLM (add_user_turn called).
    pub(crate) stt_result_pending: AtomicBool,
    /// True when a background tool has delivered its result and the session
    /// already contains the tool_call + tool_result exchange. The LLM task
    /// must continue the turn (call the LLM again) without adding a new user
    /// message — the model will generate its natural continuation from the
    /// injected tool result.
    pub(crate) pending_tool_response: AtomicBool,
}

impl SharedSession {
    fn new() -> Self {
        Self {
            transliterated_text: Mutex::new(String::new()),
            assistant_text: Mutex::new(String::new()),
            sentences: Mutex::new(VecDeque::new()),
            llm_post_finished: AtomicBool::new(false),
            llm_busy: AtomicBool::new(false),
            text_input_pending: AtomicBool::new(false),
            t_vad_end: Mutex::new(None),
            t_llm_post_send: Mutex::new(None),
            first_speech_played: AtomicBool::new(false),
            consolidation_active: AtomicBool::new(false),
            stt_result_pending: AtomicBool::new(false),
            pending_tool_response: AtomicBool::new(false),
        }
    }
}

/// Signals and events for inter-task communication.
/// Documented in doc/PROCESS_ARCHITECTURE.md.
pub(crate) struct PipelineEvents {
    /// VAD_DETECTED: broadcast cancellation. All tasks must stop immediately.
    cancel_tx: broadcast::Sender<()>,
    /// VAD_FINISH: silence detected; LLM task should start processing.
    pub(crate) vad_finish: Arc<Notify>,
    /// LLM_POST_RECEIVED: a token arrived from the LLM stream.
    llm_post_received: Arc<Notify>,
    /// SENTENCE_READY: a sentence has been pushed to shared.sentences.
    sentence_ready: Arc<Notify>,
    /// LLM_POST_FINISHED: the LLM has streamed its complete response.
    llm_post_finished: Arc<Notify>,
}

impl PipelineEvents {
    fn new() -> Self {
        let (cancel_tx, _) = broadcast::channel(16);
        Self {
            cancel_tx,
            vad_finish: Arc::new(Notify::new()),
            llm_post_received: Arc::new(Notify::new()),
            sentence_ready: Arc::new(Notify::new()),
            llm_post_finished: Arc::new(Notify::new()),
        }
    }
}

use crate::agents::ProactiveEvent;
use crate::profile::extract_facts;
use crate::audio::audio_capture::{AudioCapture, AudioChunk};
use crate::audio::buffer::AudioBuffer;
use crate::audio::output::AudioOutput;
use crate::audio::ambient_buffer::AmbientBuffer;
use crate::audio::speaker::{SpeakerVerdict, SpeakerVerifier};
// Temporarily disabled during refactoring
// use crate::stt::{VadEvent, VoiceActivityDetector as SttVoiceActivityDetector};
use crate::config::Config;
use crate::db::{Database, Memory};
use crate::llm::{OpenAIClient, LlmSession, StreamToken};
use crate::memory::{build_memory_context, extract_memories};
use crate::profile::{build_profile_context, ProfileFact};
use crate::stt::{WhisperSTTVAD, WhisperSTTVADConfig, SpeechEvent};
// whisper-cpp-plus logs directly via printf (no config hooks available)
use crate::tools::{
    format_history, ActiveAcpTask, ConversationMode, CurrentTimeTool, HermesAcpWriter, JsonRpcMessage,
    McpToolProxy, OpenAppTool, ReadClipboardTool, RunAgentTool, RunShellTool, SetClipboardTool,
    SetConversationModeTool, TakeScreenshotTool, ToolRegistry, WebSearchTool,
};
use crate::tts::{SentenceSplitter, TtsEngine};
#[cfg(feature = "kokoro")]
use crate::tts::KokoroTts;
#[cfg(feature = "avspeech")]
use crate::tts::AvSpeechTts;

#[cfg(test)]
mod e2e_tests;

const AUDIO_CHANNEL_CAPACITY: usize = 200;
const MAX_SPEECH_BUFFER_SECS: u32 = 30;
const MIN_SPEECH_DURATION_MS: u32 = 300;

// When the `avspeech` feature is enabled, the main thread must run CFRunLoop
// so that AVSpeechSynthesizer buffer callbacks are delivered.  The tokio
// runtime is moved to a background thread.
#[cfg(feature = "avspeech")]
fn main() {
    unsafe extern "C" {
        fn CFRunLoopRunInMode(mode: *const std::ffi::c_void, seconds: f64, ret: u8) -> i32;
        static kCFRunLoopDefaultMode: *const std::ffi::c_void;
    }

    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
    let quit = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let quit_c = quit.clone();

    let handle = std::thread::spawn(move || {
        let result = rt.block_on(async_main());
        quit_c.store(true, std::sync::atomic::Ordering::SeqCst);
        result
    });

    // Drive the main CFRunLoop so AVSpeechSynthesizer callbacks fire.
    while !quit.load(std::sync::atomic::Ordering::SeqCst) {
        unsafe { CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.05, 0); }
    }

    if let Err(e) = handle.join().expect("tokio thread panicked") {
        eprintln!("Error: {e:#}");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "avspeech"))]
#[tokio::main]
async fn main() -> Result<()> {
    async_main().await
}

/// Assemble the full system prompt from its components.
///
/// Order: base prompt → [USER PROFILE] → [MEMORIES] → tool instructions.
/// Used at startup and after each context consolidation cycle.
fn build_system_prompt(
    base_prompt: &str,
    profile_facts: &[crate::profile::ProfileFact],
    memories: &[Memory],
    tool_section: &str,
) -> String {
    format!(
        "{}{}{}{}",
        base_prompt,
        build_profile_context(profile_facts),
        build_memory_context(memories),
        tool_section,
    )
}

async fn async_main() -> Result<()> {
    // whisper-cpp-plus logs directly via printf/stderr (no hooks available)
    // install_logging_hooks(); // Removed - not available in whisper-cpp-plus

    #[cfg(feature = "tui")]
    {
        let log_file = std::fs::File::create("voicebot.log")
            .expect("failed to create voicebot.log");
        tracing_subscriber::fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            )
            .with_writer(std::sync::Mutex::new(log_file))
            .with_ansi(false)
            .init();
    }
    #[cfg(not(feature = "tui"))]
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    dotenvy::dotenv().ok();

    info!(target: "voicebot", "Starting voicebot...");
    let config = Config::from_env()?;

    // ── Device listing shortcut ───────────────────────────────────────────────
    let list_devices = config.list_devices
        || std::env::args().any(|a| a == "--list-devices" || a == "list-devices");
    if list_devices {
        AudioCapture::print_devices()?;
        return Ok(());
    }

    // ── Voice listing shortcut ────────────────────────────────────────────────
    let list_voices = config.list_voices
        || std::env::args().any(|a| a == "--list-voices" || a == "list-voices");
    if list_voices {
        match config.tts_provider.as_str() {
            #[cfg(feature = "avspeech")]
            "avspeech" => {
                AvSpeechTts::list_voices();
            }
            #[cfg(not(feature = "avspeech"))]
            "avspeech" => {
                eprintln!("TTS_PROVIDER=avspeech requires the 'avspeech' feature: cargo run --features avspeech");
                std::process::exit(1);
            }
            #[cfg(feature = "kokoro")]
            "kokoro" => {
                let k = KokoroTts::new(
                    &config.kokoro_model,
                    &config.kokoro_voices,
                    &config.kokoro_voice,
                    &config.kokoro_language,
                )
                .await?;
                k.list_voices();
            }
            #[cfg(not(feature = "kokoro"))]
            "kokoro" => {
                eprintln!("TTS_PROVIDER=kokoro requires the 'kokoro' feature: cargo run --features kokoro");
                std::process::exit(1);
            }
            _ => {
                eprintln!("Unknown TTS_PROVIDER '{}'. Available: avspeech, kokoro", config.tts_provider);
                std::process::exit(1);
            }
        }
        return Ok(());
    }

    info!(target: "voicebot", "Language: {}", config.language);

    // ── Proactive event channel ───────────────────────────────────────────────
    let (proactive_tx, proactive_rx) = mpsc::channel::<ProactiveEvent>(32);

    // ── LLM client ────────────────────────────────────────────────────────────
    // (Primary client is built further below; secondary is built here so tools
    //  that need it can be registered before the rest of the pipeline starts.)
    //
    // Secondary LLM client — vision, summarization, profile extraction.
    // Built early so TakeScreenshotTool can be registered in the tool registry.
    let secondary_llm_client: Option<OpenAIClient> =
        config.secondary_llm_url.as_ref().map(|url| {
            OpenAIClient::new(url, &config.secondary_llm_model, config.secondary_llm_max_tokens, 0.3)
                .with_api_key(&config.secondary_llm_api_key)
                .with_thinking(config.secondary_llm_thinking)
        });
    if secondary_llm_client.is_some() {
        info!(
            target: "llm",
            "Secondary LLM endpoint: {} (model={})",
            config.secondary_llm_url.as_deref().unwrap_or(""),
            config.secondary_llm_model,
        );
    }

    // ── Tools ─────────────────────────────────────────────────────────────────
    // `shared_history` is updated after every user turn so the agent always
    // receives full conversational context via `hermes -q "{history}"`.
    let shared_history: Arc<RwLock<String>> = Arc::new(RwLock::new(String::new()));
    let mut tool_registry = ToolRegistry::new();

    // ── Conversation mode shared state ────────────────────────────────────────
    // Shared between the VAD loop (reads it) and SetConversationModeTool (writes it).
    let conv_mode: Arc<Mutex<ConversationMode>> = Arc::new(Mutex::new(ConversationMode::Active));

    // Always available
    tool_registry.register(CurrentTimeTool);
    tool_registry.register(ReadClipboardTool);
    tool_registry.register(SetClipboardTool);
    tool_registry.register(OpenAppTool);
    tool_registry.register(SetConversationModeTool::new(Arc::clone(&conv_mode)));

    // Shell command execution — enabled by SHELL_ENABLED=1
    if config.shell_enabled {
        tool_registry.register(RunShellTool::new(config.shell_timeout_secs));
        info!(target: "voicebot", "run_shell tool enabled (timeout={}s)", config.shell_timeout_secs);
    }

    // Vision (screenshot) — enabled when SECONDARY_LLM_URL is set
    if let Some(ref sec_client) = secondary_llm_client {
        info!(
            target: "voicebot",
            "Vision tool enabled via secondary LLM (model={})",
            config.secondary_llm_model,
        );
        tool_registry.register(TakeScreenshotTool::new(sec_client.clone()));
    }

    // Web search (SearXNG) — enabled when SEARXNG_URL is set and WEB_SEARCH_ENABLED != 0
    if config.web_search_enabled
        && let Some(ref searxng_url) = config.searxng_url {
            let mut wst = WebSearchTool::new(searxng_url.clone(), config.searxng_secret.clone());
            if let Some(ref sec) = secondary_llm_client {
                wst = wst.with_synthesis(std::sync::Arc::new(sec.clone()));
                info!(target: "voicebot", "web_search synthesis via secondary LLM enabled");
            }
            tool_registry.register(wst);
            info!(target: "voicebot", "web_search tool enabled (url={})", searxng_url);
    }

    // External agent delegation — unified RunAgentTool (CLI or ACP mode)
    let acp_writer: Arc<tokio::sync::Mutex<Option<HermesAcpWriter>>> =
        Arc::new(tokio::sync::Mutex::new(None));
    let acp_inbound: Arc<tokio::sync::Mutex<Option<mpsc::Receiver<JsonRpcMessage>>>> =
        Arc::new(tokio::sync::Mutex::new(None));
    let active_task: Arc<tokio::sync::Mutex<Option<ActiveAcpTask>>> =
        Arc::new(tokio::sync::Mutex::new(None));

    if config.agent_mode == "acp" || config.agent_command.is_some() {
        let mode = config.agent_mode.clone();
        let agent_cmd = config.agent_command.clone();
        let acp_cmd = config.agent_acp_command.clone();
        info!(
            target: "voicebot",
            "Agent delegation enabled (mode={})",
            mode
        );
        let mut run_agent_tool = RunAgentTool::new(
            agent_cmd,
            Arc::clone(&acp_writer),
            Arc::clone(&acp_inbound),
            Arc::clone(&active_task),
            shared_history.clone(),
            proactive_tx.clone(),
            mode,
            acp_cmd,
        );
        if let Some(ref sec) = secondary_llm_client {
            run_agent_tool = run_agent_tool.with_synthesis(std::sync::Arc::new(sec.clone()));
            info!(target: "voicebot", "run_agent result synthesis via secondary LLM enabled");
        }
        tool_registry.register(run_agent_tool);
    }

    // ── ACP pre-warm ──────────────────────────────────────────────────────────
    // When running in ACP mode, spawn the hermes acp process and perform the
    // initialize + session/new handshake in the background at startup. This
    // populates acp_writer / acp_inbound so the first run_agent call skips
    // the cold-start delay. Optionally send a warmup prompt (AGENT_ACP_WARMUP=1)
    // to force model load before the user's first real request.
    if config.agent_mode == "acp" {
        let acp_cmd     = config.agent_acp_command.clone();
        let warmup      = config.agent_acp_warmup;
        let writer_arc  = Arc::clone(&acp_writer);
        let inbound_arc = Arc::clone(&acp_inbound);

        tokio::spawn(async move {
            info!(target: "agent", "ACP pre-warm: spawning {}…", acp_cmd);
            let (mut writer, mut rx) = match HermesAcpWriter::spawn(&acp_cmd).await {
                Ok(pair) => pair,
                Err(e) => {
                    warn!(target: "agent", "ACP pre-warm: spawn failed: {e}");
                    return;
                }
            };
            let cwd = std::env::current_dir()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            match writer.initialize(&mut rx, &cwd).await {
                Ok(sid) => info!(target: "agent", "ACP pre-warm: session ready (sid={sid})"),
                Err(e)  => {
                    warn!(target: "agent", "ACP pre-warm: init failed: {e}");
                    return;
                }
            }

            // Store so run_acp() finds the session already initialised.
            *writer_arc.lock().await  = Some(writer);
            *inbound_arc.lock().await = Some(rx);

            if warmup {
                info!(target: "agent", "ACP pre-warm: sending warmup prompt…");
                let mut taken_rx = inbound_arc.lock().await.take();
                if let Some(rx) = taken_rx.as_mut() {
                    let prompt_id = {
                        let mut w = writer_arc.lock().await;
                        let sid = w.as_ref()
                            .and_then(|w| w.session_id.clone())
                            .unwrap_or_default();
                        match w.as_mut() {
                            Some(w) => w.send_prompt(&sid, "hola").await.ok(),
                            None    => None,
                        }
                    };
                    if let Some(id) = prompt_id {
                        let deadline = tokio::time::sleep(std::time::Duration::from_secs(30));
                        tokio::pin!(deadline);
                        loop {
                            tokio::select! {
                                _ = &mut deadline => {
                                    warn!(target: "agent", "ACP pre-warm: warmup timed out");
                                    break;
                                }
                                msg = rx.recv() => match msg {
                                    Some(JsonRpcMessage::Response { id: resp_id, .. })
                                        if resp_id == id =>
                                    {
                                        info!(target: "agent", "ACP pre-warm: warmup complete");
                                        break;
                                    }
                                    Some(_) => continue,
                                    None    => break,
                                }
                            }
                        }
                    }
                }
                *inbound_arc.lock().await = taken_rx;
            }
        });
    }

    // ── MCP tools ─────────────────────────────────────────────────────────────
    // If MCP_COMMAND is set, spawn the MCP server, run the initialize handshake,
    // and register all discovered tools into the registry as McpToolProxy entries.
    // Each MCP tool is background (is_background = true) — it runs in a spawned
    // task and delivers the result via ProactiveEvent.
    if let Some(ref mcp_cmd) = config.mcp_command {
        info!(target: "mcp", "Spawning MCP server: {}", mcp_cmd);
        match mcp::McpClient::spawn_and_init(mcp_cmd, config.mcp_tool_timeout_secs).await {
            Ok((client, tool_defs)) => {
                let client = std::sync::Arc::new(client);
                let count = tool_defs.len();
                for def in tool_defs {
                    info!(
                        target: "mcp",
                        "Tool `{}`: schema={}",
                        def.name,
                        serde_json::to_string(&def.input_schema).unwrap_or_default(),
                    );
                    tool_registry.register(McpToolProxy::new(
                        def.name,
                        def.description,
                        def.input_schema,
                        std::sync::Arc::clone(&client),
                    ));
                }
                info!(target: "mcp", "Registered {} MCP tool(s)", count);
            }
            Err(e) => {
                warn!(target: "mcp", "MCP server failed to start — MCP tools disabled: {e}");
            }
        }
    }

    let tools = Arc::new(tool_registry);

    // ── Database ─────────────────────────────────────────────────────────────
    let db = Database::new(&config.db_path).await?;
    let session_id = db.get_or_create_session().await?;
    let (summary, history) = db.get_session_context(session_id, config.llm_history_load_limit).await?;
    info!(
        target: "db",
        "Loaded {} messages from history (summary: {})",
        history.len(),
        if summary.is_some() { "yes" } else { "no" }
    );

    // ── User profile ──────────────────────────────────────────────────────────
    let profile_facts: Vec<ProfileFact> = db
        .load_user_profile()
        .await?
        .into_iter()
        .map(|(key, value, confidence)| ProfileFact { key, value, confidence })
        .collect();
    if !profile_facts.is_empty() {
        info!(target: "profile", "Loaded {} user profile facts", profile_facts.len());
    }

    // ── Persistent memories ──────────────────────────────────────────────────
    let memories: Vec<Memory> = db.load_active_memories().await?;
    if !memories.is_empty() {
        info!(target: "memory", "Loaded {} persistent memories", memories.len());
    }

    // ── LLM session ───────────────────────────────────────────────────────────
    let tool_section = tools.system_prompt_section();
    let system_prompt = build_system_prompt(
        &config.llm_system_prompt,
        &profile_facts,
        &memories,
        &tool_section,
    );
    let llm_session = Arc::new(Mutex::new(LlmSession::from_history(
        &system_prompt,
        summary.as_deref(),
        &history,
    )));

    // ── LLM client ────────────────────────────────────────────────────────────
    let llm_client = OpenAIClient::new(
        &config.llm_url,
        &config.llm_model,
        config.llm_max_tokens,
        config.llm_temperature,
    )
    .with_api_key(&config.llm_api_key);
    info!(target: "llm", "LLM endpoint: {}", config.llm_url);

    // Background tasks use the secondary client when available, otherwise
    // fall back to the primary (preserves behavior when SECONDARY_LLM_URL is unset).
    let background_client =
        secondary_llm_client.clone().unwrap_or_else(|| llm_client.clone());

    // ── Inference daemon ──────────────────────────────────────────────────────
    if config.daemon_enabled {
        info!(
            target: "daemon",
            "Inference daemon enabled (interval={}s)",
            config.daemon_interval_secs
        );
        daemon::InferenceDaemon {
            interval_secs: config.daemon_interval_secs,
            llm_client: llm_client.clone(),
            llm_session: Arc::clone(&llm_session),
            proactive_tx: proactive_tx.clone(),
        }
        .spawn();
    }

    // ── EYES (visual awareness) ───────────────────────────────────────────────
    if config.eyes_interval_secs > 0 {
        if let Some(ref sec_client) = secondary_llm_client {
            info!(
                target: "eyes",
                "EYES enabled (interval={}s, model={})",
                config.eyes_interval_secs,
                config.secondary_llm_model,
            );
            eyes::EyesDaemon {
                interval_secs: config.eyes_interval_secs,
                vision_client: sec_client.clone(),
                proactive_tx: proactive_tx.clone(),
            }
            .spawn();
        } else {
            warn!(
                target: "eyes",
                "EYES_INTERVAL_SECS={} but SECONDARY_LLM_URL is not set — EYES disabled",
                config.eyes_interval_secs
            );
        }
    }

    // ── STT + VAD unified processor ────────────────────────────────────────────
    let sttvad_config = WhisperSTTVADConfig {
        whisper_model: config.whisper_model.clone(),
        vad_model: config.vad_model.clone(),
        language: config.language.clone(),
        silence_ms: config.vad_silence_ms,
    };
    let mut sttvad = WhisperSTTVAD::new(sttvad_config)?;
    info!(target: "stt", "Initialized unified WhisperSTTVAD (whisper: {}, vad: {})", 
        config.whisper_model, config.vad_model);

    // ── Speaker verifier ──────────────────────────────────────────────────────
    let mut speaker_verifier: Option<SpeakerVerifier> =
        if let Some(ref model_path) = config.speaker_model {
            match SpeakerVerifier::new(
                model_path,
                std::path::Path::new(&config.speaker_enrollment_path),
                config.speaker_similarity_min,
                config.speaker_max_profiles,
            ) {
                Ok(sv) => {
                    info!(
                        target: "speaker",
                        "Speaker verification enabled (threshold={})",
                        config.speaker_similarity_min
                    );
                    Some(sv)
                }
                Err(e) => {
                    warn!(target: "speaker", "Speaker verification disabled — model load failed: {e}");
                    None
                }
            }
        } else {
            info!(
                target: "speaker",
                "Speaker verification disabled \
                 (place model at models/speaker_embedding.onnx to enable)"
            );
            None
        };

    // ── Ambient context buffer ────────────────────────────────────────────────
    // Always running: buffers transcripts from all non-main speakers (and the
    // main user's non-wake-word utterances in Ambient mode) for LLM context.
    let ambient_buffer = Arc::new(std::sync::Mutex::new(AmbientBuffer::new(
        config.ambient_buffer_max_entries,
        config.ambient_buffer_minutes,
    )));
    info!(
        target: "pipeline",
        "Ambient buffer: {}min / {} entries max",
        config.ambient_buffer_minutes,
        config.ambient_buffer_max_entries
    );

    // ── TTS ───────────────────────────────────────────────────────────────────
    let tts: TtsEngine = match config.tts_provider.as_str() {
        #[cfg(feature = "avspeech")]
        "avspeech" => {
            info!(target: "tts", "TTS provider: AVSpeechSynthesizer (voice={}, rate={:.2})", config.avspeech_voice, config.avspeech_rate);
            let voice = config.avspeech_voice.clone();
            let rate = config.avspeech_rate;
            let t = tokio::task::spawn_blocking(move || AvSpeechTts::new(&voice, rate)).await??;
            TtsEngine::AvSpeech(t)
        }
        #[cfg(not(feature = "avspeech"))]
        "avspeech" => {
            anyhow::bail!(
                "TTS_PROVIDER=avspeech requires the 'avspeech' feature: cargo run --features avspeech"
            );
        }
        #[cfg(feature = "kokoro")]
        "kokoro" => {
            info!(target: "tts", "TTS provider: Kokoro (voice={}, lang={})", config.kokoro_voice, config.kokoro_language);
            let k = KokoroTts::new(
                &config.kokoro_model,
                &config.kokoro_voices,
                &config.kokoro_voice,
                &config.kokoro_language,
            )
            .await?;
            TtsEngine::Kokoro(k)
        }
        #[cfg(not(feature = "kokoro"))]
        "kokoro" => {
            anyhow::bail!(
                "TTS_PROVIDER=kokoro requires the 'kokoro' feature: cargo run --features kokoro"
            );
        }
        _ => {
            anyhow::bail!(
                "Unknown TTS_PROVIDER '{}'. Available: avspeech, kokoro",
                config.tts_provider
            );
        }
    };
    let tts_sample_rate = tts.sample_rate();
    let tts = Arc::new(tts);

    // ── Audio output ──────────────────────────────────────────────────────────
    let audio_output = Arc::new(AudioOutput::new(config.audio_output_device.as_deref())?);
    info!(
        target: "audio",
        "Audio output: {}Hz, {}ch",
        audio_output.sample_rate(),
        audio_output.channels()
    );

    // ── Audio capture ─────────────────────────────────────────────────────────
    let audio_capture = AudioCapture::new(config.audio_input_device.as_deref())?;
    let source_sample_rate = audio_capture.sample_rate();
    info!(target: "audio", "Audio input: {}Hz", source_sample_rate);

    let samples_per_chunk = config.samples_per_chunk();
    let (tx, rx) = bounded(AUDIO_CHANNEL_CAPACITY);
    let _stream = audio_capture.start_capture(tx.clone(), samples_per_chunk)?;

    // Event channel for STT+VAD events  
    let (stt_tx, mut stt_rx) = mpsc::channel::<SpeechEvent>(32);
    
    let mut speech_buffer = AudioBuffer::new(source_sample_rate, MAX_SPEECH_BUFFER_SECS);
    let mut t_speech_start: Option<Instant> = None;

    // ── Continuous audio accumulation ─────────────────────────────────────────
    let turn_commit_counter = Arc::new(AtomicU64::new(0));
    let mut last_cleared_commit: u64 = 0;
    let mut speech_buffer_start_offset: usize = 0;

    // ── Ambient state machine ─────────────────────────────────────────────────
    // Counts consecutive VAD segments where the speaker was NOT the enrolled user.
    // When it reaches config.speaker_ambient_trigger the bot silently switches to Ambient.
    let mut non_user_streak: u8 = 0;
    // Tracks when speech last arrived; used to auto-return from Ambient to Active.
    let mut last_speech_at: Instant = Instant::now();

    // ── Pipeline shared state & events ────────────────────────────────────────
    let shared = Arc::new(SharedSession::new());
    let events = Arc::new(PipelineEvents::new());
    // play_cancel: AtomicBool for AudioOutput::play_blocking, which runs in
    // spawn_blocking and cannot be abort()'ed from async code directly.
    let play_cancel = Arc::new(AtomicBool::new(false));

    // TTS mute toggle (controlled from TUI).
    let tts_muted = Arc::new(AtomicBool::new(false));

    // Remote device: shared sender for routing TTS audio to WebSocket.
    // When Some, tts_task sends audio here instead of CPAL play_blocking.
    #[cfg(feature = "remote")]
    let remote_tts_tx: Arc<tokio::sync::Mutex<Option<tokio::sync::mpsc::Sender<remote::protocol::TtsAudioPacket>>>> =
        Arc::new(tokio::sync::Mutex::new(None));

    // TUI event channel — pipeline tasks send events here for the TUI to render.
    #[cfg(feature = "tui")]
    let (tui_tx, tui_rx) = tokio::sync::mpsc::unbounded_channel::<tui::events::TuiEvent>();

    // Agent results that arrived while LLM was busy — processed in order when idle.
    let mut pending_agent_results: std::collections::VecDeque<(String, String)> =
        std::collections::VecDeque::new();
    // Tracks the agent result currently being announced so it can be re-queued
    // if a barge-in interrupts it before the user hears it.
    let mut current_agent_announcement: Option<(String, String)> = None;
    // ACP permission gate: when Some, the next STT result is routed to this
    // sender as the user's yes/no answer rather than starting the LLM pipeline.
    let mut pending_agent_question: Option<tokio::sync::oneshot::Sender<String>> = None;

    // Utterance epoch: incremented on every SpeechStart.  The spawned
    // STT→vad_finish task captures the epoch at spawn time and checks it
    // before firing — if a newer SpeechStart occurred, the result is stale.
    let utterance_epoch = Arc::new(AtomicU64::new(0));

    // ── Spawn permanent pipeline tasks ────────────────────────────────────────
    {
        let shared_c            = Arc::clone(&shared);
        let events_c            = Arc::clone(&events);
        let llm_session_c       = Arc::clone(&llm_session);
        let llm_client_c        = llm_client.clone();
        let db_c                = db.clone();
        let tools_c             = Arc::clone(&tools);
        let shared_history_c    = Arc::clone(&shared_history);
        let turn_commit_c       = Arc::clone(&turn_commit_counter);
        let proactive_tx_c      = proactive_tx.clone();
        #[cfg(feature = "tui")]
        let tui_tx_c = tui_tx.clone();
        tokio::spawn(async move {
            llm_task(
                shared_c, events_c, llm_session_c, llm_client_c,
                db_c, session_id, tools_c, shared_history_c, turn_commit_c,
                proactive_tx_c,
                #[cfg(feature = "tui")]
                tui_tx_c,
            ).await;
        });
    }
    {
        let shared_c = Arc::clone(&shared);
        let events_c = Arc::clone(&events);
        tokio::spawn(async move { sen_task(shared_c, events_c).await; });
    }
    {
        let shared_c      = Arc::clone(&shared);
        let events_c      = Arc::clone(&events);
        let tts_c         = Arc::clone(&tts);
        let audio_out_c   = Arc::clone(&audio_output);
        let play_cancel_c = Arc::clone(&play_cancel);
        let tts_muted_c   = Arc::clone(&tts_muted);
        #[cfg(feature = "tui")]
        let tui_tx_c = tui_tx.clone();
        #[cfg(feature = "remote")]
        let remote_tts_tx_c = Arc::clone(&remote_tts_tx);
        tokio::spawn(async move {
            tts_task(shared_c, events_c, tts_c, audio_out_c, tts_sample_rate, play_cancel_c,
                     tts_muted_c,
                     #[cfg(feature = "tui")]
                     tui_tx_c,
                     #[cfg(feature = "remote")]
                     remote_tts_tx_c,
            ).await;
        });
    }
    {
        let shared_c      = Arc::clone(&shared);
        let events_c      = Arc::clone(&events);
        let llm_session_c = Arc::clone(&llm_session);
        let background_client_c = background_client.clone();
        let db_c          = db.clone();
        let context_tokens = config.llm_context_tokens;
        let keep_turns    = config.llm_summary_keep_turns;
        let threshold_pct = config.llm_consolidation_threshold_pct;
        let idle_secs     = config.llm_idle_consolidation_secs;
        let idle_min_pct  = config.llm_idle_min_context_pct;
        let base_prompt   = config.llm_system_prompt.clone();
        let tool_section_c = tool_section.clone();
        tokio::spawn(async move {
            consolidation_task(
                shared_c, events_c, llm_session_c, background_client_c, db_c,
                session_id, context_tokens, keep_turns, threshold_pct, idle_secs, idle_min_pct,
                base_prompt, tool_section_c,
            ).await;
        });
    }

    info!(target: "voicebot", "Ready. Speak to interact...");

    // ── TUI ─────────────────────────────────────────────────────────────────
    #[cfg(feature = "tui")]
    {
        let shared_c = Arc::clone(&shared);
        let events_c = Arc::clone(&events);
        let tts_muted_c = Arc::clone(&tts_muted);
        let conv_mode_tui = Arc::clone(&conv_mode);
        tokio::spawn(async move {
            if let Err(e) = tui::run(tui_rx, shared_c, events_c, tts_muted_c, conv_mode_tui).await {
                tracing::error!("TUI error: {e}");
            }
            // TUI quit → exit process.
            std::process::exit(0);
        });
    }

    // ── Remote device WebSocket server ─────────────────────────────────────
    #[cfg(feature = "remote")]
    if let Some(ws_port) = config.ws_port {
        let remote_state = Arc::new(remote::server::RemoteState {
            audio_tx: tx.clone(),
            samples_per_chunk,
            cancel_tx: events.cancel_tx.clone(),
            play_cancel: Arc::clone(&play_cancel),
            tts_audio_tx: Arc::clone(&remote_tts_tx),
            connected: AtomicBool::new(false),
        });
        tokio::spawn(async move {
            if let Err(e) = remote::server::start_server(ws_port, remote_state).await {
                error!(target: "remote", "WebSocket server error: {e}");
            }
        });
    }

    // ── Startup consolidation (if context already exceeds idle threshold) ────
    {
        let needs = {
            let s = llm_session.lock().unwrap();
            s.needs_consolidation(config.llm_context_tokens, config.llm_idle_min_context_pct)
        };
        if needs {
            info!(target: "memory", "Startup: context exceeds idle threshold — running silent consolidation before greeting");
            run_consolidation_cycle(
                &background_client, &db, session_id, &llm_session,
                config.llm_summary_keep_turns, &config.llm_system_prompt, &tool_section,
            ).await;
        }
    }

    // ── Startup greeting ──────────────────────────────────────────────────────
    {
        let now = chrono::Local::now();
        let time_str = now.format("%H:%M").to_string();
        let date_str = now.format("%d/%m/%Y").to_string();
        let notification = format!(
            "[Sistema: el voicebot acaba de arrancar. Son las {time_str}, del día {date_str}\n\
             Saluda al usuario de forma natural y muy concisa.]"
        );
        *shared.transliterated_text.lock().unwrap() = notification;
        events.vad_finish.notify_one();
    }

    let mut proactive_rx = proactive_rx;
    tokio::select! {
        _ = async {
            loop {
                // If idle and there are pending agent results, inject the next one.
                if !shared.llm_busy.load(Ordering::SeqCst) && current_agent_announcement.is_none()
                    && let Some((task, result)) = pending_agent_results.pop_front() {
                        let notification = format!(
                            "[Sistema: una tarea en segundo plano ha terminado.]\n\
                             Tarea: {task}\n\
                             Resultado: {result}\n\
                             Informa al usuario de forma natural y concisa."
                        );
                        *shared.transliterated_text.lock().unwrap() = notification;
                        current_agent_announcement = Some((task, result));
                        events.vad_finish.notify_one();
                    }
                    // Clear announcement tracker once LLM becomes idle again.
                    if current_agent_announcement.is_some() && !shared.llm_busy.load(Ordering::SeqCst) {
                    current_agent_announcement = None;
                }

                let chunk: AudioChunk = tokio::select! {
                    result = rx.recv() => match result {
                        Ok(c) => c,
                        Err(e) => {
                            error!(target: "audio", "Audio channel closed: {}", e);
                            #[cfg(feature = "tui")]
                            tui_tx.send(tui::events::TuiEvent::Error(format!("Audio channel closed: {e}"))).ok();
                            break;
                        }
                    },
                    Some(event) = proactive_rx.recv() => {
                        match event {
                            ProactiveEvent::AgentResult { task, result, tool_call_id } => {
                                if let Some(id) = tool_call_id {
                                    // Background tool call: inject a proper OpenAI tool
                                    // result message so the LLM continues naturally from
                                    // its own tool_call instead of being re-prompted via a
                                    // synthetic user message (which led the model to
                                    // re-call the same tool in a loop).
                                    let tool_result_msg = serde_json::json!({
                                        "role": "tool",
                                        "tool_call_id": id,
                                        "content": result,
                                    });
                                    {
                                        let mut s = llm_session.lock().unwrap();
                                        s.add_tool_exchange(vec![tool_result_msg.clone()]);
                                    }
                                    {
                                        let db_c = db.clone();
                                        let exchange = vec![tool_result_msg];
                                        tokio::spawn(async move {
                                            if let Err(e) = db_c.save_tool_exchanges(session_id, &exchange).await {
                                                warn!(target: "db", "Failed to save tool_result exchange: {}", e);
                                            }
                                        });
                                    }
                                    shared.pending_tool_response.store(true, Ordering::SeqCst);
                                    if !shared.llm_busy.load(Ordering::SeqCst) {
                                        events.vad_finish.notify_one();
                                    }
                                    // If the LLM is busy, don't interrupt — when the
                                    // active turn ends, llm_busy drops and the next
                                    // SpeechEnd or idle cycle will see
                                    // pending_tool_response and trigger a fresh turn.
                                } else if !shared.llm_busy.load(Ordering::SeqCst) {
                                    pending_agent_results.push_front((task, result));
                                } else {
                                    pending_agent_results.push_back((task, result));
                                }
                            }
                            ProactiveEvent::InferenceDaemon { .. } => {}
                            ProactiveEvent::AgentQuestion { question, options, response_tx } => {
                                // Cancel active pipeline so the bot can ask the permission question.
                                if shared.llm_busy.load(Ordering::SeqCst) {
                                    events.cancel_tx.send(()).ok();
                                    play_cancel.store(true, Ordering::SeqCst);
                                    tokio::time::sleep(std::time::Duration::from_millis(25)).await;
                                    play_cancel.store(false, Ordering::SeqCst);
                                    if let Some(announcement) = current_agent_announcement.take() {
                                        pending_agent_results.push_front(announcement);
                                    }
                                }
                                pending_agent_question = Some(response_tx);
                                let opts_str = options.join(" / ");
                                let prompt = format!(
                                    "[Sistema: el agente ACP necesita permiso para realizar una acción.]\n\
                                     Acción solicitada: {question}\n\
                                     Opciones: {opts_str}\n\
                                     Pregunta al usuario de forma natural si desea permitirlo (sí/no)."
                                );
                                *shared.transliterated_text.lock().unwrap() = prompt;
                                events.vad_finish.notify_one();
                            }
                        }
                        continue;
                    },
                };

                // Downmix to mono
                let mono: Vec<f32> = if chunk.channels > 1 {
                    chunk.samples
                        .chunks(chunk.channels as usize)
                        .map(|f| f.iter().sum::<f32>() / chunk.channels as f32)
                        .collect()
                } else {
                    chunk.samples
                };

                // Process audio through unified STT+VAD - dispatches events asynchronously
                sttvad.process_audio(&mono, &stt_tx).await.ok();

                // Consume events from the channel immediately
                while let Ok(event) = stt_rx.try_recv() {
                    match event {
                        SpeechEvent::SpeechStart => {
                            t_speech_start = Some(Instant::now());
                            info!(target: "performance", "[+0ms] SpeechStart");
                            #[cfg(feature = "tui")]
                            tui_tx.send(tui::events::TuiEvent::StateChange(
                                tui::events::PipelineState::Listening,
                            )).ok();
                            last_speech_at = Instant::now();

                            // Always fire VAD_DETECTED on new speech to cancel active LLM/TTS.
                            info!(target: "pipeline", "SpeechStart — firing VAD_DETECTED");
                            events.cancel_tx.send(()).ok();
                            play_cancel.store(true, Ordering::SeqCst);
                            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                            play_cancel.store(false, Ordering::SeqCst);
                            if let Some(announcement) = current_agent_announcement.take() {
                                info!(target: "pipeline", "SpeechStart interrupted agent announcement — re-queueing");
                                pending_agent_results.push_front(announcement);
                            }

                            // If a turn was committed since the last clear, start fresh.
                            let current_commits = turn_commit_counter.load(Ordering::SeqCst);
                            if current_commits > last_cleared_commit {
                                speech_buffer.clear();
                                last_cleared_commit = current_commits;
                            }
                            speech_buffer_start_offset = speech_buffer.sample_count();
                            speech_buffer.push(&mono);

                            // Invalidate any stale results from prior utterance.
                            utterance_epoch.fetch_add(1, Ordering::SeqCst);
                        }
                        SpeechEvent::Speech(partial_text) => {
                            speech_buffer.push(&mono);
                            debug!(target: "stt", "Partial: {}", partial_text);
                            // #[cfg(feature = "tui")]
                            // tui_tx.send(tui::events::TuiEvent::UserMessage(partial_text, InputSource::Voice)).ok();
                        }
                        SpeechEvent::SpeechEnd(final_stream_text) => {
                            speech_buffer.push(&mono);

                            let segment_duration_ms = t_speech_start.as_ref()
                                .map(|t| t.elapsed().as_millis() as u32)
                                .unwrap_or(0);

                            if segment_duration_ms < MIN_SPEECH_DURATION_MS {
                                debug!(target: "pipeline", "Too short ({}ms), skipping", segment_duration_ms);
                                continue;
                            }

                            let current_commits = turn_commit_counter.load(Ordering::SeqCst);
                            if current_commits > last_cleared_commit {
                                last_cleared_commit = current_commits;
                            }
                            let audio = speech_buffer.get_samples_from(speech_buffer_start_offset);
                            let duration_ms = audio.len() as u32 * 1000 / source_sample_rate;

                            info!(target: "pipeline", "Speech: {}ms (segment {}ms)", duration_ms, segment_duration_ms);
                            
                            // Use streaming transcription result - it's already complete when SpeechEnd fires
                            let mut segment_text = final_stream_text;
                            
                            // Fallback to transcribe_complete if streaming didn't produce text
                            if segment_text.trim().is_empty()
                                && let Ok(text) = sttvad.transcribe_complete(&audio) {
                                    segment_text = text;
                                }

                            #[cfg(feature = "tui")]
                            tui_tx.send(tui::events::TuiEvent::StateChange(
                                tui::events::PipelineState::Transcribing,
                            )).ok();
                            
                            let vad_elapsed = t_speech_start.take()
                                .map(|t| t.elapsed().as_millis()).unwrap_or(0);
                            info!(target: "performance", "[+{}ms] VAD end ({}ms speech)", vad_elapsed, duration_ms);
                            *shared.t_vad_end.lock().unwrap() = Some(Instant::now());

                            last_speech_at = Instant::now();

                            // ── Speaker verification ──────────────────────────────
                            let mut is_main_speaker = true;
                            let mut speaker_label = "Usuario".to_string();

                            if let Some(ref mut sv) = speaker_verifier {
                                match sv.verify(config.sample_rate, &audio) {
                                    SpeakerVerdict::Enrolled { id, ref label } => {
                                        speaker_label = label.clone();
                                        if id == 0 {
                                            info!(target: "speaker", "Main speaker enrolled — processing utterance");
                                            non_user_streak = 0;
                                        } else {
                                            info!(target: "speaker", "Speaker {} enrolled — buffering", label);
                                            is_main_speaker = false;
                                        }
                                    }
                                    SpeakerVerdict::Known { id, ref label, similarity } => {
                                        speaker_label = label.clone();
                                        if id == 0 {
                                            debug!(target: "speaker", "Main speaker verified (similarity={similarity:.3})");
                                            non_user_streak = 0;
                                        } else {
                                            info!(
                                                target: "speaker",
                                                "Speaker {} (similarity={similarity:.3}) — buffering \
                                                 (streak={}/{})",
                                                label, non_user_streak, config.speaker_ambient_trigger
                                            );
                                            is_main_speaker = false;
                                            non_user_streak = non_user_streak.saturating_add(1);
                                            if non_user_streak >= config.speaker_ambient_trigger {
                                                let mut mode = conv_mode.lock().unwrap();
                                                if *mode == ConversationMode::Active {
                                                    *mode = ConversationMode::Ambient;
                                                    info!(
                                                        target: "pipeline",
                                                        "Ambient mode: {} consecutive non-user voices — switching automatically",
                                                        non_user_streak
                                                    );
                                                }
                                            }
                                        }
                                    }
                                    SpeakerVerdict::Unknown { similarity } => {
                                        speaker_label = "Ambiente".to_string();
                                        non_user_streak = non_user_streak.saturating_add(1);
                                        info!(
                                            target: "speaker",
                                            "Unknown speaker (similarity={similarity:.3}) — buffering \
                                             (streak={non_user_streak}/{})",
                                            config.speaker_ambient_trigger
                                        );
                                        is_main_speaker = false;
                                        if non_user_streak >= config.speaker_ambient_trigger {
                                            let mut mode = conv_mode.lock().unwrap();
                                            if *mode == ConversationMode::Active {
                                                *mode = ConversationMode::Ambient;
                                                info!(
                                                    target: "pipeline",
                                                    "Ambient mode: {} consecutive non-user voices — switching automatically",
                                                    non_user_streak
                                                );
                                            }
                                        }
                                    }
                                }
                            }

                            // ── Non-main speaker: spawn background transcription ─────
                            if !is_main_speaker {
                                let amb_c = Arc::clone(&ambient_buffer);
                                let label = speaker_label.clone();
                                let audio_for_task = audio.clone();
                                let lang = config.language.clone();
                                let wm = config.whisper_model.clone();
                                let vm = config.vad_model.clone();
                                let sms = config.vad_silence_ms;
                                
                                // Spawn background transcription using transcribe_complete  
                                tokio::spawn(async move {
                                    let t0 = Instant::now();
                                    let config = WhisperSTTVADConfig {
                                        whisper_model: wm,
                                        vad_model: vm,
                                        language: lang,
                                        silence_ms: sms,
                                    };
                                    if let Ok(vad) = WhisperSTTVAD::new(config)
                                        && let Ok(text) = vad.transcribe_complete(&audio_for_task)
                                            && !text.is_empty() {
                                                amb_c.lock().unwrap().push(label.clone(), text.clone());
                                                debug!(target: "pipeline", "Ambient buffer ← {label}: {text} ({}ms)", t0.elapsed().as_millis());
                                        }
                                });
                                continue;
                            }

                            let mode_snapshot = conv_mode.lock().unwrap().clone();
                            let ambient_locked = mode_snapshot == ConversationMode::AmbientLocked;
                            let ambient_auto   = mode_snapshot == ConversationMode::Ambient;
                            let wake_word_check = config.wake_word.clone();

                            // ── ACP permission gate ───────────────────────────────
                            if let Some(resp_tx) = pending_agent_question.take() {
                                let audio_for_task = audio.clone();
                                let wm = config.whisper_model.clone();
                                let vm = config.vad_model.clone();
                                let lang = config.language.clone();
                                let sms = config.vad_silence_ms;
                                
                                tokio::spawn(async move {
                                    let t0 = Instant::now();
                                    let config = WhisperSTTVADConfig {
                                        whisper_model: wm,
                                        vad_model: vm,
                                        language: lang,
                                        silence_ms: sms,
                                    };
                                    let answer = if let Ok(vad) = WhisperSTTVAD::new(config) {
                                        vad.transcribe_complete(&audio_for_task).unwrap_or_default()
                                    } else {
                                        String::new()
                                    };
                                    info!(target: "acp", "STT for permission question took {}ms", t0.elapsed().as_millis());
                                    let outcome = map_answer_to_outcome(&answer);
                                    info!(target: "acp", "Permission answer: {:?} → {}", answer, outcome);
                                    let _ = resp_tx.send(outcome);
                                });
                                continue;
                            }

                            // Use streaming result directly - no additional STT call needed
                            let stt_elapsed_ms = segment_duration_ms as u128;
                            info!(target: "performance", "[+{}ms] STT transcription complete (audio={}samples, {}chars)", 
                                stt_elapsed_ms, audio.len(), segment_text.len());
                            debug!(target: "stt", "Segment final: {}", segment_text);

                            if segment_text.trim().is_empty() {
                                debug!(target: "pipeline", "Empty transcription — skipping");
                                events.vad_finish.notify_one();
                                continue;
                            }

                            let mut final_text = segment_text;

                            // Ambient (locked) — only respond to wake word
                            if ambient_locked {
                                let lower = final_text.to_lowercase();
                                if !lower.contains(&wake_word_check) {
                                    ambient_buffer.lock().unwrap()
                                        .push("Usuario".to_string(), final_text.clone());
                                    debug!(target: "pipeline", "Ambient (locked): no wake word — buffered");
                                    events.vad_finish.notify_one();
                                    continue;
                                }
                                info!(target: "pipeline", "Ambient (locked): wake word detected");
                            } else if ambient_auto {
                                *conv_mode.lock().unwrap() = ConversationMode::Active;
                                info!(target: "pipeline", "Auto-ambient: main user spoke — returning Active");
                            }

                            // Inject ambient context if query contains a referential.
                            final_text = {
                                let buf = ambient_buffer.lock().unwrap();
                                if crate::audio::ambient_buffer::has_referential(&final_text) {
                                    if let Some(ctx) = buf.format_context() {
                                        format!("{ctx}\n---\n{final_text}")
                                    } else {
                                        final_text
                                    }
                                } else {
                                    final_text
                                }
                            };

                           if final_text.trim().is_empty() {
                                debug!(target: "pipeline", "Empty after context injection — skipping");
                                events.vad_finish.notify_one();
                                continue;
                            }

                            // Store transcript and trigger LLM processing
                            *shared.transliterated_text.lock().unwrap() = final_text;
                            shared.stt_result_pending.store(true, Ordering::SeqCst);
                            if let Some(t0) = shared.t_vad_end.lock().unwrap().as_ref() {
                                info!(target: "performance", "[+{}ms] STT done → VAD_FINISH", t0.elapsed().as_millis());
                            }
                            events.vad_finish.notify_one();
                        }
                        SpeechEvent::Silence => {
                            // Manage ambient mode transitions on silence
                            {
                                let mut mode = conv_mode.lock().unwrap();
                                if *mode == ConversationMode::Active
                                    && last_speech_at.elapsed().as_secs() >= config.ambient_clear_secs
                                {
                                    *mode = ConversationMode::Ambient;
                                    non_user_streak = 0;
                                    info!(
                                        target: "pipeline",
                                        "Ambient mode: {}s of silence — returning to Ambient",
                                        config.ambient_clear_secs
                                    );
                                }
                            }
                        }
                    }
                }
            }
        } => {}
        _ = tokio::signal::ctrl_c() => {
            info!(target: "voicebot", "Shutting down...");
            events.cancel_tx.send(()).ok();
            play_cancel.store(true, Ordering::SeqCst);
        }
    }

    Ok(())
}

/// Await a pending playback handle, logging any error. No-op if `None`.
#[allow(dead_code)]
async fn drain_play(handle: &mut Option<tokio::task::JoinHandle<anyhow::Result<()>>>) {
    if let Some(h) = handle.take() {
        match h.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => error!(target: "audio", "Playback error: {}", e),
            Err(e)     => error!(target: "audio", "Playback task panicked: {}", e),
        }
    }
}

// ── Permanent pipeline tasks ──────────────────────────────────────────────────

/// LLM task: blocks on VAD_FINISH, runs the LLM+tools pipeline, fires events.
///
/// Corresponds to the LLM thread in doc/PROCESS_ARCHITECTURE.md.
#[allow(clippy::too_many_arguments)]
async fn llm_task(
    shared: Arc<SharedSession>,
    events: Arc<PipelineEvents>,
    llm_session: Arc<Mutex<LlmSession>>,
    llm_client: OpenAIClient,
    db: Database,
    session_id: uuid::Uuid,
    tools: Arc<ToolRegistry>,
    shared_history: Arc<RwLock<String>>,
    turn_commit_counter: Arc<AtomicU64>,
    proactive_tx: mpsc::Sender<ProactiveEvent>,
    #[cfg(feature = "tui")]
    tui_tx: tui::events::TuiEventTx,
) {
    let pipeline_id = PIPELINE_RUN_ID.fetch_add(1, Ordering::SeqCst);
    let mut cancel_rx = events.cancel_tx.subscribe();

    loop {
        // Block until VAD_FINISH; ignore cancels while idle.
        loop {
            tokio::select! {
                _ = events.vad_finish.notified() => { break; }
                _ = cancel_rx.recv() => { /* stale cancel while idle — keep waiting */ }
            }
        }

        // If context consolidation is in progress, wait for it to finish before
        // processing the next user turn. Audio capture and STT continue — the
        // transcript stays in transliterated_text until we drain it below.
        while shared.consolidation_active.load(Ordering::SeqCst) {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        shared.llm_busy.store(true, Ordering::SeqCst);

         // Drain accumulated transcription.
        let text = std::mem::take(&mut *shared.transliterated_text.lock().unwrap());

        // Continuation from a background tool result: the audio loop injected
        // an OpenAI-format `tool` message into the session already, so we call
        // the LLM again without adding a synthetic user turn.
        let tool_continuation = shared.pending_tool_response.swap(false, Ordering::SeqCst);

        if text.trim().is_empty() && !tool_continuation {
            shared.llm_busy.store(false, Ordering::SeqCst);
            while cancel_rx.try_recv().is_ok() {}
            continue;
        }

        // Transcription result has been drained — allow fresh utterances now.
        shared.stt_result_pending.store(false, Ordering::SeqCst);

        // Ambient mode: text pipeline injections bypass the wake-word check
        // because the audio loop already validated them before firing vad_finish.
        // (Voice utterances are filtered inside the SpeechEnd spawned task.)
        if tool_continuation {
            info!(target: "pipeline", "[pipe={}] Tool result delivered — continuing turn", pipeline_id);
        } else {
            info!(target: "pipeline", "[pipe={}] User: {}", pipeline_id, text);
        }

        #[cfg(feature = "tui")]
        if !tool_continuation {
            let source = if shared.text_input_pending.swap(false, Ordering::SeqCst) {
                tui::events::InputSource::Text
            } else {
                tui::events::InputSource::Voice
            };
            tui_tx.send(tui::events::TuiEvent::UserMessage {
                text: text.clone(),
                source,
            }).ok();
            tui_tx.send(tui::events::TuiEvent::StateChange(
                tui::events::PipelineState::Thinking,
            )).ok();
        }

        let messages_snapshot = llm_session.lock().unwrap().messages.clone();

        if !tool_continuation {
            // Add user turn and signal the audio loop to clear the speech buffer.
            {
                let mut s = llm_session.lock().unwrap();
                s.add_user_turn(&text);
                turn_commit_counter.fetch_add(1, Ordering::SeqCst);
                if let Ok(mut h) = shared_history.write() {
                    *h = format_history(&s.messages);
                }
            }

            // Save user message to DB (non-blocking).
            {
                let db_c = db.clone();
                let text_c = text.clone();
                tokio::spawn(async move {
                    if let Err(e) = db_c.save_message(session_id, "User", &text_c).await {
                        warn!(target: "db", "Failed to save user message: {}", e);
                    }
                });
            }
        } else {
            // Tool continuation: session history already has the tool result
            // appended by the audio loop. Keep shared_history in sync.
            if let Ok(mut h) = shared_history.write() {
                let s = llm_session.lock().unwrap();
                *h = format_history(&s.messages);
            }
            turn_commit_counter.fetch_add(1, Ordering::SeqCst);
        }

        // Reset post-finished flag and clear assistant text buffer for new turn.
        shared.llm_post_finished.store(false, Ordering::SeqCst);
        shared.assistant_text.lock().unwrap().clear();

        // Tool call loop — allows the model to call tools before its spoken response.
        let tool_defs = tools.tool_definitions();
        info!(target: "pipeline", "LLM request: {} tool(s) available: {:?}",
            tool_defs.len(),
            tool_defs.iter().filter_map(|t| t["function"]["name"].as_str()).collect::<Vec<_>>()
        );
        let mut messages = llm_session.lock().unwrap().all_messages_api();
        let base_msg_len = messages.len();
        let mut final_response = String::new();
        let mut committed = false;
        let mut cancelled = false;
        let mut first_token_logged = false;

        'pipeline: {
            'tool_loop: for iter in 0..MAX_TOOL_ITERATIONS {
                info!(target: "performance", "LLM request [pipe={}]", pipeline_id);
                let (token_rx, stream_handle) = match llm_client.stream(&messages, &tool_defs).await {
                    Ok(r)  => r,
                    Err(e) => {
                        error!(target: "llm", "LLM error: {}", e);
                        #[cfg(feature = "tui")]
                        tui_tx.send(tui::events::TuiEvent::Error(format!("LLM error: {e}"))).ok();
                        shared.sentences.lock().unwrap()
                            .push_back("Lo siento, no pude conectar con el modelo de lenguaje.".to_string());
                        events.sentence_ready.notify_one();
                        break 'pipeline;
                    }
                };

                *shared.t_llm_post_send.lock().unwrap() = Some(Instant::now());

                let mut token_rx = token_rx;
                let mut llm_text = String::new();
                let mut tool_call: Option<(String, String)> = None;

                // Stream tokens; forward each to SEN via shared.assistant_text + event.
                loop {
                    tokio::select! {
                        token = token_rx.recv() => {
                            match token {
                                Some(StreamToken::Content(t)) => {
                                    // Strip leading newlines from the first token of each
                                    // turn. Qwen3 in thinking mode emits \n\n after </think>
                                    // before the actual response starts.
                                    let t = if llm_text.is_empty() {
                                        t.trim_start_matches('\n').to_string()
                                    } else {
                                        t
                                    };
                                    if t.is_empty() { continue; }
                                    if !first_token_logged {
                                        first_token_logged = true;
                                        if let Some(t0) = shared.t_llm_post_send.lock().unwrap().as_ref() {
                                            info!(target: "performance", "[+{}ms] LLM first token (TTFT)", t0.elapsed().as_millis());
                                        }
                                    }
                                    llm_text.push_str(&t);
                                    shared.assistant_text.lock().unwrap().push_str(&t);
                                    events.llm_post_received.notify_one();
                                    #[cfg(feature = "tui")]
                                    tui_tx.send(tui::events::TuiEvent::AssistantToken(t)).ok();
                                }
                                Some(StreamToken::ToolCall { name, args }) => {
                                    info!(target: "pipeline", "ToolCall received: name={} args={}", name, args);
                                    tool_call = Some((name, args));
                                    break;
                                }
                                None => {
                                    // Stream finished — fire LLM_POST_FINISHED.
                                    shared.llm_post_finished.store(true, Ordering::SeqCst);
                                    events.llm_post_received.notify_one(); // wake SEN to flush
                                    events.llm_post_finished.notify_one(); // wake SUM
                                    #[cfg(feature = "tui")]
                                    tui_tx.send(tui::events::TuiEvent::AssistantDone).ok();
                                    break;
                                }
                            }
                        }
                        _ = cancel_rx.recv() => {
                            cancelled = true;
                            drop(token_rx);
                            stream_handle.abort();
                            break;
                        }
                    }
                }

                if cancelled { break 'pipeline; }

                match tool_call {
                    Some((name, args)) => {
                        if tools.is_background(&name) {
                            // Background tool: speak ACK via TTS (not persisted), commit the
                            // tool_call to session + DB so the later tool_result lines up by id,
                            // then spawn the task and exit without an assistant turn. When the
                            // result arrives, the audio loop appends the matching tool_result
                            // message and triggers llm_task to produce the real response.
                            let ack = match name.as_str() {
                                "web_search" => "Buscando.",
                                "run_shell"  => "Ejecutando.",
                                _            => "Procesando en segundo plano, le aviso al terminar.",
                            };
                            {
                                let mut text = shared.assistant_text.lock().unwrap();
                                text.push_str(ack);
                            }
                            events.llm_post_received.notify_one();
                            shared.llm_post_finished.store(true, Ordering::SeqCst);
                            events.llm_post_received.notify_one(); // flush SEN
                            events.llm_post_finished.notify_one(); // wake consolidation

                            let tc_id = format!("bg_{}_{}_{}", pipeline_id, iter, name);
                            let tool_call_msg = serde_json::json!({
                                "role": "assistant",
                                "content": serde_json::Value::Null,
                                "tool_calls": [{
                                    "id": tc_id,
                                    "type": "function",
                                    "function": {"name": &name, "arguments": &args}
                                }]
                            });
                            messages.push(tool_call_msg);

                            // Persist tool_call to session + DB so `all_messages_api()` on the
                            // continuation turn has the matching assistant tool_call entry.
                            {
                                let tool_exchanges = messages[base_msg_len..].to_vec();
                                {
                                    let mut s = llm_session.lock().unwrap();
                                    s.add_tool_exchange(tool_exchanges.clone());
                                    if let Ok(mut h) = shared_history.write() {
                                        *h = format_history(&s.messages);
                                    }
                                }
                                let db_c = db.clone();
                                tokio::spawn(async move {
                                    if let Err(e) = db_c.save_tool_exchanges(session_id, &tool_exchanges).await {
                                        warn!(target: "db", "Failed to save tool_call exchange: {}", e);
                                    }
                                });
                            }

                            let tools_c     = Arc::clone(&tools);
                            let name_c      = name.clone();
                            let args_c      = args.clone();
                            let proactive_c = proactive_tx.clone();
                            let tc_id_c     = tc_id.clone();
                            tokio::spawn(async move {
                                info!(target: "pipeline", "Background tool `{}` started", name_c);
                                let result = tools_c.execute(&name_c, &args_c).await;
                                info!(target: "pipeline", "Background tool `{}` finished ({} chars): {:?}", name_c, result.len(), result);
                                proactive_c.send(ProactiveEvent::AgentResult {
                                    task: name_c,
                                    result,
                                    tool_call_id: Some(tc_id_c),
                                }).await.ok();
                            });

                            // Mark committed so the cancel-rollback path is skipped; exit
                            // without the final_response commit block (no assistant turn).
                            committed = true;
                            break 'pipeline;
                        }

                        // Synchronous tool: execute inline and loop back to LLM.
                        let result = tools.execute(&name, &args).await;
                        info!(target: "pipeline", "Tool[{}] `{}` → {}", iter, name, result);
                        #[cfg(feature = "tui")]
                        tui_tx.send(tui::events::TuiEvent::ToolCall {
                            name: name.clone(),
                            result: result.clone(),
                        }).ok();

                        let tool_call_id = format!("call_{}_{}", name, iter);
                        messages.push(serde_json::json!({
                            "role": "assistant",
                            "content": serde_json::Value::Null,
                            "tool_calls": [{
                                "id": tool_call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": args}
                            }]
                        }));
                        messages.push(serde_json::json!({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": result
                        }));

                        if cancel_rx.try_recv().is_ok() {
                            cancelled = true;
                            break 'pipeline;
                        }
                    }
                    None => {
                        final_response = llm_text;
                        break 'tool_loop;
                    }
                }
            }

            if final_response.is_empty() || cancelled { break 'pipeline; }

            info!(target: "pipeline", "[pipe={}] Assistant: {}", pipeline_id, final_response);

            // Commit assistant response to session and DB.
            {
                let db_c = db.clone();
                let resp_c = final_response.clone();
                let tool_exchanges_c = messages[base_msg_len..].to_vec();
                tokio::spawn(async move {
                    // Persist tool-call exchanges BEFORE the assistant text so the DB
                    // reflects the same ordering the LLM saw: tool_calls → tool_result → response.
                    if !tool_exchanges_c.is_empty()
                        && let Err(e) = db_c.save_tool_exchanges(session_id, &tool_exchanges_c).await {
                            warn!(target: "db", "Failed to save tool exchanges: {}", e);
                        }
                    if let Err(e) = db_c.save_message(session_id, "Assistant", &resp_c).await {
                        warn!(target: "db", "Failed to save assistant message: {}", e);
                    }
                });
            }
            {
                let mut s = llm_session.lock().unwrap();
                let tool_exchanges = messages[base_msg_len..].to_vec();
                if !tool_exchanges.is_empty() {
                    s.add_tool_exchange(tool_exchanges);
                }
                s.add_assistant_turn(&final_response);
            }
            committed = true;
        }

        if !committed && cancelled {
            // Roll back the user turn so history stays consistent.
            llm_session.lock().unwrap().messages = messages_snapshot;
            info!(target: "pipeline", "[pipe={}] Cancelled — session rolled back", pipeline_id);
        }

        shared.llm_busy.store(false, Ordering::SeqCst);
        #[cfg(feature = "tui")]
        tui_tx.send(tui::events::TuiEvent::StateChange(
            tui::events::PipelineState::Idle,
        )).ok();

        // Drain stale cancels before going back to sleep.
        while cancel_rx.try_recv().is_ok() {}
    }
}

/// SEN task: blocks on LLM_POST_RECEIVED, splits assistant_text into sentences.
///
/// Corresponds to the SEN thread in doc/PROCESS_ARCHITECTURE.md.
async fn sen_task(shared: Arc<SharedSession>, events: Arc<PipelineEvents>) {
    let mut cancel_rx = events.cancel_tx.subscribe();
    let mut splitter = SentenceSplitter::new();
    let mut first_sentence_logged = false;

    loop {
        let cancelled = tokio::select! {
            _ = events.llm_post_received.notified() => false,
            _ = cancel_rx.recv() => true,
        };

        if cancelled {
            shared.assistant_text.lock().unwrap().clear();
            splitter = SentenceSplitter::new();
            first_sentence_logged = false;
            while cancel_rx.try_recv().is_ok() {}
            continue;
        }

        // Drain available text into the splitter.
        let new_text = std::mem::take(&mut *shared.assistant_text.lock().unwrap());

        let mut ready_sentences: Vec<String> = Vec::new();
        if !new_text.is_empty()
            && let Some(s) = splitter.push(&new_text) {
                ready_sentences.push(s);
            }

        // If LLM is done streaming, flush any remaining fragment.
        if shared.llm_post_finished.load(Ordering::SeqCst)
            && let Some(s) = splitter.flush() {
                ready_sentences.push(s);
            }

        for sentence in ready_sentences {
            if !first_sentence_logged {
                first_sentence_logged = true;
                if let Some(t0) = shared.t_vad_end.lock().unwrap().as_ref() {
                    let tts_queue_ms = t0.elapsed().as_millis();
                    // TTFT alternativo: tiempo desde VAD end hasta tener primer texto listo para TTS
                    info!(target: "performance", "[+{}ms] first sentence → TTS queue", tts_queue_ms);
                    // Latencia parcial: VAD→LLM→TTS (sin contar síntesis/playback)
                    if let Some(t_llm_sent) = shared.t_llm_post_send.lock().unwrap().as_ref() {
                        info!(target: "performance", "  └─ LLM processing: {}ms", t_llm_sent.elapsed().as_millis());
                    }
                }
            }
            shared.sentences.lock().unwrap().push_back(sentence);
            events.sentence_ready.notify_one();
        }
    }
}

/// TTS task: blocks on SENTENCE_READY, synthesizes and plays each sentence.
///
/// Corresponds to the TTS thread in doc/PROCESS_ARCHITECTURE.md.
async fn tts_task(
    shared: Arc<SharedSession>,
    events: Arc<PipelineEvents>,
    tts: Arc<TtsEngine>,
    audio_output: Arc<AudioOutput>,
    tts_sample_rate: u32,
    play_cancel: Arc<AtomicBool>,
    tts_muted: Arc<AtomicBool>,
    #[cfg(feature = "tui")]
    tui_tx: tui::events::TuiEventTx,
    #[cfg(feature = "remote")]
    remote_tts_tx: Arc<tokio::sync::Mutex<Option<tokio::sync::mpsc::Sender<remote::protocol::TtsAudioPacket>>>>,
) {
    let mut cancel_rx = events.cancel_tx.subscribe();
    let mut play_handle: Option<tokio::task::JoinHandle<anyhow::Result<()>>> = None;
    let mut first_sentence = true;

    loop {
        // Drain the queue immediately; only block when it's empty.
        // `Notify` stores at most one permit, so multiple `notify_one()` calls
        // while we're busy would be lost — we must re-check the queue first.
        let sentence = shared.sentences.lock().unwrap().pop_front();
        let sentence = if let Some(s) = sentence {
            s
        } else {
            // Queue empty — wait for the next sentence or a cancel signal.
            let cancelled = tokio::select! {
                _ = events.sentence_ready.notified() => false,
                _ = cancel_rx.recv() => true,
            };
            if cancelled {
                // Stop playback and discard queued sentences.
                play_cancel.store(true, Ordering::SeqCst);
                if let Some(h) = play_handle.take() { h.abort(); }
                shared.sentences.lock().unwrap().clear();
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                play_cancel.store(false, Ordering::SeqCst);
                first_sentence = true;
                while cancel_rx.try_recv().is_ok() {}
                continue;
            }
            match shared.sentences.lock().unwrap().pop_front() {
                Some(s) => s,
                None => continue,
            }
        };

        // If TTS is muted, skip synthesis and playback entirely.
        if tts_muted.load(Ordering::SeqCst) {
            continue;
        }

        #[cfg(feature = "tui")]
        tui_tx.send(tui::events::TuiEvent::StateChange(
            tui::events::PipelineState::Speaking,
        )).ok();

        // Start synthesis while the previous sentence is still playing.
        let tts_c = Arc::clone(&tts);
        let sentence_c = sentence.clone();
        let synth_handle = tokio::task::spawn_blocking(move || tts_c.synthesize(&sentence_c));

        // Await previous playback, interruptible by cancel.
        // play_cancel AtomicBool signals play_blocking to stop; cancel_rx wakes this select.
        if let Some(h) = play_handle.take() {
            let cancelled = tokio::select! {
                result = h => {
                    if let Ok(Err(e)) = result {
                        error!(target: "audio", "Playback error: {}", e);
                    }
                    false
                },
                _ = cancel_rx.recv() => true,
            };
            if cancelled {
                synth_handle.abort();
                play_cancel.store(true, Ordering::SeqCst);
                shared.sentences.lock().unwrap().clear();
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                play_cancel.store(false, Ordering::SeqCst);
                first_sentence = true;
                while cancel_rx.try_recv().is_ok() {}
                continue;
            }
        }

        // Check cancel once more before awaiting synthesis result.
        if cancel_rx.try_recv().is_ok() {
            synth_handle.abort();
            shared.sentences.lock().unwrap().clear();
            first_sentence = true;
            while cancel_rx.try_recv().is_ok() {}
            continue;
        }

        let samples = match synth_handle.await {
            Ok(Ok(s))  => s,
            Ok(Err(e)) => {
                error!(target: "tts", "TTS synthesis error: {}", e);
                #[cfg(feature = "tui")]
                tui_tx.send(tui::events::TuiEvent::Error(format!("TTS synthesis error: {e}"))).ok();
                continue;
            }
            Err(e) => {
                error!(target: "tts", "TTS task panicked: {}", e);
                #[cfg(feature = "tui")]
                tui_tx.send(tui::events::TuiEvent::Error(format!("TTS task panicked: {e}"))).ok();
                continue;
            }
        };

        if first_sentence {
            first_sentence = false;
            // Mark that we've started playing the first sentence (synthesis done, playback starting)
            shared.first_speech_played.store(true, Ordering::SeqCst);
            if let Some(t0) = shared.t_vad_end.lock().unwrap().as_ref() {
                let latency_ms = t0.elapsed().as_millis();
                info!(target: "performance", "[+{}ms] SpeechStart → FirstAudioPlayback", latency_ms);
            }
        }
        // Route audio: remote WebSocket if connected, otherwise local CPAL.
        #[cfg(feature = "remote")]
        {
            let maybe_tx = remote_tts_tx.lock().await.clone();
            if let Some(tx) = maybe_tx {
                let packet = remote::protocol::TtsAudioPacket {
                    samples,
                    sample_rate: tts_sample_rate,
                };
                if tx.send(packet).await.is_err() {
                    warn!(target: "remote", "Remote TTS channel closed");
                }
                continue;
            }
        }

        let out_c    = Arc::clone(&audio_output);
        let cancel_c = Arc::clone(&play_cancel);
        play_handle = Some(tokio::task::spawn_blocking(move || {
            out_c.play_blocking(&samples, tts_sample_rate, &cancel_c)
        }));
    }
}

/// Core consolidation work: extract profile facts + memories, summarize old
/// turns, rebuild the system prompt, and apply the compacted session.
///
/// Called both by `consolidation_task` (recurring) and at startup when the
/// context already exceeds `LLM_IDLE_MIN_CONTEXT_PCT`.
#[allow(clippy::too_many_arguments)]
async fn run_consolidation_cycle(
    background_client: &OpenAIClient,
    db: &Database,
    session_id: uuid::Uuid,
    llm_session: &Arc<Mutex<LlmSession>>,
    keep_turns: usize,
    base_prompt: &str,
    tool_section: &str,
) {
    let (conversation_text, summary_prompt, turns_to_summarize) = {
        let s = llm_session.lock().unwrap();
        let count = s.summarizable_turn_count(keep_turns);
        let prompt = s.build_summary_prompt(keep_turns);
        let mut conv = String::new();
          for msg in &s.messages[..count.min(s.messages.len())] {
            if let (Some(role), Some(content)) =
                (msg["role"].as_str(), msg["content"].as_str())
                && (role == "user" || role == "assistant") {
                    conv.push_str(role);
                    conv.push_str(": ");
                    conv.push_str(content);
                    conv.push_str("\n\n");
            }
        }
        (conv, prompt, count)
    };

    // Profile facts.
    if !conversation_text.is_empty() {
        let facts = extract_facts(background_client, &conversation_text, "").await;
        for fact in facts {
            if let Err(e) = db.upsert_profile_fact(&fact.key, &fact.value, fact.confidence).await {
                warn!(target: "profile", "Failed to save profile fact '{}': {}", fact.key, e);
            } else {
                debug!(target: "profile", "Profile: {} = {} ({:.0}%)", fact.key, fact.value, fact.confidence * 100.0);
            }
        }
    }

    // Persistent memories.
    let existing_memories = db.load_active_memories().await.unwrap_or_default();
    let mem_result = extract_memories(background_client, &conversation_text, &existing_memories).await;
    for id in &mem_result.archive_ids {
        if let Err(e) = db.deactivate_memory(*id).await {
            warn!(target: "memory", "Failed to archive memory id={}: {}", id, e);
        }
    }
    if !mem_result.new_memories.is_empty() {
        info!(target: "memory", "Extracted {} new memories", mem_result.new_memories.len());
        if let Err(e) = db.save_memories_batch(&mem_result.new_memories, session_id).await {
            warn!(target: "memory", "Failed to save memories: {}", e);
        }
    }
    if !mem_result.archive_ids.is_empty() {
        info!(target: "memory", "Archived {} outdated memories", mem_result.archive_ids.len());
    }

    // Summarize.
    let summary = if let Some(prompt) = summary_prompt {
        match background_client.complete(&prompt).await {
            Ok(s) if !s.is_empty() => { info!(target: "memory", "Summary: {}", s); Some(s) }
            Ok(_) => { warn!(target: "memory", "Summarization returned empty result"); None }
            Err(e) => { warn!(target: "memory", "Summarization failed: {}", e); None }
        }
    } else {
        None
    };

    // Persist summary and rebuild system prompt.
    if let Some(ref summary_text) = summary {
        let prev_through_id = db.get_summary_through_id(session_id).await.unwrap_or(0);
        let through_id = db
            .get_message_id_at_offset(session_id, prev_through_id, turns_to_summarize.saturating_sub(1))
            .await
             .ok()
             .flatten()
             .unwrap_or(0);
         if through_id > 0 && let Err(e) = db.save_summary(session_id, summary_text, through_id).await {
             warn!(target: "db", "Failed to persist summary: {}", e);
         }
     }

    let fresh_profile = db.load_user_profile().await.unwrap_or_default();
    let fresh_profile_facts: Vec<crate::profile::ProfileFact> = fresh_profile
        .into_iter()
        .map(|(key, value, confidence)| crate::profile::ProfileFact { key, value, confidence })
        .collect();
    let fresh_memories = db.load_active_memories().await.unwrap_or_default();
    let new_system_prompt = build_system_prompt(
        base_prompt, &fresh_profile_facts, &fresh_memories, tool_section,
    );

    {
        let mut s = llm_session.lock().unwrap();
        if let Some(ref summary_text) = summary {
            s.apply_summary(summary_text, keep_turns);
        }
        s.set_system_prompt(new_system_prompt);
    }

    info!(
        target: "memory",
        "Consolidation complete — prompt rebuilt ({} profile facts, {} memories, {} recent turns kept)",
        fresh_profile_facts.len(), fresh_memories.len(), keep_turns,
    );
}

/// Context consolidation task: blocks on LLM_POST_FINISHED, runs a full
/// memory consolidation cycle when the context window approaches its limit.
///
/// Replaces the old `sum_task`. Announces to the user via voice before and
/// after the process so they know the bot is temporarily unavailable.
#[allow(clippy::too_many_arguments)]
async fn consolidation_task(
    shared: Arc<SharedSession>,
    events: Arc<PipelineEvents>,
    llm_session: Arc<Mutex<LlmSession>>,
    background_client: OpenAIClient,
    db: Database,
    session_id: uuid::Uuid,
    context_tokens: usize,
    keep_turns: usize,
    threshold_pct: usize,
    idle_consolidation_secs: u64,
    idle_min_context_pct: usize,
    base_prompt: String,
    tool_section: String,
) {
    let mut cancel_rx = events.cancel_tx.subscribe();
    let mut last_turn_at = Instant::now();

    loop {
        // Wait for either: LLM finishes a turn OR idle timeout expires.
        // Cancels are drained but don't interrupt — we're idle anyway.
        let triggered_by_idle = loop {
            let idle_wait = if idle_consolidation_secs > 0 {
                let elapsed = last_turn_at.elapsed().as_secs();
                let remaining = idle_consolidation_secs.saturating_sub(elapsed);
               // Check at most every 60s so we don't spin tight.
                 Duration::from_secs(remaining.clamp(1, 60))
            } else {
                Duration::from_secs(3600) // effectively disabled
            };

            tokio::select! {
                _ = events.llm_post_finished.notified() => {
                    last_turn_at = Instant::now();
                    break false;
                }
                _ = tokio::time::sleep(idle_wait) => {
                    let elapsed = last_turn_at.elapsed().as_secs();
                    if idle_consolidation_secs > 0
                        && elapsed >= idle_consolidation_secs
                        && !shared.llm_busy.load(Ordering::SeqCst)
                    {
                        break true;
                    }
                    // Not idle enough yet — loop and recalculate wait.
                }
                _ = cancel_rx.recv() => {}
            }
        };

        // ── Context check ────────────────────────────────────────────────────
        // Post-turn: consolidate when context >= threshold_pct (hard limit, always enforced).
        // Idle:      consolidate when context >= idle_min_context_pct (proactive, lower bar).
        //            This keeps the context well below the hard limit while the user is away.
        let (needs, approx_tokens, current_pct, msg_count, effective_threshold) = {
            let s = llm_session.lock().unwrap();
            let approx = s.approx_tokens();
            let pct = if context_tokens > 0 { approx * 100 / context_tokens } else { 0 };
            let effective = if triggered_by_idle { idle_min_context_pct } else { threshold_pct };
            let needs = s.needs_consolidation(context_tokens, effective);
            (needs, approx, pct, s.messages.len(), effective)
        };
        info!(
            target: "memory",
            "Context check ({}): ~{} tokens / {} max ({}%) — threshold {}% — {} msgs — consolidation {}",
            if triggered_by_idle { "idle" } else { "post-turn" },
            approx_tokens, context_tokens, current_pct, effective_threshold,
            msg_count,
            if needs { "TRIGGERED" } else { "not needed" },
        );
        if !needs {
            while cancel_rx.try_recv().is_ok() {}
            // If triggered by idle but nothing to consolidate, reset the timer
            // so we don't re-check on every minute tick.
            if triggered_by_idle {
                last_turn_at = Instant::now();
            }
            continue;
        }

        // ── Phase 1: Announce (only for context-limit consolidation, not idle) ─
        if !triggered_by_idle {
            info!(target: "memory", "Context limit approaching — starting announced consolidation");

            // Wait for the LLM to be idle, then send the announcement.
            while shared.llm_busy.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            shared.consolidation_active.store(false, Ordering::SeqCst);
            *shared.transliterated_text.lock().unwrap() =
                "[Sistema: necesitas reorganizar tu memoria para seguir conversando. \
                 Avisa al usuario de que vuelves en unos minutos.]".to_string();
            events.vad_finish.notify_one();

            loop {
                tokio::select! {
                    _ = events.llm_post_finished.notified() => { break; }
                    _ = cancel_rx.recv() => {}
                }
            }
            tokio::time::sleep(Duration::from_secs(3)).await;
            shared.consolidation_active.store(true, Ordering::SeqCst);
            info!(target: "memory", "Pipeline paused — running consolidation...");
        } else {
            info!(target: "memory", "Idle timer — running silent consolidation...");
        }

        // ── Phase 2+3: Extract, summarize, rebuild prompt ───────────────────
        run_consolidation_cycle(
            &background_client, &db, session_id, &llm_session,
            keep_turns, &base_prompt, &tool_section,
        ).await;

        // ── Phase 4: Announce back (only for non-idle consolidation) ─────────
        if !triggered_by_idle {
            shared.consolidation_active.store(false, Ordering::SeqCst);
            let now = chrono::Local::now().format("%H:%M").to_string();
            *shared.transliterated_text.lock().unwrap() = format!(
                "[Sistema: has terminado de reorganizar tu memoria. Son las {now}. \
                 Avisa al usuario de que ya estás disponible de nuevo.]"
            );
            events.vad_finish.notify_one();
            info!(target: "memory", "Consolidation cycle finished — pipeline resumed");
        }

        // Reset idle timer so we don't immediately re-consolidate.
        last_turn_at = Instant::now();
        while cancel_rx.try_recv().is_ok() {}
    }
}

/// Maximum number of sequential tool calls allowed per user turn.
const MAX_TOOL_ITERATIONS: usize = 5;

/// Test-only: run a single pipeline turn end-to-end (transcript → LLM → TTS).
///
/// Replaces the old `run_pipeline` function for e2e tests. Spawns the 4 permanent
/// tasks, fires vad_finish with the given transcript, waits for TTS to drain, then
/// cancels all tasks.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
async fn run_pipeline(
    _audio: Vec<f32>,
    _stt_stream: Arc<crate::stt::SttStream>,
    _cancel: Arc<AtomicBool>,
    tts: Arc<crate::tts::TtsEngine>,
    audio_output: Arc<crate::audio::output::AudioOutput>,
    llm_session: Arc<Mutex<crate::llm::LlmSession>>,
    llm_client: crate::llm::OpenAIClient,
    db: crate::db::Database,
    session_id: uuid::Uuid,
    tts_sample_rate: u32,
    tools: Arc<crate::tools::ToolRegistry>,
    shared_history: Arc<RwLock<String>>,
    context_tokens: usize,
    summary_keep_turns: usize,
    _started_at: std::time::Instant,
    ambient: bool,
    _conv_mode: Arc<Mutex<crate::tools::ConversationMode>>,
    wake_word: String,
    turn_commit_counter: Arc<AtomicU64>,
) {
    use std::sync::atomic::Ordering;

    let transcript = match _stt_stream.await_result(_audio).await {
        Ok(t) => t,
        Err(_) => return,
    };

    if transcript.trim().is_empty() {
        return;
    }

    // Ambient (locked) wake-word check — only applies when user explicitly set ambient mode.
    if ambient {
        let lower = transcript.to_lowercase();
        if !lower.contains(&wake_word.to_lowercase()) {
            return;
        }
    }

    let shared = Arc::new(SharedSession::new());
    let events = Arc::new(PipelineEvents::new());
    let play_cancel = Arc::new(AtomicBool::new(false));

    *shared.transliterated_text.lock().unwrap() = transcript;

    // Spawn tasks.
    let turn_commit_c = Arc::clone(&turn_commit_counter);
    let h_llm = {
        let shared_c = Arc::clone(&shared);
        let events_c = Arc::clone(&events);
        let llm_session_c = Arc::clone(&llm_session);
        let llm_client_c = llm_client.clone();
        let db_c = db.clone();
        let tools_c = Arc::clone(&tools);
        let shared_history_c = Arc::clone(&shared_history);
        let (test_proactive_tx, _test_proactive_rx) = mpsc::channel::<ProactiveEvent>(8);
        tokio::spawn(async move {
            llm_task(
                shared_c, events_c, llm_session_c, llm_client_c,
                db_c, session_id, tools_c, shared_history_c, turn_commit_c,
                test_proactive_tx,
                #[cfg(feature = "tui")]
                { let (tx, _) = tokio::sync::mpsc::unbounded_channel(); tx },
            ).await;
        })
    };
    let h_sen = {
        let shared_c = Arc::clone(&shared);
        let events_c = Arc::clone(&events);
        tokio::spawn(async move { sen_task(shared_c, events_c).await; })
    };
    let h_tts = {
        let shared_c = Arc::clone(&shared);
        let events_c = Arc::clone(&events);
        let tts_c = Arc::clone(&tts);
        let audio_out_c = Arc::clone(&audio_output);
        let play_cancel_c = Arc::clone(&play_cancel);
        let tts_muted_c = Arc::new(AtomicBool::new(false));
        tokio::spawn(async move {
            tts_task(shared_c, events_c, tts_c, audio_out_c, tts_sample_rate, play_cancel_c,
                     tts_muted_c,
                     #[cfg(feature = "tui")]
                     { let (tx, _) = tokio::sync::mpsc::unbounded_channel(); tx },
                     #[cfg(feature = "remote")]
                     Arc::new(tokio::sync::Mutex::new(None)),
            ).await;
        })
    };
    let h_sum = {
        let shared_c = Arc::clone(&shared);
        let events_c = Arc::clone(&events);
        let llm_session_c = Arc::clone(&llm_session);
        let llm_client_c = llm_client.clone();
        let db_c = db.clone();
        tokio::spawn(async move {
            consolidation_task(
                shared_c, events_c, llm_session_c, llm_client_c, db_c,
                session_id, context_tokens, summary_keep_turns, 80, 0, 0,
                String::new(), String::new(),
            ).await;
        })
    };

    // Fire the initial VAD_FINISH to start the LLM task.
    events.vad_finish.notify_one();

    // Wait until LLM finishes streaming, then give TTS time to drain.
    events.llm_post_finished.notified().await;

    // Give TTS tasks time to synthesize and "play" (mock TTS is instant).
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Cancel all tasks.
    let _ = events.cancel_tx.send(());
    play_cancel.store(true, Ordering::SeqCst);

    // Wait for tasks to exit.
    let _ = tokio::time::timeout(std::time::Duration::from_secs(2), async {
        let _ = h_llm.await;
        let _ = h_sen.await;
        let _ = h_tts.await;
        let _ = h_sum.await;
    }).await;
}

/// Map a spoken yes/no transcript to an ACP permission outcome string.
fn map_answer_to_outcome(transcript: &str) -> String {
    let t = transcript.to_lowercase();
    if t.contains("sí")
        || t.contains("si")
        || t.contains("yes")
        || t.contains("claro")
        || t.contains("dale")
        || t.contains("ok")
        || t.contains("adelante")
        || t.contains("permite")
        || t.contains("permiso")
        || t.contains("autorizo")
    {
        "allow_once".to_string()
    } else {
        "reject_once".to_string()
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// Integration test for summarization using a real LLM server.
    ///
    /// Requires a running LLM server (default http://localhost:8000, e.g. mlx-lm or oMLX).
    /// Override with `LLM_URL` env var.
    ///
    /// Run manually:
    /// ```sh
    /// cargo test test_summarize_real_llm -- --ignored --nocapture
    /// ```
    #[tokio::test]
    #[ignore]
    async fn test_summarize_real_llm() {
        let _ = tracing_subscriber::fmt()
            .with_env_filter("info")
            .try_init();

        // ── Setup temp DB ────────────────────────────────────────────────────
        let db_dir = tempfile::TempDir::new().unwrap();
        let db_path = db_dir.path().join("test_summarize.db");
        let db = crate::db::Database::new(db_path.to_str().unwrap())
            .await
            .unwrap();
        let session_id = db.get_or_create_session().await.unwrap();

        // ── Load .env and setup real LLM client ─────────────────────────────
        let _ = dotenvy::dotenv();
        let llm_url = std::env::var("LLM_URL")
            .unwrap_or_else(|_| "http://localhost:8080".to_string());
        let llm_model = std::env::var("LLM_MODEL")
            .unwrap_or_else(|_| "local-model".to_string());
        let llm_api_key = std::env::var("LLM_API_KEY").unwrap_or_default();
        let llm_client = crate::llm::OpenAIClient::new(
            &llm_url,
            &llm_model,
            400,       // max_tokens
            0.3,       // temperature
        )
        .with_api_key(&llm_api_key);

        // ── Populate session with enough turns to trigger summarization ──────
        let system_prompt = "You are a helpful assistant.";
        let mut session = crate::llm::LlmSession::new(system_prompt);

        let turns = vec![
            ("What is the capital of France?", "The capital of France is Paris, a city known for the Eiffel Tower and its rich cultural heritage."),
            ("Tell me about the Rust programming language.", "Rust is a systems programming language focused on safety, speed, and concurrency. It was created by Mozilla."),
            ("What is photosynthesis?", "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen."),
            ("Who wrote Don Quixote?", "Don Quixote was written by Miguel de Cervantes and published in two parts in 1605 and 1615."),
            ("Explain quantum computing briefly.", "Quantum computing uses quantum bits or qubits that can exist in superposition, enabling parallel computation for certain problems."),
            ("What is the tallest mountain on Earth?", "Mount Everest is the tallest mountain on Earth at 8,849 meters above sea level, located in the Himalayas."),
            ("How does a combustion engine work?", "A combustion engine burns fuel in cylinders, creating expanding gases that push pistons connected to a crankshaft to produce rotary motion."),
            ("What is the speed of light?", "The speed of light in a vacuum is approximately 299,792,458 meters per second, often rounded to 300,000 km/s."),
        ];

        for (user_msg, assistant_msg) in &turns {
            session.add_user_turn(user_msg);
            session.add_assistant_turn(assistant_msg);
            db.save_message(session_id, "User", user_msg).await.unwrap();
            db.save_message(session_id, "Assistant", assistant_msg).await.unwrap();
        }

        let msg_count_before = session.messages.len();
        println!("Messages before summarization: {}", msg_count_before);

        let context_tokens: usize = 300;
        let keep_turns: usize = 2;

        assert!(
            session.needs_consolidation(context_tokens, 75),
            "Session should need consolidation with context_tokens={} but doesn't.",
            context_tokens
        );

        // ── Run summarization directly ──────────────────────────────────────
        let prompt = session.build_summary_prompt(keep_turns).unwrap();
        let summary = llm_client.complete(&prompt).await.unwrap();
        assert!(!summary.is_empty(), "Summary should not be empty");

        let turns_to_summarize = session.summarizable_turn_count(keep_turns);
        let through_id = db
            .get_message_id_at_offset(session_id, 0, turns_to_summarize - 1)
            .await
            .unwrap()
            .unwrap();
        db.save_summary(session_id, &summary, through_id).await.unwrap();
        session.apply_summary(&summary, keep_turns);

        // ── Assertions ──────────────────────────────────────────────────────
        let msg_count_after = session.messages.len();
        println!("Messages after summarization: {}", msg_count_after);
        assert!(msg_count_after < msg_count_before);
        assert_eq!(msg_count_after, keep_turns);

        let all_msgs = session.all_messages_api();
        let system_content = all_msgs[0]["content"].as_str().unwrap();
        assert!(system_content.contains("[CONVERSATION SUMMARY]"));

        let (db_summary, db_recent) = db.get_session_context(session_id, 0).await.unwrap();
        assert!(db_summary.is_some());
        assert!(!db_recent.is_empty());

        println!("\n✓ summarize integration test passed");
    }

    /// Benchmark: measures TTFT before and after summarization to detect
    /// KV-cache invalidation caused by prompt compaction.
    ///
    /// Run manually:
    /// ```sh
    /// cargo test test_kv_cache_after_summarize --bin voicebot -- --ignored --nocapture
    /// ```
    #[tokio::test]
    #[ignore]
    async fn test_kv_cache_after_summarize() {
        use std::time::Instant;

        let _ = tracing_subscriber::fmt()
            .with_env_filter("info")
            .try_init();

        // ── Setup ────────────────────────────────────────────────────────────
        let _ = dotenvy::dotenv();
        let llm_url = std::env::var("LLM_URL")
            .unwrap_or_else(|_| "http://localhost:8080".to_string());
        let llm_model = std::env::var("LLM_MODEL")
            .unwrap_or_else(|_| "local-model".to_string());
        let llm_api_key = std::env::var("LLM_API_KEY").unwrap_or_default();
        let llm_client = crate::llm::OpenAIClient::new(
            &llm_url, &llm_model, 400, 0.3,
        )
        .with_api_key(&llm_api_key);

        let db_dir = tempfile::TempDir::new().unwrap();
        let db_path = db_dir.path().join("bench_kv.db");
        let db = crate::db::Database::new(db_path.to_str().unwrap())
            .await
            .unwrap();
        let session_id = db.get_or_create_session().await.unwrap();

        // ── Populate session with conversation turns ─────────────────────────
        let system_prompt = "You are a helpful assistant. Answer briefly.";
        let mut session = crate::llm::LlmSession::new(system_prompt);

        let turns = vec![
            ("What is the capital of France?", "The capital of France is Paris."),
            ("Tell me about Rust.", "Rust is a systems programming language focused on safety and performance."),
            ("What is photosynthesis?", "Photosynthesis converts sunlight, water, and CO2 into glucose and oxygen."),
            ("Who wrote Don Quixote?", "Miguel de Cervantes wrote Don Quixote, published in 1605 and 1615."),
            ("Explain quantum computing.", "Quantum computing uses qubits in superposition for parallel computation."),
            ("What is the tallest mountain?", "Mount Everest at 8,849 meters above sea level."),
            ("How does a combustion engine work?", "It burns fuel in cylinders, pushing pistons to create rotary motion."),
            ("What is the speed of light?", "About 299,792,458 meters per second in a vacuum."),
        ];

        for (user_msg, assistant_msg) in &turns {
            session.add_user_turn(user_msg);
            session.add_assistant_turn(assistant_msg);
            db.save_message(session_id, "User", user_msg).await.unwrap();
            db.save_message(session_id, "Assistant", assistant_msg).await.unwrap();
        }

        // ── Helper: measure TTFT for a simple message ────────────────────────
        async fn measure_ttft(
            client: &crate::llm::OpenAIClient,
            session: &crate::llm::LlmSession,
            probe_msg: &str,
        ) -> (u128, u128, String) {
            let mut session_clone = session.clone();
            session_clone.add_user_turn(probe_msg);
            let messages = session_clone.all_messages_api();

            let t = Instant::now();
            let (mut rx, _stream_handle) = client.stream(&messages, &[]).await
                .expect("Failed to start LLM stream");
            let mut ttft_ms: Option<u128> = None;
            let mut response = String::new();

            while let Some(token) = rx.recv().await {
                match token {
                    StreamToken::Content(s) => {
                        if ttft_ms.is_none() && !s.is_empty() {
                            ttft_ms = Some(t.elapsed().as_millis());
                        }
                        response.push_str(&s);
                    }
                    StreamToken::ToolCall { .. } => {}
                }
            }
            let total_ms = t.elapsed().as_millis();
            (ttft_ms.unwrap_or(total_ms), total_ms, response)
        }

        let probe = "Hola, ¿qué tal?";

        // ── Warmup: prime the KV-cache with the full conversation ────────────
        // First call populates the cache; second call measures the warm TTFT.
        println!("\n── Warmup (priming KV-cache) ──");
        let (warmup_ttft, warmup_total, _) = measure_ttft(&llm_client, &session, probe).await;
        println!("  Warmup TTFT: {}ms  (total {}ms)", warmup_ttft, warmup_total);

        // ── BEFORE: measure TTFT with warm KV-cache ─────────────────────────
        println!("\n── BEFORE summarization (warm KV-cache) ──");
        let (before_ttft, before_total, before_response) =
            measure_ttft(&llm_client, &session, probe).await;
        println!("  TTFT:     {}ms", before_ttft);
        println!("  Total:    {}ms", before_total);
        println!("  Response: {:?}", &before_response[..before_response.len().min(80)]);
        assert!(!before_response.is_empty(), "LLM should produce a response before summarization");

        // ── Run summarization ───────────────────────────────────────────────
        println!("\n── Running summarization ──");
        let keep_turns: usize = 2;
        let prompt = session.build_summary_prompt(keep_turns).unwrap();
        let summary = llm_client.complete(&prompt).await.unwrap();
        session.apply_summary(&summary, keep_turns);

        let session_after = session.clone();
        let msg_count = session_after.messages.len();
        println!("  Session compacted: {} messages remaining", msg_count);

        // ── AFTER: measure TTFT with invalidated KV-cache ───────────────────
        println!("\n── AFTER summarization (KV-cache likely invalidated) ──");
        let (after_ttft, after_total, after_response) =
            measure_ttft(&llm_client, &session_after, probe).await;
        println!("  TTFT:     {}ms", after_ttft);
        println!("  Total:    {}ms", after_total);
        println!("  Response: {:?}", &after_response[..after_response.len().min(80)]);
        assert!(!after_response.is_empty(), "LLM should produce a response after summarization");

        // ── Comparison ──────────────────────────────────────────────────────
        let delta = after_ttft as i128 - before_ttft as i128;
        let ratio = if before_ttft > 0 {
            after_ttft as f64 / before_ttft as f64
        } else {
            f64::NAN
        };

        println!("\n{}", "=".repeat(60));
        println!("  KV-CACHE BENCHMARK RESULTS");
        println!("{}", "=".repeat(60));
        println!("  TTFT before summarization:  {:>6}ms", before_ttft);
        println!("  TTFT after  summarization:  {:>6}ms", after_ttft);
        println!("  Delta:                      {:>+6}ms", delta);
        println!("  Ratio (after/before):       {:>6.2}x", ratio);
        println!("{}", "=".repeat(60));
        if delta > 0 {
            println!("  → KV-cache was likely INVALIDATED by summarization.");
            println!("    The LLM had to re-process the entire prompt.");
        } else {
            println!("  → KV-cache appears to still be effective.");
        }
        println!();
    }
}
