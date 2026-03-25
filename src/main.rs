mod agents;
mod audio;
mod config;
mod daemon;
mod db;
mod llm;
mod profile;
mod stt;
mod tools;
#[cfg(feature = "tui")]
mod tui;
mod tts;

use anyhow::Result;
use async_channel::bounded;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;
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
    sentences: Mutex<VecDeque<String>>,
    /// True when the current LLM streaming POST has finished.
    llm_post_finished: AtomicBool,
    /// True while the LLM task is actively processing a turn.
    llm_busy: AtomicBool,
    /// True when the pending transliterated_text came from TUI text input (not voice).
    pub(crate) text_input_pending: AtomicBool,
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
use uuid::Uuid;

use crate::agents::ProactiveEvent;
use crate::profile::extract_facts;
use crate::audio::audio_capture::{AudioCapture, AudioChunk};
use crate::audio::buffer::AudioBuffer;
use crate::audio::output::AudioOutput;
use crate::audio::ambient_buffer::AmbientBuffer;
use crate::audio::speaker::{SpeakerVerdict, SpeakerVerifier};
use crate::audio::vad::{VadResult, VoiceActivityDetector};
use crate::config::Config;
use crate::db::Database;
use crate::llm::{LlamaClient, LlmSession, StreamToken};
use crate::profile::{build_profile_context, ProfileFact};
use crate::stt::{WhisperStt, SttStream};
use whisper_rs::install_logging_hooks;
use crate::tools::{
    format_history, ActiveAcpTask, ConversationMode, CurrentTimeTool, HermesAcpWriter, JsonRpcMessage,
    OpenAppTool, ReadClipboardTool, RunAgentTool, SetClipboardTool, SetConversationModeTool,
    TakeScreenshotTool, ToolRegistry,
};
use crate::tts::{SayTts, SentenceSplitter, TtsEngine};
#[cfg(feature = "kokoro")]
use crate::tts::KokoroTts;
#[cfg(feature = "avspeech")]
use crate::tts::AvSpeechTts;

#[cfg(test)]
mod e2e_tests;

const AUDIO_CHANNEL_CAPACITY: usize = 200;
const MAX_SPEECH_BUFFER_SECS: u32 = 30;
const MIN_SPEECH_DURATION_MS: u32 = 300;
/// Pre-roll chunks kept before speech onset to recover VAD onset delay (~250ms).
const PRE_ROLL_CHUNKS: usize = 15;

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

async fn async_main() -> Result<()> {
    // Disable whisper prints into console
    install_logging_hooks();

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
                SayTts::list_voices()?;
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
    let secondary_llm_client: Option<LlamaClient> =
        config.secondary_llm_url.as_ref().map(|url| {
            LlamaClient::new(url, &config.secondary_llm_model, config.secondary_llm_max_tokens, 0.3, 0, -1)
                .with_provider(&config.secondary_llm_provider)
                .with_api_key(&config.secondary_llm_api_key)
        });
    if secondary_llm_client.is_some() {
        info!(
            target: "llm",
            "Secondary LLM endpoint: {} (model={}, provider={})",
            config.secondary_llm_url.as_deref().unwrap_or(""),
            config.secondary_llm_model,
            config.secondary_llm_provider,
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

    // Vision (screenshot) — enabled when SECONDARY_LLM_URL is set
    if let Some(ref sec_client) = secondary_llm_client {
        info!(
            target: "voicebot",
            "Vision tool enabled via secondary LLM (model={})",
            config.secondary_llm_model,
        );
        tool_registry.register(TakeScreenshotTool::new(sec_client.clone()));
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
        tool_registry.register(RunAgentTool::new(
            agent_cmd,
            Arc::clone(&acp_writer),
            Arc::clone(&acp_inbound),
            Arc::clone(&active_task),
            shared_history.clone(),
            proactive_tx.clone(),
            mode,
            acp_cmd,
        ));
    }

    let tools = Arc::new(tool_registry);

    // ── Database ─────────────────────────────────────────────────────────────
    let db = Database::new(&config.db_path).await?;
    let session_id = db.get_or_create_session().await?;
    let (summary, history) = db.get_session_context(session_id).await?;
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

    // ── LLM session ───────────────────────────────────────────────────────────
    // System prompt = base + user profile context + tool instructions.
    let system_prompt = format!(
        "{}{}{}",
        config.llm_system_prompt,
        build_profile_context(&profile_facts),
        tools.system_prompt_section()
    );
    let llm_session = Arc::new(Mutex::new(LlmSession::from_history(
        &system_prompt,
        config.llm_slot_id,
        summary.as_deref(),
        &history,
    )));

    // ── LLM client ────────────────────────────────────────────────────────────
    let llm_client = LlamaClient::new(
        &config.llm_url,
        &config.llm_model,
        config.llm_max_tokens,
        config.llm_temperature,
        config.llm_slot_id,
        config.llm_background_slot_id,
    )
    .with_provider(&config.llm_provider)
    .with_api_key(&config.llm_api_key);
    info!(target: "llm", "LLM endpoint: {} (provider: {})", config.llm_url, config.llm_provider);

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

    // ── STT (whisper) ─────────────────────────────────────────────────────────
    let whisper_model = config.whisper_model.clone();
    let whisper_language = config.language.clone();
    let stt = tokio::task::spawn_blocking(move || {
        WhisperStt::new(&whisper_model, &whisper_language)
    })
    .await??;
    let stt = Arc::new(stt);
    // Always-running STT: Whisper starts as soon as the user begins speaking
    // so the result is ready (or nearly ready) when VAD fires SpeechEnd.
    let stt_stream = SttStream::new(Arc::clone(&stt));

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
            info!(target: "tts", "TTS provider: say (voice={}, rate={}wpm)", config.say_voice, config.say_rate);
            let say_voice = config.say_voice.clone();
            let say_rate = config.say_rate;
            let s = tokio::task::spawn_blocking(move || SayTts::new(&say_voice, say_rate)).await??;
            TtsEngine::Say(s)
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
    let _stream = audio_capture.start_capture(tx, samples_per_chunk)?;

    let mut vad = VoiceActivityDetector::new(source_sample_rate, config.vad_silence_ms)?;
    info!(target: "audio", "VAD silence threshold: {}ms", config.vad_silence_ms);
    let mut speech_buffer = AudioBuffer::new(source_sample_rate, MAX_SPEECH_BUFFER_SECS);
    let mut pre_roll: VecDeque<Vec<f32>> = VecDeque::with_capacity(PRE_ROLL_CHUNKS + 1);
    let mut t_speech_start: Option<Instant> = None;

    // ── Continuous audio accumulation ─────────────────────────────────────────
    // speech_buffer is no longer cleared at SpeechEnd; it accumulates audio
    // across short mid-thought pauses so all speech reaches Whisper as one chunk.
    // It is cleared at the next SpeechStart only after a turn has been committed
    // (i.e. add_user_turn was called in run_pipeline).
    let turn_commit_counter = Arc::new(AtomicU64::new(0));
    let mut last_cleared_commit: u64 = 0;
    // Sample count in speech_buffer at the start of the current speech segment
    // (recorded before pre-roll is flushed).  Used to trim old committed audio
    // when a turn commits mid-segment.
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
        #[cfg(feature = "tui")]
        let tui_tx_c = tui_tx.clone();
        tokio::spawn(async move {
            llm_task(
                shared_c, events_c, llm_session_c, llm_client_c,
                db_c, session_id, tools_c, shared_history_c, turn_commit_c,
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
        tokio::spawn(async move {
            tts_task(shared_c, events_c, tts_c, audio_out_c, tts_sample_rate, play_cancel_c,
                     tts_muted_c,
                     #[cfg(feature = "tui")]
                     tui_tx_c,
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
        tokio::spawn(async move {
            sum_task(shared_c, events_c, llm_session_c, background_client_c, db_c,
                     session_id, context_tokens, keep_turns).await;
        });
    }

    info!(target: "voicebot", "Ready. Speak to interact...");

    // ── TUI ─────────────────────────────────────────────────────────────────
    #[cfg(feature = "tui")]
    {
        let shared_c = Arc::clone(&shared);
        let events_c = Arc::clone(&events);
        let tts_muted_c = Arc::clone(&tts_muted);
        tokio::spawn(async move {
            if let Err(e) = tui::run(tui_rx, shared_c, events_c, tts_muted_c).await {
                tracing::error!("TUI error: {e}");
            }
            // TUI quit → exit process.
            std::process::exit(0);
        });
    }

    // ── Startup greeting ──────────────────────────────────────────────────────
    {
        let now = chrono::Local::now();
        let time_str = now.format("%H:%M").to_string();
        let notification = format!(
            "[Sistema: el voicebot acaba de arrancar. Son las {time_str}.\n\
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
                if !shared.llm_busy.load(Ordering::SeqCst) && current_agent_announcement.is_none() {
                    if let Some((task, result)) = pending_agent_results.pop_front() {
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
                }
                // Clear announcement tracker once LLM becomes idle again.
                if current_agent_announcement.is_some() && !shared.llm_busy.load(Ordering::SeqCst) {
                    current_agent_announcement = None;
                }

                let chunk: AudioChunk = tokio::select! {
                    result = rx.recv() => match result {
                        Ok(c) => c,
                        Err(e) => { error!(target: "audio", "Audio channel closed: {}", e); break; }
                    },
                    Some(event) = proactive_rx.recv() => {
                        match event {
                            ProactiveEvent::AgentResult { task, result } => {
                                if !shared.llm_busy.load(Ordering::SeqCst) {
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

                match vad.process(&mono) {
                    VadResult::SpeechStart => {
                        t_speech_start = Some(Instant::now());
                        info!(target: "performance", "[+0ms] SpeechStart");
                        #[cfg(feature = "tui")]
                        tui_tx.send(tui::events::TuiEvent::StateChange(
                            tui::events::PipelineState::Listening,
                        )).ok();
                        last_speech_at = Instant::now();
                        // ── VAD_DETECTED: cancel any active LLM/TTS pipeline ──
                        // Fired unconditionally: llm_busy becomes false when the LLM
                        // finishes streaming, but TTS may still be playing sentences.
                        // All tasks handle stale cancels gracefully.
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

                        for pre in pre_roll.drain(..) {
                            speech_buffer.push(&pre);
                        }
                        speech_buffer.push(&mono);

                        // Invalidate any stale STT→vad_finish tasks from a prior utterance.
                        utterance_epoch.fetch_add(1, Ordering::SeqCst);
                    }
                    VadResult::Speech => {
                        speech_buffer.push(&mono);
                    }
                    VadResult::SpeechEnd => {
                        speech_buffer.push(&mono);
                        pre_roll.clear();

                        let segment_duration_ms = t_speech_start.as_ref()
                            .map(|t| t.elapsed().as_millis() as u32)
                            .unwrap_or(0);

                        if segment_duration_ms < MIN_SPEECH_DURATION_MS {
                            debug!(target: "pipeline", "Too short ({}ms), skipping", segment_duration_ms);
                            continue;
                        }

                        let current_commits = turn_commit_counter.load(Ordering::SeqCst);
                        let audio = if current_commits > last_cleared_commit {
                            last_cleared_commit = current_commits;
                            speech_buffer.get_samples_from(speech_buffer_start_offset)
                        } else {
                            speech_buffer.get_samples()
                        };
                        let duration_ms = audio.len() as u32 * 1000 / source_sample_rate;

                        info!(target: "pipeline", "Speech: {}ms (segment {}ms)", duration_ms, segment_duration_ms);
                        #[cfg(feature = "tui")]
                        tui_tx.send(tui::events::TuiEvent::StateChange(
                            tui::events::PipelineState::Transcribing,
                        )).ok();
                        let vad_elapsed = t_speech_start.take()
                            .map(|t| t.elapsed().as_millis()).unwrap_or(0);
                        info!(target: "performance", "[+{}ms] VAD end ({}ms speech)", vad_elapsed, duration_ms);

                        last_speech_at = Instant::now();

                        // ── Speaker verification ──────────────────────────────
                        // Determines whether this segment is from the main user
                        // (id=0) or someone else. Non-main speakers are buffered
                        // for context but never routed to the LLM directly.
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

                        // Submit audio to STT for all speakers — ambient buffer
                        // needs transcripts from non-main speakers too.
                        let min_stt_gen = stt_stream.submit(audio);

                        // ── Non-main speaker: buffer transcript, skip LLM ─────
                        if !is_main_speaker {
                            let stt_c = Arc::clone(&stt_stream);
                            let amb_c = Arc::clone(&ambient_buffer);
                            let label = speaker_label.clone();
                            tokio::spawn(async move {
                                let text = stt_c.await_result(min_stt_gen).await;
                                if !text.is_empty() {
                                    amb_c.lock().unwrap().push(label.clone(), text.clone());
                                    debug!(target: "pipeline", "Ambient buffer ← {label}: {text}");
                                }
                            });
                            vad.reset();
                            continue;
                        }

                        let ambient = *conv_mode.lock().unwrap() == ConversationMode::Ambient;
                        let wake_word_check = config.wake_word.clone();

                        // ── ACP permission gate ───────────────────────────────
                        if let Some(resp_tx) = pending_agent_question.take() {
                            let stt_c = Arc::clone(&stt_stream);
                            tokio::spawn(async move {
                                let answer = stt_c.await_result(min_stt_gen).await;
                                let outcome = map_answer_to_outcome(&answer);
                                info!(target: "acp", "Permission answer: {:?} → {}", answer, outcome);
                                let _ = resp_tx.send(outcome);
                            });
                            continue;
                        }

                        // ── Fire VAD_FINISH after final STT result is ready ────
                        // Awaiting in a spawned task keeps the audio loop unblocked.
                        let stt_c      = Arc::clone(&stt_stream);
                        let shared_c   = Arc::clone(&shared);
                        let events_c   = Arc::clone(&events);
                        let amb_c      = Arc::clone(&ambient_buffer);
                        let epoch      = utterance_epoch.load(Ordering::SeqCst);
                        let epoch_ref  = Arc::clone(&utterance_epoch);
                        tokio::spawn(async move {
                            let text = stt_c.await_result(min_stt_gen).await;
                            if text.is_empty() { return; }

                            // Stale check: if a new SpeechStart fired since this
                            // task was spawned, the user interrupted — discard.
                            if epoch_ref.load(Ordering::SeqCst) != epoch {
                                debug!(target: "pipeline", "Stale STT result (epoch changed) — discarding: {:?}", &text[..text.len().min(40)]);
                                return;
                            }

                            // Ambient mode: buffer non-wake-word utterances from
                            // the main user too (e.g. their side of a conversation).
                            // Wake-word fires one response but does NOT change mode —
                            // only an explicit "modo activo" command switches to Active.
                            if ambient {
                                let lower = text.to_lowercase();
                                if lower.contains(&wake_word_check) {
                                    info!(target: "pipeline", "Ambient: wake word detected — responding (staying Ambient)");
                                } else {
                                    // Buffer the main user's ambient utterance for context.
                                    amb_c.lock().unwrap().push("Usuario".to_string(), text.clone());
                                    debug!(target: "pipeline", "Ambient: no wake word — buffered as context");
                                    return;
                                }
                            }

                            // Inject ambient context when the query contains a referential.
                            let final_text = {
                                let buf = amb_c.lock().unwrap();
                                if crate::audio::ambient_buffer::has_referential(&text) {
                                    if let Some(ctx) = buf.format_context() {
                                        format!("{ctx}\n---\n{text}")
                                    } else {
                                        text
                                    }
                                } else {
                                    text
                                }
                            };

                            // Store final transcript and wake the LLM task.
                            *shared_c.transliterated_text.lock().unwrap() = final_text;
                            events_c.vad_finish.notify_one();
                        });
                    }
                    VadResult::Silence => {
                        pre_roll.push_back(mono);
                        if pre_roll.len() > PRE_ROLL_CHUNKS {
                            pre_roll.pop_front();
                        }
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
    llm_client: LlamaClient,
    db: Database,
    session_id: uuid::Uuid,
    tools: Arc<ToolRegistry>,
    shared_history: Arc<RwLock<String>>,
    turn_commit_counter: Arc<AtomicU64>,
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

        shared.llm_busy.store(true, Ordering::SeqCst);

        // Drain accumulated transcription.
        let text = std::mem::take(&mut *shared.transliterated_text.lock().unwrap());

        if text.trim().is_empty() {
            shared.llm_busy.store(false, Ordering::SeqCst);
            while cancel_rx.try_recv().is_ok() {}
            continue;
        }

        // Ambient mode: text pipeline injections bypass the wake-word check
        // because the audio loop already validated them before firing vad_finish.
        // (Voice utterances are filtered inside the SpeechEnd spawned task.)
        info!(target: "pipeline", "[pipe={}] User: {}", pipeline_id, text);

        #[cfg(feature = "tui")]
        {
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

        // Reset post-finished flag and clear assistant text buffer for new turn.
        shared.llm_post_finished.store(false, Ordering::SeqCst);
        shared.assistant_text.lock().unwrap().clear();

        // Speculative KV-cache prefill (llama.cpp only).
        // Fire-and-forget: initiate the request so llama.cpp starts prefilling
        // the full prompt (with user turn), then abort after 5ms — the server
        // continues prefilling in the background while we set up the real stream.
        let use_prefill = llm_client.supports_prefill_warm();
        let spec_msgs = if use_prefill { llm_session.lock().unwrap().all_messages_api() } else { vec![] };
        let spec_client = llm_client.clone();
        let mut spec_handle = tokio::spawn(async move {
            if use_prefill {
                if let Err(e) = spec_client.prefill_warm(spec_msgs).await {
                    debug!(target: "llm", "Speculative prefill ended: {e}");
                }
            }
        });
        tokio::select! {
            _ = &mut spec_handle => {}
            _ = tokio::time::sleep(std::time::Duration::from_millis(5)) => {
                spec_handle.abort();
                let _ = spec_handle.await;
            }
        }

        // Tool call loop — allows the model to call tools before its spoken response.
        let mut messages = llm_session.lock().unwrap().all_messages_api();
        let base_msg_len = messages.len();
        let tool_defs = tools.tool_definitions();
        let mut final_response = String::new();
        let mut committed = false;
        let mut cancelled = false;

        'pipeline: {
            'tool_loop: for iter in 0..MAX_TOOL_ITERATIONS {
                info!(target: "performance", "LLM request [pipe={}]", pipeline_id);
                let (token_rx, stream_handle) = match llm_client.stream(&messages, &tool_defs).await {
                    Ok(r)  => r,
                    Err(e) => { error!(target: "llm", "LLM error: {}", e); break 'pipeline; }
                };

                let mut token_rx = token_rx;
                let mut llm_text = String::new();
                let mut tool_call: Option<(String, String)> = None;

                // Stream tokens; forward each to SEN via shared.assistant_text + event.
                loop {
                    tokio::select! {
                        token = token_rx.recv() => {
                            match token {
                                Some(StreamToken::Content(t)) => {
                                    llm_text.push_str(&t);
                                    shared.assistant_text.lock().unwrap().push_str(&t);
                                    events.llm_post_received.notify_one();
                                    #[cfg(feature = "tui")]
                                    tui_tx.send(tui::events::TuiEvent::AssistantToken(t)).ok();
                                }
                                Some(StreamToken::ToolCall { name, args }) => {
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
                tokio::spawn(async move {
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

    loop {
        let cancelled = tokio::select! {
            _ = events.llm_post_received.notified() => false,
            _ = cancel_rx.recv() => true,
        };

        if cancelled {
            shared.assistant_text.lock().unwrap().clear();
            splitter = SentenceSplitter::new();
            while cancel_rx.try_recv().is_ok() {}
            continue;
        }

        // Drain available text into the splitter.
        let new_text = std::mem::take(&mut *shared.assistant_text.lock().unwrap());

        let mut ready_sentences: Vec<String> = Vec::new();
        if !new_text.is_empty() {
            if let Some(s) = splitter.push(&new_text) {
                ready_sentences.push(s);
            }
        }

        // If LLM is done streaming, flush any remaining fragment.
        if shared.llm_post_finished.load(Ordering::SeqCst) {
            if let Some(s) = splitter.flush() {
                ready_sentences.push(s);
            }
        }

        for sentence in ready_sentences {
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
) {
    let mut cancel_rx = events.cancel_tx.subscribe();
    let mut play_handle: Option<tokio::task::JoinHandle<anyhow::Result<()>>> = None;

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
                while cancel_rx.try_recv().is_ok() {}
                continue;
            }
        }

        // Check cancel once more before awaiting synthesis result.
        if cancel_rx.try_recv().is_ok() {
            synth_handle.abort();
            shared.sentences.lock().unwrap().clear();
            while cancel_rx.try_recv().is_ok() {}
            continue;
        }

        let samples = match synth_handle.await {
            Ok(Ok(s))  => s,
            Ok(Err(e)) => { error!(target: "tts", "TTS synthesis error: {}", e); continue; }
            Err(e)     => { error!(target: "tts", "TTS task panicked: {}", e); continue; }
        };

        let out_c    = Arc::clone(&audio_output);
        let cancel_c = Arc::clone(&play_cancel);
        play_handle = Some(tokio::task::spawn_blocking(move || {
            out_c.play_blocking(&samples, tts_sample_rate, &cancel_c)
        }));
    }
}

/// SUM task: blocks on LLM_POST_FINISHED, runs summarization when needed.
///
/// Corresponds to the SUM thread in doc/PROCESS_ARCHITECTURE.md.
#[allow(clippy::too_many_arguments)]
async fn sum_task(
    shared: Arc<SharedSession>,
    events: Arc<PipelineEvents>,
    llm_session: Arc<Mutex<LlmSession>>,
    background_client: LlamaClient,
    db: Database,
    session_id: uuid::Uuid,
    context_tokens: usize,
    keep_turns: usize,
) {
    let mut cancel_rx = events.cancel_tx.subscribe();

    loop {
        // Block until LLM finishes a response; ignore cancels while idle.
        loop {
            tokio::select! {
                _ = events.llm_post_finished.notified() => { break; }
                _ = cancel_rx.recv() => {}
            }
        }

        // Run summarization (uses background slot — does not touch main cache).
        // Bail out early if a barge-in arrives.
        tokio::select! {
            _ = maybe_summarize(&llm_session, &background_client, &db, session_id, context_tokens, keep_turns, "", "") => {}
            _ = cancel_rx.recv() => {
                debug!(target: "llm", "Summarization cancelled by barge-in");
                // Drop shared reference to suppress unused warning if not otherwise used.
                let _ = &shared;
            }
        }

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
    _min_stt_gen: u64,
    _stt_stream: Arc<crate::stt::SttStream>,
    _cancel: Arc<AtomicBool>,
    tts: Arc<crate::tts::TtsEngine>,
    audio_output: Arc<crate::audio::output::AudioOutput>,
    llm_session: Arc<Mutex<crate::llm::LlmSession>>,
    llm_client: crate::llm::LlamaClient,
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

    let transcript = {
        let s = _stt_stream.await_result(_min_stt_gen).await;
        s
    };

    if transcript.trim().is_empty() {
        return;
    }

    // Ambient mode wake-word check.
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
        tokio::spawn(async move {
            llm_task(
                shared_c, events_c, llm_session_c, llm_client_c,
                db_c, session_id, tools_c, shared_history_c, turn_commit_c,
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
            sum_task(shared_c, events_c, llm_session_c, llm_client_c, db_c, session_id, context_tokens, summary_keep_turns).await;
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


/// Summarize old conversation turns if the prompt is approaching the context limit.
///
/// Runs after every completed pipeline turn. Builds a summary of old turns,
/// injects it into the session prompt, and persists it to the DB so future
/// restarts can restore the compact context.
async fn maybe_summarize(
    llm_session: &Arc<Mutex<LlmSession>>,
    background_client: &LlamaClient,
    db: &Database,
    session_id: Uuid,
    context_tokens: usize,
    keep_turns: usize,
    user_text: &str,
    assistant_text: &str,
) {
    let needs = llm_session.lock().unwrap().needs_summarization(context_tokens);
    if !needs {
        return;
    }

    // Extract user profile facts from the current turn before summarizing.
    // Only runs when summarization is needed, avoiding an LLM call every turn.
    let facts = extract_facts(background_client, user_text, assistant_text).await;
    for fact in facts {
        if let Err(e) = db.upsert_profile_fact(&fact.key, &fact.value, fact.confidence).await {
            warn!(target: "profile", "Failed to save profile fact '{}': {}", fact.key, e);
        } else {
            debug!(target: "profile", "Profile: {} = {} ({:.0}%)", fact.key, fact.value, fact.confidence * 100.0);
        }
    }

    let (summary_prompt, turns_to_summarize) = {
        let s = llm_session.lock().unwrap();
        let prompt = s.build_summary_prompt(keep_turns);
        let count = s.summarizable_turn_count(keep_turns);
        (prompt, count)
    };

    let Some(prompt) = summary_prompt else {
        return;
    };

    info!(target: "llm", "Context limit approaching — summarizing {} old turns...", turns_to_summarize);

    let summary = match background_client.complete(&prompt).await {
        Ok(s) if !s.is_empty() => s,
        Ok(_) => {
            warn!(target: "llm", "Summarization returned empty result, skipping");
            return;
        }
        Err(e) => {
            warn!(target: "llm", "Summarization failed: {}", e);
            return;
        }
    };

    info!(target: "llm", "Summary: {}", summary);

    // Find the DB message id of the last turn that is being summarized.
    // Each turn in `turns` corresponds to one row in messages (alternating User/Assistant),
    // so the last summarized message is at 0-based offset (turns_to_summarize - 1).
    let through_id = match db
        .get_message_id_at_offset(session_id, turns_to_summarize - 1)
        .await
    {
        Ok(Some(id)) => id,
        Ok(None) => {
            warn!(target: "llm", "Could not find message offset for summary cutpoint, skipping");
            return;
        }
        Err(e) => {
            warn!(target: "llm", "DB error finding summary cutpoint: {}", e);
            return;
        }
    };

    if let Err(e) = db.save_summary(session_id, &summary, through_id).await {
        warn!(target: "db", "Failed to persist summary: {}", e);
    }

    llm_session.lock().unwrap().apply_summary(&summary, keep_turns);

    info!(
        target: "llm",
        "Summarization complete — prompt compacted (keeping {} recent turns)",
        keep_turns
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// Integration test for `maybe_summarize` using a real LLM server.
    ///
    /// Requires a running llama.cpp server (default http://localhost:8080).
    /// Override with `LLM_URL` env var.
    ///
    /// Run manually:
    /// ```sh
    /// cargo test test_maybe_summarize_real_llm -- --ignored --nocapture
    /// ```
    #[tokio::test]
    #[ignore]
    async fn test_maybe_summarize_real_llm() {
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
        let llm_provider = std::env::var("LLM_PROVIDER")
            .unwrap_or_else(|_| "llama".to_string());
        let llm_api_key = std::env::var("LLM_API_KEY").unwrap_or_default();
        let llm_client = crate::llm::LlamaClient::new(
            &llm_url,
            &llm_model,
            400,       // max_tokens
            0.3,       // temperature
            0,         // slot_id
            -1,        // background_slot_id
        )
        .with_provider(&llm_provider)
        .with_api_key(&llm_api_key);

        // ── Populate session with enough turns to trigger summarization ──────
        let system_prompt = "You are a helpful assistant.";
        let mut session = crate::llm::LlmSession::new(system_prompt, 0);

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

        // Use a small context_tokens to force summarization, keep only 2 recent turns.
        // Token estimate ≈ total_chars * 10 / 35; threshold = context_tokens * 3 / 4.
        let context_tokens: usize = 300;
        let keep_turns: usize = 2;

        assert!(
            session.needs_summarization(context_tokens),
            "Session should need summarization with context_tokens={} but doesn't. \
             Add more turns or reduce context_tokens.",
            context_tokens
        );

        // ── Call maybe_summarize ─────────────────────────────────────────────
        let llm_session = Arc::new(Mutex::new(session));
        maybe_summarize(
            &llm_session,
            &llm_client,
            &db,
            session_id,
            context_tokens,
            keep_turns,
            "",
            "",
        )
        .await;

        // ── Assertions ──────────────────────────────────────────────────────
        let session_after = llm_session.lock().unwrap();

        // 1. Messages should be compacted to keep_turns * 2 (user+assistant pairs)
        let msg_count_after = session_after.messages.len();
        println!("Messages after summarization: {}", msg_count_after);
        assert!(
            msg_count_after < msg_count_before,
            "Message count should decrease after summarization: before={}, after={}",
            msg_count_before, msg_count_after
        );
        assert_eq!(
            msg_count_after,
            keep_turns,
            "Should keep exactly {} messages (keep_turns={}), got {}",
            keep_turns,
            keep_turns,
            msg_count_after
        );

        // 2. System message should now contain the summary
        let all_msgs = session_after.all_messages_api();
        let system_content = all_msgs[0]["content"].as_str().unwrap();
        println!("System message after summarization:\n{}", system_content);
        assert!(
            system_content.contains("[CONVERSATION SUMMARY]"),
            "System message should contain [CONVERSATION SUMMARY] marker"
        );

        // 3. Summary should be persisted in DB
        let (db_summary, db_recent) = db.get_session_context(session_id).await.unwrap();
        assert!(
            db_summary.is_some(),
            "DB should have a summary after summarization"
        );
        let summary_text = db_summary.unwrap();
        println!("Summary from DB:\n{}", summary_text);
        assert!(
            !summary_text.is_empty(),
            "Summary text should not be empty"
        );
        assert!(
            !db_recent.is_empty(),
            "DB should still have recent messages after summary cutoff"
        );

        println!("\n✓ maybe_summarize integration test passed");
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
        let llm_provider = std::env::var("LLM_PROVIDER")
            .unwrap_or_else(|_| "llama".to_string());
        let llm_api_key = std::env::var("LLM_API_KEY").unwrap_or_default();
        let llm_client = crate::llm::LlamaClient::new(
            &llm_url, &llm_model, 400, 0.3, 0, -1,
        )
        .with_provider(&llm_provider)
        .with_api_key(&llm_api_key);

        let db_dir = tempfile::TempDir::new().unwrap();
        let db_path = db_dir.path().join("bench_kv.db");
        let db = crate::db::Database::new(db_path.to_str().unwrap())
            .await
            .unwrap();
        let session_id = db.get_or_create_session().await.unwrap();

        // ── Populate session with conversation turns ─────────────────────────
        let system_prompt = "You are a helpful assistant. Answer briefly.";
        let mut session = crate::llm::LlmSession::new(system_prompt, 0);

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
            client: &crate::llm::LlamaClient,
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
        println!("\n── Running maybe_summarize ──");
        let context_tokens: usize = 200;
        let keep_turns: usize = 2;
        let llm_session = Arc::new(Mutex::new(session));
        maybe_summarize(
            &llm_session, &llm_client, &db, session_id,
            context_tokens, keep_turns, "", "",
        ).await;

        let session_after = llm_session.lock().unwrap().clone();
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
