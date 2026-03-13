mod agents;
mod audio;
mod config;
mod daemon;
mod db;
mod llm;
mod profile;
mod stt;
mod system_state;
mod tools;
mod tts;

use anyhow::Result;
use async_channel::bounded;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

use crate::agents::ProactiveEvent;
use crate::audio::audio_capture::{AudioCapture, AudioChunk};
use crate::audio::buffer::AudioBuffer;
use crate::audio::output::AudioOutput;
use crate::audio::speaker::{SpeakerVerdict, SpeakerVerifier};
use crate::audio::vad::{VadResult, VoiceActivityDetector};
use crate::config::Config;
use crate::db::Database;
use crate::llm::{LlamaClient, LlmSession, StreamToken};
use crate::profile::{build_profile_context, extract_facts, ProfileFact};
use crate::stt::WhisperStt;
use whisper_rs::install_logging_hooks;
use crate::tools::{
    format_history, CalendarCreateTool, CalendarGetEventsTool, CurrentTimeTool,
    OpenAppTool, ReadClipboardTool, ReadFileTool, RunAgentAsyncTool, RunShellTool,
    SendNotificationTool, SetClipboardTool, TakeScreenshotTool, ToolRegistry,
};
use crate::tts::{SayTts, SentenceSplitter, TtsEngine};
#[cfg(feature = "kokoro")]
use crate::tts::KokoroTts;

const AUDIO_CHANNEL_CAPACITY: usize = 200;
const MAX_SPEECH_BUFFER_SECS: u32 = 30;
const MIN_SPEECH_DURATION_MS: u32 = 800;
/// Pre-roll chunks kept before speech onset to recover VAD onset delay (~250ms).
const PRE_ROLL_CHUNKS: usize = 15;

#[tokio::main]
async fn main() -> Result<()> {
    // Disable whisper prints into console
    install_logging_hooks();
    
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

    info!(target: "voicebot", "Language: {}", config.language);

    // ── Proactive event channel ───────────────────────────────────────────────
    let (proactive_tx, proactive_rx) = mpsc::channel::<ProactiveEvent>(32);

    // ── Tools ─────────────────────────────────────────────────────────────────
    // `shared_history` is updated after every user turn so the agent always
    // receives full conversational context via `hermes -q "{history}"`.
    let shared_history: Arc<RwLock<String>> = Arc::new(RwLock::new(String::new()));
    let mut tool_registry = ToolRegistry::new();

    // Always available
    tool_registry.register(CurrentTimeTool);
    tool_registry.register(CalendarGetEventsTool);
    tool_registry.register(CalendarCreateTool);
    tool_registry.register(ReadClipboardTool);
    tool_registry.register(SetClipboardTool);
    tool_registry.register(ReadFileTool);
    tool_registry.register(OpenAppTool);
    tool_registry.register(SendNotificationTool);

    // Shell — enabled via SHELL_ENABLED=1
    if config.shell_enabled {
        info!(target: "voicebot", "Shell tool enabled (timeout={}s)", config.shell_timeout_secs);
        tool_registry.register(RunShellTool::new(config.shell_timeout_secs));
    }

    // Vision (screenshot) — enabled when VISION_URL is set
    if let Some(ref vision_url) = config.vision_url {
        info!(target: "voicebot", "Vision tool enabled: {} (model={})", vision_url, config.vision_model);
        tool_registry.register(TakeScreenshotTool::new(
            vision_url,
            &config.vision_model,
            config.vision_max_tokens,
        ));
    }

    // External agent delegation — enabled when AGENT_COMMAND is set
    if let Some(ref agent_command) = config.agent_command {
        info!(target: "voicebot", "Agent delegation enabled: {}", agent_command);
        tool_registry.register(RunAgentAsyncTool::new(
            agent_command,
            shared_history.clone(),
            proactive_tx.clone(),
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
    );
    info!(target: "llm", "LLM endpoint: {}", config.llm_url);

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

    // ── Speaker verifier ──────────────────────────────────────────────────────
    let mut speaker_verifier: Option<SpeakerVerifier> =
        if let Some(ref model_path) = config.speaker_model {
            match SpeakerVerifier::new(
                model_path,
                std::path::Path::new(&config.speaker_enrollment_path),
                config.speaker_similarity_min,
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

    // ── TTS ───────────────────────────────────────────────────────────────────
    let tts: TtsEngine = match config.tts_provider.as_str() {
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
            info!(target: "tts", "TTS provider: say (voice={})", config.say_voice);
            let say_voice = config.say_voice.clone();
            let s = tokio::task::spawn_blocking(move || SayTts::new(&say_voice)).await??;
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

    // ── Barge-in state ────────────────────────────────────────────────────────
    let cancel = Arc::new(AtomicBool::new(false));
    let mut pipeline_handle: Option<tokio::task::JoinHandle<()>> = None;
    // Agent results that arrived while the voicebot was busy — processed in order when idle.
    let mut pending_agent_results: std::collections::VecDeque<(String, String)> =
        std::collections::VecDeque::new();

    info!(target: "voicebot", "Ready. Speak to interact...");

    // ── Startup greeting ──────────────────────────────────────────────────────
    {
        let now = chrono::Local::now();
        let time_str = now.format("%H:%M").to_string();
        let notification = format!(
            "[Sistema: el voicebot acaba de arrancar. Son las {time_str}.\n\
             Saluda al usuario de forma natural y muy concisa.\n\
             Si conoces su nombre (aparece en el perfil de usuario), úsalo.\n\
             Si no lo conoces, preséntate y pregúntale su nombre.]"
        );
        let cancel_c      = Arc::clone(&cancel);
        let tts_c         = Arc::clone(&tts);
        let audio_out_c   = Arc::clone(&audio_output);
        let llm_session_c = Arc::clone(&llm_session);
        let llm_client_c  = llm_client.clone();
        let db_c          = db.clone();
        let tools_c       = Arc::clone(&tools);
        tokio::spawn(async move {
            run_text_pipeline(
                notification, cancel_c, tts_c, audio_out_c,
                llm_session_c, llm_client_c, db_c, session_id, tts_sample_rate, tools_c,
            )
            .await;
        });
    }

    let mut proactive_rx = proactive_rx;
    tokio::select! {
        _ = async {
            loop {
                // If the active pipeline finished naturally, release it so we look idle.
                if pipeline_handle.as_ref().map(|h| h.is_finished()).unwrap_or(false) {
                    pipeline_handle = None;
                }

                // If idle and there are pending agent results, process the next one.
                if pipeline_handle.is_none() {
                    if let Some((task, result)) = pending_agent_results.pop_front() {
                        let notification = format!(
                            "[Sistema: una tarea en segundo plano ha terminado.]\n\
                             Tarea: {task}\n\
                             Resultado: {result}\n\
                             Informa al usuario de forma natural y concisa."
                        );
                        cancel.store(false, Ordering::SeqCst);
                        let cancel_c      = Arc::clone(&cancel);
                        let tts_c         = Arc::clone(&tts);
                        let audio_out_c   = Arc::clone(&audio_output);
                        let llm_session_c = Arc::clone(&llm_session);
                        let llm_client_c  = llm_client.clone();
                        let db_c          = db.clone();
                        let tools_c       = Arc::clone(&tools);
                        pipeline_handle = Some(tokio::spawn(async move {
                            run_text_pipeline(
                                notification, cancel_c, tts_c, audio_out_c,
                                llm_session_c, llm_client_c, db_c, session_id,
                                tts_sample_rate, tools_c,
                            )
                            .await;
                        }));
                    }
                }

                let chunk: AudioChunk = tokio::select! {
                    result = rx.recv() => match result {
                        Ok(c) => c,
                        Err(e) => { error!(target: "audio", "Audio channel closed: {}", e); break; }
                    },
                    Some(event) = proactive_rx.recv() => {
                        match event {
                            ProactiveEvent::AgentResult { task, result } => {
                                if pipeline_handle.is_none() {
                                    // Idle: put at front so it's picked up next iteration.
                                    pending_agent_results.push_front((task, result));
                                } else {
                                    // Busy: queue for later.
                                    pending_agent_results.push_back((task, result));
                                }
                            }
                            ProactiveEvent::InferenceDaemon { .. } => {
                                // Daemon events ignored in the queue-based pipeline.
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
                        // ── Barge-in ─────────────────────────────────────────
                        if let Some(h) = pipeline_handle.take() {
                            info!(target: "pipeline", "Barge-in detected — cancelling active pipeline");
                            cancel.store(true, Ordering::SeqCst);
                            h.abort();
                        }

                        for pre in pre_roll.drain(..) {
                            speech_buffer.push(&pre);
                        }
                        speech_buffer.push(&mono);
                    }
                    VadResult::Speech => {
                        speech_buffer.push(&mono);
                    }
                    VadResult::SpeechEnd => {
                        let t_speech_end = Instant::now();
                        speech_buffer.push(&mono);
                        let audio = speech_buffer.get_samples();
                        let duration_ms = speech_buffer.duration_ms();
                        speech_buffer.clear();
                        pre_roll.clear();

                        if duration_ms < MIN_SPEECH_DURATION_MS {
                            debug!(target: "pipeline", "Too short ({}ms), skipping", duration_ms);
                            continue;
                        }

                        info!(target: "pipeline", "Speech: {}ms — starting pipeline", duration_ms);
                        let vad_elapsed = t_speech_start.take()
                            .map(|t| t.elapsed().as_millis()).unwrap_or(0);
                        info!(target: "performance", "[+{}ms] VAD end ({}ms speech)", vad_elapsed, duration_ms);

                        // ── Speaker verification ──────────────────────────────
                        if let Some(ref mut sv) = speaker_verifier {
                            match sv.verify(config.sample_rate, &audio) {
                                SpeakerVerdict::Enrolled => {
                                    info!(target: "speaker", "Main speaker enrolled — processing utterance");
                                }
                                SpeakerVerdict::IsMainSpeaker { similarity } => {
                                    debug!(target: "speaker", "Speaker verified (similarity={similarity:.3})");
                                }
                                SpeakerVerdict::OtherSpeaker { similarity } => {
                                    info!(
                                        target: "speaker",
                                        "Unknown speaker (similarity={similarity:.3}) — discarding"
                                    );
                                    vad.reset();
                                    continue;
                                }
                            }
                        }

                        if let Some(h) = pipeline_handle.take() {
                            cancel.store(true, Ordering::SeqCst);
                            h.abort();
                        }

                        cancel.store(false, Ordering::SeqCst);

                        let cancel_c           = Arc::clone(&cancel);
                        let stt_c              = Arc::clone(&stt);
                        let tts_c              = Arc::clone(&tts);
                        let audio_out_c        = Arc::clone(&audio_output);
                        let llm_session_c      = Arc::clone(&llm_session);
                        let llm_client_c       = llm_client.clone();
                        let db_c               = db.clone();
                        let tools_c            = Arc::clone(&tools);
                        let shared_history_c   = Arc::clone(&shared_history);
                        let context_tokens     = config.llm_context_tokens;
                        let keep_turns         = config.llm_summary_keep_turns;
                        let inject_system_data = config.inject_system_data;

                        pipeline_handle = Some(tokio::spawn(async move {
                            run_pipeline(
                                audio,
                                cancel_c,
                                stt_c,
                                tts_c,
                                audio_out_c,
                                llm_session_c,
                                llm_client_c,
                                db_c,
                                session_id,
                                tts_sample_rate,
                                tools_c,
                                shared_history_c,
                                context_tokens,
                                keep_turns,
                                inject_system_data,
                                t_speech_end,
                            )
                            .await;
                        }));
                    }
                    VadResult::Silence => {
                        pre_roll.push_back(mono);
                        if pre_roll.len() > PRE_ROLL_CHUNKS {
                            pre_roll.pop_front();
                        }
                    }
                }
            }
        } => {}
        _ = tokio::signal::ctrl_c() => {
            info!(target: "voicebot", "Shutting down...");
            cancel.store(true, Ordering::SeqCst);
            if let Some(h) = pipeline_handle.take() {
                h.abort();
            }
        }
    }

    Ok(())
}

/// Stream LLM tokens into TTS, sentence by sentence.
///
/// Returns `(full_response, tool_call, last_play)`.
/// - `tool_call`: if the LLM emitted a tool call, its text was NOT sent to TTS.
/// - `last_play`: the still-running playback task for the last sentence, or
///   `None` if already finished (cancelled / tool-call path). The caller is
///   responsible for awaiting or aborting it — this allows the caller to do
///   CPU/GPU work (DB writes, summarization) concurrently with the tail audio.
///
/// Synthesis and playback are overlapped: as soon as sentence N's text is
/// ready its synthesis starts in a blocking task, while sentence N-1 is still
/// playing. By the time N-1 finishes (typically 1–3 s), N is already
/// synthesised, so the gap between sentences is near zero.
async fn stream_and_tts(
    mut token_rx: mpsc::Receiver<StreamToken>,
    cancel: &Arc<AtomicBool>,
    tts: &Arc<TtsEngine>,
    audio_output: &Arc<AudioOutput>,
    tts_sample_rate: u32,
    t_llm_start: Instant,
) -> (String, Option<(String, String)>, Option<tokio::task::JoinHandle<anyhow::Result<()>>>) {
    let mut sentence_buf = SentenceSplitter::new();
    let mut full_response = String::new();

    // Playback task for the sentence that is currently playing.
    // We hold it here so the next sentence's synthesis can run in parallel.
    let mut play_handle: Option<tokio::task::JoinHandle<anyhow::Result<()>>> = None;

    // Drain a finished play_handle, logging any error.
    macro_rules! await_play {
        ($h:expr) => {
            if let Some(h) = $h.take() {
                match h.await {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => error!(target: "audio", "Playback error: {}", e),
                    Err(e) => error!(target: "audio", "Playback task panicked: {}", e),
                }
            }
        };
    }

    let mut t_prev = t_llm_start;
    let mut first_sentence = true;
    let mut first_token_seen = false;

    loop {
        if cancel.load(Ordering::SeqCst) {
            if let Some(h) = play_handle { h.abort(); }
            return (full_response, None, None);
        }

        let event = token_rx.recv().await;

        // ── Tool call ──────────────────────────────────────────────────────────
        if let Some(StreamToken::ToolCall { name, args }) = &event {
            await_play!(play_handle);
            return (full_response, Some((name.clone(), args.clone())), None);
        }

        let is_done = event.is_none();
        let token = match event {
            Some(StreamToken::Content(s)) => s,
            _ => String::new(),
        };

        if !first_token_seen && !token.is_empty() {
            info!(target: "performance", "[+{}ms] LLM first token", t_prev.elapsed().as_millis());
            t_prev = Instant::now();
            first_token_seen = true;
        }

        full_response.push_str(&token);

        let sentences_to_play: Vec<String> = if is_done {
            let mut v = Vec::new();
            if let Some(s) = sentence_buf.push(&token) { v.push(s); }
            if let Some(s) = sentence_buf.flush()      { v.push(s); }
            v
        } else if let Some(s) = sentence_buf.push(&token) {
            vec![s]
        } else {
            vec![]
        };

        for sentence in sentences_to_play {
            if cancel.load(Ordering::SeqCst) {
                if let Some(h) = play_handle { h.abort(); }
                return (full_response, None, None);
            }

            info!(target: "tts", "TTS: {:?}", sentence);

            // ── Start synthesis immediately (runs while previous sentence plays) ──
            if first_sentence {
                info!(target: "performance", "[+{}ms] TTS start", t_prev.elapsed().as_millis());
                t_prev = Instant::now();
            }
            let t_synth = Instant::now();
            let tts_c = Arc::clone(tts);
            let sentence_c = sentence.clone();
            let synth_handle = tokio::task::spawn_blocking(move || {
                tts_c.synthesize(&sentence_c)
            });

            // ── Wait for the previous sentence to finish playing ──────────────────
            await_play!(play_handle);

            if cancel.load(Ordering::SeqCst) {
                synth_handle.abort();
                return (full_response, None, None);
            }

            // ── Collect synthesis result (usually already done) ───────────────────
            let samples = match synth_handle.await {
                Ok(Ok(s)) => s,
                Ok(Err(e)) => { error!(target: "tts", "TTS error: {}", e); continue; }
                Err(e) => { error!(target: "tts", "TTS task panicked: {}", e); continue; }
            };
            if first_sentence {
                info!(target: "performance", "[+{}ms] TTS end", t_prev.elapsed().as_millis());
                t_prev = Instant::now();
                first_sentence = false;
            } else {
                debug!(target: "performance", "TTS IN: {}ms", t_synth.elapsed().as_millis());
            }

            if cancel.load(Ordering::SeqCst) {
                return (full_response, None, None);
            }

            // ── Start playback without awaiting — next sentence can synthesise now ─
            let out_c = Arc::clone(audio_output);
            let cancel_c = Arc::clone(cancel);
            play_handle = Some(tokio::task::spawn_blocking(move || {
                out_c.play_blocking(&samples, tts_sample_rate, &cancel_c)
            }));
        }

        if is_done {
            break;
        }
    }

    debug!(target: "performance", "LLM TG: {}ms", t_llm_start.elapsed().as_millis());

    // Return the last playback handle to the caller so it can overlap
    // GPU/DB work (summarization, profile extraction) with tail audio.
    (full_response, None, play_handle)
}

/// Maximum number of sequential tool calls allowed per user turn.
const MAX_TOOL_ITERATIONS: usize = 5;

/// Full STT → LLM → (tools →)* TTS pipeline for a single utterance.
async fn run_pipeline(
    audio: Vec<f32>,
    cancel: Arc<AtomicBool>,
    stt: Arc<WhisperStt>,
    tts: Arc<TtsEngine>,
    audio_output: Arc<AudioOutput>,
    llm_session: Arc<Mutex<LlmSession>>,
    llm_client: LlamaClient,
    db: Database,
    session_id: Uuid,
    tts_sample_rate: u32,
    tools: Arc<ToolRegistry>,
    shared_history: Arc<RwLock<String>>,
    context_tokens: usize,
    summary_keep_turns: usize,
    inject_system_data: bool,
    t_speech_end: Instant,
) {
    macro_rules! check_cancel {
        () => {
            if cancel.load(Ordering::SeqCst) {
                debug!(target: "pipeline", "Pipeline cancelled");
                return;
            }
        };
    }

    let mut t_prev = t_speech_end;

    // ── Speculative KV-cache warm-up — runs in parallel with STT ──────────────
    // Fire a 1-token streaming request with the current session messages so
    // llama.cpp starts computing KV vectors while Whisper transcribes audio.
    // Aborted as soon as STT is done; the partial cache is kept by the server.
    let spec_msgs = llm_session.lock().unwrap().all_messages_api();
    let spec_client = llm_client.clone();
    let mut spec_handle = tokio::spawn(async move {
        if let Err(e) = spec_client.prefill_warm(spec_msgs).await {
            debug!(target: "llm", "Speculative prefill ended: {e}");
        }
    });

    // ── system_state::build() — also in parallel with STT ─────────────────────
    let state_handle = if inject_system_data {
        Some(tokio::spawn(system_state::build()))
    } else {
        None
    };

    // ── STT ───────────────────────────────────────────────────────────────────
    info!(target: "performance", "[+{}ms] STT start", t_prev.elapsed().as_millis());
    t_prev = Instant::now();
    let transcript = match tokio::task::spawn_blocking(move || stt.transcribe(&audio)).await {
        Ok(Ok(t)) => t,
        Ok(Err(e)) => { error!(target: "stt", "STT error: {}", e); spec_handle.abort(); return; }
        Err(e)     => { error!(target: "stt", "STT task panicked: {}", e); spec_handle.abort(); return; }
    };
    info!(target: "performance", "[+{}ms] STT end", t_prev.elapsed().as_millis());
    t_prev = Instant::now();

    // Wait briefly for speculative prefill to finish cleanly (max_tokens=1 means
    // it completes in ~200ms for typical histories). A clean finish lets llama.cpp
    // release the slot without cleanup delay; abort only if still running after window.
    tokio::select! {
        _ = &mut spec_handle => {
            debug!(target: "llm", "Speculative prefill completed cleanly");
        }
        _ = tokio::time::sleep(std::time::Duration::from_millis(80)) => {
            debug!(target: "llm", "Speculative prefill still running after STT — aborting");
            spec_handle.abort();
            let _ = spec_handle.await;
        }
    }

    check_cancel!();

    if transcript.is_empty() {
        debug!(target: "stt", "Empty transcript, skipping");
        return;
    }

    info!(target: "pipeline", "User: {}", transcript);

    let messages_snapshot = llm_session.lock().unwrap().messages.clone();

    let session_for_llm = {
        let mut s = llm_session.lock().unwrap();
        s.add_user_turn(&transcript);
        // Update shared history so the agent tool has the current conversation
        // context (including this user turn) when run_agent_async is called.
        if let Ok(mut h) = shared_history.write() {
            *h = format_history(&s.messages);
        }
        s.clone()
    };

    // Save user message in background — don't block LLM start on a DB write.
    {
        let db_c = db.clone();
        let transcript_c = transcript.clone();
        tokio::spawn(async move {
            if let Err(e) = db_c.save_message(session_id, "User", &transcript_c).await {
                warn!(target: "db", "Failed to save user message: {}", e);
            }
        });
    }

    check_cancel!();

    // ── LLM streaming + tool call loop ────────────────────────────────────────
    let mut session_snapshot = session_for_llm;
    let mut final_response = String::new();
    // Last playback handle returned by stream_and_tts (last sentence still playing).
    // Awaited after committing DB/session so GPU work overlaps with tail audio.
    let mut last_play: Option<tokio::task::JoinHandle<anyhow::Result<()>>> = None;
    let mut committed = false;

    'pipeline: {
        let tool_defs = tools.tool_definitions();

        // Build the initial message list. Inject ambient system state as a
        // prefix to the current user message (ephemeral — not stored in session).
        let mut messages = session_snapshot.all_messages_api();
        if let Some(h) = state_handle {
            let state = h.await.unwrap_or_default();
            if let Some(last_msg) = messages.last_mut() {
                if last_msg["role"] == "user" {
                    let original = last_msg["content"].as_str().unwrap_or("").to_string();
                    last_msg["content"] = serde_json::Value::String(
                        format!("{state}\n\n{original}")
                    );
                }
            }
            debug!(target: "pipeline", "System state injected: {}", state);
        }

        // Tool call loop — allows the model to call multiple tools sequentially
        // before producing its final spoken response (max MAX_TOOL_ITERATIONS).
        'tool_loop: for iter in 0..MAX_TOOL_ITERATIONS {
            info!(target: "performance", "[+{}ms] LLM request", t_prev.elapsed().as_millis());
            t_prev = Instant::now();
            let t_llm_start = t_prev;
            let token_rx = match llm_client.stream(&messages, &tool_defs).await {
                Ok(r)  => r,
                Err(e) => { error!(target: "llm", "LLM stream error: {}", e); break 'pipeline; }
            };

            let (llm_text, tool_call, play) =
                stream_and_tts(token_rx, &cancel, &tts, &audio_output, tts_sample_rate, t_llm_start).await;

            if cancel.load(Ordering::SeqCst) {
                if let Some(h) = play { h.abort(); }
                break 'pipeline;
            }

            match tool_call {
                Some((name, args)) => {
                    // play is None here — tool-call path already awaited it.
                    let result = tools.execute(&name, &args).await;
                    info!(target: "pipeline", "Tool[{}] `{}` → {}", iter, name, result);

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

                    if cancel.load(Ordering::SeqCst) { break 'pipeline; }
                    t_prev = Instant::now();
                    // Continue to next iteration for the model's follow-up.
                }
                None => {
                    // No tool call — this is the final spoken response.
                    final_response = llm_text;
                    last_play = play;
                    break 'tool_loop;
                }
            }
        }

        if final_response.is_empty() {
            break 'pipeline;
        }

        info!(target: "pipeline", "Assistant: {}", final_response);

        // ── Commit to DB and session while last TTS sentence plays ────────────
        // save_message is a fast SQLite write; it runs concurrently with last_play.
        if let Err(e) = db.save_message(session_id, "Assistant", &final_response).await {
            warn!(target: "db", "Failed to save assistant message: {}", e);
        }
        llm_session.lock().unwrap().add_assistant_turn(&final_response);
        committed = true;

        // ── Background GPU tasks — overlap with tail audio playback ───────────
        // Both calls use cache_prompt=false so they don't touch the main slot.
        {
            let llm_session_c = Arc::clone(&llm_session);
            let llm_client_c  = llm_client.clone();
            let db_c          = db.clone();
            tokio::spawn(async move {
                maybe_summarize(&llm_session_c, &llm_client_c, &db_c, session_id, context_tokens, summary_keep_turns).await;
            });
        }
        {
            let llm_client_c = llm_client.clone();
            let db_c         = db.clone();
            let transcript_c = transcript.clone();
            let response_c   = final_response.clone();
            tokio::spawn(async move {
                let facts = extract_facts(&llm_client_c, &transcript_c, &response_c).await;
                for fact in facts {
                    if let Err(e) = db_c.upsert_profile_fact(&fact.key, &fact.value, fact.confidence).await {
                        warn!(target: "profile", "Failed to save profile fact '{}': {}", fact.key, e);
                    } else {
                        debug!(target: "profile", "Profile: {} = {} ({:.0}%)", fact.key, fact.value, fact.confidence * 100.0);
                    }
                }
            });
        }

        // ── Await last TTS sentence — keeps pipeline_handle alive for barge-in ─
        if let Some(h) = last_play {
            match h.await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => error!(target: "audio", "Playback error: {}", e),
                Err(e)     => error!(target: "audio", "Playback task panicked: {}", e),
            }
        }
    }

    // ── Roll back session if cancelled before commit ───────────────────────────
    if !committed && cancel.load(Ordering::SeqCst) {
        llm_session.lock().unwrap().messages = messages_snapshot;
        info!(target: "pipeline", "Pipeline cancelled — session rolled back");
    }
}

/// Summarize old conversation turns if the prompt is approaching the context limit.
///
/// Runs after every completed pipeline turn. Builds a summary of old turns,
/// injects it into the session prompt, and persists it to the DB so future
/// restarts can restore the compact context.
async fn maybe_summarize(
    llm_session: &Arc<Mutex<LlmSession>>,
    llm_client: &LlamaClient,
    db: &Database,
    session_id: Uuid,
    context_tokens: usize,
    keep_turns: usize,
) {
    let needs = llm_session.lock().unwrap().needs_summarization(context_tokens);
    if !needs {
        return;
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

    let summary = match llm_client.complete(&prompt).await {
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

/// Speak a notification, persisting it to session and DB.
///
/// Adds the notification as a user turn, calls the LLM, speaks the response,
/// then commits the assistant turn. Used for agent results and startup greeting.
async fn run_text_pipeline(
    notification: String,
    cancel: Arc<AtomicBool>,
    tts: Arc<TtsEngine>,
    audio_output: Arc<AudioOutput>,
    llm_session: Arc<Mutex<LlmSession>>,
    llm_client: LlamaClient,
    db: Database,
    session_id: uuid::Uuid,
    tts_sample_rate: u32,
    tools: Arc<ToolRegistry>,
) {
    if cancel.load(Ordering::SeqCst) {
        return;
    }

    info!(target: "pipeline", "Text pipeline: {}", &notification[..notification.len().min(80)]);

    // Add the notification as a user turn so the LLM has it in context.
    {
        let mut s = llm_session.lock().unwrap();
        s.add_user_turn(&notification);
    }
    {
        let db_c = db.clone();
        let notif_c = notification.clone();
        tokio::spawn(async move {
            if let Err(e) = db_c.save_message(session_id, "User", &notif_c).await {
                warn!(target: "db", "Failed to save text-pipeline user message: {}", e);
            }
        });
    }

    let messages_api: Vec<serde_json::Value> = {
        let s = llm_session.lock().unwrap();
        s.all_messages_api()
    };

    let tool_defs = tools.tool_definitions();
    let mut messages = messages_api;
    let mut last_play: Option<tokio::task::JoinHandle<anyhow::Result<()>>> = None;
    let mut llm_text_final = String::new();

    'tool_loop: for iter in 0..MAX_TOOL_ITERATIONS {
        let t_llm_start = Instant::now();
        let token_rx = match llm_client.stream(&messages, &tool_defs).await {
            Ok(r) => r,
            Err(e) => {
                error!(target: "llm", "Text pipeline LLM error: {}", e);
                return;
            }
        };

        let (llm_text, tool_call, play) =
            stream_and_tts(token_rx, &cancel, &tts, &audio_output, tts_sample_rate, t_llm_start).await;

        if cancel.load(Ordering::SeqCst) {
            if let Some(h) = play { h.abort(); }
            return;
        }

        match tool_call {
            Some((name, args)) => {
                let result = tools.execute(&name, &args).await;
                info!(target: "pipeline", "Text pipeline tool[{}] `{}` → {}", iter, name, result);

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

                if cancel.load(Ordering::SeqCst) { return; }
            }
            None => {
                if !llm_text.is_empty() {
                    info!(target: "pipeline", "Text pipeline: {}", llm_text);
                }
                llm_text_final = llm_text;
                last_play = play;
                break 'tool_loop;
            }
        }
    }

    // Persist assistant response to session and DB.
    if !llm_text_final.is_empty() {
        llm_session.lock().unwrap().add_assistant_turn(&llm_text_final);
        let db_c = db.clone();
        let text_c = llm_text_final.clone();
        tokio::spawn(async move {
            if let Err(e) = db_c.save_message(session_id, "Assistant", &text_c).await {
                warn!(target: "db", "Failed to save text-pipeline assistant message: {}", e);
            }
        });
    }

    if let Some(h) = last_play {
        match h.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => error!(target: "audio", "Text pipeline playback error: {}", e),
            Err(e)     => error!(target: "audio", "Text pipeline playback task panicked: {}", e),
        }
    }
}
