mod agents;
mod audio;
mod config;
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
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

use crate::agents::ProactiveEvent;
use crate::audio::audio_capture::{AudioCapture, AudioChunk};
use crate::audio::buffer::AudioBuffer;
use crate::audio::output::AudioOutput;
use crate::audio::vad::{VadResult, VoiceActivityDetector};
use crate::config::Config;
use crate::db::Database;
use crate::llm::{LlamaClient, LlmSession, StreamToken};
use crate::profile::{build_profile_context, extract_facts, ProfileFact};
use crate::stt::WhisperStt;
use crate::tools::{CurrentTimeTool, RunAgentAsyncTool, RunAgentTool, RunShellTool, TakeScreenshotTool, ToolRegistry};
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
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    dotenvy::dotenv().ok();

    info!("Starting voicebot...");
    let config = Config::from_env()?;

    // ── Device listing shortcut ───────────────────────────────────────────────
    let list_devices = config.list_devices
        || std::env::args().any(|a| a == "--list-devices" || a == "list-devices");
    if list_devices {
        AudioCapture::print_devices()?;
        return Ok(());
    }

    info!("Language: {}", config.language);

    // ── Proactive event channel ───────────────────────────────────────────────
    let (proactive_tx, proactive_rx) = mpsc::channel::<ProactiveEvent>(32);

    // ── Tools ─────────────────────────────────────────────────────────────────
    let mut tool_registry = ToolRegistry::new();
    tool_registry.register(CurrentTimeTool);
    if config.shell_enabled {
        info!("Shell tool enabled (timeout={}s)", config.shell_timeout_secs);
        tool_registry.register(RunShellTool::new(config.shell_timeout_secs));
    }
    if let Some(ref vision_url) = config.vision_url {
        info!("Vision tool enabled: {} (model={})", vision_url, config.vision_model);
        tool_registry.register(TakeScreenshotTool::new(
            vision_url,
            &config.vision_model,
            config.vision_max_tokens,
        ));
    }
    if let Some(ref agent_url) = config.agent_url {
        info!("Agent delegation enabled: {}", agent_url);
        tool_registry.register(RunAgentTool::new(
            agent_url,
            &config.agent_model,
            config.agent_max_tokens,
        ));
        tool_registry.register(RunAgentAsyncTool::new(
            agent_url,
            &config.agent_model,
            config.agent_max_tokens,
            proactive_tx,
        ));
    }
    let tools = Arc::new(tool_registry);

    // ── Database ─────────────────────────────────────────────────────────────
    let db = Database::new(&config.db_path).await?;
    let session_id = db.get_or_create_session().await?;
    let (summary, history) = db.get_session_context(session_id).await?;
    info!(
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
        info!("Loaded {} user profile facts", profile_facts.len());
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
    );
    info!("LLM endpoint: {}", config.llm_url);

    // ── STT (whisper) ─────────────────────────────────────────────────────────
    let whisper_model = config.whisper_model.clone();
    let whisper_language = config.language.clone();
    let stt = tokio::task::spawn_blocking(move || {
        WhisperStt::new(&whisper_model, &whisper_language)
    })
    .await??;
    let stt = Arc::new(stt);

    // ── TTS ───────────────────────────────────────────────────────────────────
    let tts: TtsEngine = match config.tts_provider.as_str() {
        #[cfg(feature = "kokoro")]
        "kokoro" => {
            info!("TTS provider: Kokoro (voice={}, lang={})", config.kokoro_voice, config.kokoro_language);
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
            info!("TTS provider: say (voice={})", config.say_voice);
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
        "Audio output: {}Hz, {}ch",
        audio_output.sample_rate(),
        audio_output.channels()
    );

    // ── Audio capture ─────────────────────────────────────────────────────────
    let audio_capture = AudioCapture::new(config.audio_input_device.as_deref())?;
    let source_sample_rate = audio_capture.sample_rate();
    info!("Audio input: {}Hz", source_sample_rate);

    let samples_per_chunk = config.samples_per_chunk();
    let (tx, rx) = bounded(AUDIO_CHANNEL_CAPACITY);
    let _stream = audio_capture.start_capture(tx, samples_per_chunk)?;

    let mut vad = VoiceActivityDetector::new(source_sample_rate, config.vad_silence_ms)?;
    info!("VAD silence threshold: {}ms", config.vad_silence_ms);
    let mut speech_buffer = AudioBuffer::new(source_sample_rate, MAX_SPEECH_BUFFER_SECS);
    let mut pre_roll: VecDeque<Vec<f32>> = VecDeque::with_capacity(PRE_ROLL_CHUNKS + 1);

    // ── Barge-in state ────────────────────────────────────────────────────────
    let cancel = Arc::new(AtomicBool::new(false));
    let mut pipeline_handle: Option<tokio::task::JoinHandle<()>> = None;

    info!("Ready. Speak to interact...");

    // ── Startup greeting ──────────────────────────────────────────────────────
    // The voicebot greets the user once at startup. It uses the proactive
    // pipeline so it doesn't modify the conversation history.
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
        let tools_c       = Arc::clone(&tools);
        tokio::spawn(async move {
            run_proactive_pipeline(
                notification, cancel_c, tts_c, audio_out_c,
                llm_session_c, llm_client_c, tts_sample_rate, tools_c,
            )
            .await;
        });
    }

    let mut proactive_rx = proactive_rx;
    tokio::select! {
        _ = async {
            loop {
                let chunk: AudioChunk = tokio::select! {
                    result = rx.recv() => match result {
                        Ok(c) => c,
                        Err(e) => { error!("Audio channel closed: {}", e); break; }
                    },
                    Some(event) = proactive_rx.recv() => {
                        let ProactiveEvent::AgentResult { task, result } = event;
                        let notification = format!(
                            "[Sistema: una tarea en segundo plano ha terminado.]\n\
                             Tarea: {task}\n\
                             Resultado: {result}\n\
                             Informa al usuario de forma natural y concisa."
                        );
                        let cancel_c      = Arc::clone(&cancel);
                        let tts_c         = Arc::clone(&tts);
                        let audio_out_c   = Arc::clone(&audio_output);
                        let llm_session_c = Arc::clone(&llm_session);
                        let llm_client_c  = llm_client.clone();
                        let tools_c       = Arc::clone(&tools);
                        tokio::spawn(async move {
                            run_proactive_pipeline(
                                notification, cancel_c, tts_c, audio_out_c,
                                llm_session_c, llm_client_c, tts_sample_rate, tools_c,
                            )
                            .await;
                        });
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
                        // ── Barge-in ─────────────────────────────────────────
                        if let Some(h) = pipeline_handle.take() {
                            info!("Barge-in detected — cancelling active pipeline");
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
                        speech_buffer.push(&mono);
                        let audio = speech_buffer.get_samples();
                        let duration_ms = speech_buffer.duration_ms();
                        speech_buffer.clear();
                        pre_roll.clear();

                        if duration_ms < MIN_SPEECH_DURATION_MS {
                            debug!("Too short ({}ms), skipping", duration_ms);
                            continue;
                        }

                        info!("Speech: {}ms — starting pipeline", duration_ms);

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
                                context_tokens,
                                keep_turns,
                                inject_system_data,
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
            info!("Shutting down...");
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
                    Ok(Err(e)) => error!("Playback error: {}", e),
                    Err(e) => error!("Playback task panicked: {}", e),
                }
            }
        };
    }

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

            info!("TTS: {:?}", sentence);

            // ── Start synthesis immediately (runs while previous sentence plays) ──
            let tts_c = Arc::clone(tts);
            let sentence_c = sentence.clone();
            let synth_handle =
                tokio::task::spawn_blocking(move || tts_c.synthesize(&sentence_c));

            // ── Wait for the previous sentence to finish playing ──────────────────
            await_play!(play_handle);

            if cancel.load(Ordering::SeqCst) {
                synth_handle.abort();
                return (full_response, None, None);
            }

            // ── Collect synthesis result (usually already done) ───────────────────
            let samples = match synth_handle.await {
                Ok(Ok(s)) => s,
                Ok(Err(e)) => { error!("TTS error: {}", e); continue; }
                Err(e) => { error!("TTS task panicked: {}", e); continue; }
            };

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
    context_tokens: usize,
    summary_keep_turns: usize,
    inject_system_data: bool,
) {
    macro_rules! check_cancel {
        () => {
            if cancel.load(Ordering::SeqCst) {
                debug!("Pipeline cancelled");
                return;
            }
        };
    }

    // ── STT ───────────────────────────────────────────────────────────────────
    let transcript = match tokio::task::spawn_blocking(move || stt.transcribe(&audio)).await {
        Ok(Ok(t)) => t,
        Ok(Err(e)) => { error!("STT error: {}", e); return; }
        Err(e)     => { error!("STT task panicked: {}", e); return; }
    };

    check_cancel!();

    if transcript.is_empty() {
        debug!("Empty transcript, skipping");
        return;
    }

    info!("User: {}", transcript);

    let messages_snapshot = llm_session.lock().unwrap().messages.clone();

    let session_for_llm = {
        let mut s = llm_session.lock().unwrap();
        s.add_user_turn(&transcript);
        s.clone()
    };

    // Save user message in background — don't block LLM start on a DB write.
    {
        let db_c = db.clone();
        let transcript_c = transcript.clone();
        tokio::spawn(async move {
            if let Err(e) = db_c.save_message(session_id, "User", &transcript_c).await {
                warn!("Failed to save user message: {}", e);
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
        if inject_system_data {
            let state = system_state::build().await;
            if let Some(last_msg) = messages.last_mut() {
                if last_msg["role"] == "user" {
                    let original = last_msg["content"].as_str().unwrap_or("").to_string();
                    last_msg["content"] = serde_json::Value::String(
                        format!("{state}\n\n{original}")
                    );
                }
            }
            debug!("System state injected: {}", state);
        }

        // Tool call loop — allows the model to call multiple tools sequentially
        // before producing its final spoken response (max MAX_TOOL_ITERATIONS).
        'tool_loop: for iter in 0..MAX_TOOL_ITERATIONS {
            let token_rx = match llm_client.stream(&messages, &tool_defs).await {
                Ok(r)  => r,
                Err(e) => { error!("LLM stream error: {}", e); break 'pipeline; }
            };

            let (llm_text, tool_call, play) =
                stream_and_tts(token_rx, &cancel, &tts, &audio_output, tts_sample_rate).await;

            if cancel.load(Ordering::SeqCst) {
                if let Some(h) = play { h.abort(); }
                break 'pipeline;
            }

            match tool_call {
                Some((name, args)) => {
                    // play is None here — tool-call path already awaited it.
                    let result = tools.execute(&name, &args).await;
                    info!("Tool[{}] `{}` → {}", iter, name, result);

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

        info!("Assistant: {}", final_response);

        // ── Commit to DB and session while last TTS sentence plays ────────────
        // save_message is a fast SQLite write; it runs concurrently with last_play.
        if let Err(e) = db.save_message(session_id, "Assistant", &final_response).await {
            warn!("Failed to save assistant message: {}", e);
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
                        warn!("Failed to save profile fact '{}': {}", fact.key, e);
                    } else {
                        debug!("Profile: {} = {} ({:.0}%)", fact.key, fact.value, fact.confidence * 100.0);
                    }
                }
            });
        }

        // ── Await last TTS sentence — keeps pipeline_handle alive for barge-in ─
        if let Some(h) = last_play {
            match h.await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => error!("Playback error: {}", e),
                Err(e)     => error!("Playback task panicked: {}", e),
            }
        }
    }

    // ── Roll back session if cancelled before commit ───────────────────────────
    if !committed && cancel.load(Ordering::SeqCst) {
        llm_session.lock().unwrap().messages = messages_snapshot;
        info!("Pipeline cancelled — session rolled back");
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

    info!("Context limit approaching — summarizing {} old turns...", turns_to_summarize);

    let summary = match llm_client.complete(&prompt).await {
        Ok(s) if !s.is_empty() => s,
        Ok(_) => {
            warn!("Summarization returned empty result, skipping");
            return;
        }
        Err(e) => {
            warn!("Summarization failed: {}", e);
            return;
        }
    };

    info!("Summary: {}", summary);

    // Find the DB message id of the last turn that is being summarized.
    // Each turn in `turns` corresponds to one row in messages (alternating User/Assistant),
    // so the last summarized message is at 0-based offset (turns_to_summarize - 1).
    let through_id = match db
        .get_message_id_at_offset(session_id, turns_to_summarize - 1)
        .await
    {
        Ok(Some(id)) => id,
        Ok(None) => {
            warn!("Could not find message offset for summary cutpoint, skipping");
            return;
        }
        Err(e) => {
            warn!("DB error finding summary cutpoint: {}", e);
            return;
        }
    };

    if let Err(e) = db.save_summary(session_id, &summary, through_id).await {
        warn!("Failed to persist summary: {}", e);
    }

    llm_session.lock().unwrap().apply_summary(&summary, keep_turns);

    info!(
        "Summarization complete — prompt compacted (keeping {} recent turns)",
        keep_turns
    );
}

/// Speak a proactive notification without a preceding user utterance.
///
/// Builds a temporary message list (current session context + a synthetic
/// notification message), calls the LLM to produce a natural-language
/// announcement, and streams the result straight to TTS.
/// The response is NOT committed to the session or database.
async fn run_proactive_pipeline(
    notification: String,
    cancel: Arc<AtomicBool>,
    tts: Arc<TtsEngine>,
    audio_output: Arc<AudioOutput>,
    llm_session: Arc<Mutex<LlmSession>>,
    llm_client: LlamaClient,
    tts_sample_rate: u32,
    tools: Arc<ToolRegistry>,
) {
    if cancel.load(Ordering::SeqCst) {
        return;
    }

    info!("Proactive pipeline: {}", &notification[..notification.len().min(80)]);

    // Build a temporary message list that asks the LLM to respond.
    let messages_api: Vec<serde_json::Value> = {
        let s = llm_session.lock().unwrap();
        let mut msgs = s.all_messages_api();
        msgs.push(serde_json::json!({
            "role": "user",
            "content": notification,
        }));
        msgs
    };

    let token_rx = match llm_client.stream(&messages_api, &[]).await {
        Ok(r) => r,
        Err(e) => {
            error!("Proactive LLM error: {}", e);
            return;
        }
    };

    let (response, _, last_play) =
        stream_and_tts(token_rx, &cancel, &tts, &audio_output, tts_sample_rate).await;

    if !response.is_empty() {
        info!("Proactive: {}", response);
    }

    if let Some(h) = last_play {
        match h.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => error!("Proactive playback error: {}", e),
            Err(e)     => error!("Proactive playback task panicked: {}", e),
        }
    }
}
