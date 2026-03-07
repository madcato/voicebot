mod audio;
mod config;
mod db;
mod llm;
mod stt;
mod tts;

use anyhow::Result;
use async_channel::bounded;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tracing::{debug, error, info, warn};
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

use crate::audio::audio_capture::{AudioCapture, AudioChunk};
use crate::audio::buffer::AudioBuffer;
use crate::audio::output::AudioOutput;
use crate::audio::vad::{VadResult, VoiceActivityDetector};
use crate::config::Config;
use crate::db::Database;
use crate::llm::{LlamaClient, LlmSession};
use crate::stt::WhisperStt;
use crate::tts::{SayTts, SentenceSplitter};

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

    // ── Database ─────────────────────────────────────────────────────────────
    let db = Database::new(&config.db_path).await?;
    let session_id = db.get_or_create_session().await?;
    let history = db.get_session_messages(session_id).await?;
    info!("Loaded {} messages from history", history.len());

    // ── LLM session ───────────────────────────────────────────────────────────
    let llm_session = Arc::new(Mutex::new(LlmSession::from_history(
        &config.llm_system_prompt,
        config.llm_slot_id,
        &history,
    )));

    // ── LLM client ────────────────────────────────────────────────────────────
    let llm_client = LlamaClient::new(
        &config.llm_url,
        config.llm_max_tokens,
        config.llm_temperature,
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

    // ── TTS (say) ─────────────────────────────────────────────────────────────
    let say_voice = config.say_voice.clone();
    let tts = tokio::task::spawn_blocking(move || SayTts::new(&say_voice)).await??;
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
    let audio_capture = AudioCapture::new(config.audio_device.as_deref())?;
    let source_sample_rate = audio_capture.sample_rate();
    info!("Audio input: {}Hz", source_sample_rate);

    let samples_per_chunk = config.samples_per_chunk();
    let (tx, rx) = bounded(AUDIO_CHANNEL_CAPACITY);
    let _stream = audio_capture.start_capture(tx, samples_per_chunk)?;

    let mut vad = VoiceActivityDetector::new(source_sample_rate)?;
    let mut speech_buffer = AudioBuffer::new(source_sample_rate, MAX_SPEECH_BUFFER_SECS);
    let mut pre_roll: VecDeque<Vec<f32>> = VecDeque::with_capacity(PRE_ROLL_CHUNKS + 1);

    // ── Barge-in state ────────────────────────────────────────────────────────
    // Shared cancellation flag: set to true to stop the active pipeline immediately.
    let cancel = Arc::new(AtomicBool::new(false));
    let mut pipeline_handle: Option<tokio::task::JoinHandle<()>> = None;

    info!("Ready. Speak to interact...");

    tokio::select! {
        _ = async {
            loop {
                let chunk: AudioChunk = match rx.recv().await {
                    Ok(c) => c,
                    Err(e) => { error!("Audio channel closed: {}", e); break; }
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
                        // User speaks while pipeline is active: cancel it immediately.
                        if let Some(h) = pipeline_handle.take() {
                            info!("Barge-in detected — cancelling active pipeline");
                            cancel.store(true, Ordering::SeqCst);
                            h.abort();
                        }

                        // Prepend pre-roll so the utterance onset is not clipped
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

                        // Cancel any still-running pipeline before starting the new one
                        if let Some(h) = pipeline_handle.take() {
                            cancel.store(true, Ordering::SeqCst);
                            h.abort();
                        }

                        // Arm a fresh cancellation token for the new pipeline
                        cancel.store(false, Ordering::SeqCst);

                        // Clone everything the pipeline task needs
                        let cancel_c       = Arc::clone(&cancel);
                        let stt_c          = Arc::clone(&stt);
                        let tts_c          = Arc::clone(&tts);
                        let audio_out_c    = Arc::clone(&audio_output);
                        let llm_session_c  = Arc::clone(&llm_session);
                        let llm_client_c   = llm_client.clone();
                        let db_c           = db.clone();

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
                            )
                            .await;
                        }));
                    }
                    VadResult::Silence => {
                        // Keep a rolling window of recent audio for pre-roll
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
            // Cancel any active pipeline gracefully
            cancel.store(true, Ordering::SeqCst);
            if let Some(h) = pipeline_handle.take() {
                h.abort();
            }
        }
    }

    Ok(())
}

/// Full STT → LLM → TTS pipeline for a single utterance.
///
/// Runs as an independent tokio task so the VAD loop stays responsive.
/// Checks `cancel` at every stage; if set, rolls back the LLM session
/// state and returns immediately so the next utterance starts clean.
async fn run_pipeline(
    audio: Vec<f32>,
    cancel: Arc<AtomicBool>,
    stt: Arc<WhisperStt>,
    tts: Arc<SayTts>,
    audio_output: Arc<AudioOutput>,
    llm_session: Arc<Mutex<LlmSession>>,
    llm_client: LlamaClient,
    db: Database,
    session_id: Uuid,
    tts_sample_rate: u32,
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

    // Snapshot accumulated_prompt so we can roll back if cancelled mid-response
    let prompt_snapshot = llm_session.lock().unwrap().accumulated_prompt.clone();

    // Add user turn; get a clone for the LLM call (avoids holding the lock during streaming)
    let session_for_llm = {
        let mut s = llm_session.lock().unwrap();
        s.add_user_turn(&transcript);
        s.clone()
    };

    // Persist user message
    if let Err(e) = db.save_message(session_id, "User", &transcript).await {
        warn!("Failed to save user message: {}", e);
    }

    check_cancel!();

    // ── LLM streaming ─────────────────────────────────────────────────────────
    let mut token_rx = match llm_client.stream(&session_for_llm).await {
        Ok(r)  => r,
        Err(e) => { error!("LLM stream error: {}", e); return; }
    };

    let mut sentence_buf = SentenceSplitter::new();
    let mut full_response = String::new();

    'token_loop: loop {
        check_cancel!();

        let token = token_rx.recv().await;
        let is_done = token.is_none();
        let token = token.unwrap_or_default();

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

        // ── TTS + playback, sentence by sentence ──────────────────────────────
        for sentence in sentences_to_play {
            check_cancel!();

            info!("TTS: {:?}", sentence);

            let tts_c = Arc::clone(&tts);
            let sentence_c = sentence.clone();
            let samples = match tokio::task::spawn_blocking(move || tts_c.synthesize(&sentence_c)).await {
                Ok(Ok(s)) => s,
                Ok(Err(e)) => { error!("TTS error: {}", e); continue; }
                Err(e)     => { error!("TTS task panicked: {}", e); continue; }
            };

            check_cancel!();

            let audio_out_c = Arc::clone(&audio_output);
            let cancel_c    = Arc::clone(&cancel);
            if let Err(e) = tokio::task::spawn_blocking(move || {
                audio_out_c.play_blocking(&samples, tts_sample_rate, &cancel_c)
            })
            .await
            .expect("playback task panicked")
            {
                error!("Playback error: {}", e);
            }
        }

        if is_done { break 'token_loop; }
    }

    // ── Finalise or roll back ─────────────────────────────────────────────────
    if cancel.load(Ordering::SeqCst) {
        // Interrupted mid-response: undo the user turn so the next request
        // starts from a consistent session state.
        llm_session.lock().unwrap().accumulated_prompt = prompt_snapshot;
        info!("Pipeline cancelled — session rolled back");
        return;
    }

    info!("Assistant: {}", full_response);

    if let Err(e) = db.save_message(session_id, "Assistant", &full_response).await {
        warn!("Failed to save assistant message: {}", e);
    }

    llm_session.lock().unwrap().add_assistant_turn(&full_response);
}
