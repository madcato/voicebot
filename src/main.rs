mod audio;
mod config;
mod db;
mod llm;
mod stt;
mod tts;

use anyhow::Result;
use async_channel::bounded;
use std::collections::VecDeque;
use std::sync::Arc;
use tracing::{debug, error, info, warn};
use tracing_subscriber::EnvFilter;

use crate::audio::audio_capture::{AudioCapture, AudioChunk};
use crate::audio::buffer::AudioBuffer;
use crate::audio::output::AudioOutput;
use crate::audio::vad::{VadResult, VoiceActivityDetector};
use crate::config::Config;
use crate::db::Database;
use crate::llm::{LlamaClient, LlmSession};
use crate::stt::WhisperStt;
use crate::tts::{PiperTts, SentenceSplitter};

const AUDIO_CHANNEL_CAPACITY: usize = 200;
const MAX_SPEECH_BUFFER_SECS: u32 = 30;
const MIN_SPEECH_DURATION_MS: u32 = 800;
/// Pre-roll: chunks of audio kept before speech starts and prepended on SpeechStart.
/// Covers the VAD onset delay (~256ms) plus margin. At 512 samples/chunk @16kHz ≈ 32ms/chunk → 15 chunks ≈ 480ms.
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

    // ── LLM session (restore accumulated prompt from history) ─────────────────
    let mut llm_session = LlmSession::from_history(
        &config.llm_system_prompt,
        config.llm_slot_id,
        &history,
    );

    // ── LLM client ────────────────────────────────────────────────────────────
    let llm_client = LlamaClient::new(
        &config.llm_url,
        config.llm_max_tokens,
        config.llm_temperature,
    );
    info!("LLM endpoint: {}", config.llm_url);

    // ── STT (whisper) — load in blocking thread, heavy init ──────────────────
    let whisper_model = config.whisper_model.clone();
    let whisper_language = config.language.clone();
    let stt = tokio::task::spawn_blocking(move || {
        WhisperStt::new(&whisper_model, &whisper_language)
    })
    .await??;
    let stt = Arc::new(stt);

    // ── TTS (piper) — load in blocking thread ────────────────────────────────
    let piper_config = config.piper_model_path().to_string();
    let tts = tokio::task::spawn_blocking(move || PiperTts::new(&piper_config)).await??;
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
                        // Prepend buffered pre-roll audio so the onset is not clipped
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

                        if duration_ms < MIN_SPEECH_DURATION_MS {
                            debug!("Too short ({}ms), skipping", duration_ms);
                            continue;
                        }

                        info!("Speech: {}ms — transcribing...", duration_ms);

                        // ── STT ──────────────────────────────────────────────
                        let stt_ref = Arc::clone(&stt);
                        let transcript = match tokio::task::spawn_blocking(move || {
                            stt_ref.transcribe(&audio)
                        })
                        .await
                        {
                            Ok(Ok(t)) => t,
                            Ok(Err(e)) => { error!("STT error: {}", e); continue; }
                            Err(e) => { error!("STT task panicked: {}", e); continue; }
                        };

                        if transcript.is_empty() {
                            debug!("Empty transcript, skipping");
                            continue;
                        }

                        info!("User: {}", transcript);

                        // Persist user turn
                        if let Err(e) = db.save_message(session_id, "User", &transcript).await {
                            warn!("Failed to save user message: {}", e);
                        }

                        // Update LLM session with new user turn
                        llm_session.add_user_turn(&transcript);

                        // ── LLM → TTS streaming ───────────────────────────────
                        let mut token_rx = match llm_client.stream(&llm_session).await {
                            Ok(r) => r,
                            Err(e) => { error!("LLM stream error: {}", e); continue; }
                        };

                        let mut sentence_buf = SentenceSplitter::new();
                        let mut full_response = String::new();

                        loop {
                            let token = token_rx.recv().await;
                            let is_done = token.is_none();
                            let token = token.unwrap_or_default();

                            full_response.push_str(&token);

                            let sentences_to_play: Vec<String> = if is_done {
                                // Push last token then flush
                                let mut v = Vec::new();
                                if let Some(s) = sentence_buf.push(&token) {
                                    v.push(s);
                                }
                                if let Some(s) = sentence_buf.flush() {
                                    v.push(s);
                                }
                                v
                            } else if let Some(s) = sentence_buf.push(&token) {
                                vec![s]
                            } else {
                                vec![]
                            };

                            for sentence in sentences_to_play {
                                let tts_ref = Arc::clone(&tts);
                                let audio_out = Arc::clone(&audio_output);
                                let sentence_clone = sentence.clone();

                                info!("TTS synthesizing: {:?}", sentence_clone);

                                // Synthesize sentence — piper returns f32 samples directly
                                let samples_f32 = match tokio::task::spawn_blocking(move || {
                                    tts_ref.synthesize(&sentence_clone)
                                })
                                .await
                                {
                                    Ok(Ok(s)) => s,
                                    Ok(Err(e)) => { error!("TTS error: {}", e); continue; }
                                    Err(e) => { error!("TTS task panicked: {}", e); continue; }
                                };

                                info!("TTS samples: {} ({}ms @{}Hz)", samples_f32.len(), samples_f32.len() as u32 * 1000 / tts_sample_rate, tts_sample_rate);

                                if let Err(e) = tokio::task::spawn_blocking(move || {
                                    audio_out.play_blocking(&samples_f32, tts_sample_rate)
                                })
                                .await
                                .expect("playback task panicked")
                                {
                                    error!("Playback error: {}", e);
                                }
                            }

                            if is_done { break; }
                        }

                        info!("Assistant: {}", full_response);

                        // Persist assistant turn
                        if let Err(e) = db.save_message(session_id, "Assistant", &full_response).await {
                            warn!("Failed to save assistant message: {}", e);
                        }

                        // Update LLM session with completed assistant turn
                        llm_session.add_assistant_turn(&full_response);

                        // Drain stale audio accumulated during processing and reset VAD
                        let mut drained = 0usize;
                        while rx.try_recv().is_ok() { drained += 1; }
                        if drained > 0 {
                            debug!("Drained {} stale audio chunks", drained);
                        }
                        vad.reset();
                        speech_buffer.clear();
                        pre_roll.clear();
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
        }
    }

    Ok(())
}
