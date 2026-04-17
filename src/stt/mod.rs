//! Streaming STT + VAD on top of whisper-cpp-plus.
//!
//! The input is a continuous stream of audio chunks of arbitrary size (CPAL
//! delivers ~100 ms per chunk). This module accumulates them, runs the Silero
//! VAD on fixed-size probe windows, and fires transcription inside a speech→
//! silence state machine. Logic mirrors `WhisperStreamPcm::process_step_vad`
//! in whisper-cpp-plus, re-implemented here so it can be driven from an async
//! tokio channel instead of a blocking `Read` source.

use anyhow::{Context, Result};
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::mpsc;
use whisper_cpp_plus::{
    FullParams, SamplingStrategy, WhisperContext, WhisperVadProcessor,
};

/// Events emitted while processing the audio stream.
#[derive(Debug, Clone)]
pub enum SpeechEvent {
    SpeechStart,
    #[allow(dead_code)]
    Speech(String),
    SpeechEnd(String),
    #[allow(dead_code)]
    Silence,
}

#[derive(Clone)]
pub struct WhisperSTTVADConfig {
    pub whisper_model: String,
    pub vad_model: String,
    pub language: String,
    /// Milliseconds of continuous silence required to close a speech segment.
    pub silence_ms: u32,
}

impl Default for WhisperSTTVADConfig {
    fn default() -> Self {
        Self {
            whisper_model: "models/ggml-large-v3-turbo.bin".to_string(),
            vad_model: "models/ggml-silero-vad.bin".to_string(),
            language: "es".to_string(),
            silence_ms: 500,
        }
    }
}

const SAMPLE_RATE: usize = 16_000;
/// Probe size for the VAD. Silero prefers 20–200 ms windows; 200 ms gives good
/// accuracy without adding too much latency.
const VAD_PROBE_MS: usize = 200;
const VAD_PROBE_SAMPLES: usize = SAMPLE_RATE * VAD_PROBE_MS / 1000;
/// Audio retained before the VAD onset so the first phoneme isn't clipped.
const PRE_ROLL_MS: usize = 300;
const PRE_ROLL_SAMPLES: usize = SAMPLE_RATE * PRE_ROLL_MS / 1000;
/// Hard cap on a single speech segment before forcing a cut.
const MAX_SEGMENT_MS: usize = 20_000;
const MAX_SEGMENT_SAMPLES: usize = SAMPLE_RATE * MAX_SEGMENT_MS / 1000;
/// VAD probability threshold: average Silero speech prob above this = speech.
const VAD_THRESHOLD: f32 = 0.5;

pub struct WhisperSTTVAD {
    ctx: Arc<WhisperContext>,
    vad: WhisperVadProcessor,
    language: String,

    // State machine
    in_speech: bool,
    speech_buf: Vec<f32>,
    pre_roll: VecDeque<f32>,
    silence_samples: usize,
    silence_samples_threshold: usize,

    // Leftover samples that didn't fill a probe window.
    probe_carry: Vec<f32>,
}

impl WhisperSTTVAD {
    pub fn new(config: WhisperSTTVADConfig) -> Result<Self> {
        let ctx = Arc::new(
            WhisperContext::new(&config.whisper_model).context("Failed to load Whisper model")?,
        );

        let vad = WhisperVadProcessor::new(&config.vad_model)
            .context("Failed to load VAD model")?;

        tracing::info!(
            target: "sttvad",
            "WhisperSTTVAD ready (whisper: {}, vad: {}, lang: {}, silence_ms: {})",
            config.whisper_model, config.vad_model, config.language, config.silence_ms
        );

        let silence_samples_threshold = SAMPLE_RATE * config.silence_ms as usize / 1000;

        Ok(Self {
            ctx,
            vad,
            language: config.language,
            in_speech: false,
            speech_buf: Vec::with_capacity(MAX_SEGMENT_SAMPLES),
            pre_roll: VecDeque::with_capacity(PRE_ROLL_SAMPLES),
            silence_samples: 0,
            silence_samples_threshold,
            probe_carry: Vec::with_capacity(VAD_PROBE_SAMPLES),
        })
    }

    /// Feed a chunk of 16 kHz mono f32 audio. Emits events as the VAD/state
    /// machine advances. Transcription happens synchronously on the caller
    /// thread (blocking); it's acceptable for a single-user interactive loop.
    pub async fn process_audio(
        &mut self,
        audio: &[f32],
        tx: &mpsc::Sender<SpeechEvent>,
    ) -> Result<()> {
        if audio.is_empty() {
            return Ok(());
        }

        self.probe_carry.extend_from_slice(audio);

        while self.probe_carry.len() >= VAD_PROBE_SAMPLES {
            let chunk: Vec<f32> = self.probe_carry.drain(..VAD_PROBE_SAMPLES).collect();
            self.process_probe(&chunk, tx).await?;
        }

        Ok(())
    }

    async fn process_probe(
        &mut self,
        chunk: &[f32],
        tx: &mpsc::Sender<SpeechEvent>,
    ) -> Result<()> {
        let has_speech = self.vad.detect_speech(chunk);
        let silence = if !has_speech {
            true
        } else {
            let probs = self.vad.get_probs();
            if probs.is_empty() {
                true
            } else {
                let avg = probs.iter().sum::<f32>() / probs.len() as f32;
                avg < VAD_THRESHOLD
            }
        };

        if !self.in_speech {
            if !silence {
                self.in_speech = true;
                self.silence_samples = 0;
                self.speech_buf.clear();
                self.speech_buf.extend(self.pre_roll.iter().copied());
                self.speech_buf.extend_from_slice(chunk);
                let _ = tx.send(SpeechEvent::SpeechStart).await;
                tracing::debug!(target: "sttvad", "SpeechStart");
            }
        } else {
            self.speech_buf.extend_from_slice(chunk);
            if silence {
                self.silence_samples += chunk.len();
            } else {
                self.silence_samples = 0;
            }

            let should_finalize = self.silence_samples >= self.silence_samples_threshold
                || self.speech_buf.len() >= MAX_SEGMENT_SAMPLES;

            if should_finalize {
                let audio = std::mem::take(&mut self.speech_buf);
                self.in_speech = false;
                self.silence_samples = 0;

                tracing::debug!(
                    target: "sttvad",
                    "Finalizing segment: {:.2}s",
                    audio.len() as f32 / SAMPLE_RATE as f32
                );

                let ctx = Arc::clone(&self.ctx);
                let language = self.language.clone();
                let text = tokio::task::spawn_blocking(move || -> Result<String> {
                    transcribe(&ctx, &language, &audio)
                })
                .await
                .context("transcription task join")??;

                tracing::info!(target: "sttvad", "SpeechEnd: {}", text);
                let _ = tx.send(SpeechEvent::SpeechEnd(text)).await;
            }
        }

        for &s in chunk {
            if self.pre_roll.len() >= PRE_ROLL_SAMPLES {
                self.pre_roll.pop_front();
            }
            self.pre_roll.push_back(s);
        }

        Ok(())
    }

    /// Blocking one-shot transcription (used as a fallback / sanity check).
    #[allow(dead_code)]
    pub fn transcribe_complete(&self, audio: &[f32]) -> Result<String> {
        transcribe(&self.ctx, &self.language, audio)
    }
}

fn transcribe(ctx: &WhisperContext, language: &str, audio: &[f32]) -> Result<String> {
    if audio.is_empty() {
        return Ok(String::new());
    }

    let mut state = ctx.create_state()?;
    let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 })
        .language(language)
        .print_special(false)
        .print_progress(false)
        .print_realtime(false)
        .print_timestamps(false)
        .no_timestamps(true)
        .single_segment(true);

    state.full(params, audio)?;

    let n = state.full_n_segments();
    let mut text = String::new();
    for i in 0..n {
        if let Ok(seg) = state.full_get_segment_text(i) {
            text.push_str(seg.trim());
            text.push(' ');
        }
    }
    Ok(text.trim().to_string())
}
