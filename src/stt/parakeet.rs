use anyhow::{Context, Result, anyhow, ensure};
use async_trait::async_trait;
use parakeet_rs::{ParakeetTDT, Transcriber};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use whisper_cpp_plus::WhisperVadProcessor;

use super::SpeechEvent;
use super::provider::SttProvider;
use super::whisper::WhisperSTTVADConfig;

const SAMPLE_RATE: usize = 16_000;
const VAD_PROBE_MS: usize = 200;
const VAD_PROBE_SAMPLES: usize = SAMPLE_RATE * VAD_PROBE_MS / 1000;
const PRE_ROLL_MS: usize = 300;
const PRE_ROLL_SAMPLES: usize = SAMPLE_RATE * PRE_ROLL_MS / 1000;
const MAX_SEGMENT_MS: usize = 20_000;
const MAX_SEGMENT_SAMPLES: usize = SAMPLE_RATE * MAX_SEGMENT_MS / 1000;

pub struct ParakeetSttProvider {
    model: Arc<Mutex<ParakeetTDT>>,
    vad: WhisperVadProcessor,
    vad_start_threshold: f32,
    vad_end_threshold: f32,

    // State machine
    in_speech: bool,
    speech_buf: Vec<f32>,
    pre_roll: VecDeque<f32>,
    silence_samples: usize,
    silence_samples_threshold: usize,

    // Leftover samples that didn't fill a probe window.
    probe_carry: Vec<f32>,
}

impl ParakeetSttProvider {
    pub fn new(base: WhisperSTTVADConfig, parakeet_model_dir: Option<&str>) -> Result<Self> {
        ensure!(
            (0.0..=1.0).contains(&base.vad_start_threshold),
            "vad_start_threshold must be in [0.0, 1.0], got {}",
            base.vad_start_threshold
        );
        ensure!(
            (0.0..=1.0).contains(&base.vad_end_threshold),
            "vad_end_threshold must be in [0.0, 1.0], got {}",
            base.vad_end_threshold
        );

        let vad_end_threshold = if base.vad_end_threshold > base.vad_start_threshold {
            tracing::warn!(
                target: "sttvad",
                "VAD end threshold ({}) is higher than start threshold ({}). Clamping end to start.",
                base.vad_end_threshold,
                base.vad_start_threshold
            );
            base.vad_start_threshold
        } else {
            base.vad_end_threshold
        };

        let model_dir = parakeet_model_dir
            .ok_or_else(|| anyhow!("PARAKEET_MODEL_DIR must be set when STT_PROVIDER=parakeet"))?;
        let model = ParakeetTDT::from_pretrained(model_dir, None)
            .with_context(|| format!("Failed to load Parakeet TDT model from: {}\n\nThe model directory must contain ONNX Runtime files (encoder-model.onnx, decoder_joint-model.onnx, vocab.txt).\nYou may have downloaded the wrong format — parakeet-rs requires the ONNX export, not the HuggingFace Transformers model (.safetensors) or NeMo checkpoint (.nemo).\n\nSolution: download the correct ONNX model from HuggingFace:\n  mkdir -p {}\n  cd {}\n  wget https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/encoder-model.onnx\n  wget https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/decoder_joint-model.onnx\n  wget https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/vocab.txt\n  # Optional external data (if present):\n  wget https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/encoder-model.onnx.data\n", model_dir, model_dir, model_dir))?;

        let vad = WhisperVadProcessor::new(&base.vad_model).context("Failed to load VAD model")?;

        tracing::info!(
            target: "sttvad",
            "ParakeetSttProvider ready (parakeet: {}, vad: {}, lang: {}, silence_ms: {}, start_thr: {:.2}, end_thr: {:.2})",
            model_dir,
            base.vad_model,
            base.language,
            base.silence_ms,
            base.vad_start_threshold,
            vad_end_threshold
        );

        let silence_samples_threshold = SAMPLE_RATE * base.silence_ms as usize / 1000;

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            vad,
            vad_start_threshold: base.vad_start_threshold,
            vad_end_threshold,
            in_speech: false,
            speech_buf: Vec::with_capacity(MAX_SEGMENT_SAMPLES),
            pre_roll: VecDeque::with_capacity(PRE_ROLL_SAMPLES),
            silence_samples: 0,
            silence_samples_threshold,
            probe_carry: Vec::with_capacity(VAD_PROBE_SAMPLES),
        })
    }

    pub async fn process_audio(&mut self, audio: &[f32], tx: &mpsc::Sender<SpeechEvent>) -> Result<()> {
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

    async fn process_probe(&mut self, chunk: &[f32], tx: &mpsc::Sender<SpeechEvent>) -> Result<()> {
        let has_speech = self.vad.detect_speech(chunk);
        let avg_prob = if !has_speech {
            0.0
        } else {
            let probs = self.vad.get_probs();
            if probs.is_empty() {
                0.0
            } else {
                probs.iter().sum::<f32>() / probs.len() as f32
            }
        };
        let threshold = if self.in_speech {
            self.vad_end_threshold
        } else {
            self.vad_start_threshold
        };
        let silence = avg_prob < threshold;

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

                let model = Arc::clone(&self.model);
                let text = tokio::task::spawn_blocking(move || -> Result<String> {
                    transcribe(&model, &audio)
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

    pub fn transcribe_complete(&self, audio: &[f32]) -> Result<String> {
        transcribe(&self.model, audio)
    }
}

#[async_trait]
impl SttProvider for ParakeetSttProvider {
    fn provider_name(&self) -> &'static str {
        "parakeet"
    }

    async fn process_audio(&mut self, audio: &[f32], tx: &mpsc::Sender<SpeechEvent>) -> Result<()> {
        ParakeetSttProvider::process_audio(self, audio, tx).await
    }

    fn transcribe_complete(&self, audio: &[f32]) -> Result<String> {
        ParakeetSttProvider::transcribe_complete(self, audio)
    }
}

fn transcribe(model: &Arc<Mutex<ParakeetTDT>>, audio: &[f32]) -> Result<String> {
    if audio.is_empty() {
        return Ok(String::new());
    }

    let mut model = model
        .lock()
        .map_err(|_| anyhow!("Parakeet model lock poisoned"))?;

    let result = model
        .transcribe_samples(audio.to_vec(), SAMPLE_RATE as u32, 1, None)
        .context("Parakeet transcription failed")?;

    Ok(result.text.trim().to_string())
}
