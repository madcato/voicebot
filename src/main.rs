mod audio;
mod config;
mod s2s;

use anyhow::Result;
use async_channel::bounded;
use std::sync::Arc;
use tracing::{error, info, debug};
use tracing_subscriber::EnvFilter;

use crate::audio::audio_capture::{AudioCapture, AudioChunk};
use crate::audio::buffer::AudioBuffer;
use crate::audio::output::AudioOutput;
use crate::audio::vad::{VadResult, VoiceActivityDetector};
use crate::s2s::adapter::{S2SAdapter, S2SRequest};
use crate::s2s::models::{ModelConfig, ModelType};
use config::Config;

const AUDIO_CHANNEL_CAPACITY: usize = 100;
const MAX_SPEECH_BUFFER_SECS: u32 = 30;
/// Minimum captured speech duration to send to the model (avoids reacting to short noise bursts).
const MIN_SPEECH_DURATION_MS: u32 = 800;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    info!("Starting voicebot...");

    let config = Config::from_env()?;

    let list_devices = config.list_devices
        || std::env::args().any(|a| a == "--list-devices" || a == "list-devices");

    if list_devices {
        AudioCapture::print_devices()?;
        return Ok(());
    }

    info!("Configuration loaded: {:?}", config);

    // Initialize audio capture
    let audio_capture = AudioCapture::new(config.audio_device.as_deref())?;
    let source_sample_rate = audio_capture.sample_rate();
    let source_channels = audio_capture.channels();

    info!(
        "Audio source: {} Hz, {} channels",
        source_sample_rate, source_channels
    );

    // Initialize S2S adapter
    let model_config = ModelConfig::default();
    let mut s2s = S2SAdapter::new(ModelType::LFM, model_config).await?;
    info!("S2S model initialized: {}", s2s.model_info().model_type.as_str());

    // Initialize audio output
    let audio_output = Arc::new(AudioOutput::new()?);
    info!(
        "Audio output: {} Hz, {} ch",
        audio_output.sample_rate(),
        audio_output.channels()
    );

    // Audio pipeline: capture -> channel -> VAD -> buffer -> S2S
    let samples_per_chunk = config.samples_per_chunk();
    let (tx, rx) = bounded(AUDIO_CHANNEL_CAPACITY);
    let _stream = audio_capture.start_capture(tx, samples_per_chunk)?;

    let mut vad = VoiceActivityDetector::new(source_sample_rate)?;
    let mut speech_buffer = AudioBuffer::new(source_sample_rate, MAX_SPEECH_BUFFER_SECS);

    info!("Audio pipeline running. Speak to interact...");

    tokio::select! {
        _ = async {
            loop {
                let chunk: AudioChunk = match rx.recv().await {
                    Ok(c) => c,
                    Err(e) => {
                        error!("Audio channel closed: {}", e);
                        break;
                    }
                };

                // Downmix to mono if the device returns multi-channel audio
                let mono: Vec<f32> = if chunk.channels > 1 {
                    chunk.samples
                        .chunks(chunk.channels as usize)
                        .map(|frame| frame.iter().sum::<f32>() / chunk.channels as f32)
                        .collect()
                } else {
                    chunk.samples
                };

                match vad.process(&mono) {
                    VadResult::SpeechStart | VadResult::Speech => {
                        speech_buffer.push(&mono);
                    }
                    VadResult::SpeechEnd => {
                        speech_buffer.push(&mono);
                        let audio = speech_buffer.get_samples();
                        let duration_ms = speech_buffer.duration_ms();
                        speech_buffer.clear();

                        if duration_ms < MIN_SPEECH_DURATION_MS {
                            debug!("Speech too short ({}ms), skipping", duration_ms);
                            continue;
                        }

                        info!("Speech captured: {}ms — sending to S2S", duration_ms);

                        let request = S2SRequest {
                            audio,
                            sample_rate: source_sample_rate,
                            context: vec![],
                            tools: None,
                            stream: false,
                        };

                        match s2s.process(request).await {
                            Ok(response) => {
                                if let Some(text) = &response.output_text {
                                    info!("S2S response: {}", text);
                                }
                                info!(
                                    "Playing {} samples @ {} Hz",
                                    response.audio.len(),
                                    response.sample_rate,
                                );
                                let out = Arc::clone(&audio_output);
                                let audio = response.audio;
                                let sr = response.sample_rate;
                                if let Err(e) = tokio::task::spawn_blocking(move || {
                                    out.play_blocking(&audio, sr)
                                })
                                .await
                                .expect("playback task panicked")
                                {
                                    error!("Audio playback error: {}", e);
                                }
                            }
                            Err(e) => error!("S2S processing error: {}", e),
                        }

                        // Drain audio accumulated during S2S + playback and reset
                        // VAD state so we don't react to speech captured while the
                        // bot was processing or speaking.
                        let drained = {
                            let mut n = 0usize;
                            while rx.try_recv().is_ok() { n += 1; }
                            n
                        };
                        if drained > 0 {
                            debug!("Drained {} stale audio chunks after response", drained);
                        }
                        vad.reset();
                        speech_buffer.clear();
                    }
                    VadResult::Silence => {
                        debug!("SILENCE")
                    }
                }
            }
        } => {}
        _ = tokio::signal::ctrl_c() => {
            info!("Received shutdown signal");
        }
    }

    info!("Shutting down...");
    Ok(())
}
