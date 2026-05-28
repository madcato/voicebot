use anyhow::{Result, bail};
use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::config::Config;

use super::SpeechEvent;
use super::whisper::{WhisperSTTVADConfig, WhisperSttProvider};

#[async_trait]
pub trait SttProvider: Send {
    fn provider_name(&self) -> &'static str;
    async fn process_audio(&mut self, audio: &[f32], tx: &mpsc::Sender<SpeechEvent>) -> Result<()>;
    fn transcribe_complete(&self, audio: &[f32]) -> Result<String>;
}

pub fn create_provider(config: &Config) -> Result<Box<dyn SttProvider>> {
    let whisper_cfg = WhisperSTTVADConfig {
        whisper_model: config.whisper_model.clone(),
        vad_model: config.vad_model.clone(),
        language: config.language.clone(),
        silence_ms: config.vad_silence_ms,
        vad_start_threshold: config.vad_start_threshold,
        vad_end_threshold: config.vad_end_threshold,
    };

    match config.stt_provider.to_lowercase().as_str() {
        "whisper" => Ok(Box::new(WhisperSttProvider::new(whisper_cfg)?)),
        "parakeet" => {
            #[cfg(feature = "parakeet")]
            {
                let provider = super::parakeet::ParakeetSttProvider::new(
                    whisper_cfg,
                    config.parakeet_model_dir.as_deref(),
                )?;
                return Ok(Box::new(provider));
            }

            #[cfg(not(feature = "parakeet"))]
            {
                bail!(
                    "STT_PROVIDER=parakeet requested but the 'parakeet' feature is not enabled. Rebuild with: cargo run --features parakeet"
                );
            }
        }
        other => bail!(
            "Invalid STT_PROVIDER '{other}'. Supported values: whisper, parakeet"
        ),
    }
}
