use anyhow::{Context, Result};
use std::env;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Config {
    /// Target sample rate for output audio
    pub sample_rate: u32,
    /// Number of audio channels (1 = mono, 2 = stereo)
    pub channels: u16,
    /// Bit depth for audio samples
    pub bit_depth: u16,
    /// Audio format identifier
    pub audio_format: String,
    /// Audio chunk duration in milliseconds
    pub chunk_ms: u32,
    /// Audio input device name (None = use default device)
    pub audio_device: Option<String>,
    /// List available audio devices and exit
    pub list_devices: bool,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            sample_rate: env::var("AUDIO_SAMPLE_RATE")
                .unwrap_or_else(|_| "16000".to_string())
                .parse()
                .context("Invalid AUDIO_SAMPLE_RATE")?,
            channels: env::var("AUDIO_CHANNELS")
                .unwrap_or_else(|_| "1".to_string())
                .parse()
                .context("Invalid AUDIO_CHANNELS")?,
            bit_depth: env::var("AUDIO_BIT_DEPTH")
                .unwrap_or_else(|_| "16".to_string())
                .parse()
                .context("Invalid AUDIO_BIT_DEPTH")?,
            audio_format: env::var("AUDIO_FORMAT").unwrap_or_else(|_| "pcm_s16le".to_string()),
            chunk_ms: env::var("AUDIO_CHUNK_MS")
                .unwrap_or_else(|_| "100".to_string())
                .parse()
                .context("Invalid AUDIO_CHUNK_MS")?,
            audio_device: env::var("AUDIO_DEVICE").ok(),
            list_devices: env::var("LIST_AUDIO_DEVICES")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(false),
        })
    }

    /// Calculate the number of samples per chunk based on sample rate and chunk duration
    pub fn samples_per_chunk(&self) -> usize {
        (self.sample_rate as usize * self.chunk_ms as usize) / 1000
    }
}
