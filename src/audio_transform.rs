use anyhow::{Context, Result};
use rubato::{FftFixedIn, Resampler};
use tracing::{debug, info};

use crate::audio_capture::AudioChunk;
use crate::config::Config;

/// Transformed audio ready to be sent to services
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TransformedAudio {
    /// PCM audio data as bytes (format depends on config)
    pub data: Vec<u8>,
    pub sample_rate: u32,
    pub channels: u16,
    pub bit_depth: u16,
}

#[allow(dead_code)]
pub struct AudioTransformer {
    target_sample_rate: u32,
    target_channels: u16,
    target_bit_depth: u16,
    resampler: Option<FftFixedIn<f32>>,
    source_sample_rate: u32,
    source_channels: u16,
    chunk_size: usize,
}

impl AudioTransformer {
    pub fn new(config: &Config, source_sample_rate: u32, source_channels: u16) -> Result<Self> {
        let chunk_size = config.samples_per_chunk();

        let resampler = if source_sample_rate != config.sample_rate {
            info!(
                "Creating resampler: {} Hz -> {} Hz",
                source_sample_rate, config.sample_rate
            );
            Some(
                FftFixedIn::<f32>::new(
                    source_sample_rate as usize,
                    config.sample_rate as usize,
                    chunk_size,
                    2, // Sub-chunks for better quality
                    config.channels as usize,
                )
                .context("Failed to create resampler")?,
            )
        } else {
            None
        };

        Ok(Self {
            target_sample_rate: config.sample_rate,
            target_channels: config.channels,
            target_bit_depth: config.bit_depth,
            resampler,
            source_sample_rate,
            source_channels,
            chunk_size,
        })
    }

    /// Transform audio chunk to target format
    pub fn transform(&mut self, chunk: AudioChunk) -> Result<TransformedAudio> {
        // Step 1: Convert to mono if needed
        let mono_samples = if chunk.channels > 1 && self.target_channels == 1 {
            self.to_mono(&chunk.samples, chunk.channels)
        } else if chunk.channels == 1 && self.target_channels > 1 {
            // Duplicate mono to stereo
            self.to_stereo(&chunk.samples)
        } else {
            chunk.samples
        };

        // Step 2: Resample if needed
        let resampled = if let Some(ref mut resampler) = self.resampler {
            Self::resample(resampler, &mono_samples, self.target_channels)?
        } else {
            mono_samples
        };

        // Step 3: Convert to target bit depth and format
        let data = self.to_pcm_bytes(&resampled);

        debug!(
            "Transformed {} samples to {} bytes",
            resampled.len(),
            data.len()
        );

        Ok(TransformedAudio {
            data,
            sample_rate: self.target_sample_rate,
            channels: self.target_channels,
            bit_depth: self.target_bit_depth,
        })
    }

    /// Convert multi-channel audio to mono by averaging channels
    fn to_mono(&self, samples: &[f32], channels: u16) -> Vec<f32> {
        samples
            .chunks(channels as usize)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    }

    /// Convert mono to stereo by duplicating samples
    fn to_stereo(&self, samples: &[f32]) -> Vec<f32> {
        samples.iter().flat_map(|&s| [s, s]).collect()
    }

    /// Resample audio to target sample rate
    fn resample(
        resampler: &mut FftFixedIn<f32>,
        samples: &[f32],
        target_channels: u16,
    ) -> Result<Vec<f32>> {
        // rubato expects Vec<Vec<f32>> where outer vec is channels
        let channels_data = if target_channels == 1 {
            vec![samples.to_vec()]
        } else {
            // Split interleaved samples into channels
            let mut channels: Vec<Vec<f32>> = vec![Vec::new(); target_channels as usize];
            for (i, &sample) in samples.iter().enumerate() {
                channels[i % target_channels as usize].push(sample);
            }
            channels
        };

        // Pad or truncate to expected chunk size
        let chunk_size = resampler.input_frames_max();
        let channels_data: Vec<Vec<f32>> = channels_data
            .into_iter()
            .map(|mut ch| {
                ch.resize(chunk_size, 0.0);
                ch
            })
            .collect();

        let resampled = resampler
            .process(&channels_data, None)
            .context("Resampling failed")?;

        // Interleave channels back
        if resampled.len() == 1 {
            Ok(resampled.into_iter().next().unwrap())
        } else {
            let len = resampled[0].len();
            let mut interleaved = Vec::with_capacity(len * resampled.len());
            for i in 0..len {
                for ch in &resampled {
                    interleaved.push(ch[i]);
                }
            }
            Ok(interleaved)
        }
    }

    /// Convert f32 samples to PCM bytes based on target bit depth
    fn to_pcm_bytes(&self, samples: &[f32]) -> Vec<u8> {
        match self.target_bit_depth {
            16 => {
                // Convert to 16-bit signed little-endian PCM
                samples
                    .iter()
                    .flat_map(|&s| {
                        let clamped = s.clamp(-1.0, 1.0);
                        let scaled = (clamped * i16::MAX as f32) as i16;
                        scaled.to_le_bytes()
                    })
                    .collect()
            }
            24 => {
                // Convert to 24-bit signed little-endian PCM
                samples
                    .iter()
                    .flat_map(|&s| {
                        let clamped = s.clamp(-1.0, 1.0);
                        let scaled = (clamped * 8388607.0) as i32; // 2^23 - 1
                        let bytes = scaled.to_le_bytes();
                        [bytes[0], bytes[1], bytes[2]] // Take only 3 bytes
                    })
                    .collect()
            }
            32 => {
                // Convert to 32-bit float little-endian
                samples.iter().flat_map(|&s| s.to_le_bytes()).collect()
            }
            _ => {
                // Default to 16-bit
                samples
                    .iter()
                    .flat_map(|&s| {
                        let clamped = s.clamp(-1.0, 1.0);
                        let scaled = (clamped * i16::MAX as f32) as i16;
                        scaled.to_le_bytes()
                    })
                    .collect()
            }
        }
    }
}
