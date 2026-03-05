use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, StreamConfig};
use rubato::{FftFixedIn, Resampler};
use std::sync::{Arc, Condvar, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::info;

pub struct AudioOutput {
    device: Device,
    config: StreamConfig,
}

impl AudioOutput {
    pub fn new() -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .context("No output device available")?;

        let supported = device
            .default_output_config()
            .context("Failed to get default output config")?;

        info!(
            "Output device: {} Hz, {} ch",
            supported.sample_rate(),
            supported.channels()
        );

        let config = StreamConfig {
            channels: supported.channels(),
            sample_rate: supported.sample_rate(),
            buffer_size: cpal::BufferSize::Default,
        };

        Ok(Self { device, config })
    }

    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    pub fn channels(&self) -> u16 {
        self.config.channels
    }

    /// Resample mono `samples` from `source_rate` to `target_rate`, then
    /// duplicate to `channels`. Returns interleaved f32 PCM ready for CPAL.
    ///
    /// Exposed as `pub` so tests can exercise it without audio hardware.
    pub fn prepare(
        samples: &[f32],
        source_rate: u32,
        target_rate: u32,
        channels: u16,
    ) -> Result<Vec<f32>> {
        // Step 1 — resample
        let resampled = if source_rate != target_rate {
            resample(samples, source_rate, target_rate)
                .context("Failed to resample audio for playback")?
        } else {
            samples.to_vec()
        };

        // Step 2 — expand mono to device channel count
        if channels <= 1 {
            return Ok(resampled);
        }
        let ch = channels as usize;
        let mut interleaved = Vec::with_capacity(resampled.len() * ch);
        for s in resampled {
            for _ in 0..ch {
                interleaved.push(s);
            }
        }
        Ok(interleaved)
    }

    /// Play mono f32 samples (at `source_rate`) through the default output
    /// device. Resamples and expands channels as needed. Blocks the calling
    /// thread until every sample has been consumed by CPAL.
    pub fn play_blocking(&self, samples: &[f32], source_rate: u32) -> Result<()> {
        let prepared =
            Self::prepare(samples, source_rate, self.sample_rate(), self.channels())?;

        if prepared.is_empty() {
            return Ok(());
        }

        let total = prepared.len();
        let buf = Arc::new(prepared);
        let pos = Arc::new(AtomicUsize::new(0));
        let done = Arc::new((Mutex::new(false), Condvar::new()));

        let buf_cb = Arc::clone(&buf);
        let pos_cb = Arc::clone(&pos);
        let done_cb = Arc::clone(&done);

        let stream = self
            .device
            .build_output_stream(
                &self.config,
                move |data: &mut [f32], _| {
                    let p = pos_cb.load(Ordering::Relaxed);
                    let n = data.len().min(total.saturating_sub(p));
                    data[..n].copy_from_slice(&buf_cb[p..p + n]);
                    data[n..].fill(0.0);
                    let new_pos = p + n;
                    pos_cb.store(new_pos, Ordering::Relaxed);

                    if new_pos >= total {
                        let (lock, cvar) = &*done_cb;
                        *lock.lock().unwrap() = true;
                        cvar.notify_one();
                    }
                },
                |err| eprintln!("Audio output error: {err}"),
                None,
            )
            .context("Failed to build output stream")?;

        stream.play().context("Failed to start output stream")?;

        // Block until the stream callback signals completion.
        let (lock, cvar) = &*done;
        let mut finished = lock.lock().unwrap();
        while !*finished {
            finished = cvar.wait(finished).unwrap();
        }

        Ok(())
    }
}

// ── Resampling ─────────────────────────────────────────────────────────────────

const RESAMPLE_CHUNK: usize = 1024;

fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    let expected_out =
        (samples.len() as f64 * to_rate as f64 / from_rate as f64).ceil() as usize;

    let mut resampler = FftFixedIn::<f32>::new(
        from_rate as usize,
        to_rate as usize,
        RESAMPLE_CHUNK,
        2,
        1,
    )
    .context("Failed to create resampler")?;

    // Pad to a multiple of RESAMPLE_CHUNK so every chunk is full.
    let padded_len = samples.len().div_ceil(RESAMPLE_CHUNK) * RESAMPLE_CHUNK;
    let mut padded = samples.to_vec();
    padded.resize(padded_len, 0.0);

    let mut output = Vec::with_capacity(expected_out + RESAMPLE_CHUNK);
    for chunk in padded.chunks(RESAMPLE_CHUNK) {
        let out = resampler
            .process(&[chunk.to_vec()], None)
            .context("Resampling chunk failed")?;
        output.extend_from_slice(&out[0]);
    }

    output.truncate(expected_out);
    Ok(output)
}
