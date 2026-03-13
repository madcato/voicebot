use anyhow::{anyhow, Context, Result};
use async_channel::Sender;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, StreamConfig};
use std::sync::Arc;
use tracing::{debug, error, info, warn, trace};

/// Raw audio data from the microphone
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AudioChunk {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

pub struct AudioCapture {
    device: Device,
    config: StreamConfig,
    sample_format: SampleFormat,
}

impl AudioCapture {
    /// Create a new AudioCapture instance.
    ///
    /// # Arguments
    /// * `device_name` - Optional device name to use. If None, uses the default input device.
    ///                   The name can be a partial match (case-insensitive).
    pub fn new(device_name: Option<&str>) -> Result<Self> {
        let host = cpal::default_host();

        let device = match device_name {
            Some(name) => Self::find_device_by_name(&host, name)?,
            None => host
                .default_input_device()
                .ok_or_else(|| anyhow!("No input device available"))?,
        };

        let actual_device_name = device.description()?;
        info!(target: "audio", "Using input device: {}", actual_device_name);

        let supported_config = device
            .default_input_config()
            .context("Failed to get default input config")?;

        info!(
            target: "audio",
            "Default input config: {:?} Hz, {:?} channels, format: {:?}",
            supported_config.sample_rate(),
            supported_config.channels(),
            supported_config.sample_format()
        );

        let sample_format = supported_config.sample_format();
        let config: StreamConfig = supported_config.into();

        Ok(Self {
            device,
            config,
            sample_format,
        })
    }

    /// Find an input device by name (case-insensitive partial match)
    fn find_device_by_name(host: &cpal::Host, name: &str) -> Result<Device> {
        let name_lower = name.to_lowercase();

        let devices = host
            .input_devices()
            .context("Failed to enumerate input devices")?;

        for device in devices {
            if let Ok(device_name) = device.description() {
                if device_name.name().to_lowercase().contains(&name_lower) {
                    info!(target: "audio", "Found matching device: {}", device_name);
                    return Ok(device);
                }
            }
        }

        Err(anyhow!(
            "No input device found matching '{}'. Use LIST_AUDIO_DEVICES=1 to see available devices.",
            name
        ))
    }

    /// List all available input devices
    // pub fn list_devices() -> Result<Vec<String>> {
    //     let host = cpal::default_host();
    //     let devices = host
    //         .input_devices()
    //         .context("Failed to enumerate input devices")?;

    //     let mut device_names = Vec::new();
    //     for device in devices {
    //         if let Ok(name) = device.name() {
    //             device_names.push(name);
    //         }
    //     }

    //     Ok(device_names)
    // }

    /// Print all available input devices to stdout
    pub fn print_devices() -> Result<()> {
        let host = cpal::default_host();

        // Print default device
        if let Some(default_device) = host.default_input_device() {
            println!(
                "Default input device: {}",
                default_device.description()?.name()
            );
        }

        println!("\nAvailable input devices:");

        let devices = host
            .input_devices()
            .context("Failed to enumerate input devices")?;

        for (idx, device) in devices.enumerate() {
            if let Ok(device_description) = device.description() {
                // Try to get supported config info
                let config_info = device
                    .default_input_config()
                    .map(|c| {
                        format!(
                            "{} Hz, {} ch, {:?}",
                            c.sample_rate(),
                            c.channels(),
                            c.sample_format()
                        )
                    })
                    .unwrap_or_else(|_| "config unavailable".to_string());

                println!(
                    "  [{}] {} ({})",
                    idx,
                    device_description.name(),
                    config_info
                );
            }
        }

        Ok(())
    }

    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    pub fn channels(&self) -> u16 {
        self.config.channels
    }

    /// Start capturing audio and send chunks through the provided channel
    pub fn start_capture(
        &self,
        tx: Sender<AudioChunk>,
        samples_per_chunk: usize,
    ) -> Result<cpal::Stream> {
        let channels = self.config.channels;
        let sample_rate = self.config.sample_rate;

        let err_fn = |err| error!(target: "audio", "Audio stream error: {}", err);

        // Buffer to accumulate samples
        let buffer = Arc::new(std::sync::Mutex::new(Vec::with_capacity(
            samples_per_chunk * channels as usize,
        )));
        let buffer_clone = Arc::clone(&buffer);

        let stream = match self.sample_format {
            SampleFormat::I16 => {
                let tx = tx.clone();
                self.device.build_input_stream(
                    &self.config,
                    move |data: &[i16], _: &cpal::InputCallbackInfo| {
                        // Convert i16 to f32
                        let samples: Vec<f32> =
                            data.iter().map(|&s| s as f32 / i16::MAX as f32).collect();
                        Self::process_samples(
                            &buffer_clone,
                            samples,
                            &tx,
                            samples_per_chunk,
                            sample_rate,
                            channels,
                        );
                    },
                    err_fn,
                    None,
                )?
            }
            SampleFormat::U16 => {
                let tx = tx.clone();
                self.device.build_input_stream(
                    &self.config,
                    move |data: &[u16], _: &cpal::InputCallbackInfo| {
                        // Convert u16 to f32
                        let samples: Vec<f32> = data
                            .iter()
                            .map(|&s| (s as f32 / u16::MAX as f32) * 2.0 - 1.0)
                            .collect();
                        Self::process_samples(
                            &buffer_clone,
                            samples,
                            &tx,
                            samples_per_chunk,
                            sample_rate,
                            channels,
                        );
                    },
                    err_fn,
                    None,
                )?
            }
            SampleFormat::F32 => {
                let tx = tx.clone();
                self.device.build_input_stream(
                    &self.config,
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        Self::process_samples(
                            &buffer_clone,
                            data.to_vec(),
                            &tx,
                            samples_per_chunk,
                            sample_rate,
                            channels,
                        );
                    },
                    err_fn,
                    None,
                )?
            }
            sample_format => {
                return Err(anyhow!("Unsupported sample format: {:?}", sample_format));
            }
        };

        stream.play().context("Failed to start audio stream")?;
        info!(target: "audio", "Audio capture started");

        Ok(stream)
    }

    fn process_samples(
        buffer: &Arc<std::sync::Mutex<Vec<f32>>>,
        samples: Vec<f32>,
        tx: &Sender<AudioChunk>,
        samples_per_chunk: usize,
        sample_rate: u32,
        channels: u16,
    ) {
        let mut buf = buffer.lock().unwrap();
        buf.extend(samples);

        // Total samples needed including all channels
        let total_samples_needed = samples_per_chunk * channels as usize;

        while buf.len() >= total_samples_needed {
            let chunk_samples: Vec<f32> = buf.drain(..total_samples_needed).collect();

            let chunk = AudioChunk {
                samples: chunk_samples,
                sample_rate,
                channels,
            };

            // Non-blocking send - drop chunk if channel is full
            if let Err(e) = tx.try_send(chunk) {
                warn!(target: "audio", "Failed to send audio chunk: {}", e);
            } else {
                trace!(target: "audio", "Sent audio chunk with {} samples", total_samples_needed);
            }
        }
    }
}
