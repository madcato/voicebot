use anyhow::{Context, Result};
use async_trait::async_trait;
use base64::Engine as _;
use std::env;

use crate::s2s::adapter::{S2SModel, S2SRequest, S2SResponse};
use crate::s2s::models::ModelConfig;

/// Default API endpoint — local provider server (see provider/server.py)
const DEFAULT_API_URL: &str = "http://127.0.0.1:8000/v1/chat/completions";
const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful voice assistant. \
    Listen carefully to what the user says and respond naturally in spoken language. \
    Keep your answers concise and conversational. \
    Do not use markdown, bullet points, or any formatting — speak as you would in a real conversation.";

/// LFM2.5-Audio model — calls the Liquid AI API (OpenAI-compatible audio chat completions)
pub struct LFMModel {
    config: ModelConfig,
    client: reqwest::Client,
    api_url: String,
    api_key: String,
    system_prompt: String,
}

impl LFMModel {
    pub async fn new(config: &ModelConfig) -> Result<Self> {
        let api_url = env::var("LFM_API_URL")
            .unwrap_or_else(|_| DEFAULT_API_URL.to_string());
        // LFM_API_KEY is optional when running the local provider server.
        // Set it to a non-empty value when using a hosted endpoint that requires auth.
        let api_key = env::var("LFM_API_KEY").unwrap_or_default();
        let system_prompt = env::var("LFM_SYSTEM_PROMPT")
            .unwrap_or_else(|_| DEFAULT_SYSTEM_PROMPT.to_string());

        tracing::info!("Initializing LFM2.5-Audio model (endpoint: {})", api_url);

        Ok(Self {
            config: config.clone(),
            client: reqwest::Client::new(),
            api_url,
            api_key,
            system_prompt,
        })
    }

    /// Encode mono f32 samples as a 16-bit PCM WAV file (no extra dependencies).
    fn encode_wav(samples: &[f32], sample_rate: u32) -> Vec<u8> {
        let num_channels: u16 = 1;
        let bits_per_sample: u16 = 16;
        let byte_rate = sample_rate * num_channels as u32 * bits_per_sample as u32 / 8;
        let block_align = num_channels * bits_per_sample / 8;
        let data_size = (samples.len() * block_align as usize) as u32;

        let mut wav = Vec::with_capacity(44 + data_size as usize);

        // RIFF header
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_size).to_le_bytes());
        wav.extend_from_slice(b"WAVE");

        // fmt chunk
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&num_channels.to_le_bytes());
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&byte_rate.to_le_bytes());
        wav.extend_from_slice(&block_align.to_le_bytes());
        wav.extend_from_slice(&bits_per_sample.to_le_bytes());

        // data chunk
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_size.to_le_bytes());

        for &s in samples {
            let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
            wav.extend_from_slice(&v.to_le_bytes());
        }

        wav
    }

    /// Decode a WAV file to mono f32 samples. Mixes down multi-channel to mono.
    fn decode_wav(data: &[u8]) -> Result<(Vec<f32>, u32)> {
        if data.len() < 44 {
            anyhow::bail!("WAV data too short ({} bytes)", data.len());
        }
        if &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
            anyhow::bail!("Response is not a valid WAV file");
        }

        let mut offset = 12usize;
        let mut sample_rate = 0u32;
        let mut bits_per_sample = 0u16;
        let mut num_channels = 0u16;
        let mut data_start = 0usize;
        let mut data_size = 0u32;

        while offset + 8 <= data.len() {
            let chunk_id = &data[offset..offset + 4];
            let chunk_size =
                u32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap());

            if chunk_id == b"fmt " {
                num_channels =
                    u16::from_le_bytes(data[offset + 10..offset + 12].try_into().unwrap());
                sample_rate =
                    u32::from_le_bytes(data[offset + 12..offset + 16].try_into().unwrap());
                bits_per_sample =
                    u16::from_le_bytes(data[offset + 22..offset + 24].try_into().unwrap());
            } else if chunk_id == b"data" {
                data_start = offset + 8;
                data_size = chunk_size;
                break;
            }

            offset += 8 + chunk_size as usize;
        }

        if data_start == 0 {
            anyhow::bail!("No data chunk found in WAV response");
        }

        let pcm = &data[data_start..data_start + data_size as usize];
        let bytes_per_sample = bits_per_sample / 8;
        let frame_size = num_channels as usize * bytes_per_sample as usize;

        let samples: Vec<f32> = pcm
            .chunks_exact(frame_size)
            .map(|frame| {
                // Take only the first channel (mono mix-down)
                match bits_per_sample {
                    16 => {
                        let v = i16::from_le_bytes(frame[0..2].try_into().unwrap());
                        v as f32 / 32767.0
                    }
                    32 => f32::from_le_bytes(frame[0..4].try_into().unwrap()),
                    _ => 0.0,
                }
            })
            .collect();

        Ok((samples, sample_rate))
    }

    async fn call_api(
        &self,
        audio: &[f32],
        sample_rate: u32,
        context: &[String],
    ) -> Result<(Vec<f32>, u32, Option<String>, Option<String>)> {
        // Encode input audio
        let wav_bytes = Self::encode_wav(audio, sample_rate);
        let audio_b64 = base64::engine::general_purpose::STANDARD.encode(&wav_bytes);

        // Build message list
        let mut messages = vec![serde_json::json!({
            "role": "system",
            "content": self.system_prompt
        })];

        // Interleave context as user/assistant turns
        for (i, msg) in context.iter().enumerate() {
            let role = if i % 2 == 0 { "user" } else { "assistant" };
            messages.push(serde_json::json!({ "role": role, "content": msg }));
        }

        // Append the current audio turn
        messages.push(serde_json::json!({
            "role": "user",
            "content": [{
                "type": "input_audio",
                "input_audio": { "data": audio_b64, "format": "wav" }
            }]
        }));

        let payload = serde_json::json!({
            "model": "lfm-2.5-audio",
            "modalities": ["text", "audio"],
            "audio": { "voice": "alloy", "format": "wav" },
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": 1024,
        });

        tracing::debug!(
            "LFM API request: {} samples @ {} Hz → {}",
            audio.len(),
            sample_rate,
            self.api_url
        );

        let mut req = self.client.post(&self.api_url).json(&payload);
        if !self.api_key.is_empty() {
            req = req.header("Authorization", format!("Bearer {}", self.api_key));
        }
        let response = req
            .send()
            .await
            .context("Failed to reach LFM API")?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("LFM API error {}: {}", status, body);
        }

        let resp: serde_json::Value = response
            .json()
            .await
            .context("Failed to parse LFM API response")?;

        let message = &resp["choices"][0]["message"];

        // Text transcript of output
        let output_text = message["content"].as_str().map(|s| s.to_string());

        // Input transcription (what the user said)
        let input_text = message["audio"]["transcript"]
            .as_str()
            .map(|s| s.to_string());

        // Decode response audio
        let audio_b64_out = message["audio"]["data"]
            .as_str()
            .context("LFM API response missing audio.data")?;

        let audio_bytes = base64::engine::general_purpose::STANDARD
            .decode(audio_b64_out)
            .context("Failed to decode base64 audio from LFM response")?;

        let (output_audio, out_sr) =
            Self::decode_wav(&audio_bytes).context("Failed to decode WAV from LFM response")?;

        tracing::debug!(
            "LFM API response: {} samples @ {} Hz",
            output_audio.len(),
            out_sr
        );

        Ok((output_audio, out_sr, input_text, output_text))
    }
}

#[async_trait]
impl S2SModel for LFMModel {
    async fn process(&mut self, request: S2SRequest) -> Result<S2SResponse> {
        let (audio, out_sr, input_text, output_text) = self
            .call_api(&request.audio, request.sample_rate, &request.context)
            .await
            .context("LFM2.5-Audio inference failed")?;

        Ok(S2SResponse {
            audio,
            sample_rate: out_sr,
            input_text,
            output_text,
            tool_calls: None,
        })
    }

    fn supports_streaming(&self) -> bool {
        false
    }

    fn supports_tools(&self) -> bool {
        self.config.enable_tools
    }

    fn name(&self) -> &str {
        "LFM2.5-Audio"
    }
}
