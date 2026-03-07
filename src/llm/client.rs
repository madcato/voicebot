use anyhow::{Context, Result};
use futures_util::StreamExt;
use tokio::sync::mpsc;

use super::LlmSession;

#[derive(Clone)]
pub struct LlamaClient {
    client: reqwest::Client,
    completion_url: String,
    max_tokens: u32,
    temperature: f32,
}

impl LlamaClient {
    pub fn new(base_url: &str, max_tokens: u32, temperature: f32) -> Self {
        Self {
            client: reqwest::Client::new(),
            completion_url: format!("{}/completion", base_url.trim_end_matches('/')),
            // completion_url: format!("{}/v1/chat/completions", base_url.trim_end_matches('/')), // for mlx-lm
            max_tokens,
            temperature,
        }
    }

    /// Stream completion tokens from llama.cpp.
    ///
    /// Returns a channel receiver that yields text tokens as they are generated.
    /// The channel closes when generation is complete or an error occurs.
    /// Uses `cache_prompt: true` so the KV-cache for previous turns is reused.
    pub async fn stream(&self, session: &LlmSession) -> Result<mpsc::Receiver<String>> {
        let payload = serde_json::json!({
            "prompt": session.prompt(),
            "n_predict": self.max_tokens,
            "cache_prompt": true,
            "slot_id": session.slot_id,
            "temperature": self.temperature,
            "top_p": 0.95,
            "stream": true,
            "stop": ["<|im_end|>", "<|im_start|>"],
        });

        let response = self
            .client
            .post(&self.completion_url)
            .json(&payload)
            .send()
            .await
            .context("Failed to reach llama.cpp server")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("llama.cpp error {}: {}", status, body);
        }

        let (tx, rx) = mpsc::channel::<String>(256);

        tokio::spawn(async move {
            let mut stream = response.bytes_stream();
            let mut buf = String::new();

            while let Some(chunk) = stream.next().await {
                let bytes = match chunk {
                    Ok(b) => b,
                    Err(e) => {
                        tracing::error!("LLM stream error: {}", e);
                        break;
                    }
                };

                buf.push_str(&String::from_utf8_lossy(&bytes));

                // Process all complete SSE lines in buffer
                loop {
                    let Some(newline) = buf.find('\n') else { break };
                    let line = buf[..newline].trim().to_string();
                    buf = buf[newline + 1..].to_string();

                    let Some(data) = line.strip_prefix("data: ") else { continue };

                    let Ok(json) = serde_json::from_str::<serde_json::Value>(data) else {
                        continue;
                    };

                    if json["stop"].as_bool().unwrap_or(false) {
                        return;
                    }

                    if let Some(content) = json["content"].as_str() {
                        if !content.is_empty() && tx.send(content.to_string()).await.is_err() {
                            return; // receiver dropped
                        }
                    }
                }
            }
        });

        Ok(rx)
    }

    /// Check if the llama.cpp server is reachable.
    pub async fn health_check(&self, base_url: &str) -> bool {
        let url = format!("{}/health", base_url.trim_end_matches('/'));
        self.client.get(&url).send().await.map(|r| r.status().is_success()).unwrap_or(false)
    }
}
