use anyhow::{Context, Result};
use futures_util::StreamExt;
use tokio::sync::mpsc;

use super::session::Message;

#[derive(Clone)]
pub struct LlamaClient {
    client: reqwest::Client,
    chat_url: String,
    model: String,
    max_tokens: u32,
    temperature: f32,
}

impl LlamaClient {
    pub fn new(base_url: &str, model: &str, max_tokens: u32, temperature: f32) -> Self {
        Self {
            client: reqwest::Client::new(),
            chat_url: format!("{}/v1/chat/completions", base_url.trim_end_matches('/')),
            model: model.to_string(),
            max_tokens,
            temperature,
        }
    }

    /// Stream completion tokens from an OpenAI-compatible endpoint.
    ///
    /// Returns a channel receiver that yields text tokens as they arrive.
    pub async fn stream(&self, messages: &[Message]) -> Result<mpsc::Receiver<String>> {
        let payload = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": 0.95,
            "stream": true,
        });

        let response = self
            .client
            .post(&self.chat_url)
            .json(&payload)
            .send()
            .await
            .context("Failed to reach LLM server")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("LLM error {}: {}", status, body);
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

                loop {
                    let Some(newline) = buf.find('\n') else { break };
                    let line = buf[..newline].trim().to_string();
                    buf = buf[newline + 1..].to_string();

                    let Some(data) = line.strip_prefix("data: ") else { continue };

                    if data == "[DONE]" {
                        return;
                    }

                    let Ok(json) = serde_json::from_str::<serde_json::Value>(data) else {
                        continue;
                    };

                    if let Some(content) = json["choices"][0]["delta"]["content"].as_str() {
                        if !content.is_empty() && tx.send(content.to_string()).await.is_err() {
                            return;
                        }
                    }
                }
            }
        });

        Ok(rx)
    }

    /// One-shot (non-streaming) completion. Used for summarization.
    pub async fn complete(&self, messages: &[Message]) -> Result<String> {
        let payload = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.3,
            "stream": false,
        });

        let response = self
            .client
            .post(&self.chat_url)
            .json(&payload)
            .send()
            .await
            .context("Failed to reach LLM server for summarization")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("LLM summarization error {}: {}", status, body);
        }

        let json: serde_json::Value = response.json().await?;
        Ok(json["choices"][0]["message"]["content"].as_str().unwrap_or("").trim().to_string())
    }

    /// One-shot completion with a short token budget. Used for structured
    /// extractions (profile facts, etc.) that produce brief outputs.
    pub async fn complete_short(&self, messages: &[Message]) -> Result<String> {
        let payload = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "max_tokens": 256,
            "temperature": 0.1,
            "stream": false,
        });

        let response = self
            .client
            .post(&self.chat_url)
            .json(&payload)
            .send()
            .await
            .context("Failed to reach LLM server for extraction")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("LLM extraction error {}: {}", status, body);
        }

        let json: serde_json::Value = response.json().await?;
        Ok(json["choices"][0]["message"]["content"].as_str().unwrap_or("").trim().to_string())
    }

    /// Check if the server is reachable.
    pub async fn health_check(&self, base_url: &str) -> bool {
        let url = format!("{}/health", base_url.trim_end_matches('/'));
        self.client.get(&url).send().await.map(|r| r.status().is_success()).unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn make_messages() -> Vec<Message> {
        vec![
            Message::system("You are a summarizer."),
            Message::user("Summarize this conversation."),
        ]
    }

    // ── complete (non-streaming) ───────────────────────────────────────────────

    #[tokio::test]
    async fn complete_parses_openai_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": "This is the summary."}}]
            })))
            .mount(&server)
            .await;

        let client = LlamaClient::new(&server.uri(), "test-model", 512, 0.3);
        let result = client.complete(&make_messages()).await.unwrap();
        assert_eq!(result, "This is the summary.");
    }

    #[tokio::test]
    async fn complete_trims_whitespace() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": "  trimmed  \n"}}]
            })))
            .mount(&server)
            .await;

        let client = LlamaClient::new(&server.uri(), "test-model", 512, 0.3);
        let result = client.complete(&make_messages()).await.unwrap();
        assert_eq!(result, "trimmed");
    }

    #[tokio::test]
    async fn complete_returns_error_on_server_failure() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(500).set_body_string("Internal error"))
            .mount(&server)
            .await;

        let client = LlamaClient::new(&server.uri(), "test-model", 512, 0.3);
        let result = client.complete(&make_messages()).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("500"));
    }

    // ── complete_short ────────────────────────────────────────────────────────

    #[tokio::test]
    async fn complete_short_parses_openai_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": "[{\"key\": \"name\", \"value\": \"Daniel\", \"confidence\": 0.95}]"}}]
            })))
            .mount(&server)
            .await;

        let client = LlamaClient::new(&server.uri(), "test-model", 256, 0.1);
        let result = client.complete_short(&make_messages()).await.unwrap();
        assert!(result.contains("Daniel"));
    }

    // ── stream (SSE) ──────────────────────────────────────────────────────────

    #[tokio::test]
    async fn stream_collects_tokens_until_done() {
        let server = MockServer::start().await;
        // Simulate OpenAI SSE: two content tokens then [DONE]
        let sse_body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n",
            "data: [DONE]\n\n",
        );
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(sse_body)
                    .append_header("content-type", "text/event-stream"),
            )
            .mount(&server)
            .await;

        let client = LlamaClient::new(&server.uri(), "test-model", 400, 0.7);
        let mut rx = client.stream(&make_messages()).await.unwrap();

        let mut collected = String::new();
        while let Some(token) = rx.recv().await {
            collected.push_str(&token);
        }
        assert_eq!(collected, "Hello world");
    }

    #[tokio::test]
    async fn stream_skips_empty_content_tokens() {
        let server = MockServer::start().await;
        let sse_body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n",
            "data: [DONE]\n\n",
        );
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(sse_body)
                    .append_header("content-type", "text/event-stream"),
            )
            .mount(&server)
            .await;

        let client = LlamaClient::new(&server.uri(), "test-model", 400, 0.7);
        let mut rx = client.stream(&make_messages()).await.unwrap();

        let mut collected = String::new();
        while let Some(token) = rx.recv().await {
            collected.push_str(&token);
        }
        assert_eq!(collected, "Hi");
    }

    // ── end-to-end summarization (client + session) ───────────────────────────

    // ── tool call detection via stream ────────────────────────────────────────

    #[tokio::test]
    async fn stream_delivers_tool_call_tokens() {
        // The LLM emits a tool call split across multiple SSE chunks, as happens
        // in practice when tokens arrive one-by-one.
        let server = MockServer::start().await;
        let sse_body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"<tool_call>\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"current_time\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"</tool_call>\"}}]}\n\n",
            "data: [DONE]\n\n",
        );
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(sse_body)
                    .append_header("content-type", "text/event-stream"),
            )
            .mount(&server)
            .await;

        let client = LlamaClient::new(&server.uri(), "test-model", 400, 0.7);
        let mut rx = client.stream(&make_messages()).await.unwrap();

        let mut full = String::new();
        while let Some(token) = rx.recv().await {
            full.push_str(&token);
        }

        // The full response contains the complete tool call XML
        assert_eq!(full, "<tool_call>current_time</tool_call>");

        // ToolRegistry can detect and route it
        let mut registry = super::super::session::LlmSession::new("", 0); // just to use the module
        let _ = registry; // unused, registry test is below

        use crate::tools::{CurrentTimeTool, ToolRegistry};
        let mut reg = ToolRegistry::new();
        reg.register(CurrentTimeTool);
        let name = reg.parse_tool_call(&full).expect("should detect current_time tool call");
        assert_eq!(name, "current_time");
        let result = reg.execute(&name);
        assert!(!result.is_empty());
        assert!(result.contains(':'), "expected time in result: {result:?}");
    }

    #[tokio::test]
    async fn summarization_cycle_with_mock_llm() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": "El usuario intercambió varios mensajes con el asistente."}}]
            })))
            .mount(&server)
            .await;

        let client = LlamaClient::new(&server.uri(), "test-model", 512, 0.3);

        // Build a session with enough history to trigger summarization
        let mut session = super::super::session::LlmSession::new("Eres Jarvis.", 0);
        for i in 0..5 {
            session.add_user_turn(&format!("Pregunta {i} del usuario"));
            session.add_assistant_turn(&format!("Respuesta {i} del asistente"));
        }

        let keep_n = 4;
        assert!(session.needs_summarization(50), "should trigger at small context window");

        // Build the summarization request and call the mock LLM
        let summary_messages = session.build_summary_prompt(keep_n).unwrap();
        let summary = client.complete(&summary_messages).await.unwrap();
        assert!(!summary.is_empty());

        // Apply the summary
        session.apply_summary(&summary, keep_n);

        // Verify the compacted session
        assert_eq!(session.messages.len(), keep_n);

        let all = session.all_messages();
        assert_eq!(all.len(), 1 + keep_n); // system + kept messages
        assert!(all[0].content.contains("[CONVERSATION SUMMARY]"));
        assert!(all[0].content.contains("El usuario intercambió"));

        // Recent turns are preserved
        assert!(all[1].content.contains("Pregunta 3"));
        assert!(all[2].content.contains("Respuesta 3"));
        assert!(all[3].content.contains("Pregunta 4"));
        assert!(all[4].content.contains("Respuesta 4"));
    }
}
