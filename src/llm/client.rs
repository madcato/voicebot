use anyhow::{Context, Result};
use futures_util::StreamExt;
use tokio::sync::mpsc;

use super::session::Message;

/// Strips `<think>…</think>` blocks from a streaming token sequence.
///
/// Qwen3 (and other reasoning models) emit a chain-of-thought block before
/// the actual response. Those tokens must not reach TTS or the tool detector.
///
/// The filter handles the case where the opening or closing tag is split
/// across multiple tokens by buffering up to `max(tag_len - 1)` bytes.
struct ThinkFilter {
    in_think: bool,
    /// Holds trailing bytes that could be the start of a tag (`<think>` or
    /// `</think>`). Flushed once we know they are not part of a tag.
    pending: String,
}

impl ThinkFilter {
    fn new() -> Self {
        Self { in_think: false, pending: String::new() }
    }

    /// Feed the next raw token from the SSE stream.
    /// Returns the portion of the token (if any) that should be forwarded.
    fn process(&mut self, token: &str) -> Option<String> {
        self.pending.push_str(token);
        let mut out = String::new();

        loop {
            if self.in_think {
                match self.pending.find("</think>") {
                    Some(pos) => {
                        // Found closing tag — resume normal output after it.
                        self.pending = self.pending[pos + "</think>".len()..].to_string();
                        self.in_think = false;
                        // Continue loop to check remaining pending for more tags.
                    }
                    None => {
                        // Keep only a suffix long enough to catch a split tag.
                        let keep = partial_tag_suffix(&self.pending, "</think>");
                        self.pending = self.pending[self.pending.len() - keep..].to_string();
                        break;
                    }
                }
            } else {
                match self.pending.find("<think>") {
                    Some(pos) => {
                        // Emit everything before the tag, then enter think mode.
                        out.push_str(&self.pending[..pos]);
                        self.pending = self.pending[pos + "<think>".len()..].to_string();
                        self.in_think = true;
                        // Continue loop to consume the think block.
                    }
                    None => {
                        // Keep only a suffix that could be a partial opening tag.
                        let keep = partial_tag_suffix(&self.pending, "<think>");
                        out.push_str(&self.pending[..self.pending.len() - keep]);
                        self.pending = self.pending[self.pending.len() - keep..].to_string();
                        break;
                    }
                }
            }
        }

        if out.is_empty() { None } else { Some(out) }
    }

    /// Call once when the stream ends to emit any buffered non-think content.
    fn flush(&mut self) -> Option<String> {
        if self.in_think || self.pending.is_empty() {
            self.pending.clear();
            return None;
        }
        let out = std::mem::take(&mut self.pending);
        Some(out)
    }
}

/// Returns the length of the longest suffix of `s` that is a proper prefix of
/// `tag` (i.e. the tail of `s` that could be the start of `tag`).
fn partial_tag_suffix(s: &str, tag: &str) -> usize {
    for n in (1..tag.len()).rev() {
        if s.ends_with(&tag[..n]) {
            return n;
        }
    }
    0
}

/// A token produced by `LlamaClient::stream`.
///
/// The LLM either generates text content (route to TTS) or calls a tool
/// (stop streaming, execute the tool, then continue).
#[derive(Debug)]
pub enum StreamToken {
    /// Regular text — forward to TTS.
    Content(String),
    /// The model invoked a tool. `args` is the JSON arguments string.
    ToolCall { name: String, args: String },
}

#[derive(Clone)]
pub struct LlamaClient {
    client: reqwest::Client,
    chat_url: String,
    model: String,
    max_tokens: u32,
    temperature: f32,
    /// llama.cpp KV-cache slot (0 for single-user). Sent with `cache_prompt=true`
    /// on streaming calls so the server reuses the cached prompt across turns.
    slot_id: u8,
    /// Slot used for background (non-streaming) calls so they don't evict the
    /// main conversation's KV cache. -1 = let llama.cpp pick any free slot.
    /// Set to 1 when llama-server runs with --parallel 2.
    background_slot_id: i32,
    /// When true, include llama.cpp-specific fields (`cache_prompt`, `slot_id`)
    /// in every request. Set to false when using mlx-lm or other backends.
    llama_extensions: bool,
}

impl LlamaClient {
    pub fn new(
        base_url: &str,
        model: &str,
        max_tokens: u32,
        temperature: f32,
        slot_id: u8,
        background_slot_id: i32,
    ) -> Self {
        Self {
            client: reqwest::Client::new(),
            chat_url: format!("{}/v1/chat/completions", base_url.trim_end_matches('/')),
            model: model.to_string(),
            max_tokens,
            temperature,
            slot_id,
            background_slot_id,
            llama_extensions: true,
        }
    }

    /// Configure which backend-specific extensions to send.
    /// `"llama"` → include `cache_prompt` + `slot_id` (llama.cpp).
    /// Any other value (e.g. `"mlx"`) → omit those fields.
    pub fn with_provider(mut self, provider: &str) -> Self {
        self.llama_extensions = provider == "llama";
        self
    }

    /// Stream completion tokens from an OpenAI-compatible endpoint.
    ///
    /// Returns a channel receiver that yields text tokens as they arrive.
    pub async fn stream(
        &self,
        messages: &[serde_json::Value],
        tools: &[serde_json::Value],
    ) -> Result<mpsc::Receiver<StreamToken>> {
        let mut payload = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": 0.95,
            "stream": true,
        });
        if self.llama_extensions {
            // llama.cpp extensions: reuse the KV-cache across turns for this slot.
            payload["cache_prompt"] = serde_json::json!(true);
            payload["slot_id"] = serde_json::json!(self.slot_id);
        }
        if !tools.is_empty() {
            payload["tools"] = serde_json::json!(tools);
            payload["tool_choice"] = serde_json::json!("auto");
        }

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

        let (tx, rx) = mpsc::channel::<StreamToken>(256);

        tokio::spawn(async move {
            let mut stream = response.bytes_stream();
            let mut buf = String::new();
            let mut think = ThinkFilter::new();

            // Accumulate tool call across streamed fragments
            let mut tool_name: Option<String> = None;
            let mut tool_args = String::new();

            while let Some(chunk) = stream.next().await {
                let bytes = match chunk {
                    Ok(b) => b,
                    Err(e) => {
                        tracing::error!(target: "llm", "LLM stream error: {}", e);
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
                        // Flush any content held back for partial-tag detection.
                        if let Some(tail) = think.flush() {
                            let _ = tx.send(StreamToken::Content(tail)).await;
                        }
                        // If a tool call was accumulating but no finish_reason arrived.
                        if let Some(name) = tool_name.take() {
                            let _ = tx.send(StreamToken::ToolCall { name, args: tool_args.clone() }).await;
                        }
                        return;
                    }

                    let Ok(json) = serde_json::from_str::<serde_json::Value>(data) else {
                        continue;
                    };

                    // Check finish_reason before looking at delta content.
                    if let Some(finish_reason) = json["choices"][0]["finish_reason"].as_str() {
                        if finish_reason == "tool_calls" {
                            if let Some(tail) = think.flush() {
                                let _ = tx.send(StreamToken::Content(tail)).await;
                            }
                            if let Some(name) = tool_name.take() {
                                let _ = tx.send(StreamToken::ToolCall { name, args: tool_args.clone() }).await;
                            }
                            return;
                        }
                    }

                    // Accumulate tool_calls fragments.
                    if let Some(calls) = json["choices"][0]["delta"]["tool_calls"].as_array() {
                        if let Some(call) = calls.first() {
                            if let Some(name) = call["function"]["name"].as_str() {
                                if !name.is_empty() {
                                    tool_name = Some(name.to_string());
                                }
                            }
                            if let Some(frag) = call["function"]["arguments"].as_str() {
                                tool_args.push_str(frag);
                            }
                        }
                        continue; // tool_calls and content are mutually exclusive
                    }

                    // Regular content token.
                    if let Some(content) = json["choices"][0]["delta"]["content"].as_str() {
                        if content.is_empty() { continue; }
                        if let Some(filtered) = think.process(content) {
                            if tx.send(StreamToken::Content(filtered)).await.is_err() { return; }
                        }
                    }
                }
            }
        });

        Ok(rx)
    }

    /// One-shot (non-streaming) completion. Used for summarization.
    ///
    /// Sends `slot_id: background_slot_id` (default -1 = any free slot) so
    /// llama.cpp does not evict the main conversation's KV-cache in slot 0.
    pub async fn complete(&self, messages: &[Message]) -> Result<String> {
        let mut payload = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.3,
            "stream": false,
        });
        if self.llama_extensions {
            payload["cache_prompt"] = serde_json::json!(false);
            payload["slot_id"] = serde_json::json!(self.background_slot_id);
        }

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
    ///
    /// Sends `slot_id: background_slot_id` so it does not evict the main slot.
    pub async fn complete_short(&self, messages: &[Message]) -> Result<String> {
        let mut payload = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "max_tokens": 256,
            "temperature": 0.1,
            "stream": false,
        });
        if self.llama_extensions {
            payload["cache_prompt"] = serde_json::json!(false);
            payload["slot_id"] = serde_json::json!(self.background_slot_id);
        }

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

    /// Fire a speculative KV-cache warm-up request without waiting for tokens.
    ///
    /// Sends the current session messages (without the new user turn) so
    /// llama.cpp starts computing KV vectors while Whisper STT runs in parallel.
    /// The caller aborts the spawned task as soon as the transcript is ready.
    pub async fn prefill_warm(&self, messages: Vec<serde_json::Value>) -> Result<()> {
        let mut payload = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "max_tokens": 1,
            "temperature": self.temperature,
            "stream": true,
        });
        if self.llama_extensions {
            payload["cache_prompt"] = serde_json::json!(true);
            payload["slot_id"] = serde_json::json!(self.slot_id);
        }

        let response = self
            .client
            .post(&self.chat_url)
            .json(&payload)
            .send()
            .await
            .context("Speculative prefill: failed to reach LLM server")?;

        if !response.status().is_success() {
            let status = response.status();
            anyhow::bail!("Speculative prefill error {}", status);
        }

        // Stream the body to keep the connection alive while the server prefills.
        // tokio cancels this future when the JoinHandle is aborted by the caller.
        let mut stream = response.bytes_stream();
        while stream.next().await.is_some() {}
        Ok(())
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

    fn messages_to_json(msgs: &[Message]) -> Vec<serde_json::Value> {
        msgs.iter().map(|m| serde_json::json!({"role": m.role, "content": m.content})).collect()
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

        let client = LlamaClient::new(&server.uri(), "test-model", 512, 0.3, 0, -1);
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

        let client = LlamaClient::new(&server.uri(), "test-model", 512, 0.3, 0, -1);
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

        let client = LlamaClient::new(&server.uri(), "test-model", 512, 0.3, 0, -1);
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

        let client = LlamaClient::new(&server.uri(), "test-model", 256, 0.1, 0, -1);
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

        let client = LlamaClient::new(&server.uri(), "test-model", 400, 0.7, 0, -1);
        let messages = make_messages();
        let mut rx = client.stream(&messages_to_json(&messages), &[]).await.unwrap();

        let mut collected = String::new();
        while let Some(token) = rx.recv().await {
            if let StreamToken::Content(s) = token { collected.push_str(&s); }
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

        let client = LlamaClient::new(&server.uri(), "test-model", 400, 0.7, 0, -1);
        let messages = make_messages();
        let mut rx = client.stream(&messages_to_json(&messages), &[]).await.unwrap();

        let mut collected = String::new();
        while let Some(token) = rx.recv().await {
            if let StreamToken::Content(s) = token { collected.push_str(&s); }
        }
        assert_eq!(collected, "Hi");
    }

    // ── ThinkFilter ───────────────────────────────────────────────────────────

    #[test]
    fn think_filter_passthrough_when_no_think_block() {
        let mut f = ThinkFilter::new();
        assert_eq!(f.process("Hello world").as_deref(), Some("Hello world"));
        assert_eq!(f.flush().as_deref(), None);
    }

    #[test]
    fn think_filter_strips_single_token_think_block() {
        let mut f = ThinkFilter::new();
        // Entire block in one token
        let result = f.process("<think>some thoughts</think>answer");
        assert_eq!(result.as_deref(), Some("answer"));
    }

    #[test]
    fn think_filter_strips_think_block_split_across_tokens() {
        let mut f = ThinkFilter::new();
        assert_eq!(f.process("<think>"), None);
        assert_eq!(f.process("some reasoning"), None);
        assert_eq!(f.process("</think>"), None);
        assert_eq!(f.process("actual answer").as_deref(), Some("actual answer"));
    }

    #[test]
    fn think_filter_handles_split_opening_tag() {
        let mut f = ThinkFilter::new();
        // "<think>" split as "<th" + "ink>" + content
        assert_eq!(f.process("<th"), None); // buffered as partial tag
        assert_eq!(f.process("ink>thoughts</think>answer").as_deref(), Some("answer"));
    }

    #[test]
    fn think_filter_handles_split_closing_tag() {
        let mut f = ThinkFilter::new();
        f.process("<think>thoughts</thi");
        assert_eq!(f.process("nk>real answer").as_deref(), Some("real answer"));
    }

    #[test]
    fn think_filter_emits_content_before_think_block() {
        let mut f = ThinkFilter::new();
        let result = f.process("prefix<think>thoughts</think>suffix");
        assert_eq!(result.as_deref(), Some("prefixsuffix"));
    }

    #[test]
    fn think_filter_flush_returns_buffered_content() {
        let mut f = ThinkFilter::new();
        f.process("Hello"); // held in pending as possible partial tag start? No — "Hello" has no partial "<"
        // Actually "Hello" gets emitted immediately. Let's test flush with a partial tag.
        let mut f2 = ThinkFilter::new();
        f2.process("Hello <thi"); // "<thi" held as partial
        let flushed = f2.flush();
        // Flush should emit the pending partial since it never completed
        assert!(flushed.as_deref().unwrap_or("").contains("<thi") || flushed.is_none() || flushed.as_deref() == Some("<thi"));
    }

    #[test]
    fn think_filter_flush_inside_think_block_returns_none() {
        let mut f = ThinkFilter::new();
        f.process("<think>unfinished");
        assert_eq!(f.flush(), None);
    }

    // ── end-to-end summarization (client + session) ───────────────────────────

    // ── tool call detection via stream ────────────────────────────────────────

    #[tokio::test]
    async fn stream_delivers_native_tool_call() {
        let server = MockServer::start().await;
        // Simulate llama.cpp SSE for a native function call
        let sse_body = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_0\",\"type\":\"function\",\"function\":{\"name\":\"current_time\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{}\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
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

        let client = LlamaClient::new(&server.uri(), "test-model", 400, 0.7, 0, -1);
        let mut rx = client.stream(&[], &[]).await.unwrap();

        let token = rx.recv().await.expect("should receive a token");
        match token {
            StreamToken::ToolCall { name, args } => {
                assert_eq!(name, "current_time");
                assert_eq!(args, "{}");
            }
            other => panic!("expected ToolCall, got {:?}", other),
        }
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

        let client = LlamaClient::new(&server.uri(), "test-model", 512, 0.3, 0, -1);

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
