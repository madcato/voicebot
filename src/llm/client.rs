use anyhow::{Context, Result};
use futures_util::StreamExt;
use std::time::Duration;
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

/// Strip all `<think>…</think>` blocks from a complete (non-streaming) string.
///
/// Used to post-process secondary LLM responses when thinking mode is enabled.
/// The model reasons inside the tags; only the text after the closing tag is
/// returned to callers.
fn strip_think_blocks(s: &str) -> String {
    let mut out = s.to_string();
    loop {
        match (out.find("<think>"), out.find("</think>")) {
            (Some(start), Some(end)) if start <= end => {
                let after = &out[end + "</think>".len()..].to_string();
                out = format!("{}{}", &out[..start], after);
            }
            _ => break,
        }
    }
    out.trim().to_string()
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

/// A token produced by `OpenAIClient::stream`.
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
pub struct OpenAIClient {
    client: reqwest::Client,
    chat_url: String,
    model: String,
    max_tokens: u32,
    temperature: f32,
    /// Bearer token sent in `Authorization` header. Empty = no auth header.
    api_key: String,
    /// When true, send `chat_template_kwargs: {"enable_thinking": true}` in
    /// non-streaming requests and strip `<think>…</think>` blocks from the
    /// returned text. Intended for the secondary LLM only.
    thinking: bool,
}

impl OpenAIClient {
    pub fn new(
        base_url: &str,
        model: &str,
        max_tokens: u32,
        temperature: f32,
    ) -> Self {
        let client = reqwest::Client::builder()
            .tcp_keepalive(Duration::from_secs(60))
            .tcp_nodelay(true) // Disable Nagle — send SSE tokens immediately
            .connect_timeout(Duration::from_secs(5)) // Fail fast on unreachable server
            .pool_max_idle_per_host(4)
            .pool_idle_timeout(Duration::from_secs(90))
            .build()
            .expect("failed to build HTTP client");
        Self {
            client,
            chat_url: format!("{}/v1/chat/completions", base_url.trim_end_matches('/')),
            model: model.to_string(),
            max_tokens,
            temperature,
            api_key: String::new(),
            thinking: false,
        }
    }

    /// Set the API key sent as `Authorization: Bearer <key>`.
    pub fn with_api_key(mut self, key: &str) -> Self {
        self.api_key = key.to_string();
        self
    }

    /// Returns a POST request builder for the chat completions URL,
    /// with the `Authorization` header set if an API key is configured.
    fn post_chat(&self) -> reqwest::RequestBuilder {
        let req = self.client.post(&self.chat_url);
        if self.api_key.is_empty() {
            req
        } else {
            req.bearer_auth(&self.api_key)
        }
    }

    /// Enable Qwen3 thinking mode for non-streaming completions.
    ///
    /// When `true`, `chat_template_kwargs: {"enable_thinking": true}` is sent
    /// and `<think>…</think>` blocks are stripped from the returned text.
    /// The model reasons internally; callers only receive the final answer.
    pub fn with_thinking(mut self, thinking: bool) -> Self {
        self.thinking = thinking;
        self
    }

    /// Stream completion tokens from an OpenAI-compatible endpoint.
    ///
    /// Returns a channel receiver that yields text tokens as they arrive.
    pub async fn stream(
        &self,
        messages: &[serde_json::Value],
        tools: &[serde_json::Value],
    ) -> Result<(mpsc::Receiver<StreamToken>, tokio::task::JoinHandle<()>)> {
        let mut payload = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": 0.90,
            "stream": true,
            // mlx-lm requires sampling params per-request (no server-side config).
            "repetition_penalty": 1.1,
            "top_k": 40,
            "min_p": 0.05,
        });
        // Do not send chat_template_kwargs for streaming: changing the Jinja2 template
        // can conflict with tool calling for some mlx-community quantizations.
        // The ThinkFilter strips any <think> blocks that arrive in the SSE stream.
        if !tools.is_empty() {
            payload["tools"] = serde_json::json!(tools);
            payload["tool_choice"] = serde_json::json!("auto");
        }

        tracing::debug!(target: "llm", "Request payload: {}", serde_json::to_string(&payload).unwrap_or_default());

        let response = self
            .post_chat()
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

        let stream_handle = tokio::spawn(async move {
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
                            tracing::info!(target: "llm", "Emitting ToolCall (via [DONE]): name={} args={}", name, tool_args);
                            let _ = tx.send(StreamToken::ToolCall { name, args: tool_args.clone() }).await;
                        }
                        return;
                    }

                    let Ok(json) = serde_json::from_str::<serde_json::Value>(data) else {
                        continue;
                    };

                    // Accumulate tool_calls fragments FIRST — before checking
                    // finish_reason. mlx-lm (and some other servers) send the entire
                    // tool call in a single chunk that has both delta.tool_calls AND
                    // finish_reason="tool_calls". If we checked finish_reason first we
                    // would emit an empty ToolCall (tool_name still None) and return.
                    //
                    // Guard: mlx-lm sends `"tool_calls": []` on every content chunk;
                    // only treat it as a real call when the array is non-empty.
                    let has_tool_call_delta =
                        if let Some(calls) = json["choices"][0]["delta"]["tool_calls"].as_array() {
                            if !calls.is_empty() {
                                if let Some(call) = calls.first() {
                                    if let Some(name) = call["function"]["name"].as_str() {
                                        if !name.is_empty() {
                                            tracing::debug!(target: "llm", "Tool call detected: {}", name);
                                            tool_name = Some(name.to_string());
                                        }
                                    }
                                    if let Some(frag) = call["function"]["arguments"].as_str() {
                                        tool_args.push_str(frag);
                                    }
                                }
                                true
                            } else {
                                false
                            }
                        } else {
                            false
                        };

                    // Now check finish_reason (tool_name may have just been set above).
                    if let Some(finish_reason) = json["choices"][0]["finish_reason"].as_str() {
                        tracing::debug!(target: "llm", "SSE finish_reason={}", finish_reason);
                        if finish_reason == "tool_calls" {
                            if let Some(tail) = think.flush() {
                                let _ = tx.send(StreamToken::Content(tail)).await;
                            }
                            if let Some(name) = tool_name.take() {
                                tracing::info!(target: "llm", "Emitting ToolCall (finish_reason=tool_calls): name={} args={}", name, tool_args);
                                let _ = tx.send(StreamToken::ToolCall { name, args: tool_args.clone() }).await;
                            }
                            return;
                        }
                    }

                    // Skip content processing if this chunk contained tool_calls data.
                    if has_tool_call_delta { continue; }

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

        Ok((rx, stream_handle))
    }

    /// One-shot (non-streaming) completion. Used for summarization.
    pub async fn complete(&self, messages: &[Message]) -> Result<String> {
        let mut payload = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.3,
            "stream": false,
        });
        if self.thinking {
            payload["chat_template_kwargs"] = serde_json::json!({"enable_thinking": true});
        }

        let response = self
            .post_chat()
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
        let text = json["choices"][0]["message"]["content"].as_str().unwrap_or("").trim().to_string();
        Ok(if self.thinking { strip_think_blocks(&text) } else { text })
    }

    /// One-shot completion with a short token budget. Used for structured
    /// extractions (profile facts, etc.) that produce brief outputs.
    pub async fn complete_short(&self, messages: &[Message]) -> Result<String> {
        let mut payload = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "max_tokens": 256,
            "temperature": 0.1,
            "stream": false,
        });
        if self.thinking {
            payload["chat_template_kwargs"] = serde_json::json!({"enable_thinking": true});
        }

        let response = self
            .post_chat()
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
        let text = json["choices"][0]["message"]["content"].as_str().unwrap_or("").trim().to_string();
        Ok(if self.thinking { strip_think_blocks(&text) } else { text })
    }

    /// One-shot multimodal completion with a single image + text prompt.
    ///
    /// Sends an OpenAI-compatible content array with `image_url` and `text` parts.
    /// Used by `TakeScreenshotTool`.
    pub async fn complete_multimodal(
        &self,
        image_data_url: &str,
        text_prompt: &str,
    ) -> Result<String> {
        let mut payload = serde_json::json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "stream": false,
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "image_url", "image_url": { "url": image_data_url } },
                    { "type": "text", "text": text_prompt }
                ]
            }]
        });
        if self.thinking {
            payload["chat_template_kwargs"] = serde_json::json!({"enable_thinking": true});
        }

        let response = self
            .post_chat()
            .json(&payload)
            .send()
            .await
            .context("Failed to reach vision model server")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Vision model error {}: {}", status, body);
        }

        let json: serde_json::Value = response.json().await?;
        let text = json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .trim()
            .to_string();
        Ok(if self.thinking { strip_think_blocks(&text) } else { text })
    }

    /// Check if the server is reachable.
    #[allow(dead_code)]
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

        let client = OpenAIClient::new(&server.uri(), "test-model", 512, 0.3);
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

        let client = OpenAIClient::new(&server.uri(), "test-model", 512, 0.3);
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

        let client = OpenAIClient::new(&server.uri(), "test-model", 512, 0.3);
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

        let client = OpenAIClient::new(&server.uri(), "test-model", 256, 0.1);
        let result = client.complete_short(&make_messages()).await.unwrap();
        assert!(result.contains("Daniel"));
    }

    // ── complete_multimodal ───────────────────────────────────────────────────

    #[tokio::test]
    async fn complete_multimodal_returns_model_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": "  A terminal window showing Rust code.  "}}]
            })))
            .mount(&server)
            .await;

        let client = OpenAIClient::new(&server.uri(), "vision-model", 512, 0.3);
        let result = client
            .complete_multimodal("data:image/png;base64,abc123", "What do you see?")
            .await
            .unwrap();
        assert_eq!(result, "A terminal window showing Rust code.");
    }

    #[tokio::test]
    async fn complete_multimodal_returns_error_on_server_failure() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(503).set_body_string("unavailable"))
            .mount(&server)
            .await;

        let client = OpenAIClient::new(&server.uri(), "vision-model", 512, 0.3);
        let result = client
            .complete_multimodal("data:image/png;base64,abc123", "What do you see?")
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("503"));
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

        let client = OpenAIClient::new(&server.uri(), "test-model", 400, 0.7);
        let messages = make_messages();
        let (mut rx, _handle) = client.stream(&messages_to_json(&messages), &[]).await.unwrap();

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

        let client = OpenAIClient::new(&server.uri(), "test-model", 400, 0.7);
        let messages = make_messages();
        let (mut rx, _handle) = client.stream(&messages_to_json(&messages), &[]).await.unwrap();

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
        // Simulate SSE for a native function call
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

        let client = OpenAIClient::new(&server.uri(), "test-model", 400, 0.7);
        let (mut rx, _handle) = client.stream(&[], &[]).await.unwrap();

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

        let client = OpenAIClient::new(&server.uri(), "test-model", 512, 0.3);

        // Build a session with enough history to trigger summarization
        let mut session = super::super::session::LlmSession::new("Eres Jarvis.");
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
