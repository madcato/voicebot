use async_trait::async_trait;
use tracing::{info, warn};

use super::Tool;
use crate::agents::ProactiveEvent;

// ── Shared HTTP helper ────────────────────────────────────────────────────────

async fn call_agent(
    client: &reqwest::Client,
    chat_url: &str,
    model: &str,
    max_tokens: u32,
    task: &str,
) -> String {
    let payload = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": task}],
        "max_tokens": max_tokens,
        "stream": false,
    });

    match client.post(chat_url).json(&payload).send().await {
        Ok(resp) if resp.status().is_success() => match resp.json::<serde_json::Value>().await {
            Ok(json) => json["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("")
                .trim()
                .to_string(),
            Err(e) => {
                warn!("Agent response parse error: {}", e);
                format!("Agent response parse error: {e}")
            }
        },
        Ok(resp) => {
            warn!("Agent returned HTTP {}", resp.status());
            format!("Agent error: HTTP {}", resp.status())
        }
        Err(e) => {
            warn!("Agent unreachable: {}", e);
            format!("Agent unreachable: {e}")
        }
    }
}

// ── RunAgentTool (synchronous) ────────────────────────────────────────────────

/// Synchronous agent tool: calls any OpenAI-compatible remote agent and waits
/// for the result. Suitable for tasks expected to complete in under ~10 seconds.
pub struct RunAgentTool {
    client: reqwest::Client,
    chat_url: String,
    model: String,
    max_tokens: u32,
}

impl RunAgentTool {
    pub fn new(base_url: &str, model: &str, max_tokens: u32) -> Self {
        Self {
            client: reqwest::Client::new(),
            chat_url: format!("{}/v1/chat/completions", base_url.trim_end_matches('/')),
            model: model.to_string(),
            max_tokens,
        }
    }
}

#[async_trait]
impl Tool for RunAgentTool {
    fn name(&self) -> &str {
        "run_agent"
    }

    fn description(&self) -> &str {
        "Delegates a task to a remote AI agent and waits for the result (< 10 s). \
         Use for tasks needing more reasoning than the local model."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Description of the task to delegate to the agent"
                }
            },
            "required": ["task"]
        })
    }

    async fn run(&self, args: &str) -> String {
        let task = serde_json::from_str::<serde_json::Value>(args)
            .ok()
            .and_then(|v| v["task"].as_str().map(String::from))
            .unwrap_or_else(|| args.to_string());
        if task.is_empty() {
            return "Error: run_agent requires a task description.".to_string();
        }
        info!("RunAgentTool: delegating task: {:?}", task);
        call_agent(&self.client, &self.chat_url, &self.model, self.max_tokens, &task).await
    }
}

// ── RunAgentAsyncTool (fire-and-forget) ───────────────────────────────────────

/// Asynchronous agent tool: spawns the agent task in the background and
/// returns an acknowledgment immediately. The result is delivered via the
/// proactive channel when the agent finishes.
pub struct RunAgentAsyncTool {
    client: reqwest::Client,
    chat_url: String,
    model: String,
    max_tokens: u32,
    proactive_tx: tokio::sync::mpsc::Sender<ProactiveEvent>,
}

impl RunAgentAsyncTool {
    pub fn new(
        base_url: &str,
        model: &str,
        max_tokens: u32,
        proactive_tx: tokio::sync::mpsc::Sender<ProactiveEvent>,
    ) -> Self {
        Self {
            client: reqwest::Client::new(),
            chat_url: format!("{}/v1/chat/completions", base_url.trim_end_matches('/')),
            model: model.to_string(),
            max_tokens,
            proactive_tx,
        }
    }
}

#[async_trait]
impl Tool for RunAgentAsyncTool {
    fn name(&self) -> &str {
        "run_agent_async"
    }

    fn description(&self) -> &str {
        "Delegates a long-running task to a remote AI agent in the background and returns \
         immediately. The result will be delivered proactively when the agent finishes. \
         Use for tasks that take more than 10 seconds (research, analysis, code generation)."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Description of the long-running task to delegate in the background"
                }
            },
            "required": ["task"]
        })
    }

    async fn run(&self, args: &str) -> String {
        let task = serde_json::from_str::<serde_json::Value>(args)
            .ok()
            .and_then(|v| v["task"].as_str().map(String::from))
            .unwrap_or_else(|| args.to_string());
        if task.is_empty() {
            return "Error: run_agent_async requires a task description.".to_string();
        }

        let client = self.client.clone();
        let chat_url = self.chat_url.clone();
        let model = self.model.clone();
        let max_tokens = self.max_tokens;
        let proactive_tx = self.proactive_tx.clone();

        tokio::spawn(async move {
            info!("RunAgentAsyncTool: background task started: {:?}", task);
            let result = call_agent(&client, &chat_url, &model, max_tokens, &task).await;
            info!(
                "RunAgentAsyncTool: task complete ({} chars), notifying proactive channel",
                result.len()
            );
            let _ = proactive_tx.send(ProactiveEvent::AgentResult { task, result }).await;
        });

        "[Tarea delegada al agente. El resultado llegará en breve.]".to_string()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use tokio::sync::mpsc;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    // ── RunAgentTool (sync) ───────────────────────────────────────────────────

    #[test]
    fn sync_name_and_description() {
        let tool = RunAgentTool::new("http://localhost:8080", "test-model", 2048);
        assert_eq!(tool.name(), "run_agent");
        assert!(!tool.description().is_empty(), "description should be non-empty");
    }

    #[tokio::test]
    async fn sync_empty_args_returns_error() {
        let tool = RunAgentTool::new("http://localhost:8080", "test-model", 2048);
        let result = tool.run("").await;
        assert!(result.to_lowercase().contains("error"), "got: {result:?}");
    }

    #[tokio::test]
    async fn sync_calls_agent_and_returns_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": "Agent result."}}]
            })))
            .mount(&server)
            .await;

        let tool = RunAgentTool::new(&server.uri(), "test-model", 2048);
        let result = tool.run("Summarise the Rust 2024 edition").await;
        assert_eq!(result, "Agent result.");
    }

    #[tokio::test]
    async fn sync_trims_response_whitespace() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": "  trimmed  \n"}}]
            })))
            .mount(&server)
            .await;

        let tool = RunAgentTool::new(&server.uri(), "test-model", 2048);
        assert_eq!(tool.run("task").await, "trimmed");
    }

    #[tokio::test]
    async fn sync_handles_server_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;

        let tool = RunAgentTool::new(&server.uri(), "test-model", 2048);
        let result = tool.run("task").await;
        assert!(
            result.contains("500") || result.to_lowercase().contains("error"),
            "should describe the error: {result:?}"
        );
    }

    #[tokio::test]
    async fn sync_handles_unreachable_server() {
        // Port with nothing listening
        let tool = RunAgentTool::new("http://127.0.0.1:19998", "test-model", 2048);
        let result = tool.run("task").await;
        assert!(!result.is_empty(), "should return an error message, not panic");
        assert!(result.to_lowercase().contains("error") || result.to_lowercase().contains("agent"));
    }

    // ── RunAgentAsyncTool ─────────────────────────────────────────────────────

    #[test]
    fn async_name_and_description() {
        let (tx, _rx) = mpsc::channel::<ProactiveEvent>(8);
        let tool = RunAgentAsyncTool::new("http://localhost:8080", "test-model", 2048, tx);
        assert_eq!(tool.name(), "run_agent_async");
        assert!(!tool.description().is_empty(), "description should be non-empty");
    }

    #[tokio::test]
    async fn async_empty_args_returns_error() {
        let (tx, _rx) = mpsc::channel::<ProactiveEvent>(8);
        let tool = RunAgentAsyncTool::new("http://localhost:8080", "test-model", 2048, tx);
        let result = tool.run("").await;
        assert!(result.to_lowercase().contains("error"), "got: {result:?}");
    }

    #[tokio::test]
    async fn async_returns_acknowledgment_immediately() {
        let server = MockServer::start().await;
        // Simulate a slow agent (200ms) — the tool must return well before that.
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({
                        "choices": [{"message": {"content": "Done."}}]
                    }))
                    .set_delay(std::time::Duration::from_millis(200)),
            )
            .mount(&server)
            .await;

        let (tx, _rx) = mpsc::channel::<ProactiveEvent>(8);
        let tool = RunAgentAsyncTool::new(&server.uri(), "test-model", 2048, tx);

        let start = std::time::Instant::now();
        let result = tool.run("Research something").await;
        let elapsed = start.elapsed();

        assert!(
            elapsed.as_millis() < 100,
            "should return in < 100 ms, took {}ms",
            elapsed.as_millis()
        );
        assert!(!result.is_empty(), "should return acknowledgment text: {result:?}");
    }

    #[tokio::test]
    async fn async_delivers_result_to_proactive_channel() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": "Rust is fast and memory-safe."}}]
            })))
            .mount(&server)
            .await;

        let (tx, mut rx) = mpsc::channel::<ProactiveEvent>(8);
        let tool = RunAgentAsyncTool::new(&server.uri(), "test-model", 2048, tx);

        tool.run("Research Rust performance").await;

        let event = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
            .await
            .expect("timed out waiting for proactive event")
            .expect("channel closed unexpectedly");

        match event {
            ProactiveEvent::AgentResult { task, result } => {
                assert!(task.contains("Research Rust"), "task: {task:?}");
                assert!(result.contains("Rust is fast"), "result: {result:?}");
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[tokio::test]
    async fn async_delivers_error_message_on_agent_failure() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(503))
            .mount(&server)
            .await;

        let (tx, mut rx) = mpsc::channel::<ProactiveEvent>(8);
        let tool = RunAgentAsyncTool::new(&server.uri(), "test-model", 2048, tx);
        tool.run("Do something").await;

        let event = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
            .await
            .expect("timed out")
            .expect("channel closed");

        match event {
            ProactiveEvent::AgentResult { result, .. } => {
                assert!(!result.is_empty(), "error result must not be empty");
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }

    // ── round-trip: parse_tool_call → execute → check result ─────────────────

    #[tokio::test]
    async fn sync_round_trip_via_registry() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": "Research complete."}}]
            })))
            .mount(&server)
            .await;

        let mut registry = crate::tools::ToolRegistry::new();
        registry.register(RunAgentTool::new(&server.uri(), "test-model", 2048));

        let llm_output = "<tool_call>run_agent: Explain async Rust</tool_call>";
        let (name, args) = registry.parse_tool_call(llm_output).expect("should parse run_agent");

        assert_eq!(name, "run_agent");
        assert_eq!(args, "Explain async Rust");

        let result = registry.execute(&name, &args).await;
        assert_eq!(result, "Research complete.");
    }

    #[tokio::test]
    async fn async_round_trip_via_registry() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": "Background research done."}}]
            })))
            .mount(&server)
            .await;

        let (tx, mut rx) = mpsc::channel::<ProactiveEvent>(8);
        let mut registry = crate::tools::ToolRegistry::new();
        registry.register(RunAgentAsyncTool::new(&server.uri(), "test-model", 2048, tx));

        let llm_output =
            "<tool_call>run_agent_async: Investigate climate data trends</tool_call>";
        let (name, args) = registry.parse_tool_call(llm_output).expect("should parse");
        assert_eq!(name, "run_agent_async");
        assert_eq!(args, "Investigate climate data trends");

        let ack = registry.execute(&name, &args).await;
        assert!(ack.contains("breve") || !ack.is_empty(), "should return acknowledgment");

        // Result arrives asynchronously in the proactive channel
        let event = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
            .await
            .expect("timed out")
            .expect("channel closed");
        match event {
            ProactiveEvent::AgentResult { task, result } => {
                assert!(task.contains("climate"), "task: {task:?}");
                assert!(result.contains("Background research done"), "result: {result:?}");
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }
}
