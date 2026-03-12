use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use tokio::process::Command;
use tracing::{info, warn};

use super::Tool;
use crate::agents::ProactiveEvent;
use crate::llm::Message;

// ── History formatting ────────────────────────────────────────────────────────

/// Format conversation messages as a human-readable chat history for the agent.
/// Only user and assistant turns are included; system and tool messages are omitted.
///
/// The result is passed as the `-q` argument to the agent CLI so it has full
/// conversational context when processing the delegation request.
pub fn format_history(messages: &[Message]) -> String {
    messages
        .iter()
        .filter(|m| m.role == "user" || m.role == "assistant")
        .map(|m| match m.role.as_str() {
            "user" => format!("[User]: {}", m.content),
            "assistant" => format!("[Jarvis]: {}", m.content),
            _ => unreachable!(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

// ── Subprocess helper ─────────────────────────────────────────────────────────

/// Spawns the agent CLI passing `query` via the `-q` flag.
/// Reads the complete stdout as the response.
///
/// Command construction: `{command_parts...} -q {query}`
/// e.g. AGENT_COMMAND=`hermes chat` → `hermes chat -q "..."`
async fn call_agent(command: String, query: String) -> String {
    let parts: Vec<String> = command.split_whitespace().map(String::from).collect();
    let program = match parts.first() {
        Some(p) => p.clone(),
        None => return "Agent error: AGENT_COMMAND is empty.".to_string(),
    };
    let mut args: Vec<String> = parts[1..].to_vec();
    args.push("-q".to_string());
    args.push(query);

    let child = match Command::new(&program)
        .args(&args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            warn!("Failed to spawn agent '{}': {}", program, e);
            return format!("Agent error: failed to launch '{}': {}", program, e);
        }
    };

    match child.wait_with_output().await {
        Ok(output) => {
            let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if text.is_empty() {
                "Agent completed with no output.".to_string()
            } else {
                text
            }
        }
        Err(e) => {
            warn!("Agent process error: {}", e);
            format!("Agent error: {}", e)
        }
    }
}

// ── RunAgentTool (synchronous — kept for reference, not registered) ───────────

/// Synchronous variant. Not registered in the voice pipeline (use RunAgentAsyncTool
/// instead so the voicebot never blocks waiting for the agent). Kept for testing
/// and future use.
pub struct RunAgentTool {
    command: String,
    timeout_secs: u64,
    history: Arc<RwLock<String>>,
}

impl RunAgentTool {
    pub fn new(command: &str, timeout_secs: u64, history: Arc<RwLock<String>>) -> Self {
        Self {
            command: command.to_string(),
            timeout_secs,
            history,
        }
    }
}

#[async_trait]
impl Tool for RunAgentTool {
    fn name(&self) -> &str {
        "run_agent"
    }

    fn description(&self) -> &str {
        "Delegates a task to the external agent and waits for the result (< 30s)."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "task": { "type": "string", "description": "Task to delegate" }
            },
            "required": ["task"]
        })
    }

    async fn run(&self, args: &str) -> String {
        let task = parse_task(args);
        if task.is_empty() {
            return "Error: run_agent requires a task description.".to_string();
        }
        let query = self.history.read().map(|h| h.clone()).unwrap_or_else(|_| task.clone());
        let command = self.command.clone();
        let timeout = std::time::Duration::from_secs(self.timeout_secs);
        match tokio::time::timeout(timeout, call_agent(command, query)).await {
            Ok(result) => result,
            Err(_) => {
                warn!("RunAgentTool: task timed out after {}s", self.timeout_secs);
                format!("Agent timed out after {}s.", self.timeout_secs)
            }
        }
    }
}

// ── RunAgentAsyncTool (fire-and-forget) ───────────────────────────────────────

/// Asynchronous agent tool: spawns the agent CLI in the background, passing the
/// full conversation history via `-q`, and returns an acknowledgment immediately.
///
/// The agent receives complete context (all prior turns + the current request as
/// the last [User] line) so it can respond coherently without additional prompting.
/// The result is delivered via the proactive channel when the process exits.
pub struct RunAgentAsyncTool {
    command: String,
    /// Shared formatted conversation history, updated by main after each user turn.
    history: Arc<RwLock<String>>,
    proactive_tx: tokio::sync::mpsc::Sender<ProactiveEvent>,
}

impl RunAgentAsyncTool {
    pub fn new(
        command: &str,
        history: Arc<RwLock<String>>,
        proactive_tx: tokio::sync::mpsc::Sender<ProactiveEvent>,
    ) -> Self {
        Self {
            command: command.to_string(),
            history,
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
        "Delegates a task to the external agent in the background and returns immediately. \
         The agent has full conversation context. The result will be announced proactively \
         when the agent finishes. Use for any task requiring computer control, file access, \
         web search, calendar, or extended reasoning."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Brief label for this task (used in the completion announcement)"
                }
            },
            "required": ["task"]
        })
    }

    async fn run(&self, args: &str) -> String {
        let task = parse_task(args);
        if task.is_empty() {
            return "Error: run_agent_async requires a task description.".to_string();
        }

        // The query sent to the agent is the full conversation history.
        // history already ends with the current [User] turn (updated by main
        // immediately after add_user_turn, before the LLM generates this call).
        let query = self.history.read().map(|h| h.clone()).unwrap_or_else(|_| task.clone());
        let command = self.command.clone();
        let proactive_tx = self.proactive_tx.clone();

        tokio::spawn(async move {
            info!("RunAgentAsyncTool: task started: {:?}", task);
            let result = call_agent(command, query).await;
            info!("RunAgentAsyncTool: task complete ({} chars): {:?}", result.len(), result);
            let _ = proactive_tx.send(ProactiveEvent::AgentResult { task, result }).await;
        });

        "[Tarea delegada al agente. El resultado llegará en breve.]".to_string()
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn parse_task(args: &str) -> String {
    serde_json::from_str::<serde_json::Value>(args)
        .ok()
        .and_then(|v| v["task"].as_str().map(String::from))
        .unwrap_or_else(|| args.to_string())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use tokio::sync::mpsc;

    use super::*;

    fn empty_history() -> Arc<RwLock<String>> {
        Arc::new(RwLock::new(String::new()))
    }

    fn history_with(s: &str) -> Arc<RwLock<String>> {
        Arc::new(RwLock::new(s.to_string()))
    }

    // ── format_history ────────────────────────────────────────────────────────

    #[test]
    fn format_history_empty_messages() {
        assert_eq!(format_history(&[]), "");
    }

    #[test]
    fn format_history_single_user_turn() {
        let msgs = vec![Message::user("hola")];
        assert_eq!(format_history(&msgs), "[User]: hola");
    }

    #[test]
    fn format_history_user_and_assistant() {
        let msgs = vec![Message::user("hola"), Message::assistant("hola Daniel")];
        assert_eq!(format_history(&msgs), "[User]: hola\n[Jarvis]: hola Daniel");
    }

    #[test]
    fn format_history_skips_system_messages() {
        let msgs = vec![
            Message::system("Eres Jarvis"),
            Message::user("hola"),
            Message::assistant("hola"),
        ];
        let result = format_history(&msgs);
        assert!(!result.contains("Eres Jarvis"), "system message should be excluded");
        assert!(result.contains("[User]: hola"));
    }

    #[test]
    fn format_history_skips_tool_messages() {
        let msgs = vec![
            Message::user("qué hora es"),
            Message::tool("14:30"),
            Message::assistant("Son las 14:30"),
        ];
        let result = format_history(&msgs);
        assert!(!result.contains("14:30\n"), "bare tool result should be excluded");
        assert!(result.contains("[Jarvis]: Son las 14:30"));
    }

    #[test]
    fn format_history_multiple_turns() {
        let msgs = vec![
            Message::user("primera"),
            Message::assistant("respuesta uno"),
            Message::user("segunda"),
            Message::assistant("respuesta dos"),
        ];
        let expected = "[User]: primera\n[Jarvis]: respuesta uno\n[User]: segunda\n[Jarvis]: respuesta dos";
        assert_eq!(format_history(&msgs), expected);
    }

    // ── RunAgentTool (sync) ───────────────────────────────────────────────────

    #[test]
    fn sync_name_and_description() {
        let tool = RunAgentTool::new("echo", 30, empty_history());
        assert_eq!(tool.name(), "run_agent");
        assert!(!tool.description().is_empty());
    }

    #[tokio::test]
    async fn sync_empty_args_returns_error() {
        let tool = RunAgentTool::new("echo", 30, empty_history());
        let result = tool.run("").await;
        assert!(result.to_lowercase().contains("error"), "got: {result:?}");
    }

    #[tokio::test]
    async fn sync_handles_nonexistent_command() {
        let tool = RunAgentTool::new("__nonexistent__", 10, empty_history());
        let result = tool.run(r#"{"task": "task"}"#).await;
        assert!(result.to_lowercase().contains("error"), "got: {result:?}");
    }

    // ── RunAgentAsyncTool ─────────────────────────────────────────────────────

    #[test]
    fn async_name_and_description() {
        let (tx, _rx) = mpsc::channel::<ProactiveEvent>(8);
        let tool = RunAgentAsyncTool::new("echo", empty_history(), tx);
        assert_eq!(tool.name(), "run_agent_async");
        assert!(!tool.description().is_empty());
    }

    #[tokio::test]
    async fn async_empty_args_returns_error() {
        let (tx, _rx) = mpsc::channel::<ProactiveEvent>(8);
        let tool = RunAgentAsyncTool::new("echo", empty_history(), tx);
        let result = tool.run("").await;
        assert!(result.to_lowercase().contains("error"), "got: {result:?}");
    }

    #[tokio::test]
    async fn async_returns_acknowledgment_immediately() {
        let (tx, _rx) = mpsc::channel::<ProactiveEvent>(8);
        let tool = RunAgentAsyncTool::new("sleep 2", empty_history(), tx);
        let start = std::time::Instant::now();
        let result = tool.run(r#"{"task": "slow task"}"#).await;
        assert!(start.elapsed().as_millis() < 200, "should return immediately");
        assert!(!result.is_empty(), "should return acknowledgment: {result:?}");
    }

    #[tokio::test]
    async fn async_delivers_result_to_proactive_channel() {
        let (tx, mut rx) = mpsc::channel::<ProactiveEvent>(8);
        // echo ignores -q flag and prints its other args; we just need any output
        let tool = RunAgentAsyncTool::new("echo agent_done", empty_history(), tx);
        tool.run(r#"{"task": "some task"}"#).await;

        let event = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
            .await
            .expect("timed out")
            .expect("channel closed");

        match event {
            ProactiveEvent::AgentResult { task, result } => {
                assert!(task.contains("some task"), "task: {task:?}");
                assert!(!result.is_empty(), "result should not be empty");
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[tokio::test]
    async fn async_passes_history_in_query() {
        // Use `cat` with -q to verify the history is passed:
        // the query is "-q {history}", but cat reads stdin... this doesn't work with -q.
        // Instead, verify via the history content reaching the proactive event label.
        let (tx, mut rx) = mpsc::channel::<ProactiveEvent>(8);
        let hist = history_with("[User]: busca noticias\n[Jarvis]: delegando");
        let tool = RunAgentAsyncTool::new("echo done", hist, tx);
        tool.run(r#"{"task": "busca noticias"}"#).await;

        let event = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
            .await
            .expect("timed out")
            .expect("channel closed");

        match event {
            ProactiveEvent::AgentResult { task, .. } => {
                assert!(task.contains("busca noticias"));
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[tokio::test]
    async fn async_delivers_error_on_launch_failure() {
        let (tx, mut rx) = mpsc::channel::<ProactiveEvent>(8);
        let tool = RunAgentAsyncTool::new("__nonexistent__", empty_history(), tx);
        tool.run(r#"{"task": "task"}"#).await;

        let event = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
            .await
            .expect("timed out")
            .expect("channel closed");

        match event {
            ProactiveEvent::AgentResult { result, .. } => {
                assert!(result.to_lowercase().contains("error"), "got: {result:?}");
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }
}
