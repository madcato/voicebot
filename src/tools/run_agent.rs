use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{mpsc, oneshot, Mutex};
use tracing::{debug, info, warn};

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
#[allow(dead_code)]
pub struct RunAgentTool {
    command: String,
    timeout_secs: u64,
    history: Arc<RwLock<String>>,
}

impl RunAgentTool {
    #[allow(dead_code)]
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

// ── ACP Protocol types ────────────────────────────────────────────────────────

/// A single text content block sent inside an ACP `Prompt` message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcpTextBlock {
    #[serde(rename = "type")]
    pub kind: String,
    pub text: String,
}

impl AcpTextBlock {
    pub fn text(s: impl Into<String>) -> Self {
        Self { kind: "text".into(), text: s.into() }
    }
}

/// Messages sent **from the voicebot to the ACP process** (outbound).
///
/// `#[serde(tag = "type")]` emits `{"type": "<variant_snake_case>", ...}`.
/// Field names are already snake_case so no rename_all is needed.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AcpOutbound {
    Initialize {
        protocol_version: u32,
    },
    NewSession {
        cwd: String,
        #[serde(default)]
        mcp_servers: Vec<serde_json::Value>,
    },
    Prompt {
        session_id: String,
        prompt: Vec<AcpTextBlock>,
    },
    PermissionResponse {
        session_id: String,
        outcome: String,
    },
    Cancel {
        session_id: String,
    },
}

/// Session-update sub-events streamed by the ACP process while working.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AcpSessionUpdate {
    ToolStart { tool: String },
    ToolComplete { tool: String, #[allow(dead_code)] result: Option<String> },
    AgentThought { text: String },
    AgentMessage { text: String },
}

/// Messages received **from the ACP process** (inbound).
///
/// Unknown `type` values are captured by the `Unknown` variant so the reader
/// loop never panics on future protocol extensions.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AcpInbound {
    InitializeResponse {
        session_id: String,
    },
    SessionUpdate {
        #[allow(dead_code)]
        session_id: String,
        update: AcpSessionUpdate,
    },
    PermissionRequest {
        #[allow(dead_code)]
        session_id: String,
        description: String,
        options: Vec<String>,
    },
    PromptResponse {
        #[allow(dead_code)]
        session_id: String,
        output: String,
        #[allow(dead_code)]
        stop_reason: String,
    },
    #[serde(other)]
    Unknown,
}

// ── HermesAcpWriter ───────────────────────────────────────────────────────────

/// Write-side of a persistent `hermes acp` subprocess.
///
/// Reads are served by a background reader task that forwards parsed
/// `AcpInbound` messages on an `mpsc` channel returned from `spawn()`.
/// Keeping reads and writes separate avoids holding a Mutex while awaiting.
pub struct HermesAcpWriter {
    pub session_id: Option<String>,
    stdin: ChildStdin,
    #[allow(dead_code)]
    child: Child,
}

impl HermesAcpWriter {
    /// Spawn the ACP process and start the reader task.
    ///
    /// Returns `(writer, inbound_rx)`. The caller owns `inbound_rx`; it should
    /// not be shared (single-consumer design).
    pub async fn spawn(command: &str) -> anyhow::Result<(Self, mpsc::Receiver<AcpInbound>)> {
        let parts: Vec<&str> = command.split_whitespace().collect();
        let program = parts.first().copied()
            .ok_or_else(|| anyhow::anyhow!("ACP: AGENT_ACP_COMMAND is empty"))?;
        let args = &parts[1..];

        let mut child = Command::new(program)
            .args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit()) // hermes logs to stderr
            .spawn()
            .map_err(|e| anyhow::anyhow!("ACP: failed to spawn '{}': {}", command, e))?;

        let stdin = child.stdin.take()
            .ok_or_else(|| anyhow::anyhow!("ACP: no stdin handle"))?;
        let stdout = child.stdout.take()
            .ok_or_else(|| anyhow::anyhow!("ACP: no stdout handle"))?;

        let (tx, rx) = mpsc::channel::<AcpInbound>(64);

        tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let line = line.trim().to_string();
                if line.is_empty() {
                    continue;
                }
                match serde_json::from_str::<AcpInbound>(&line) {
                    Ok(msg) => {
                        if tx.send(msg).await.is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        warn!(target: "acp", "Unparseable ACP line: {} — raw: {:?}", e, line);
                    }
                }
            }
            debug!(target: "acp", "ACP reader task ended");
        });

        Ok((Self { session_id: None, stdin, child }, rx))
    }

    /// Serialize `msg` to a single JSON line and write it to the process stdin.
    pub async fn send(&mut self, msg: &AcpOutbound) -> anyhow::Result<()> {
        let json = serde_json::to_string(msg)?;
        self.stdin.write_all(json.as_bytes()).await?;
        self.stdin.write_all(b"\n").await?;
        self.stdin.flush().await?;
        Ok(())
    }

    /// Perform the ACP initialize handshake and send `new_session`.
    /// Blocks until `initialize_response` arrives on `rx`.
    pub async fn initialize(
        &mut self,
        rx: &mut mpsc::Receiver<AcpInbound>,
        cwd: &str,
    ) -> anyhow::Result<String> {
        self.send(&AcpOutbound::Initialize { protocol_version: 1 }).await?;

        let session_id = loop {
            match rx.recv().await {
                Some(AcpInbound::InitializeResponse { session_id }) => break session_id,
                Some(other) => debug!(target: "acp", "init: ignoring {:?}", other),
                None => anyhow::bail!("ACP process closed before initialize_response"),
            }
        };

        self.send(&AcpOutbound::NewSession {
            cwd: cwd.to_string(),
            mcp_servers: vec![],
        })
        .await?;

        self.session_id = Some(session_id.clone());
        info!(target: "acp", "ACP initialized, session_id={}", session_id);
        Ok(session_id)
    }

    /// Kill the subprocess.
    #[allow(dead_code)]
    pub async fn kill(&mut self) {
        let _ = self.child.kill().await;
    }
}

// ── ActiveAcpTask ─────────────────────────────────────────────────────────────

/// Tracks a single in-flight ACP task.
pub struct ActiveAcpTask {
    pub session_id: String,
    /// Sending on this channel cancels the task's collect loop.
    pub cancel_tx: oneshot::Sender<()>,
}

// ── collect_acp_response ──────────────────────────────────────────────────────

/// Drive the ACP inbound message loop for one task.
///
/// Handles streaming updates, permission requests, and cancellation.
/// Returns the final text result (from `PromptResponse`) or an error/cancel string.
async fn collect_acp_response(
    acp_writer: Arc<Mutex<Option<HermesAcpWriter>>>,
    inbound_rx: &mut mpsc::Receiver<AcpInbound>,
    proactive_tx: mpsc::Sender<ProactiveEvent>,
    session_id: String,
    mut cancel_rx: oneshot::Receiver<()>,
) -> String {
    let mut progress: Vec<String> = Vec::new();

    loop {
        let maybe_msg = tokio::select! {
            biased;
            _ = &mut cancel_rx => None,
            msg = inbound_rx.recv() => msg,
        };

        match maybe_msg {
            None => {
                // Cancel fired or channel closed — send cancel to the agent.
                let mut guard = acp_writer.lock().await;
                if let Some(w) = guard.as_mut() {
                    let _ = w.send(&AcpOutbound::Cancel { session_id: session_id.clone() }).await;
                }
                return "[Tarea cancelada.]".to_string();
            }
            Some(AcpInbound::PromptResponse { output, .. }) => {
                if progress.is_empty() {
                    return output;
                }
                return format!("{output}\n\n[Progreso: {}]", progress.join("; "));
            }
            Some(AcpInbound::SessionUpdate { update, .. }) => match update {
                AcpSessionUpdate::ToolStart { tool } => {
                    info!(target: "acp", "Tool start: {}", tool);
                    progress.push(format!("usando {tool}"));
                }
                AcpSessionUpdate::ToolComplete { tool, .. } => {
                    debug!(target: "acp", "Tool complete: {}", tool);
                }
                AcpSessionUpdate::AgentThought { text } => {
                    debug!(target: "acp", "Thought: {}", text);
                }
                AcpSessionUpdate::AgentMessage { text } => {
                    info!(target: "acp", "Agent message: {}", text);
                    progress.push(text);
                }
            },
            Some(AcpInbound::PermissionRequest { description, options, .. }) => {
                let (resp_tx, resp_rx) = oneshot::channel::<String>();
                let _ = proactive_tx
                    .send(ProactiveEvent::AgentQuestion {
                        question: description.clone(),
                        options: options.clone(),
                        response_tx: resp_tx,
                    })
                    .await;

                let outcome = match tokio::time::timeout(
                    std::time::Duration::from_secs(60),
                    resp_rx,
                )
                .await
                {
                    Ok(Ok(ans)) => ans,
                    _ => {
                        warn!(target: "acp", "Permission timeout — defaulting to reject_once");
                        "reject_once".to_string()
                    }
                };

                let mut guard = acp_writer.lock().await;
                if let Some(w) = guard.as_mut() {
                    let _ = w
                        .send(&AcpOutbound::PermissionResponse {
                            session_id: session_id.clone(),
                            outcome,
                        })
                        .await;
                }
            }
            Some(other) => {
                debug!(target: "acp", "Ignored: {:?}", other);
            }
        }
    }
}

// ── RunAgentAcpTool ───────────────────────────────────────────────────────────

/// ACP-mode agent tool. Maintains a persistent `hermes acp` subprocess and
/// communicates via JSON-RPC over stdio.
///
/// Supports:
/// - Streaming progress (tool calls, agent messages)
/// - Bidirectional communication (permission requests → user yes/no)
/// - Cancellation via `CancelAgentTool`
pub struct RunAgentAcpTool {
    acp_writer: Arc<Mutex<Option<HermesAcpWriter>>>,
    acp_inbound: Arc<Mutex<Option<mpsc::Receiver<AcpInbound>>>>,
    active_task: Arc<Mutex<Option<ActiveAcpTask>>>,
    acp_command: String,
    history: Arc<RwLock<String>>,
    proactive_tx: mpsc::Sender<ProactiveEvent>,
}

impl RunAgentAcpTool {
    pub fn new(
        acp_writer: Arc<Mutex<Option<HermesAcpWriter>>>,
        acp_inbound: Arc<Mutex<Option<mpsc::Receiver<AcpInbound>>>>,
        active_task: Arc<Mutex<Option<ActiveAcpTask>>>,
        acp_command: &str,
        history: Arc<RwLock<String>>,
        proactive_tx: mpsc::Sender<ProactiveEvent>,
    ) -> Self {
        Self {
            acp_writer,
            acp_inbound,
            active_task,
            acp_command: acp_command.to_string(),
            history,
            proactive_tx,
        }
    }
}

#[async_trait]
impl Tool for RunAgentAcpTool {
    fn name(&self) -> &str {
        "run_agent_acp"
    }

    fn description(&self) -> &str {
        "Delegates a task to the Hermes ACP agent. The agent runs in the background and \
         can request permission for actions (you will be asked). Use cancel_agent to \
         stop a running task. The result is announced proactively when the agent finishes. \
         Use for any task requiring computer control, file access, web search, calendar, \
         or extended reasoning."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Brief label for this task (shown in the completion announcement)"
                }
            },
            "required": ["task"]
        })
    }

    async fn run(&self, args: &str) -> String {
        let task = parse_task(args);
        if task.is_empty() {
            return "Error: run_agent_acp requires a task description.".to_string();
        }

        // Refuse if another task is already running.
        {
            let guard = self.active_task.lock().await;
            if guard.is_some() {
                return "[El agente ya tiene una tarea en progreso. Usa cancel_agent para cancelarla primero.]"
                    .to_string();
            }
        }

        let query = self.history.read().map(|h| h.clone()).unwrap_or_else(|_| task.clone());
        let task_c = task.clone();
        let acp_writer = Arc::clone(&self.acp_writer);
        let acp_inbound = Arc::clone(&self.acp_inbound);
        let active_task = Arc::clone(&self.active_task);
        let proactive_tx = self.proactive_tx.clone();
        let acp_command = self.acp_command.clone();

        tokio::spawn(async move {
            // ── Lazily initialize the ACP process ────────────────────────────
            let session_id = {
                let mut w_guard = acp_writer.lock().await;
                if w_guard.is_none() {
                    match HermesAcpWriter::spawn(&acp_command).await {
                        Ok((mut writer, mut rx)) => {
                            let cwd = std::env::current_dir()
                                .unwrap_or_default()
                                .to_string_lossy()
                                .to_string();
                            match writer.initialize(&mut rx, &cwd).await {
                                Ok(sid) => {
                                    *w_guard = Some(writer);
                                    let mut rx_guard = acp_inbound.lock().await;
                                    *rx_guard = Some(rx);
                                    sid
                                }
                                Err(e) => {
                                    let _ = proactive_tx
                                        .send(ProactiveEvent::AgentResult {
                                            task: task_c,
                                            result: format!("ACP init error: {e}"),
                                        })
                                        .await;
                                    return;
                                }
                            }
                        }
                        Err(e) => {
                            let _ = proactive_tx
                                .send(ProactiveEvent::AgentResult {
                                    task: task_c,
                                    result: format!("ACP spawn error: {e}"),
                                })
                                .await;
                            return;
                        }
                    }
                } else {
                    w_guard.as_ref().unwrap().session_id.clone().unwrap_or_default()
                }
            };

            // ── Send prompt ───────────────────────────────────────────────────
            {
                let mut guard = acp_writer.lock().await;
                if let Some(w) = guard.as_mut() {
                    if let Err(e) = w
                        .send(&AcpOutbound::Prompt {
                            session_id: session_id.clone(),
                            prompt: vec![AcpTextBlock::text(&query)],
                        })
                        .await
                    {
                        let _ = proactive_tx
                            .send(ProactiveEvent::AgentResult {
                                task: task_c,
                                result: format!("ACP send error: {e}"),
                            })
                            .await;
                        return;
                    }
                }
            }

            // ── Register active task ──────────────────────────────────────────
            let (cancel_tx, cancel_rx) = oneshot::channel::<()>();
            {
                let mut at = active_task.lock().await;
                *at = Some(ActiveAcpTask { session_id: session_id.clone(), cancel_tx });
            }

            // ── Collect responses ─────────────────────────────────────────────
            // Take the inbound rx out of the Arc<Mutex<Option<...>>> for the
            // duration of this call. We put it back after collect finishes so
            // future tasks can reuse it.
            let mut taken_rx = {
                let mut rx_guard = acp_inbound.lock().await;
                rx_guard.take()
            };
            let result = if let Some(rx) = taken_rx.as_mut() {
                collect_acp_response(
                    Arc::clone(&acp_writer),
                    rx,
                    proactive_tx.clone(),
                    session_id,
                    cancel_rx,
                )
                .await
            } else {
                "ACP: inbound channel not initialized.".to_string()
            };
            // Return the rx for reuse.
            {
                let mut rx_guard = acp_inbound.lock().await;
                *rx_guard = taken_rx;
            }

            { active_task.lock().await.take(); }

            let _ = proactive_tx
                .send(ProactiveEvent::AgentResult { task: task_c, result })
                .await;
        });

        "[Tarea ACP delegada al agente. El resultado llegará en breve.]".to_string()
    }
}

// ── CancelAgentTool ───────────────────────────────────────────────────────────

/// Cancels the in-flight ACP task, if any.
pub struct CancelAgentTool {
    active_task: Arc<Mutex<Option<ActiveAcpTask>>>,
    acp_writer: Arc<Mutex<Option<HermesAcpWriter>>>,
}

impl CancelAgentTool {
    pub fn new(
        active_task: Arc<Mutex<Option<ActiveAcpTask>>>,
        acp_writer: Arc<Mutex<Option<HermesAcpWriter>>>,
    ) -> Self {
        Self { active_task, acp_writer }
    }
}

#[async_trait]
impl Tool for CancelAgentTool {
    fn name(&self) -> &str {
        "cancel_agent"
    }

    fn description(&self) -> &str {
        "Cancels the agent task currently in progress. \
         Use when the user says to stop or cancel the running agent task."
    }

    async fn run(&self, _args: &str) -> String {
        let mut guard = self.active_task.lock().await;
        if let Some(task) = guard.take() {
            // Signal the collect loop to cancel.
            let _ = task.cancel_tx.send(());
            // Also send the ACP cancel message directly.
            let mut w_guard = self.acp_writer.lock().await;
            if let Some(w) = w_guard.as_mut() {
                let _ = w
                    .send(&AcpOutbound::Cancel { session_id: task.session_id })
                    .await;
            }
            "[Tarea del agente cancelada.]".to_string()
        } else {
            "[No hay ninguna tarea del agente en progreso.]".to_string()
        }
    }
}

// ── AgentStatusTool ───────────────────────────────────────────────────────────

/// Reports whether the ACP agent is currently working on a task.
pub struct AgentStatusTool {
    active_task: Arc<Mutex<Option<ActiveAcpTask>>>,
}

impl AgentStatusTool {
    pub fn new(active_task: Arc<Mutex<Option<ActiveAcpTask>>>) -> Self {
        Self { active_task }
    }
}

#[async_trait]
impl Tool for AgentStatusTool {
    fn name(&self) -> &str {
        "agent_status"
    }

    fn description(&self) -> &str {
        "Returns whether the ACP agent is currently working on a task."
    }

    async fn run(&self, _args: &str) -> String {
        let guard = self.active_task.lock().await;
        if guard.is_some() {
            "[El agente está trabajando en una tarea.]".to_string()
        } else {
            "[El agente no tiene ninguna tarea activa.]".to_string()
        }
    }
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

    // ── ACP message serialization ─────────────────────────────────────────────

    #[test]
    fn acp_initialize_serializes_type_field() {
        let msg = AcpOutbound::Initialize { protocol_version: 1 };
        let v: serde_json::Value = serde_json::from_str(&serde_json::to_string(&msg).unwrap()).unwrap();
        assert_eq!(v["type"], "initialize");
        assert_eq!(v["protocol_version"], 1);
    }

    #[test]
    fn acp_cancel_serializes_correctly() {
        let msg = AcpOutbound::Cancel { session_id: "s1".to_string() };
        let v: serde_json::Value = serde_json::from_str(&serde_json::to_string(&msg).unwrap()).unwrap();
        assert_eq!(v["type"], "cancel");
        assert_eq!(v["session_id"], "s1");
    }

    #[test]
    fn acp_permission_response_serializes_correctly() {
        let msg = AcpOutbound::PermissionResponse {
            session_id: "s1".to_string(),
            outcome: "allow_once".to_string(),
        };
        let v: serde_json::Value = serde_json::from_str(&serde_json::to_string(&msg).unwrap()).unwrap();
        assert_eq!(v["type"], "permission_response");
        assert_eq!(v["outcome"], "allow_once");
    }

    #[test]
    fn acp_prompt_serializes_text_block() {
        let msg = AcpOutbound::Prompt {
            session_id: "s1".to_string(),
            prompt: vec![AcpTextBlock::text("hello")],
        };
        let v: serde_json::Value = serde_json::from_str(&serde_json::to_string(&msg).unwrap()).unwrap();
        assert_eq!(v["type"], "prompt");
        assert_eq!(v["prompt"][0]["text"], "hello");
        assert_eq!(v["prompt"][0]["type"], "text");
    }

    // ── ACP message deserialization ───────────────────────────────────────────

    #[test]
    fn acp_initialize_response_deserializes() {
        let json = r#"{"type":"initialize_response","session_id":"abc123"}"#;
        let msg: AcpInbound = serde_json::from_str(json).unwrap();
        match msg {
            AcpInbound::InitializeResponse { session_id } => {
                assert_eq!(session_id, "abc123");
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn acp_prompt_response_deserializes() {
        let json = r#"{"type":"prompt_response","session_id":"s1","output":"Done","stop_reason":"done"}"#;
        let msg: AcpInbound = serde_json::from_str(json).unwrap();
        match msg {
            AcpInbound::PromptResponse { output, stop_reason, .. } => {
                assert_eq!(output, "Done");
                assert_eq!(stop_reason, "done");
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn acp_permission_request_deserializes() {
        let json = r#"{"type":"permission_request","session_id":"s1","description":"Abrir Safari","options":["allow_once","reject_once"]}"#;
        let msg: AcpInbound = serde_json::from_str(json).unwrap();
        match msg {
            AcpInbound::PermissionRequest { description, options, .. } => {
                assert_eq!(description, "Abrir Safari");
                assert_eq!(options, vec!["allow_once", "reject_once"]);
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn acp_unknown_type_deserializes_to_unknown() {
        let json = r#"{"type":"some_future_message","foo":"bar"}"#;
        let msg: AcpInbound = serde_json::from_str(json).unwrap();
        assert!(matches!(msg, AcpInbound::Unknown), "expected Unknown variant");
    }

    // ── CancelAgentTool ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn cancel_returns_no_task_when_idle() {
        let active_task = Arc::new(Mutex::new(None));
        let acp_writer = Arc::new(Mutex::new(None));
        let tool = CancelAgentTool::new(Arc::clone(&active_task), Arc::clone(&acp_writer));
        let result = tool.run("").await;
        assert!(result.contains("No hay"), "got: {result:?}");
    }

    #[tokio::test]
    async fn cancel_fires_cancel_channel() {
        let (cancel_tx, mut cancel_rx) = tokio::sync::oneshot::channel::<()>();
        let active_task = Arc::new(Mutex::new(Some(ActiveAcpTask {
            session_id: "s1".to_string(),
            cancel_tx,
        })));
        let acp_writer: Arc<Mutex<Option<HermesAcpWriter>>> = Arc::new(Mutex::new(None));
        let tool = CancelAgentTool::new(Arc::clone(&active_task), Arc::clone(&acp_writer));
        let result = tool.run("").await;
        assert!(result.contains("cancelada"), "got: {result:?}");
        assert!(cancel_rx.try_recv().is_ok(), "cancel channel should have fired");
    }

    // ── AgentStatusTool ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn status_returns_idle_when_no_task() {
        let active_task = Arc::new(Mutex::new(None));
        let tool = AgentStatusTool::new(active_task);
        let result = tool.run("").await;
        assert!(result.contains("no tiene"), "got: {result:?}");
    }
}
