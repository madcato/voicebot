use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{mpsc, oneshot, Mutex};
use tracing::{debug, info, warn};

use super::Tool;
use crate::agents::ProactiveEvent;

// ── History formatting ────────────────────────────────────────────────────────

/// Format conversation messages as a human-readable chat history for the agent.
/// Only user and assistant turns with text content are included; system,
/// tool-call, and tool-result messages are omitted.
///
/// The result is passed as the `-q` argument to the agent CLI so it has full
/// conversational context when processing the delegation request.
pub fn format_history(messages: &[serde_json::Value]) -> String {
    messages
        .iter()
        .filter_map(|m| {
            let role = m["role"].as_str()?;
            let content = m["content"].as_str()?; // skips null-content tool_call messages
            match role {
                "user" => Some(format!("[User]: {content}")),
                "assistant" => Some(format!("[Jarvis]: {content}")),
                _ => None,
            }
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

// ── JSON-RPC 2.0 helpers ─────────────────────────────────────────────────────

fn jsonrpc_request(id: u64, method: &str, params: Value) -> Value {
    serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": method,
        "params": params
    })
}

fn jsonrpc_notification(method: &str, params: Value) -> Value {
    serde_json::json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params
    })
}

/// A parsed JSON-RPC 2.0 message received from the ACP process.
#[derive(Debug, Clone)]
pub enum JsonRpcMessage {
    /// A response to a request we sent (matched by id).
    Response {
        id: u64,
        result: Option<Value>,
        error: Option<Value>,
    },
    /// A request from the server that expects a response (has id + method).
    Request {
        id: u64,
        method: String,
        params: Option<Value>,
    },
    /// A notification from the server (has method but no id).
    Notification {
        method: String,
        params: Option<Value>,
    },
}

fn parse_jsonrpc(v: &Value) -> Option<JsonRpcMessage> {
    let method = v.get("method").and_then(|m| m.as_str()).map(String::from);
    let id = v.get("id").and_then(|i| i.as_u64());

    match (method, id) {
        // Request from server (has both method and id)
        (Some(method), Some(id)) => Some(JsonRpcMessage::Request {
            id,
            method,
            params: v.get("params").cloned(),
        }),
        // Notification from server (has method, no id)
        (Some(method), None) => Some(JsonRpcMessage::Notification {
            method,
            params: v.get("params").cloned(),
        }),
        // Response to our request (has id, no method)
        (None, Some(id)) => Some(JsonRpcMessage::Response {
            id,
            result: v.get("result").cloned(),
            error: v.get("error").cloned(),
        }),
        _ => None,
    }
}

// ── RunAgentTool ──────────────────────────────────────────────────────────────

/// Unified agent delegation tool.
///
/// Supports two modes (selected by the `mode` field):
/// - `"cli"` — spawns the agent as a one-shot CLI subprocess (fire-and-forget).
/// - `"acp"` — maintains a persistent ACP subprocess via JSON-RPC 2.0 over stdio.
///
/// Additionally handles two inline commands that require no subprocess:
/// - `run_agent: cancel` — cancels the currently running ACP task.
/// - `run_agent: status` — reports whether the ACP agent is busy.
pub struct RunAgentTool {
    /// CLI executable (and optional args) — used in `"cli"` mode.
    command: Option<String>,
    /// Persistent ACP process write-side — lazily initialized on first use.
    acp_writer: Arc<Mutex<Option<HermesAcpWriter>>>,
    /// Inbound message channel from the ACP process.
    acp_inbound: Arc<Mutex<Option<mpsc::Receiver<JsonRpcMessage>>>>,
    /// Currently executing ACP task, if any.
    active_task: Arc<Mutex<Option<ActiveAcpTask>>>,
    /// Formatted conversation history shared with the agent for context.
    history: Arc<RwLock<String>>,
    /// Channel for delivering agent results back to the main pipeline.
    proactive_tx: mpsc::Sender<ProactiveEvent>,
    /// `"cli"` or `"acp"`.
    mode: String,
    /// ACP subprocess command — used in `"acp"` mode.
    acp_command: String,
}

impl RunAgentTool {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        command: Option<String>,
        acp_writer: Arc<Mutex<Option<HermesAcpWriter>>>,
        acp_inbound: Arc<Mutex<Option<mpsc::Receiver<JsonRpcMessage>>>>,
        active_task: Arc<Mutex<Option<ActiveAcpTask>>>,
        history: Arc<RwLock<String>>,
        proactive_tx: mpsc::Sender<ProactiveEvent>,
        mode: String,
        acp_command: String,
    ) -> Self {
        Self {
            command,
            acp_writer,
            acp_inbound,
            active_task,
            history,
            proactive_tx,
            mode,
            acp_command,
        }
    }

    /// Cancel the in-flight ACP task, if any.
    async fn cancel(&self) -> String {
        let mut guard = self.active_task.lock().await;
        if let Some(task) = guard.take() {
            let _ = task.cancel_tx.send(());
            let mut w_guard = self.acp_writer.lock().await;
            if let Some(w) = w_guard.as_mut() {
                let _ = w.send_cancel(task.prompt_request_id).await;
            }
            "[Tarea del agente cancelada.]".to_string()
        } else {
            "[No hay ninguna tarea del agente en progreso.]".to_string()
        }
    }

    /// Report whether the ACP agent is currently busy.
    async fn status(&self) -> String {
        let guard = self.active_task.lock().await;
        if guard.is_some() {
            "[El agente está trabajando en una tarea.]".to_string()
        } else {
            "[El agente no tiene ninguna tarea activa.]".to_string()
        }
    }

    /// CLI mode: spawn agent as one-shot subprocess, deliver result proactively.
    async fn run_cli(&self, task: String) -> String {
        let command = match &self.command {
            Some(c) => c.clone(),
            None => return "Error: CLI agent command not configured.".to_string(),
        };
        let query = self.history.read().map(|h| h.clone()).unwrap_or_else(|_| task.clone());
        let proactive_tx = self.proactive_tx.clone();

        tokio::spawn(async move {
            info!("RunAgentTool(cli): task started: {:?}", task);
            let result = call_agent(command, query).await;
            info!("RunAgentTool(cli): task complete ({} chars): {:?}", result.len(), result);
            let _ = proactive_tx.send(ProactiveEvent::AgentResult { task, result }).await;
        });

        "[Tarea delegada al agente. El resultado llegará en breve.]".to_string()
    }

    /// ACP mode: send task to persistent ACP subprocess, deliver result proactively.
    async fn run_acp(&self, task: String) -> String {
        // Refuse if another task is already running.
        {
            let guard = self.active_task.lock().await;
            if guard.is_some() {
                return "[El agente ya tiene una tarea en progreso. Usa 'run_agent: cancel' para cancelarla primero.]"
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
            let prompt_request_id = {
                let mut guard = acp_writer.lock().await;
                if let Some(w) = guard.as_mut() {
                    match w.send_prompt(&session_id, &query).await {
                        Ok(id) => id,
                        Err(e) => {
                            let _ = proactive_tx
                                .send(ProactiveEvent::AgentResult {
                                    task: task_c,
                                    result: format!("ACP send error: {e}"),
                                })
                                .await;
                            return;
                        }
                    }
                } else {
                    let _ = proactive_tx
                        .send(ProactiveEvent::AgentResult {
                            task: task_c,
                            result: "ACP: writer not initialized.".to_string(),
                        })
                        .await;
                    return;
                }
            };

            // ── Register active task ──────────────────────────────────────────
            let (cancel_tx, cancel_rx) = oneshot::channel::<()>();
            {
                let mut at = active_task.lock().await;
                *at = Some(ActiveAcpTask {
                    session_id: session_id.clone(),
                    prompt_request_id,
                    cancel_tx,
                });
            }

            // ── Collect responses ─────────────────────────────────────────────
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
                    prompt_request_id,
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

#[async_trait]
impl Tool for RunAgentTool {
    fn name(&self) -> &str {
        "run_agent"
    }

    fn description(&self) -> &str {
        "Delega una tarea al agente externo (Hermes). El agente tiene acceso a \
         herramientas de computadora, archivos, web, calendario y razonamiento extendido. \
         El resultado se anuncia de forma proactiva cuando el agente termina. \
         Para cancelar una tarea en curso usa run_agent con task='cancel'. \
         Para consultar el estado usa task='status'."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Descripción breve de la tarea a delegar, o 'cancel' para \
                                    cancelar la tarea en curso, o 'status' para consultar el estado."
                }
            },
            "required": ["task"]
        })
    }

    async fn run(&self, args: &str) -> String {
        let task = parse_task(args);
        if task.is_empty() {
            return "Error: run_agent requires a task description.".to_string();
        }

        // Inline commands — no subprocess needed.
        let lower = task.trim().to_lowercase();
        if lower.starts_with("cancel") {
            return self.cancel().await;
        }
        if lower.starts_with("status") {
            return self.status().await;
        }

        match self.mode.as_str() {
            "acp" => self.run_acp(task).await,
            _ => self.run_cli(task).await,
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn parse_task(args: &str) -> String {
    serde_json::from_str::<serde_json::Value>(args)
        .ok()
        .and_then(|v| v["task"].as_str().map(String::from))
        .unwrap_or_else(|| args.to_string())
}

// ── HermesAcpWriter ───────────────────────────────────────────────────────────

/// Write-side of a persistent `hermes acp` subprocess using JSON-RPC 2.0.
///
/// Reads are served by a background reader task that forwards parsed
/// `JsonRpcMessage` messages on an `mpsc` channel returned from `spawn()`.
pub struct HermesAcpWriter {
    pub session_id: Option<String>,
    stdin: ChildStdin,
    #[allow(dead_code)]
    child: Child,
    next_id: u64,
    /// When true, raw JSON-RPC messages are printed to stderr.
    pub verbose: Arc<AtomicBool>,
}

impl HermesAcpWriter {
    /// Spawn the ACP process and start the reader task.
    ///
    /// Returns `(writer, inbound_rx)`. The caller owns `inbound_rx`; it should
    /// not be shared (single-consumer design).
    pub async fn spawn(command: &str) -> anyhow::Result<(Self, mpsc::Receiver<JsonRpcMessage>)> {
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

        let (tx, rx) = mpsc::channel::<JsonRpcMessage>(64);
        let verbose = Arc::new(AtomicBool::new(false));
        let verbose_reader = Arc::clone(&verbose);

        tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let line = line.trim().to_string();
                if line.is_empty() {
                    continue;
                }
                if verbose_reader.load(Ordering::Relaxed) {
                    eprintln!("\x1b[2m← {line}\x1b[0m");
                }
                match serde_json::from_str::<Value>(&line) {
                    Ok(v) => {
                        if let Some(msg) = parse_jsonrpc(&v) {
                            if tx.send(msg).await.is_err() {
                                break;
                            }
                        } else {
                            warn!(target: "acp", "Unrecognized JSON-RPC message: {:?}", line);
                        }
                    }
                    Err(e) => {
                        warn!(target: "acp", "Unparseable ACP line: {} — raw: {:?}", e, line);
                    }
                }
            }
            debug!(target: "acp", "ACP reader task ended");
        });

        Ok((Self { session_id: None, stdin, child, next_id: 0, verbose }, rx))
    }

    /// Write a raw JSON value as a newline-delimited line to the process stdin.
    pub async fn write_json(&mut self, msg: &Value) -> anyhow::Result<()> {
        let json = serde_json::to_string(msg)?;
        if self.verbose.load(Ordering::Relaxed) {
            eprintln!("\x1b[2m→ {json}\x1b[0m");
        }
        self.stdin.write_all(json.as_bytes()).await?;
        self.stdin.write_all(b"\n").await?;
        self.stdin.flush().await?;
        Ok(())
    }

    /// Send a JSON-RPC request and return the assigned request id.
    pub async fn send_request(&mut self, method: &str, params: Value) -> anyhow::Result<u64> {
        let id = self.next_id;
        self.next_id += 1;
        let msg = jsonrpc_request(id, method, params);
        debug!(target: "acp", "→ {}", serde_json::to_string(&msg).unwrap_or_default());
        self.write_json(&msg).await?;
        Ok(id)
    }

    /// Send a JSON-RPC notification (no id, no response expected).
    pub async fn send_notification(&mut self, method: &str, params: Value) -> anyhow::Result<()> {
        let msg = jsonrpc_notification(method, params);
        debug!(target: "acp", "→ {}", serde_json::to_string(&msg).unwrap_or_default());
        self.write_json(&msg).await?;
        Ok(())
    }

    /// Send a JSON-RPC response to a request from the server.
    pub async fn send_response(&mut self, id: u64, result: Value) -> anyhow::Result<()> {
        let msg = serde_json::json!({"jsonrpc": "2.0", "id": id, "result": result});
        debug!(target: "acp", "→ {}", serde_json::to_string(&msg).unwrap_or_default());
        self.write_json(&msg).await?;
        Ok(())
    }

    /// Perform the full ACP initialize + session/new handshake.
    /// Blocks until both responses arrive on `rx`.
    pub async fn initialize(
        &mut self,
        rx: &mut mpsc::Receiver<JsonRpcMessage>,
        cwd: &str,
    ) -> anyhow::Result<String> {
        // ── Step 1: initialize ───────────────────────────────────────────────
        let init_id = self.send_request("initialize", serde_json::json!({
            "protocolVersion": 1,
            "clientCapabilities": {},
            "clientInfo": {"name": "voicebot", "version": "0.1.0"}
        })).await?;

        // Wait for initialize response
        loop {
            match rx.recv().await {
                Some(JsonRpcMessage::Response { id, error, .. }) if id == init_id => {
                    if let Some(err) = error {
                        anyhow::bail!("ACP initialize error: {}", err);
                    }
                    debug!(target: "acp", "initialize response received");
                    break;
                }
                Some(other) => debug!(target: "acp", "init: ignoring {:?}", other),
                None => anyhow::bail!("ACP process closed before initialize response"),
            }
        }

        // ── Step 2: session/new ──────────────────────────────────────────────
        let session_id = self.send_request("session/new", serde_json::json!({
            "cwd": cwd,
            "mcpServers": []
        })).await?;

        // Wait for session/new response with sessionId
        let sid = loop {
            match rx.recv().await {
                Some(JsonRpcMessage::Response { id, result, error, .. }) if id == session_id => {
                    if let Some(err) = error {
                        anyhow::bail!("ACP session/new error: {}", err);
                    }
                    let result = result.unwrap_or_default();
                    let sid = result["sessionId"]
                        .as_str()
                        .ok_or_else(|| anyhow::anyhow!("ACP session/new response missing sessionId"))?
                        .to_string();
                    break sid;
                }
                Some(other) => debug!(target: "acp", "session/new: ignoring {:?}", other),
                None => anyhow::bail!("ACP process closed before session/new response"),
            }
        };

        self.session_id = Some(sid.clone());
        info!(target: "acp", "ACP initialized, sessionId={}", sid);
        Ok(sid)
    }

    /// Send a session/prompt request and return the request id.
    pub async fn send_prompt(&mut self, session_id: &str, text: &str) -> anyhow::Result<u64> {
        self.send_request("session/prompt", serde_json::json!({
            "sessionId": session_id,
            "prompt": [{"type": "text", "text": text}]
        })).await
    }

    /// Send a session/cancel notification for a running prompt request.
    pub async fn send_cancel(&mut self, request_id: u64) -> anyhow::Result<()> {
        self.send_notification("session/cancel", serde_json::json!({
            "requestId": request_id
        })).await
    }

    /// Create a new session (without re-initializing the process).
    pub async fn send_new_session(&mut self, cwd: &str) -> anyhow::Result<u64> {
        self.send_request("session/new", serde_json::json!({
            "cwd": cwd,
            "mcpServers": []
        })).await
    }

    /// Fork an existing session.
    pub async fn send_fork_session(&mut self, session_id: &str, cwd: &str) -> anyhow::Result<u64> {
        self.send_request("session/fork", serde_json::json!({
            "sessionId": session_id,
            "cwd": cwd
        })).await
    }

    /// Load a previous session by ID.
    pub async fn send_load_session(&mut self, session_id: &str, cwd: &str) -> anyhow::Result<u64> {
        self.send_request("session/load", serde_json::json!({
            "sessionId": session_id,
            "cwd": cwd
        })).await
    }

    /// Resume a suspended session.
    pub async fn send_resume_session(&mut self, session_id: &str, cwd: &str) -> anyhow::Result<u64> {
        self.send_request("session/resume", serde_json::json!({
            "sessionId": session_id,
            "cwd": cwd
        })).await
    }

    /// List active sessions.
    pub async fn send_list_sessions(&mut self, cwd: &str) -> anyhow::Result<u64> {
        self.send_request("session/list", serde_json::json!({
            "cwd": cwd
        })).await
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
    /// The JSON-RPC request id for the prompt, used for cancellation.
    pub prompt_request_id: u64,
    /// Sending on this channel cancels the task's collect loop.
    pub cancel_tx: oneshot::Sender<()>,
}

// ── collect_acp_response ──────────────────────────────────────────────────────

/// Drive the ACP inbound message loop for one task.
///
/// Handles streaming session/update notifications, permission requests, and
/// cancellation. Returns the accumulated text result or an error/cancel string.
async fn collect_acp_response(
    acp_writer: Arc<Mutex<Option<HermesAcpWriter>>>,
    inbound_rx: &mut mpsc::Receiver<JsonRpcMessage>,
    proactive_tx: mpsc::Sender<ProactiveEvent>,
    _session_id: String,
    prompt_request_id: u64,
    mut cancel_rx: oneshot::Receiver<()>,
) -> String {
    let mut accumulated_text = String::new();
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
                    let _ = w.send_cancel(prompt_request_id).await;
                }
                return "[Tarea cancelada.]".to_string();
            }
            // ── Response to our prompt request → task complete ─────────────
            Some(JsonRpcMessage::Response { id, result, error }) if id == prompt_request_id => {
                if let Some(err) = error {
                    return format!("ACP error: {}", err);
                }
                let stop_reason = result
                    .as_ref()
                    .and_then(|r| r["stopReason"].as_str())
                    .unwrap_or("unknown");
                debug!(target: "acp", "Prompt complete, stopReason={}", stop_reason);

                if accumulated_text.is_empty() && !progress.is_empty() {
                    return format!("[Progreso: {}]", progress.join("; "));
                }
                if !accumulated_text.is_empty() && !progress.is_empty() {
                    return format!("{}\n\n[Progreso: {}]", accumulated_text.trim(), progress.join("; "));
                }
                if accumulated_text.is_empty() {
                    return format!("[Agente terminó con stopReason={stop_reason}]");
                }
                return accumulated_text.trim().to_string();
            }
            // ── session/update notification → streaming content ───────────
            Some(JsonRpcMessage::Notification { method, params }) if method == "session/update" => {
                let params = params.unwrap_or_default();
                let update = &params["update"];
                let session_update = update["sessionUpdate"].as_str().unwrap_or("");

                match session_update {
                    "agent_message_chunk" => {
                        if let Some(text) = update["content"]["text"].as_str() {
                            accumulated_text.push_str(text);
                            debug!(target: "acp", "Agent chunk: {}", text);
                        }
                    }
                    "agent_thought_chunk" => {
                        if let Some(text) = update["content"]["text"].as_str() {
                            debug!(target: "acp", "Thought: {}", text);
                        }
                    }
                    "tool_call" => {
                        let tool_name = update["name"].as_str().unwrap_or("unknown");
                        info!(target: "acp", "Tool start: {}", tool_name);
                        progress.push(format!("usando {tool_name}"));
                    }
                    "tool_call_update" => {
                        let tool_name = update["name"].as_str().unwrap_or("unknown");
                        debug!(target: "acp", "Tool update: {}", tool_name);
                    }
                    other => {
                        debug!(target: "acp", "Ignored session update: {}", other);
                    }
                }
            }
            // ── session/request_permission request → auto-allow or ask user ─
            Some(JsonRpcMessage::Request { id, method, params }) if method == "session/request_permission" => {
                let params = params.unwrap_or_default();

                // Extract permission options for the user
                let options: Vec<String> = params["options"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|o| o["optionId"].as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();

                // Build a description from the toolCall info
                let tool_call = &params["toolCall"];
                let description = tool_call["name"]
                    .as_str()
                    .unwrap_or("acción desconocida")
                    .to_string();

                let (resp_tx, resp_rx) = oneshot::channel::<String>();
                let _ = proactive_tx
                    .send(ProactiveEvent::AgentQuestion {
                        question: description,
                        options: options.clone(),
                        response_tx: resp_tx,
                    })
                    .await;

                let outcome_option_id = match tokio::time::timeout(
                    std::time::Duration::from_secs(60),
                    resp_rx,
                )
                .await
                {
                    Ok(Ok(ans)) => ans,
                    _ => {
                        warn!(target: "acp", "Permission timeout — defaulting to cancelled");
                        String::new() // will send cancelled outcome
                    }
                };

                // Build the response: AllowedOutcome or DeniedOutcome
                let result = if outcome_option_id.is_empty() || outcome_option_id == "cancelled" {
                    serde_json::json!({"outcome": "cancelled"})
                } else {
                    serde_json::json!({"outcome": "selected", "optionId": outcome_option_id})
                };

                let mut guard = acp_writer.lock().await;
                if let Some(w) = guard.as_mut() {
                    let _ = w.send_response(id, result).await;
                }
            }
            // ── Unmatched response (different id) ─────────────────────────
            Some(JsonRpcMessage::Response { id, .. }) => {
                debug!(target: "acp", "Ignored response for id={}", id);
            }
            // ── Other notifications / requests ────────────────────────────
            Some(other) => {
                debug!(target: "acp", "Ignored: {:?}", other);
            }
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

    fn cli_tool(command: &str, history: Arc<RwLock<String>>, tx: mpsc::Sender<ProactiveEvent>) -> RunAgentTool {
        RunAgentTool::new(
            Some(command.to_string()),
            Arc::new(Mutex::new(None)),
            Arc::new(Mutex::new(None)),
            Arc::new(Mutex::new(None)),
            history,
            tx,
            "cli".to_string(),
            String::new(),
        )
    }

    fn cancel_tool(
        active_task: Arc<Mutex<Option<ActiveAcpTask>>>,
        acp_writer: Arc<Mutex<Option<HermesAcpWriter>>>,
        tx: mpsc::Sender<ProactiveEvent>,
    ) -> RunAgentTool {
        RunAgentTool::new(
            None,
            acp_writer,
            Arc::new(Mutex::new(None)),
            active_task,
            empty_history(),
            tx,
            "acp".to_string(),
            String::new(),
        )
    }

    // ── format_history ────────────────────────────────────────────────────────

    #[test]
    fn format_history_empty_messages() {
        assert_eq!(format_history(&[]), "");
    }

    #[test]
    fn format_history_single_user_turn() {
        let msgs = vec![serde_json::json!({"role": "user", "content": "hola"})];
        assert_eq!(format_history(&msgs), "[User]: hola");
    }

    #[test]
    fn format_history_user_and_assistant() {
        let msgs = vec![
            serde_json::json!({"role": "user", "content": "hola"}),
            serde_json::json!({"role": "assistant", "content": "hola Daniel"}),
        ];
        assert_eq!(format_history(&msgs), "[User]: hola\n[Jarvis]: hola Daniel");
    }

    #[test]
    fn format_history_skips_system_messages() {
        let msgs = vec![
            serde_json::json!({"role": "system", "content": "Eres Jarvis"}),
            serde_json::json!({"role": "user", "content": "hola"}),
            serde_json::json!({"role": "assistant", "content": "hola"}),
        ];
        let result = format_history(&msgs);
        assert!(!result.contains("Eres Jarvis"), "system message should be excluded");
        assert!(result.contains("[User]: hola"));
    }

    #[test]
    fn format_history_skips_tool_messages() {
        let msgs = vec![
            serde_json::json!({"role": "user", "content": "qué hora es"}),
            serde_json::json!({"role": "tool", "tool_call_id": "c1", "content": "14:30"}),
            serde_json::json!({"role": "assistant", "content": "Son las 14:30"}),
        ];
        let result = format_history(&msgs);
        assert!(!result.contains("14:30\n"), "bare tool result should be excluded");
        assert!(result.contains("[Jarvis]: Son las 14:30"));
    }

    #[test]
    fn format_history_skips_tool_call_assistant_messages() {
        let msgs = vec![
            serde_json::json!({"role": "user", "content": "Activa el modo ambiente"}),
            serde_json::json!({"role": "assistant", "content": serde_json::Value::Null,
                "tool_calls": [{"id": "c1", "type": "function",
                    "function": {"name": "set_conversation_mode", "arguments": "{\"mode\":\"ambient\"}"}}]}),
            serde_json::json!({"role": "tool", "tool_call_id": "c1", "content": "Ambient mode activated."}),
            serde_json::json!({"role": "assistant", "content": "Modo ambiente activado."}),
        ];
        let result = format_history(&msgs);
        assert!(!result.contains("Ambient mode activated."), "tool result should be excluded");
        assert!(result.contains("[Jarvis]: Modo ambiente activado."));
        assert!(result.contains("[User]: Activa el modo ambiente"));
    }

    #[test]
    fn format_history_multiple_turns() {
        let msgs = vec![
            serde_json::json!({"role": "user", "content": "primera"}),
            serde_json::json!({"role": "assistant", "content": "respuesta uno"}),
            serde_json::json!({"role": "user", "content": "segunda"}),
            serde_json::json!({"role": "assistant", "content": "respuesta dos"}),
        ];
        let expected = "[User]: primera\n[Jarvis]: respuesta uno\n[User]: segunda\n[Jarvis]: respuesta dos";
        assert_eq!(format_history(&msgs), expected);
    }

    // ── RunAgentTool — name / description ─────────────────────────────────────

    #[test]
    fn tool_name_and_description() {
        let (tx, _rx) = mpsc::channel::<ProactiveEvent>(8);
        let tool = cli_tool("echo", empty_history(), tx);
        assert_eq!(tool.name(), "run_agent");
        assert!(!tool.description().is_empty());
    }

    // ── RunAgentTool — CLI mode ───────────────────────────────────────────────

    #[tokio::test]
    async fn cli_empty_args_returns_error() {
        let (tx, _rx) = mpsc::channel::<ProactiveEvent>(8);
        let tool = cli_tool("echo", empty_history(), tx);
        let result = tool.run("").await;
        assert!(result.to_lowercase().contains("error"), "got: {result:?}");
    }

    #[tokio::test]
    async fn cli_returns_acknowledgment_immediately() {
        let (tx, _rx) = mpsc::channel::<ProactiveEvent>(8);
        let tool = cli_tool("sleep 2", empty_history(), tx);
        let start = std::time::Instant::now();
        let result = tool.run(r#"{"task": "slow task"}"#).await;
        assert!(start.elapsed().as_millis() < 200, "should return immediately");
        assert!(!result.is_empty(), "should return acknowledgment: {result:?}");
    }

    #[tokio::test]
    async fn cli_delivers_result_to_proactive_channel() {
        let (tx, mut rx) = mpsc::channel::<ProactiveEvent>(8);
        let tool = cli_tool("echo agent_done", empty_history(), tx);
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
    async fn cli_passes_history_in_query() {
        let (tx, mut rx) = mpsc::channel::<ProactiveEvent>(8);
        let hist = history_with("[User]: busca noticias\n[Jarvis]: delegando");
        let tool = cli_tool("echo done", hist, tx);
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
    async fn cli_delivers_error_on_launch_failure() {
        let (tx, mut rx) = mpsc::channel::<ProactiveEvent>(8);
        let tool = cli_tool("__nonexistent__", empty_history(), tx);
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

    // ── RunAgentTool — cancel / status inline commands ────────────────────────

    #[tokio::test]
    async fn cancel_returns_no_task_when_idle() {
        let (tx, _rx) = mpsc::channel::<ProactiveEvent>(8);
        let active_task = Arc::new(Mutex::new(None));
        let acp_writer = Arc::new(Mutex::new(None));
        let tool = cancel_tool(active_task, acp_writer, tx);
        let result = tool.run(r#"{"task": "cancel"}"#).await;
        assert!(result.contains("No hay"), "got: {result:?}");
    }

    #[tokio::test]
    async fn cancel_fires_cancel_channel() {
        let (cancel_tx, mut cancel_rx) = tokio::sync::oneshot::channel::<()>();
        let active_task = Arc::new(Mutex::new(Some(ActiveAcpTask {
            session_id: "s1".to_string(),
            prompt_request_id: 2,
            cancel_tx,
        })));
        let acp_writer: Arc<Mutex<Option<HermesAcpWriter>>> = Arc::new(Mutex::new(None));
        let (tx, _rx) = mpsc::channel::<ProactiveEvent>(8);
        let tool = cancel_tool(active_task, acp_writer, tx);
        let result = tool.run(r#"{"task": "cancel"}"#).await;
        assert!(result.contains("cancelada"), "got: {result:?}");
        assert!(cancel_rx.try_recv().is_ok(), "cancel channel should have fired");
    }

    #[tokio::test]
    async fn status_returns_idle_when_no_task() {
        let (tx, _rx) = mpsc::channel::<ProactiveEvent>(8);
        let active_task = Arc::new(Mutex::new(None));
        let acp_writer = Arc::new(Mutex::new(None));
        let tool = cancel_tool(active_task, acp_writer, tx);
        let result = tool.run(r#"{"task": "status"}"#).await;
        assert!(result.contains("no tiene"), "got: {result:?}");
    }

    // ── JSON-RPC helpers ─────────────────────────────────────────────────────

    #[test]
    fn jsonrpc_request_has_correct_structure() {
        let msg = jsonrpc_request(0, "initialize", serde_json::json!({"protocolVersion": 1}));
        assert_eq!(msg["jsonrpc"], "2.0");
        assert_eq!(msg["id"], 0);
        assert_eq!(msg["method"], "initialize");
        assert_eq!(msg["params"]["protocolVersion"], 1);
    }

    #[test]
    fn jsonrpc_notification_has_no_id() {
        let msg = jsonrpc_notification("session/cancel", serde_json::json!({"requestId": 5}));
        assert_eq!(msg["jsonrpc"], "2.0");
        assert!(msg.get("id").is_none(), "notification must not have id");
        assert_eq!(msg["method"], "session/cancel");
        assert_eq!(msg["params"]["requestId"], 5);
    }

    #[test]
    fn parse_jsonrpc_response() {
        let v: Value = serde_json::from_str(
            r#"{"jsonrpc":"2.0","id":0,"result":{"protocolVersion":1,"agentInfo":{"name":"hermes","version":"0.1.0"}}}"#
        ).unwrap();
        let msg = parse_jsonrpc(&v).unwrap();
        match msg {
            JsonRpcMessage::Response { id, result, error } => {
                assert_eq!(id, 0);
                assert!(result.is_some());
                assert!(error.is_none());
                assert_eq!(result.unwrap()["protocolVersion"], 1);
            }
            other => panic!("expected Response, got: {:?}", other),
        }
    }

    #[test]
    fn parse_jsonrpc_notification() {
        let v: Value = serde_json::from_str(
            r#"{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"s1","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"hello"}}}}"#
        ).unwrap();
        let msg = parse_jsonrpc(&v).unwrap();
        match msg {
            JsonRpcMessage::Notification { method, params } => {
                assert_eq!(method, "session/update");
                let params = params.unwrap();
                assert_eq!(params["sessionId"], "s1");
                assert_eq!(params["update"]["sessionUpdate"], "agent_message_chunk");
                assert_eq!(params["update"]["content"]["text"], "hello");
            }
            other => panic!("expected Notification, got: {:?}", other),
        }
    }

    #[test]
    fn parse_jsonrpc_request_from_server() {
        let v: Value = serde_json::from_str(
            r#"{"jsonrpc":"2.0","id":5,"method":"session/request_permission","params":{"sessionId":"s1","options":[{"optionId":"allow","name":"Allow","kind":"allow"}],"toolCall":{"name":"bash"}}}"#
        ).unwrap();
        let msg = parse_jsonrpc(&v).unwrap();
        match msg {
            JsonRpcMessage::Request { id, method, params } => {
                assert_eq!(id, 5);
                assert_eq!(method, "session/request_permission");
                let params = params.unwrap();
                assert_eq!(params["options"][0]["optionId"], "allow");
                assert_eq!(params["toolCall"]["name"], "bash");
            }
            other => panic!("expected Request, got: {:?}", other),
        }
    }

    #[test]
    fn parse_jsonrpc_error_response() {
        let v: Value = serde_json::from_str(
            r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32600,"message":"Invalid request"}}"#
        ).unwrap();
        let msg = parse_jsonrpc(&v).unwrap();
        match msg {
            JsonRpcMessage::Response { id, result, error } => {
                assert_eq!(id, 1);
                assert!(result.is_none());
                assert!(error.is_some());
                assert_eq!(error.unwrap()["message"], "Invalid request");
            }
            other => panic!("expected Response, got: {:?}", other),
        }
    }

    // ── Initialize request format ────────────────────────────────────────────

    #[test]
    fn initialize_request_uses_camel_case() {
        let msg = jsonrpc_request(0, "initialize", serde_json::json!({
            "protocolVersion": 1,
            "clientCapabilities": {},
            "clientInfo": {"name": "voicebot", "version": "0.1.0"}
        }));
        assert_eq!(msg["params"]["protocolVersion"], 1);
        assert!(msg["params"]["clientCapabilities"].is_object());
        assert_eq!(msg["params"]["clientInfo"]["name"], "voicebot");
    }

    // ── Prompt request format ────────────────────────────────────────────────

    #[test]
    fn prompt_request_uses_session_id_camel_case() {
        let msg = jsonrpc_request(2, "session/prompt", serde_json::json!({
            "sessionId": "abc123",
            "prompt": [{"type": "text", "text": "hello"}]
        }));
        assert_eq!(msg["method"], "session/prompt");
        assert_eq!(msg["params"]["sessionId"], "abc123");
        assert_eq!(msg["params"]["prompt"][0]["type"], "text");
        assert_eq!(msg["params"]["prompt"][0]["text"], "hello");
    }

    // ── Cancel notification format ───────────────────────────────────────────

    #[test]
    fn cancel_notification_uses_request_id() {
        let msg = jsonrpc_notification("session/cancel", serde_json::json!({
            "requestId": 2
        }));
        assert_eq!(msg["method"], "session/cancel");
        assert_eq!(msg["params"]["requestId"], 2);
        assert!(msg.get("id").is_none(), "cancel must be a notification (no id)");
    }

    // ── Permission response format ───────────────────────────────────────────

    #[test]
    fn permission_response_allowed_format() {
        let result = serde_json::json!({"outcome": "selected", "optionId": "allow"});
        let msg = serde_json::json!({"jsonrpc": "2.0", "id": 5, "result": result});
        assert_eq!(msg["result"]["outcome"], "selected");
        assert_eq!(msg["result"]["optionId"], "allow");
    }

    #[test]
    fn permission_response_denied_format() {
        let result = serde_json::json!({"outcome": "cancelled"});
        let msg = serde_json::json!({"jsonrpc": "2.0", "id": 5, "result": result});
        assert_eq!(msg["result"]["outcome"], "cancelled");
    }
}

// ── Integration tests (require running `hermes acp`) ──────────────────────────

#[cfg(test)]
mod integration_tests {
    use super::*;

    /// Full initialize + session/new handshake with a real `hermes acp` process.
    ///
    /// Requires `hermes acp` to be available in PATH.
    /// Run with: `cargo test acp_initialize_handshake -- --ignored --nocapture`
    #[tokio::test]
    #[ignore]
    async fn acp_initialize_handshake() {
        let (mut writer, mut rx) = HermesAcpWriter::spawn("hermes acp")
            .await
            .expect("failed to spawn hermes acp");

        let cwd = std::env::current_dir()
            .unwrap()
            .to_string_lossy()
            .to_string();

        let session_id = writer
            .initialize(&mut rx, &cwd)
            .await
            .expect("initialize failed");

        assert!(!session_id.is_empty(), "session_id should not be empty");
        println!("Got session_id: {session_id}");

        writer.kill().await;
    }

    /// Full flow: initialize → session/new → prompt("say hello") → collect response.
    ///
    /// Requires `hermes acp` to be available in PATH.
    /// Run with: `cargo test acp_simple_prompt -- --ignored --nocapture`
    #[tokio::test]
    #[ignore]
    async fn acp_simple_prompt() {
        let (mut writer, mut rx) = HermesAcpWriter::spawn("hermes acp")
            .await
            .expect("failed to spawn hermes acp");

        let cwd = std::env::current_dir()
            .unwrap()
            .to_string_lossy()
            .to_string();

        let session_id = writer
            .initialize(&mut rx, &cwd)
            .await
            .expect("initialize failed");

        let prompt_id = writer
            .send_prompt(&session_id, "Say hello in one sentence.")
            .await
            .expect("send_prompt failed");

        println!("Sent prompt with request id={prompt_id}");

        let writer = Arc::new(Mutex::new(Some(writer)));
        let (proactive_tx, _proactive_rx) = tokio::sync::mpsc::channel(8);
        let (_cancel_tx, cancel_rx) = oneshot::channel::<()>();

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(60),
            collect_acp_response(
                writer.clone(),
                &mut rx,
                proactive_tx,
                session_id,
                prompt_id,
                cancel_rx,
            ),
        )
        .await
        .expect("timed out waiting for agent response");

        println!("Agent response: {result}");
        assert!(!result.is_empty(), "response should not be empty");
        assert!(
            !result.contains("error") && !result.contains("Error"),
            "should not be an error: {result}"
        );

        let mut guard = writer.lock().await;
        if let Some(w) = guard.as_mut() {
            w.kill().await;
        }
    }

    /// Start a prompt, immediately cancel it, verify we get the cancel result.
    ///
    /// Requires `hermes acp` to be available in PATH.
    /// Run with: `cargo test acp_cancel_running_task -- --ignored --nocapture`
    #[tokio::test]
    #[ignore]
    async fn acp_cancel_running_task() {
        let (mut writer, mut rx) = HermesAcpWriter::spawn("hermes acp")
            .await
            .expect("failed to spawn hermes acp");

        let cwd = std::env::current_dir()
            .unwrap()
            .to_string_lossy()
            .to_string();

        let session_id = writer
            .initialize(&mut rx, &cwd)
            .await
            .expect("initialize failed");

        let prompt_id = writer
            .send_prompt(&session_id, "Write a very long essay about the history of computing.")
            .await
            .expect("send_prompt failed");

        // Give the agent a moment to start processing, then cancel
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        writer
            .send_cancel(prompt_id)
            .await
            .expect("send_cancel failed");

        println!("Sent cancel for request id={prompt_id}");

        // Drain messages until we get the prompt response
        let result = tokio::time::timeout(std::time::Duration::from_secs(30), async {
            loop {
                match rx.recv().await {
                    Some(JsonRpcMessage::Response { id, result, .. }) if id == prompt_id => {
                        let stop_reason = result
                            .as_ref()
                            .and_then(|r| r["stopReason"].as_str())
                            .unwrap_or("unknown");
                        return stop_reason.to_string();
                    }
                    Some(_) => continue,
                    None => return "channel closed".to_string(),
                }
            }
        })
        .await
        .expect("timed out waiting for cancel response");

        println!("Stop reason after cancel: {result}");

        writer.kill().await;
    }
}
