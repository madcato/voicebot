//! MCP (Model Context Protocol) client — JSON-RPC 2.0 over stdio.
//!
//! Spawns an MCP server subprocess, performs the initialize handshake,
//! discovers available tools via `tools/list`, and calls them via `tools/call`.
//!
//! Compatible with any stdio-transport MCP server (e.g. `bunx apple-mcp@latest`,
//! `macOS-local-mcp-server`, etc.).
//!
//! Concurrent `call_tool()` calls are safe: each request registers a oneshot
//! channel keyed on its JSON-RPC request id; the reader task routes responses
//! to the correct waiter.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{oneshot, Mutex};
use tracing::{debug, info, warn};

// ── Response type ────────────────────────────────────────────────────────────

/// Parsed inbound JSON-RPC response (id already matched and removed from pending map).
struct RpcResponse {
    result: Option<Value>,
    error: Option<Value>,
}

/// Parse a JSON-RPC response out of a raw JSON value.
/// Returns None for notifications or unrecognized messages.
fn parse_response(v: &Value) -> Option<(u64, RpcResponse)> {
    // Responses have an "id" but no "method".
    if v.get("method").is_some() {
        return None; // notification or request from server
    }
    let id = v.get("id").and_then(|i| i.as_u64())?;
    Some((id, RpcResponse {
        result: v.get("result").cloned(),
        error: v.get("error").cloned(),
    }))
}

// ── Tool definition ──────────────────────────────────────────────────────────

/// A tool exposed by the MCP server (from `tools/list`).
#[derive(Debug, Clone)]
pub struct McpToolDef {
    pub name: String,
    pub description: String,
    /// JSON Schema for the tool input (`inputSchema` field).
    pub input_schema: Value,
}

// ── Internal writer ──────────────────────────────────────────────────────────

struct McpWriter {
    stdin: ChildStdin,
    #[allow(dead_code)]
    child: Child,
    next_id: u64,
}

impl McpWriter {
    async fn send_raw(&mut self, msg: &Value) -> Result<()> {
        let json = serde_json::to_string(msg)?;
        debug!(target: "mcp", "→ {json}");
        self.stdin.write_all(json.as_bytes()).await?;
        self.stdin.write_all(b"\n").await?;
        self.stdin.flush().await?;
        Ok(())
    }

    async fn send_request(&mut self, method: &str, params: Value) -> Result<u64> {
        let id = self.next_id;
        self.next_id += 1;
        let msg = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        self.send_raw(&msg).await?;
        Ok(id)
    }

    async fn send_notification(&mut self, method: &str, params: Value) -> Result<()> {
        let msg = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        self.send_raw(&msg).await
    }
}

// ── McpClient ────────────────────────────────────────────────────────────────

/// Persistent MCP server subprocess client.
pub struct McpClient {
    writer: Mutex<McpWriter>,
    /// In-flight request map: id → response channel.
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<RpcResponse>>>>,
    /// Hard timeout for each tool call (seconds).
    tool_timeout_secs: u64,
}

impl McpClient {
    /// Spawn the MCP server process, perform the initialize handshake, query
    /// `tools/list`, and return `(client, tool_defs)`.
    pub async fn spawn_and_init(
        command: &str,
        tool_timeout_secs: u64,
    ) -> Result<(Self, Vec<McpToolDef>)> {
        let parts: Vec<&str> = command.split_whitespace().collect();
        let program = parts.first().copied()
            .ok_or_else(|| anyhow::anyhow!("MCP_COMMAND is empty"))?;
        let args = &parts[1..];

        // Redirect server stderr to voicebot.log so it doesn't clutter TUI output.
        let stderr_sink = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("voicebot.log")
            .map(std::process::Stdio::from)
            .unwrap_or_else(|_| std::process::Stdio::null());

        let mut child = Command::new(program)
            .args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(stderr_sink)
            .spawn()
            .map_err(|e| anyhow::anyhow!("MCP: failed to spawn '{}': {}", command, e))?;

        let stdin = child.stdin.take()
            .ok_or_else(|| anyhow::anyhow!("MCP: no stdin handle"))?;
        let stdout = child.stdout.take()
            .ok_or_else(|| anyhow::anyhow!("MCP: no stdout handle"))?;

        let pending: Arc<Mutex<HashMap<u64, oneshot::Sender<RpcResponse>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let pending_reader = Arc::clone(&pending);

        // Reader task: parse newline-delimited JSON-RPC, route responses.
        tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let line = line.trim().to_string();
                if line.is_empty() { continue; }
                debug!(target: "mcp", "← {line}");
                match serde_json::from_str::<Value>(&line) {
                    Ok(v) => {
                        if let Some((id, resp)) = parse_response(&v) {
                            let tx = pending_reader.lock().await.remove(&id);
                            if let Some(tx) = tx {
                                let _ = tx.send(resp);
                            } else {
                                warn!(target: "mcp", "Unexpected response for id={id}");
                            }
                        }
                        // Notifications (initialized, etc.) are silently ignored.
                    }
                    Err(e) => warn!(target: "mcp", "Unparseable line: {e} — raw: {line:?}"),
                }
            }
            debug!(target: "mcp", "MCP reader task ended (server exited?)");
        });

        let client = Self {
            writer: Mutex::new(McpWriter { stdin, child, next_id: 0 }),
            pending,
            tool_timeout_secs,
        };

        // ── MCP handshake ────────────────────────────────────────────────────
        client.initialize().await?;
        let tools = client.list_tools().await?;

        info!(
            target: "mcp",
            "MCP server ready — {} tool(s): {:?}",
            tools.len(),
            tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
        );

        Ok((client, tools))
    }

    // ── Protocol methods ─────────────────────────────────────────────────────

    /// Send `initialize` and `notifications/initialized`.
    async fn initialize(&self) -> Result<()> {
        // Send initialize request.
        let init_id = self.writer.lock().await.send_request(
            "initialize",
            serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "voicebot", "version": "0.1.0"},
            }),
        ).await?;

        // Wait for initialize response.
        let resp = self.wait_for_response(init_id).await?;
        if let Some(err) = resp.error {
            anyhow::bail!("MCP initialize error: {err}");
        }
        debug!(target: "mcp", "initialize OK");

        // Send initialized notification (no response expected).
        self.writer.lock().await.send_notification(
            "notifications/initialized",
            serde_json::json!({}),
        ).await?;

        Ok(())
    }

    /// Call `tools/list` and return the tool definitions.
    async fn list_tools(&self) -> Result<Vec<McpToolDef>> {
        let id = self.writer.lock().await
            .send_request("tools/list", serde_json::json!({})).await?;

        let resp = self.wait_for_response(id).await?;
        if let Some(err) = resp.error {
            anyhow::bail!("MCP tools/list error: {err}");
        }

        let result = resp.result.unwrap_or_default();
        let tools_arr = result["tools"].as_array()
            .ok_or_else(|| anyhow::anyhow!("MCP tools/list: missing 'tools' array"))?;

        let defs = tools_arr
            .iter()
            .filter_map(|t| {
                let name = t["name"].as_str()?.to_string();
                let description = t["description"].as_str().unwrap_or("").to_string();
                let input_schema = t.get("inputSchema").cloned().unwrap_or_else(|| {
                    serde_json::json!({"type": "object", "properties": {}})
                });
                Some(McpToolDef { name, description, input_schema })
            })
            .collect();

        Ok(defs)
    }

    /// Call `tools/call` and return the text content of the response.
    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<String> {
        let id = self.writer.lock().await.send_request(
            "tools/call",
            serde_json::json!({
                "name": name,
                "arguments": arguments,
            }),
        ).await?;

        let resp = tokio::time::timeout(
            Duration::from_secs(self.tool_timeout_secs),
            self.wait_for_response(id),
        )
        .await
        .map_err(|_| anyhow::anyhow!("MCP tool '{}' timed out after {}s", name, self.tool_timeout_secs))??;

        if let Some(err) = resp.error {
            anyhow::bail!("MCP tool '{}' error: {err}", name);
        }

        let result = resp.result.unwrap_or_default();
        Ok(extract_text_content(&result))
    }

    // ── Internal ─────────────────────────────────────────────────────────────

    /// Register a oneshot channel for request `id` and wait for the response.
    async fn wait_for_response(&self, id: u64) -> Result<RpcResponse> {
        let (tx, rx) = oneshot::channel::<RpcResponse>();
        self.pending.lock().await.insert(id, tx);
        rx.await.map_err(|_| anyhow::anyhow!("MCP: response channel closed for id={id}"))
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Extract text content from a `tools/call` result.
///
/// MCP result format: `{"content": [{"type": "text", "text": "..."}], "isError": false}`
/// All text parts are joined with newlines.
fn extract_text_content(result: &Value) -> String {
    let content = match result["content"].as_array() {
        Some(arr) => arr,
        None => return result.to_string(),
    };

    let parts: Vec<&str> = content
        .iter()
        .filter(|item| item["type"].as_str() == Some("text"))
        .filter_map(|item| item["text"].as_str())
        .collect();

    if parts.is_empty() {
        result.to_string()
    } else {
        parts.join("\n")
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_response_ignores_notifications() {
        let notif = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        });
        assert!(parse_response(&notif).is_none());
    }

    #[test]
    fn parse_response_matches_success() {
        let resp = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 3,
            "result": {"tools": []}
        });
        let (id, r) = parse_response(&resp).unwrap();
        assert_eq!(id, 3);
        assert!(r.error.is_none());
        assert!(r.result.is_some());
    }

    #[test]
    fn parse_response_matches_error() {
        let resp = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 5,
            "error": {"code": -32601, "message": "Method not found"}
        });
        let (id, r) = parse_response(&resp).unwrap();
        assert_eq!(id, 5);
        assert!(r.error.is_some());
    }

    #[test]
    fn extract_text_content_single_part() {
        let result = serde_json::json!({
            "content": [{"type": "text", "text": "Hola mundo"}],
            "isError": false
        });
        assert_eq!(extract_text_content(&result), "Hola mundo");
    }

    #[test]
    fn extract_text_content_multiple_parts() {
        let result = serde_json::json!({
            "content": [
                {"type": "text", "text": "Parte 1"},
                {"type": "text", "text": "Parte 2"},
            ],
            "isError": false
        });
        assert_eq!(extract_text_content(&result), "Parte 1\nParte 2");
    }

    #[test]
    fn extract_text_content_skips_non_text() {
        let result = serde_json::json!({
            "content": [
                {"type": "image", "data": "base64..."},
                {"type": "text", "text": "Solo este"},
            ],
        });
        assert_eq!(extract_text_content(&result), "Solo este");
    }

    #[test]
    fn extract_text_content_empty_falls_back_to_json() {
        let result = serde_json::json!({"content": []});
        // Empty content → JSON serialization of the whole result.
        assert!(!extract_text_content(&result).is_empty());
    }
}
