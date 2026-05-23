//! `McpToolProxy` — wraps a single MCP tool as a `dyn Tool`.
//!
//! Created at startup for each tool discovered via `tools/list`. Calls are
//! always background (`is_background = true`) because MCP tool execution
//! time is unpredictable.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use super::Tool;
use crate::mcp::McpClient;

/// Proxy that exposes one MCP tool through the `Tool` trait.
pub struct McpToolProxy {
    name: String,
    description: String,
    /// The `inputSchema` from the MCP server — used as-is for the OpenAI
    /// function-calling `parameters` field.
    parameters: Value,
    client: Arc<McpClient>,
}

impl McpToolProxy {
    pub fn new(
        name: String,
        description: String,
        parameters: Value,
        client: Arc<McpClient>,
    ) -> Self {
        Self {
            name,
            description,
            parameters,
            client,
        }
    }
}

#[async_trait]
impl Tool for McpToolProxy {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters(&self) -> Value {
        self.parameters.clone()
    }

    /// All MCP tools run in a background task — execution time is unpredictable.
    fn is_background(&self) -> bool {
        true
    }

    async fn run(&self, args: &str) -> String {
        let arguments: Value =
            serde_json::from_str(args).unwrap_or(Value::Object(Default::default()));
        match self.client.call_tool(&self.name, arguments).await {
            Ok(text) => text,
            Err(e) => {
                tracing::warn!(target: "mcp", "Tool '{}' failed: {e}", self.name);
                format!("Error calling MCP tool '{}': {e}", self.name)
            }
        }
    }
}

// Note: McpToolProxy unit tests require a live MCP subprocess (Arc<McpClient>
// has no mock constructor). Behaviour is covered by:
//   - src/mcp/mod.rs tests (parsing, content extraction, protocol helpers)
//   - manual / integration tests that spin up a real MCP server
