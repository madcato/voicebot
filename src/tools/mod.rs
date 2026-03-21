pub mod calendar;
pub mod clipboard;
pub mod conversation_mode;
pub mod current_time;
pub mod open_app;
pub mod read_file;
pub mod run_agent;
pub mod run_shell;
pub mod take_screenshot;

use std::collections::HashMap;

use async_trait::async_trait;

pub use clipboard::{ReadClipboardTool, SetClipboardTool};
pub use conversation_mode::{ConversationMode, SetConversationModeTool};
pub use current_time::CurrentTimeTool;
pub use open_app::OpenAppTool;
pub use run_agent::{
    format_history, ActiveAcpTask, HermesAcpWriter, JsonRpcMessage, RunAgentTool,
};
pub use take_screenshot::TakeScreenshotTool;

/// A tool the LLM can invoke by name.
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    /// JSON Schema for this tool's parameters (OpenAI function-calling format).
    /// Default: no parameters.
    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {}})
    }
    /// Execute the tool with optional args and return the result as a string.
    async fn run(&self, args: &str) -> String;
}

/// Registry of available tools and tool-call parser.
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: HashMap::new() }
    }

    pub fn register(&mut self, tool: impl Tool + 'static) {
        self.tools.insert(tool.name().to_string(), Box::new(tool));
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Returns the tools array for the OpenAI `tools` request field.
    pub fn tool_definitions(&self) -> Vec<serde_json::Value> {
        self.tools.values().map(|t| serde_json::json!({
            "type": "function",
            "function": {
                "name": t.name(),
                "description": t.description(),
                "parameters": t.parameters(),
            }
        })).collect()
    }

    /// Returns a section to append to the system prompt describing how to call tools.
    pub fn system_prompt_section(&self) -> String {
        if self.tools.is_empty() {
            return String::new();
        }
        "\n\nREGLA CRÍTICA — USO DE HERRAMIENTAS: \
         Tienes herramientas disponibles para ejecutar acciones reales. \
         Cuando el usuario solicite una acción que corresponda a una herramienta \
         (cambiar modo de conversación, consultar la hora, crear eventos, \
         abrir apps, ejecutar comandos, etc.), DEBES llamar a la herramienta \
         inmediatamente usando la función correspondiente. \
         NUNCA simules ni finjas que ejecutaste la acción sin llamar a la herramienta. \
         Primero llama a la herramienta, luego responde al usuario con el resultado."
            .to_string()
    }

    /// Parse a tool call from LLM output.
    ///
    /// Returns `(tool_name, args)` if a registered tool is found.
    /// Content inside `<tool_call>…</tool_call>` is split on the first `:`;
    /// everything before is the tool name, everything after (trimmed) is args.
    /// Tools that take no arguments may omit the colon entirely.
    #[allow(dead_code)]
    pub fn parse_tool_call(&self, text: &str) -> Option<(String, String)> {
        let start = text.find("<tool_call>")?;
        let after = &text[start + "<tool_call>".len()..];
        let end = after.find("</tool_call>")?;
        let content = after[..end].trim();

        let (name, args) = match content.find(':') {
            Some(pos) => {
                (content[..pos].trim().to_string(), content[pos + 1..].trim().to_string())
            }
            None => (content.to_string(), String::new()),
        };

        self.tools.contains_key(&name).then_some((name, args))
    }

    /// Execute a registered tool by name with the given args.
    pub async fn execute(&self, name: &str, args: &str) -> String {
        match self.tools.get(name) {
            Some(tool) => tool.run(args).await,
            None => format!("Unknown tool: {name}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry_with_current_time() -> ToolRegistry {
        let mut r = ToolRegistry::new();
        r.register(CurrentTimeTool);
        r
    }

    // ── parse_tool_call ───────────────────────────────────────────────────────

    #[test]
    fn parse_detects_current_time_call() {
        let r = registry_with_current_time();
        let llm_output = "<tool_call>current_time</tool_call>";
        assert_eq!(
            r.parse_tool_call(llm_output),
            Some(("current_time".to_string(), String::new()))
        );
    }

    #[test]
    fn parse_detects_tool_call_with_args() {
        let r = registry_with_current_time();
        // The parser splits on ':' so any args after the colon are captured.
        let llm_output = "<tool_call>current_time: some args</tool_call>";
        assert_eq!(
            r.parse_tool_call(llm_output),
            Some(("current_time".to_string(), "some args".to_string()))
        );
    }

    #[test]
    fn parse_detects_tool_call_embedded_in_text() {
        let r = registry_with_current_time();
        let llm_output = "  <tool_call>current_time</tool_call>  ";
        assert_eq!(
            r.parse_tool_call(llm_output),
            Some(("current_time".to_string(), String::new()))
        );
    }

    #[test]
    fn parse_returns_none_for_unregistered_tool() {
        let r = registry_with_current_time();
        let llm_output = "<tool_call>nonexistent_tool</tool_call>";
        assert_eq!(r.parse_tool_call(llm_output), None);
    }

    #[test]
    fn parse_returns_none_for_missing_closing_tag() {
        let r = registry_with_current_time();
        assert_eq!(r.parse_tool_call("<tool_call>current_time"), None);
    }

    #[test]
    fn parse_returns_none_for_missing_opening_tag() {
        let r = registry_with_current_time();
        assert_eq!(r.parse_tool_call("current_time</tool_call>"), None);
    }

    #[test]
    fn parse_returns_none_for_empty_registry() {
        let r = ToolRegistry::new();
        assert_eq!(r.parse_tool_call("<tool_call>current_time</tool_call>"), None);
    }

    #[test]
    fn parse_returns_none_for_plain_text() {
        let r = registry_with_current_time();
        assert_eq!(r.parse_tool_call("What time is it?"), None);
    }

    // ── execute ───────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn execute_current_time_returns_non_empty() {
        let r = registry_with_current_time();
        let result = r.execute("current_time", "").await;
        assert!(!result.is_empty());
    }

    #[tokio::test]
    async fn execute_current_time_contains_colon_separator() {
        // Output is "HH:MM:SS, Weekday DD Month YYYY" — always has ':'
        let r = registry_with_current_time();
        let result = r.execute("current_time", "").await;
        assert!(result.contains(':'), "expected time separator ':' in {result:?}");
    }

    #[tokio::test]
    async fn execute_unknown_tool_returns_error_message() {
        let r = registry_with_current_time();
        let result = r.execute("nonexistent", "").await;
        assert!(result.contains("nonexistent"), "error message should mention the tool name");
    }

    // ── system_prompt_section ─────────────────────────────────────────────────

    #[test]
    fn system_prompt_section_empty_for_empty_registry() {
        let r = ToolRegistry::new();
        assert!(r.system_prompt_section().is_empty());
    }

    #[test]
    fn system_prompt_section_non_empty_when_tools_registered() {
        let r = registry_with_current_time();
        assert!(!r.system_prompt_section().is_empty());
        assert!(r.system_prompt_section().contains("herramienta"));
    }

    #[test]
    fn tool_definitions_contains_tool_name_and_description() {
        let r = registry_with_current_time();
        let defs = r.tool_definitions();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0]["function"]["name"], "current_time");
        assert!(!defs[0]["function"]["description"].as_str().unwrap_or("").is_empty());
    }

    #[test]
    fn tool_definitions_empty_for_empty_registry() {
        let r = ToolRegistry::new();
        assert!(r.tool_definitions().is_empty());
    }

    // ── parse → execute round-trip ────────────────────────────────────────────

    #[tokio::test]
    async fn parse_and_execute_current_time_round_trip() {
        let r = registry_with_current_time();
        let llm_output = "<tool_call>current_time</tool_call>";

        let (name, args) = r.parse_tool_call(llm_output).expect("should parse current_time");
        let result = r.execute(&name, &args).await;

        assert_eq!(name, "current_time");
        assert!(!result.is_empty());
        // Result should look like a time (contains ':')
        assert!(result.contains(':'));
    }
}
