pub mod current_time;

use std::collections::HashMap;
pub use current_time::CurrentTimeTool;

/// A tool the LLM can invoke by name.
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    /// Execute the tool and return the result as a string.
    fn run(&self) -> String;
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

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Returns a section to append to the system prompt describing how to call tools.
    pub fn system_prompt_section(&self) -> String {
        if self.tools.is_empty() {
            return String::new();
        }
        let mut s = String::from(
            "\n\n[TOOLS]\nIf you need to use a tool, output ONLY the tool call on its own — \
             no other text before or after it:\n\
             <tool_call>tool_name</tool_call>\n\n\
             Available tools:\n",
        );
        for tool in self.tools.values() {
            s.push_str(&format!("- {}: {}\n", tool.name(), tool.description()));
        }
        s
    }

    /// Parse a tool call from LLM output. Returns the tool name if found and registered.
    pub fn parse_tool_call(&self, text: &str) -> Option<String> {
        let start = text.find("<tool_call>")?;
        let after = &text[start + "<tool_call>".len()..];
        let end = after.find("</tool_call>")?;
        let name = after[..end].trim().to_string();
        self.tools.contains_key(&name).then_some(name)
    }

    /// Execute a tool by name and return its output.
    pub fn execute(&self, name: &str) -> String {
        match self.tools.get(name) {
            Some(tool) => tool.run(),
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
        assert_eq!(r.parse_tool_call(llm_output), Some("current_time".to_string()));
    }

    #[test]
    fn parse_detects_tool_call_embedded_in_text() {
        // The tool call may appear at end of stream with no surrounding text,
        // but parse should still work if there is leading whitespace.
        let r = registry_with_current_time();
        let llm_output = "  <tool_call>current_time</tool_call>  ";
        assert_eq!(r.parse_tool_call(llm_output), Some("current_time".to_string()));
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

    #[test]
    fn execute_current_time_returns_non_empty() {
        let r = registry_with_current_time();
        let result = r.execute("current_time");
        assert!(!result.is_empty());
    }

    #[test]
    fn execute_current_time_contains_colon_separator() {
        // Output is "HH:MM:SS, Weekday DD Month YYYY" — always has ':'
        let r = registry_with_current_time();
        let result = r.execute("current_time");
        assert!(result.contains(':'), "expected time separator ':' in {result:?}");
    }

    #[test]
    fn execute_unknown_tool_returns_error_message() {
        let r = registry_with_current_time();
        let result = r.execute("nonexistent");
        assert!(result.contains("nonexistent"), "error message should mention the tool name");
    }

    // ── system_prompt_section ─────────────────────────────────────────────────

    #[test]
    fn system_prompt_section_empty_for_empty_registry() {
        let r = ToolRegistry::new();
        assert!(r.system_prompt_section().is_empty());
    }

    #[test]
    fn system_prompt_section_contains_tool_name() {
        let r = registry_with_current_time();
        let section = r.system_prompt_section();
        assert!(section.contains("current_time"), "section: {section:?}");
    }

    #[test]
    fn system_prompt_section_contains_tool_description() {
        let r = registry_with_current_time();
        let section = r.system_prompt_section();
        assert!(
            section.contains(CurrentTimeTool.description()),
            "section should include the tool description: {section:?}"
        );
    }

    #[test]
    fn system_prompt_section_contains_xml_syntax_example() {
        let r = registry_with_current_time();
        let section = r.system_prompt_section();
        assert!(section.contains("<tool_call>"), "section should show the XML call syntax");
        assert!(section.contains("</tool_call>"));
    }

    // ── parse → execute round-trip ────────────────────────────────────────────

    #[test]
    fn parse_and_execute_current_time_round_trip() {
        let r = registry_with_current_time();
        let llm_output = "<tool_call>current_time</tool_call>";

        let name = r.parse_tool_call(llm_output).expect("should parse current_time");
        let result = r.execute(&name);

        assert_eq!(name, "current_time");
        assert!(!result.is_empty());
        // Result should look like a time (contains ':')
        assert!(result.contains(':'));
    }
}
