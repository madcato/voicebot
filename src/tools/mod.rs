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
