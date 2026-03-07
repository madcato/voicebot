/// Manages the accumulated prompt for a llama.cpp session.
///
/// Mirrors the stateful session pattern from butler/llm/zosia/stateful-llm-server.py:
/// the full prompt is accumulated turn by turn. llama.cpp reuses its KV-cache for the
/// common prefix across calls, so only the new user turn needs prefill on each request.
#[derive(Clone)]
pub struct LlmSession {
    pub accumulated_prompt: String,
    pub slot_id: u8,
}

impl LlmSession {
    /// Create a new session with a system prompt.
    pub fn new(system_prompt: &str, slot_id: u8) -> Self {
        let mut prompt = String::new();
        if !system_prompt.is_empty() {
            prompt.push_str("<|im_start|>system\n");
            prompt.push_str(system_prompt);
            prompt.push_str("<|im_end|>\n");
        }
        Self {
            accumulated_prompt: prompt,
            slot_id,
        }
    }

    /// Restore session from persisted message history.
    pub fn from_history(system_prompt: &str, slot_id: u8, messages: &[(String, String)]) -> Self {
        let mut session = Self::new(system_prompt, slot_id);
        for (role, content) in messages {
            match role.as_str() {
                "User" => session.add_user_turn(content),
                "Assistant" => session.add_assistant_turn(content),
                _ => {}
            }
        }
        session
    }

    /// Append a completed user turn to the prompt.
    pub fn add_user_turn(&mut self, text: &str) {
        self.accumulated_prompt.push_str("<|im_start|>user\n");
        self.accumulated_prompt.push_str(text);
        self.accumulated_prompt.push_str("<|im_end|>\n");
        self.accumulated_prompt.push_str("<|im_start|>assistant\n");
    }

    /// Append the completed assistant turn after generation.
    pub fn add_assistant_turn(&mut self, text: &str) {
        self.accumulated_prompt.push_str(text);
        self.accumulated_prompt.push_str("<|im_end|>\n");
    }

    /// Inject a tool result after the LLM emitted a tool call mid-turn.
    ///
    /// Call this after the LLM has generated `tool_call_text` (e.g. `<tool_call>current_time</tool_call>`)
    /// and before the second LLM call that should produce the spoken response.
    pub fn add_tool_result(&mut self, tool_call_text: &str, result: &str) {
        // accumulated_prompt already ends with `<|im_start|>assistant\n`
        // Append what the LLM generated, close the assistant turn, add tool turn, reopen assistant.
        self.accumulated_prompt.push_str(tool_call_text);
        self.accumulated_prompt.push_str("<|im_end|>\n");
        self.accumulated_prompt.push_str("<|im_start|>tool\n");
        self.accumulated_prompt.push_str(result);
        self.accumulated_prompt.push_str("<|im_end|>\n");
        self.accumulated_prompt.push_str("<|im_start|>assistant\n");
    }

    pub fn prompt(&self) -> &str {
        &self.accumulated_prompt
    }
}
