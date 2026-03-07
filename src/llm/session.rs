/// Manages the accumulated prompt and turn history for a llama.cpp session.
///
/// The `accumulated_prompt` (ChatML flat string) is sent to the LLM on every call.
/// `turns` mirrors the conversation as `(role, content)` pairs — used for summarization.
#[derive(Clone)]
pub struct LlmSession {
    /// The effective system prompt (including any tool instructions).
    system_prompt: String,
    /// All conversation turns in order: role is "User" or "Assistant".
    /// Tool call/result exchanges are NOT stored here (they live only in accumulated_prompt).
    turns: Vec<(String, String)>,
    /// ChatML flat string sent to llama.cpp. This is the source of truth for LLM calls.
    pub accumulated_prompt: String,
    pub slot_id: u8,
}

impl LlmSession {
    /// Create a fresh session with the given system prompt.
    pub fn new(system_prompt: &str, slot_id: u8) -> Self {
        let mut accumulated_prompt = String::new();
        if !system_prompt.is_empty() {
            accumulated_prompt.push_str("<|im_start|>system\n");
            accumulated_prompt.push_str(system_prompt);
            accumulated_prompt.push_str("<|im_end|>\n");
        }
        Self {
            system_prompt: system_prompt.to_string(),
            turns: Vec::new(),
            accumulated_prompt,
            slot_id,
        }
    }

    /// Restore a session from persisted message history.
    ///
    /// If `summary` is Some, the prompt is built with the summary injected into the
    /// system section, followed by the recent turns. The passed `messages` should
    /// already be only the turns after the summary's cutoff point.
    pub fn from_history(
        system_prompt: &str,
        slot_id: u8,
        summary: Option<&str>,
        messages: &[(String, String)],
    ) -> Self {
        let mut session = if let Some(summary_text) = summary {
            // Rebuild with summary in the system block
            let accumulated_prompt = format!(
                "<|im_start|>system\n{system_prompt}\n\n[CONVERSATION SUMMARY]\n\
                 {summary_text}<|im_end|>\n"
            );
            Self {
                system_prompt: system_prompt.to_string(),
                turns: Vec::new(),
                accumulated_prompt,
                slot_id,
            }
        } else {
            Self::new(system_prompt, slot_id)
        };

        for (role, content) in messages {
            match role.as_str() {
                "User" => session.add_user_turn(content),
                "Assistant" => session.add_assistant_turn(content),
                _ => {}
            }
        }

        session
    }

    /// Append a completed user turn. Opens the assistant turn at the end of the prompt
    /// so the LLM continues from there on the next call.
    pub fn add_user_turn(&mut self, text: &str) {
        self.turns.push(("User".to_string(), text.to_string()));
        self.accumulated_prompt.push_str("<|im_start|>user\n");
        self.accumulated_prompt.push_str(text);
        self.accumulated_prompt.push_str("<|im_end|>\n");
        self.accumulated_prompt.push_str("<|im_start|>assistant\n");
    }

    /// Close the assistant turn after generation is complete.
    pub fn add_assistant_turn(&mut self, text: &str) {
        self.turns.push(("Assistant".to_string(), text.to_string()));
        self.accumulated_prompt.push_str(text);
        self.accumulated_prompt.push_str("<|im_end|>\n");
    }

    /// Inject a tool result after the LLM emitted a tool call mid-turn.
    ///
    /// The accumulated_prompt already ends with `<|im_start|>assistant\n`.
    /// This closes that turn, adds the tool result, and re-opens the assistant turn.
    /// Tool calls are NOT added to `turns` (they are implementation details).
    pub fn add_tool_result(&mut self, tool_call_text: &str, result: &str) {
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

    // ── Summarization ─────────────────────────────────────────────────────────

    /// Returns true when the accumulated prompt is large enough to warrant summarization.
    ///
    /// Uses chars/3.5 as a rough token estimate. Triggers at 75% of the context limit,
    /// leaving headroom for the current turn and the model's response.
    pub fn needs_summarization(&self, context_limit_tokens: usize) -> bool {
        if self.turns.len() < 4 {
            return false; // Too few turns to bother
        }
        let approx_tokens = self.accumulated_prompt.len() * 10 / 35;
        approx_tokens > context_limit_tokens * 3 / 4
    }

    /// How many turns (role, content) pairs would be summarized given keep_n recent turns.
    pub fn summarizable_turn_count(&self, keep_n: usize) -> usize {
        self.turns.len().saturating_sub(keep_n)
    }

    /// Build a one-shot prompt asking the LLM to summarize old conversation turns.
    ///
    /// Returns None if there is nothing to summarize (not enough turns).
    pub fn build_summary_prompt(&self, keep_n: usize) -> Option<String> {
        let summarize_count = self.summarizable_turn_count(keep_n);
        if summarize_count == 0 {
            return None;
        }

        let mut conversation = String::new();
        for (role, content) in &self.turns[..summarize_count] {
            conversation.push_str(role);
            conversation.push_str(": ");
            conversation.push_str(content);
            conversation.push_str("\n\n");
        }

        Some(format!(
            "<|im_start|>system\n\
             You are a conversation summarizer. Summarize the following conversation \
             concisely in the same language as the conversation. Preserve all important \
             facts, names, decisions, preferences, and context. Be brief.\
             <|im_end|>\n\
             <|im_start|>user\n\
             Summarize this conversation:\n\n{conversation}\
             <|im_end|>\n\
             <|im_start|>assistant\n"
        ))
    }

    /// Replace old turns with a summary and rebuild the accumulated prompt.
    ///
    /// Keeps the last `keep_n` turns verbatim; everything before is replaced by
    /// the summary text injected into the system block.
    pub fn apply_summary(&mut self, summary: &str, keep_n: usize) {
        let keep_start = self.turns.len().saturating_sub(keep_n);
        let recent: Vec<(String, String)> = self.turns[keep_start..].to_vec();

        // Rebuild the prompt: system + summary + recent turns
        let mut prompt = format!(
            "<|im_start|>system\n{}\n\n[CONVERSATION SUMMARY]\n{}<|im_end|>\n",
            self.system_prompt, summary
        );

        for (role, content) in &recent {
            match role.as_str() {
                "User" => {
                    prompt.push_str("<|im_start|>user\n");
                    prompt.push_str(content);
                    prompt.push_str("<|im_end|>\n");
                    prompt.push_str("<|im_start|>assistant\n");
                }
                "Assistant" => {
                    prompt.push_str(content);
                    prompt.push_str("<|im_end|>\n");
                }
                _ => {}
            }
        }

        self.turns = recent;
        self.accumulated_prompt = prompt;
    }
}
