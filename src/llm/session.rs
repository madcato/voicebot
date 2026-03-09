use serde::Serialize;

/// A single message in a conversation (OpenAI chat format).
#[derive(Clone, Debug, Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: "system".into(), content: content.into() }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".into(), content: content.into() }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: "assistant".into(), content: content.into() }
    }
    pub fn tool(content: impl Into<String>) -> Self {
        Self { role: "tool".into(), content: content.into() }
    }
}

/// Conversation state for an OpenAI-compatible chat endpoint.
///
/// Stores messages as a `Vec<Message>` (user + assistant turns).
/// Tool call/result exchanges live in `messages` during a turn but are
/// not persisted to the DB — they are implementation details.
#[derive(Clone)]
pub struct LlmSession {
    /// Base system prompt (never modified after construction).
    original_system_prompt: String,
    /// Current conversation summary, injected into the system message when present.
    summary: Option<String>,
    /// Conversation turns: user, assistant, and transient tool messages.
    pub messages: Vec<Message>,
    pub slot_id: u8,
}

impl LlmSession {
    /// Create a fresh session.
    pub fn new(system_prompt: &str, slot_id: u8) -> Self {
        Self {
            original_system_prompt: system_prompt.to_string(),
            summary: None,
            messages: Vec::new(),
            slot_id,
        }
    }

    /// Restore a session from persisted message history.
    ///
    /// If `summary` is Some it is injected into the system message. The passed
    /// `history` should already be only the turns after the summary's cutoff.
    pub fn from_history(
        system_prompt: &str,
        slot_id: u8,
        summary: Option<&str>,
        history: &[(String, String)],
    ) -> Self {
        let mut session = Self {
            original_system_prompt: system_prompt.to_string(),
            summary: summary.map(String::from),
            messages: Vec::new(),
            slot_id,
        };
        for (role, content) in history {
            match role.as_str() {
                "User" => session.add_user_turn(content),
                "Assistant" => session.add_assistant_turn(content),
                _ => {}
            }
        }
        session
    }

    /// Build the full message list as JSON values for an API call.
    ///
    /// Identical to `all_messages()` but returns serde_json::Value so callers
    /// can append tool-call / tool-result messages with arbitrary fields
    /// (tool_calls, tool_call_id, null content) that Message does not model.
    pub fn all_messages_api(&self) -> Vec<serde_json::Value> {
        self.all_messages()
            .into_iter()
            .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
            .collect()
    }

    /// Build the full message list for an API call: system first, then conversation.
    pub fn all_messages(&self) -> Vec<Message> {
        let system_content = match &self.summary {
            None => self.original_system_prompt.clone(),
            Some(s) => format!(
                "{}\n\n[CONVERSATION SUMMARY]\n{}",
                self.original_system_prompt, s
            ),
        };
        let mut msgs = vec![Message::system(system_content)];
        msgs.extend(self.messages.clone());
        msgs
    }

    /// Append a user turn.
    pub fn add_user_turn(&mut self, text: &str) {
        self.messages.push(Message::user(text));
    }

    /// Append the assistant's final response for this turn.
    pub fn add_assistant_turn(&mut self, text: &str) {
        self.messages.push(Message::assistant(text));
    }

    /// Inject a tool call + result into the message list.
    ///
    /// Called when the LLM emitted a tool call. Adds the tool call as an
    /// assistant message and the result as a tool message, so the next LLM
    /// call sees the full exchange and continues from there.
    /// Tool messages are NOT persisted to DB and NOT counted for summarization.
    pub fn add_tool_result(&mut self, tool_call_text: &str, result: &str) {
        self.messages.push(Message::assistant(tool_call_text));
        self.messages.push(Message::tool(result));
    }

    // ── Summarization ──────────────────────────────────────────────────────────

    /// True when the total content size approaches the context limit.
    ///
    /// Uses chars/3.5 as a rough token estimate. Triggers at 75% of the limit.
    pub fn needs_summarization(&self, context_limit_tokens: usize) -> bool {
        if self.messages.len() < 4 {
            return false;
        }
        let total_chars: usize = self.messages.iter().map(|m| m.content.len()).sum::<usize>()
            + self.original_system_prompt.len()
            + self.summary.as_deref().map_or(0, str::len);
        let approx_tokens = total_chars * 10 / 35;
        approx_tokens > context_limit_tokens * 3 / 4
    }

    /// How many messages would be summarized if we keep the last `keep_n`.
    pub fn summarizable_turn_count(&self, keep_n: usize) -> usize {
        self.messages.len().saturating_sub(keep_n)
    }

    /// Build the messages for a summarization LLM call.
    ///
    /// Returns None if there is nothing old enough to summarize.
    pub fn build_summary_prompt(&self, keep_n: usize) -> Option<Vec<Message>> {
        let summarize_count = self.summarizable_turn_count(keep_n);
        if summarize_count == 0 {
            return None;
        }

        let mut conversation = String::new();
        for msg in &self.messages[..summarize_count] {
            conversation.push_str(&msg.role);
            conversation.push_str(": ");
            conversation.push_str(&msg.content);
            conversation.push_str("\n\n");
        }

        Some(vec![
            Message::system(
                "You are a conversation summarizer. Summarize the following conversation \
                 concisely in the same language as the conversation. Preserve all important \
                 facts, names, decisions, preferences, and context. Be brief.",
            ),
            Message::user(format!("Summarize this conversation:\n\n{conversation}")),
        ])
    }

    /// Discard old messages, keeping only the last `keep_n`, and store the summary.
    pub fn apply_summary(&mut self, summary: &str, keep_n: usize) {
        let keep_start = self.messages.len().saturating_sub(keep_n);
        self.messages = self.messages[keep_start..].to_vec();
        self.summary = Some(summary.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a session with `n` user+assistant turn pairs.
    fn session_with_turns(n: usize) -> LlmSession {
        let mut s = LlmSession::new("System prompt.", 0);
        for i in 0..n {
            s.add_user_turn(&format!("User message {i}"));
            s.add_assistant_turn(&format!("Assistant response {i}"));
        }
        s
    }

    // ── needs_summarization ───────────────────────────────────────────────────

    #[test]
    fn needs_summarization_false_when_too_few_messages() {
        // Fewer than 4 messages always returns false regardless of size.
        let mut s = LlmSession::new("System.", 0);
        s.add_user_turn("Hello");
        s.add_assistant_turn("Hi");
        assert!(!s.needs_summarization(1)); // tiny limit, still false
    }

    #[test]
    fn needs_summarization_false_below_threshold() {
        let s = session_with_turns(3); // 6 messages, short content
        assert!(!s.needs_summarization(100_000)); // enormous context → never triggers
    }

    #[test]
    fn needs_summarization_true_above_threshold() {
        // Token estimate: total_chars * 10 / 35 > context_tokens * 3 / 4
        // → total_chars > context_tokens * 2.625
        // With context=100: need > 262 chars. Each turn pair ≈ 40 chars × 6 pairs = 240+.
        let long = "x".repeat(50);
        let mut s = LlmSession::new("sys", 0);
        for _ in 0..6 {
            s.add_user_turn(&long);
            s.add_assistant_turn(&long);
        }
        assert!(s.needs_summarization(100));
    }

    // ── summarizable_turn_count ───────────────────────────────────────────────

    #[test]
    fn summarizable_turn_count_correct() {
        let s = session_with_turns(5); // 10 messages
        assert_eq!(s.summarizable_turn_count(4), 6);
        assert_eq!(s.summarizable_turn_count(10), 0);
        assert_eq!(s.summarizable_turn_count(20), 0); // saturating_sub
    }

    // ── build_summary_prompt ──────────────────────────────────────────────────

    #[test]
    fn build_summary_prompt_none_when_nothing_to_summarize() {
        let s = session_with_turns(2); // 4 messages, keep_n = 6 → nothing old
        assert!(s.build_summary_prompt(6).is_none());
    }

    #[test]
    fn build_summary_prompt_structure() {
        let s = session_with_turns(5); // 10 messages
        let prompt = s.build_summary_prompt(4).unwrap(); // summarize first 6, keep last 4
        assert_eq!(prompt.len(), 2);
        assert_eq!(prompt[0].role, "system");
        assert_eq!(prompt[1].role, "user");
        assert!(prompt[1].content.contains("Summarize"));
    }

    #[test]
    fn build_summary_prompt_includes_old_turns_not_recent() {
        let s = session_with_turns(5); // turns 0-4, keep last 4 messages = turns 3+4
        let prompt = s.build_summary_prompt(4).unwrap();
        // Old turns (0, 1, 2) must appear in the prompt body.
        assert!(prompt[1].content.contains("User message 0"));
        assert!(prompt[1].content.contains("Assistant response 2"));
        // The kept turns (3, 4) must NOT be in the summary prompt.
        assert!(!prompt[1].content.contains("User message 3"));
        assert!(!prompt[1].content.contains("User message 4"));
    }

    // ── apply_summary ─────────────────────────────────────────────────────────

    #[test]
    fn apply_summary_keeps_last_n_messages() {
        let mut s = session_with_turns(5); // 10 messages (turns 0-4)
        s.apply_summary("Summary of old turns.", 4);
        // Only the last 4 messages remain.
        assert_eq!(s.messages.len(), 4);
        assert!(s.messages[0].content.contains("User message 3"));
        assert!(s.messages[1].content.contains("Assistant response 3"));
        assert!(s.messages[2].content.contains("User message 4"));
        assert!(s.messages[3].content.contains("Assistant response 4"));
    }

    #[test]
    fn apply_summary_with_keep_larger_than_messages() {
        let mut s = session_with_turns(2); // 4 messages
        s.apply_summary("Summary.", 10); // keep more than we have → keep all
        assert_eq!(s.messages.len(), 4);
    }

    // ── all_messages / system message injection ────────────────────────────────

    #[test]
    fn all_messages_no_summary_returns_plain_system() {
        let mut s = LlmSession::new("Base prompt.", 0);
        s.add_user_turn("Hello");
        let msgs = s.all_messages();
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[0].content, "Base prompt.");
        assert!(!msgs[0].content.contains("[CONVERSATION SUMMARY]"));
    }

    #[test]
    fn all_messages_injects_summary_into_system_message() {
        let mut s = LlmSession::new("Base prompt.", 0);
        s.add_user_turn("Hello");
        s.add_assistant_turn("Hi");
        s.apply_summary("User greeted the assistant.", 2);

        let msgs = s.all_messages();
        assert_eq!(msgs[0].role, "system");
        assert!(msgs[0].content.contains("Base prompt."));
        assert!(msgs[0].content.contains("[CONVERSATION SUMMARY]"));
        assert!(msgs[0].content.contains("User greeted the assistant."));
        // Original prompt is NOT replaced, only extended.
        assert!(msgs[0].content.starts_with("Base prompt."));
    }

    #[test]
    fn all_messages_length_equals_system_plus_conversation() {
        let mut s = session_with_turns(3); // 6 conversation messages
        s.apply_summary("Summary.", 4);    // now 4 messages kept
        let msgs = s.all_messages();
        assert_eq!(msgs.len(), 1 + 4); // system + 4 kept messages
        assert_eq!(msgs[0].role, "system");
    }

    // ── from_history ──────────────────────────────────────────────────────────

    #[test]
    fn from_history_without_summary() {
        let history = vec![
            ("User".to_string(), "Hello".to_string()),
            ("Assistant".to_string(), "Hi".to_string()),
        ];
        let s = LlmSession::from_history("System.", 0, None, &history);
        assert_eq!(s.messages.len(), 2);
        let msgs = s.all_messages();
        assert_eq!(msgs[0].content, "System.");
    }

    #[test]
    fn from_history_with_summary_injects_it() {
        let history = vec![
            ("User".to_string(), "Latest question".to_string()),
            ("Assistant".to_string(), "Latest answer".to_string()),
        ];
        let s = LlmSession::from_history("System.", 0, Some("Old conversation summary."), &history);
        assert_eq!(s.messages.len(), 2);
        let msgs = s.all_messages();
        assert!(msgs[0].content.contains("[CONVERSATION SUMMARY]"));
        assert!(msgs[0].content.contains("Old conversation summary."));
        assert_eq!(msgs[1].content, "Latest question");
    }

    #[test]
    fn from_history_ignores_unknown_roles() {
        let history = vec![
            ("User".to_string(), "Hello".to_string()),
            ("Unknown".to_string(), "Ignored".to_string()),
            ("Assistant".to_string(), "Hi".to_string()),
        ];
        let s = LlmSession::from_history("System.", 0, None, &history);
        assert_eq!(s.messages.len(), 2); // Unknown role skipped
    }

    // ── Full summarization cycle ──────────────────────────────────────────────

    #[test]
    fn full_summarization_cycle() {
        let mut s = session_with_turns(5); // turns 0-4
        let keep_n = 4;

        // Build the summarization prompt — should include turns 0-5 (6 messages)
        let prompt = s.build_summary_prompt(keep_n).unwrap();
        assert_eq!(prompt[0].role, "system");

        // Simulate the LLM returning a summary
        let summary = "El usuario y el asistente intercambiaron mensajes sobre varios temas.";
        s.apply_summary(summary, keep_n);

        // Session is now compacted
        assert_eq!(s.messages.len(), keep_n);

        // all_messages has system + kept turns
        let all = s.all_messages();
        assert_eq!(all.len(), 1 + keep_n);
        assert!(all[0].content.contains("[CONVERSATION SUMMARY]"));
        assert!(all[0].content.contains(summary));

        // Recent turns are preserved verbatim
        assert!(all[1].content.contains("User message 3"));
        assert!(all[2].content.contains("Assistant response 3"));
        assert!(all[3].content.contains("User message 4"));
        assert!(all[4].content.contains("Assistant response 4"));

        // After compaction, needs_summarization resets (assuming large context)
        assert!(!s.needs_summarization(100_000));
    }
}
