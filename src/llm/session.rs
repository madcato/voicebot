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
    #[allow(dead_code)]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: "assistant".into(), content: content.into() }
    }
    #[allow(dead_code)]
    pub fn tool(content: impl Into<String>) -> Self {
        Self { role: "tool".into(), content: content.into() }
    }
}

/// Conversation state for an OpenAI-compatible chat endpoint.
///
/// Stores messages as `Vec<serde_json::Value>` (OpenAI JSON format) so that
/// tool-call exchanges (assistant messages with `tool_calls` + tool result
/// messages) can be persisted verbatim alongside regular user/assistant turns.
/// This ensures the LLM sees prior tool calls in its context window and does not
/// learn to respond verbally without calling tools.
#[derive(Clone)]
pub struct LlmSession {
    /// Base system prompt — updated at runtime after context consolidation.
    original_system_prompt: String,
    /// Current conversation summary, injected into the system message when present.
    summary: Option<String>,
    /// Conversation turns in OpenAI JSON format (user, assistant, tool messages).
    pub messages: Vec<serde_json::Value>,
}

impl LlmSession {
    /// Returns the base system prompt (before any summary injection).
    pub fn system_prompt(&self) -> &str {
        &self.original_system_prompt
    }

    /// Returns the approximate token count of the current session content.
    pub fn approx_tokens(&self) -> usize {
        let total_chars: usize = self.messages.iter().map(|m| {
            m["content"].as_str().map_or(0, str::len)
                + m.get("tool_calls").map_or(0, |tc| tc.to_string().len())
        }).sum::<usize>()
            + self.original_system_prompt.len()
            + self.summary.as_deref().map_or(0, str::len);
        total_chars * 10 / 35
    }

    /// Create a fresh session.
    #[allow(dead_code)]
    pub fn new(system_prompt: &str) -> Self {
        Self {
            original_system_prompt: system_prompt.to_string(),
            summary: None,
            messages: Vec::new(),
        }
    }

    /// Restore a session from persisted message history.
    ///
    /// If `summary` is Some it is injected into the system message. The passed
    /// `history` should already be only the turns after the summary's cutoff.
    pub fn from_history(
        system_prompt: &str,
        summary: Option<&str>,
        history: &[(String, String)],
    ) -> Self {
        let mut session = Self {
            original_system_prompt: system_prompt.to_string(),
            summary: summary.map(String::from),
            messages: Vec::new(),
        };
        for (role, content) in history {
            match role.as_str() {
                "User" => session.add_user_turn(content),
                "Assistant" => session.add_assistant_turn(content),
                "ToolExchanges" => {
                    // Deserialise and replay the tool-call + tool-result messages so the
                    // LLM sees the same context it had during the original turn.
                    if let Ok(exchanges) = serde_json::from_str::<Vec<serde_json::Value>>(content) {
                        session.add_tool_exchange(exchanges);
                    }
                }
                _ => {}
            }
        }
        session
    }

    /// Build the full message list as JSON values for an API call.
    pub fn all_messages_api(&self) -> Vec<serde_json::Value> {
        let system_content = self.system_content();
        let mut msgs = vec![serde_json::json!({"role": "system", "content": system_content})];
        msgs.extend(self.messages.clone());
        msgs
    }

    /// Build the full message list for an API call: system first, then conversation.
    ///
    /// Returns `Vec<Message>` for callers that need the legacy struct format.
    /// Tool-call messages (null content) are skipped since `Message` cannot
    /// represent them — they are only relevant to the OpenAI API format.
    #[allow(dead_code)]
    pub fn all_messages(&self) -> Vec<Message> {
        let mut msgs = vec![Message::system(self.system_content())];
        for m in &self.messages {
            if let (Some(role), Some(content)) = (m["role"].as_str(), m["content"].as_str()) {
                msgs.push(Message { role: role.to_string(), content: content.to_string() });
            }
        }
        msgs
    }

    /// Append a user turn.
    pub fn add_user_turn(&mut self, text: &str) {
        self.messages.push(serde_json::json!({"role": "user", "content": text}));
    }

    /// Append the assistant's final response for this turn.
    pub fn add_assistant_turn(&mut self, text: &str) {
        self.messages.push(serde_json::json!({"role": "assistant", "content": text}));
    }

    /// Persist tool call exchanges from a completed pipeline turn.
    ///
    /// `exchanges` contains the tool-call assistant messages and tool-result
    /// messages that were built during the turn's tool loop. Storing them in the
    /// session ensures future LLM calls see that tool calls were made here,
    /// preventing the model from learning to respond verbally without calling tools.
    pub fn add_tool_exchange(&mut self, exchanges: Vec<serde_json::Value>) {
        self.messages.extend(exchanges);
    }

    /// Inject a tool call + result into the message list (OpenAI format).
    #[allow(dead_code)]
    pub fn add_tool_result(&mut self, tool_call_id: &str, name: &str, args: &str, result: &str) {
        self.messages.push(serde_json::json!({
            "role": "assistant",
            "content": serde_json::Value::Null,
            "tool_calls": [{
                "id": tool_call_id,
                "type": "function",
                "function": {"name": name, "arguments": args}
            }]
        }));
        self.messages.push(serde_json::json!({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result
        }));
    }

    // ── Summarization ──────────────────────────────────────────────────────────

    /// True when the total content size approaches the context limit.
    ///
    /// Uses chars/3.5 as a rough token estimate. Triggers at 75% of the limit.
    /// Kept for backward compatibility — prefer [`needs_consolidation`] with an
    /// explicit threshold percentage.
    #[allow(dead_code)]
    pub fn needs_summarization(&self, context_limit_tokens: usize) -> bool {
        self.needs_consolidation(context_limit_tokens, 75)
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
            if let (Some(role), Some(content)) = (msg["role"].as_str(), msg["content"].as_str())
                && (role == "user" || role == "assistant") {
                    conversation.push_str(role);
                    conversation.push_str(": ");
                    conversation.push_str(content);
                    conversation.push_str("\n\n");
                }
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

    // ── Context consolidation ────────────────────────────────────────────────

    /// True when the total content size approaches the context limit.
    ///
    /// Uses chars/3.5 as a rough token estimate. `threshold_pct` controls
    /// what percentage of the context window triggers consolidation (e.g. 80).
    pub fn needs_consolidation(&self, context_limit_tokens: usize, threshold_pct: usize) -> bool {
        if self.messages.len() < 4 {
            return false;
        }
        let total_chars: usize = self.messages.iter().map(|m| {
            m["content"].as_str().map_or(0, str::len)
                + m.get("tool_calls").map_or(0, |tc| tc.to_string().len())
        }).sum::<usize>()
            + self.original_system_prompt.len()
            + self.summary.as_deref().map_or(0, str::len);
        let approx_tokens = total_chars * 10 / 35;
        approx_tokens > context_limit_tokens * threshold_pct / 100
    }

    /// Replace the base system prompt at runtime.
    ///
    /// Used after context consolidation to inject updated memories, profile,
    /// and summary. The next API call will send the new system message.
    pub fn set_system_prompt(&mut self, new_prompt: String) {
        self.original_system_prompt = new_prompt;
    }

    // ── Private helpers ────────────────────────────────────────────────────────

    fn system_content(&self) -> String {
        match &self.summary {
            None => self.original_system_prompt.clone(),
            Some(s) => format!(
                "{}\n\n[CONVERSATION SUMMARY]\n{}",
                self.original_system_prompt, s
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a session with `n` user+assistant turn pairs.
    fn session_with_turns(n: usize) -> LlmSession {
        let mut s = LlmSession::new("System prompt.");
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
        let mut s = LlmSession::new("System.");
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
        let mut s = LlmSession::new("sys");
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
        assert_eq!(s.messages[0]["content"], "User message 3");
        assert_eq!(s.messages[1]["content"], "Assistant response 3");
        assert_eq!(s.messages[2]["content"], "User message 4");
        assert_eq!(s.messages[3]["content"], "Assistant response 4");
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
        let mut s = LlmSession::new("Base prompt.");
        s.add_user_turn("Hello");
        let msgs = s.all_messages();
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[0].content, "Base prompt.");
        assert!(!msgs[0].content.contains("[CONVERSATION SUMMARY]"));
    }

    #[test]
    fn all_messages_injects_summary_into_system_message() {
        let mut s = LlmSession::new("Base prompt.");
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

    // ── all_messages_api ──────────────────────────────────────────────────────

    #[test]
    fn all_messages_api_includes_tool_exchanges() {
        let mut s = LlmSession::new("System.");
        s.add_user_turn("Activa el modo ambiente");
        s.add_tool_exchange(vec![
            serde_json::json!({
                "role": "assistant",
                "content": serde_json::Value::Null,
                "tool_calls": [{"id": "call_1", "type": "function",
                    "function": {"name": "set_conversation_mode", "arguments": "{\"mode\":\"ambient\"}"}}]
            }),
            serde_json::json!({"role": "tool", "tool_call_id": "call_1", "content": "Ambient mode activated."}),
        ]);
        s.add_assistant_turn("Modo ambiente activado, señor.");

        let api_msgs = s.all_messages_api();
        // system + user + tool_call_assistant + tool_result + assistant = 5
        assert_eq!(api_msgs.len(), 5);
        assert_eq!(api_msgs[0]["role"], "system");
        assert_eq!(api_msgs[1]["role"], "user");
        assert_eq!(api_msgs[2]["role"], "assistant");
        assert!(api_msgs[2]["tool_calls"].is_array());
        assert_eq!(api_msgs[3]["role"], "tool");
        assert_eq!(api_msgs[4]["role"], "assistant");
        assert_eq!(api_msgs[4]["content"], "Modo ambiente activado, señor.");
    }

    // ── from_history ──────────────────────────────────────────────────────────

    #[test]
    fn from_history_without_summary() {
        let history = vec![
            ("User".to_string(), "Hello".to_string()),
            ("Assistant".to_string(), "Hi".to_string()),
        ];
        let s = LlmSession::from_history("System.", None, &history);
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
        let s = LlmSession::from_history("System.", Some("Old conversation summary."), &history);
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
        let s = LlmSession::from_history("System.", None, &history);
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

    // ── needs_consolidation ──────────────────────────────────────────────────

    #[test]
    fn needs_consolidation_respects_threshold_percentage() {
        let long = "x".repeat(50);
        let mut s = LlmSession::new("sys");
        for _ in 0..6 {
            s.add_user_turn(&long);
            s.add_assistant_turn(&long);
        }
        // At 75% threshold, 100 context tokens should trigger
        assert!(s.needs_consolidation(100, 75));
        // At 100% threshold with a generous limit, should not trigger
        assert!(!s.needs_consolidation(100_000, 80));
    }

    #[test]
    fn needs_consolidation_false_when_few_messages() {
        let mut s = LlmSession::new("System.");
        s.add_user_turn("Hello");
        s.add_assistant_turn("Hi");
        assert!(!s.needs_consolidation(1, 50));
    }

    // ── set_system_prompt ────────────────────────────────────────────────────

    #[test]
    fn set_system_prompt_replaces_original() {
        let mut s = LlmSession::new("Old prompt.");
        s.add_user_turn("Hello");
        s.set_system_prompt("New prompt with [MEMORIES].".to_string());

        let msgs = s.all_messages();
        assert_eq!(msgs[0].content, "New prompt with [MEMORIES].");
        assert!(!msgs[0].content.contains("Old prompt"));
    }

    #[test]
    fn set_system_prompt_preserves_summary() {
        let mut s = LlmSession::new("Base.");
        s.add_user_turn("Hello");
        s.add_assistant_turn("Hi");
        s.apply_summary("Summary text.", 2);
        s.set_system_prompt("New base.".to_string());

        let msgs = s.all_messages();
        assert!(msgs[0].content.starts_with("New base."));
        assert!(msgs[0].content.contains("[CONVERSATION SUMMARY]"));
        assert!(msgs[0].content.contains("Summary text."));
    }

}
