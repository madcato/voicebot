use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::tools::Tool;

/// Whether the voicebot is actively listening or only responding to its wake word.
#[derive(Debug, Clone, PartialEq)]
pub enum ConversationMode {
    /// Default — responds to the enrolled user's voice normally.
    Active,
    /// Quiet mode — only responds when the transcript contains the wake word.
    /// Activated manually via this tool or automatically when a non-enrolled
    /// speaker is detected N times in a row.
    Ambient,
}

/// Tool that lets the LLM switch the voicebot between Active and Ambient mode.
pub struct SetConversationModeTool {
    mode: Arc<Mutex<ConversationMode>>,
}

impl SetConversationModeTool {
    pub fn new(mode: Arc<Mutex<ConversationMode>>) -> Self {
        Self { mode }
    }
}

#[async_trait]
impl Tool for SetConversationModeTool {
    fn name(&self) -> &str {
        "set_conversation_mode"
    }

    fn description(&self) -> &str {
        "Switch the voicebot between Active and Ambient listening mode. \
         Call with mode='ambient' when the user says things like 'go to sleep', \
         'activate ambient mode', 'conversation mode', 'Jarvis sleep', or \
         'activate sleep mode'. \
         In Ambient mode the bot only responds when its name (wake word) is heard; \
         all other speech is ignored. \
         Call with mode='active' to resume normal listening."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["active", "ambient"],
                    "description": "'ambient' to go quiet, 'active' to resume normal listening"
                }
            },
            "required": ["mode"]
        })
    }

    async fn run(&self, args: &str) -> String {
        let mode_str = serde_json::from_str::<serde_json::Value>(args)
            .ok()
            .and_then(|v| v["mode"].as_str().map(|s| s.to_lowercase()))
            .unwrap_or_else(|| args.trim().to_lowercase());

        let new_mode = if mode_str.contains("ambient") || mode_str.contains("sleep") {
            ConversationMode::Ambient
        } else {
            ConversationMode::Active
        };

        let msg = match new_mode {
            ConversationMode::Ambient => "Ambient mode activated.",
            ConversationMode::Active  => "Active mode restored.",
        };

        *self.mode.lock().unwrap() = new_mode;
        msg.to_string()
    }
}
