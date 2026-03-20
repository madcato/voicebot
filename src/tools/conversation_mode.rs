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
        "Switch the voicebot between Active and Ambient listening modes. \
         IMPORTANT: Always call this tool immediately when the user requests a mode change — do not just acknowledge it. \
         \
         Call with mode='ambient' when the user says things like: \
         'modo ambiente', 'modo silencio', 'activa el modo ambiente', \
         'desactiva el modo conversación', 'quédate en silencio', \
         'duerme', 'modo espera', 'go to sleep', 'ambient mode', 'sleep mode'. \
         In Ambient mode the bot ONLY responds when it hears its wake word (its name); all other speech is ignored. \
         \
         Call with mode='active' when the user says things like: \
         'modo conversación', 'activa el modo conversación', \
         'sal del modo ambiente', 'despierta', 'escúchame', \
         'active mode', 'wake up', 'conversation mode'. \
         In Active mode the bot listens and responds normally."
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

