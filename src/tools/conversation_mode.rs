use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::tools::Tool;

/// Whether the voicebot is actively listening or only responding to its wake word.
#[derive(Debug, Clone, PartialEq)]
pub enum ConversationMode {
    /// Default — responds to the enrolled user's voice normally.
    Active,
    /// Quiet mode activated automatically (silence timer or non-user streak).
    /// Any speech from the main user immediately returns the bot to Active.
    Ambient,
    /// Quiet mode activated explicitly by the user via the tool.
    /// Stays locked until the user explicitly requests Active mode — automatic
    /// triggers (silence, non-user streak) do NOT override this state.
    AmbientLocked,
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
        "Cambia el modo de escucha del asistente entre Active y Ambient. \
         mode='ambient': silencio, modo espera, duerme, go to sleep. \
         mode='active': conversación, despierta, wake up. \
         SIEMPRE llama a esta herramienta inmediatamente — nunca simules el cambio."
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
            ConversationMode::AmbientLocked
        } else {
            ConversationMode::Active
        };

        let msg = match new_mode {
            ConversationMode::AmbientLocked => "Ambient mode activated.",
            ConversationMode::Active        => "Active mode restored.",
            ConversationMode::Ambient       => unreachable!(),
        };

        *self.mode.lock().unwrap() = new_mode;
        msg.to_string()
    }
}

