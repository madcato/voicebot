use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ControlEvent {
    StateChanged {
        state: String,
        utterance_id: Option<u64>,
    },
    Transcript {
        utterance_id: u64,
        text: String,
    },
    LlmToken {
        utterance_id: u64,
        token: String,
    },
    LlmDone {
        utterance_id: u64,
        full_text: String,
    },
    TtsStart {
        utterance_id: u64,
    },
    ToolCall {
        name: String,
        result: String,
    },
    MuteChanged {
        muted: bool,
    },
    Error {
        message: String,
    },
}

#[derive(Clone)]
pub struct ControlBroadcast {
    pub tx: broadcast::Sender<ControlEvent>,
}

impl ControlBroadcast {
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self { tx }
    }

    pub fn send(&self, event: ControlEvent) {
        let _ = self.tx.send(event);
    }

    pub fn subscribe(&self) -> broadcast::Receiver<ControlEvent> {
        self.tx.subscribe()
    }
}
