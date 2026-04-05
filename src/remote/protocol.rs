use serde::{Deserialize, Serialize};

// ── Client → Server (text frames) ───────────────────────────────────────────

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ClientMessage {
    #[serde(rename = "session.start")]
    SessionStart {
        #[serde(default = "default_sample_rate")]
        sample_rate: u32,
    },
    #[serde(rename = "barge_in")]
    BargeIn,
}

fn default_sample_rate() -> u32 {
    16000
}

// ── Server → Client (text frames) ───────────────────────────────────────────

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
#[allow(dead_code)]
pub enum ServerMessage {
    #[serde(rename = "session.ready")]
    SessionReady,
    #[serde(rename = "transcript")]
    Transcript { text: String },
    #[serde(rename = "response.text")]
    ResponseText { text: String },
    #[serde(rename = "response.end")]
    ResponseEnd,
    #[serde(rename = "audio.start")]
    AudioStart,
    #[serde(rename = "audio.end")]
    AudioEnd,
    #[serde(rename = "error")]
    Error { message: String },
}

/// TTS audio packet sent from the pipeline to the WebSocket sink task.
pub struct TtsAudioPacket {
    /// Mono f32 samples at `sample_rate`.
    pub samples: Vec<f32>,
    /// Sample rate of the audio.
    pub sample_rate: u32,
}
