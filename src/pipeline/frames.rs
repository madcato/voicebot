/// Typed messages that flow between pipeline actors.
///
/// Each variant carries an `utterance_id` so every stage of a single
/// user turn can be correlated in logs and latency metrics.
#[derive(Debug, Clone)]
pub enum PipelineFrame {
    /// STT completed: transcript ready for the LLM actor.
    TranscriptReady { utterance_id: u64, text: String },

    /// A single streamed token from the LLM.
    LLMToken { utterance_id: u64, token: String },

    /// LLM stream finished; carries the full concatenated response.
    LLMResponseDone { utterance_id: u64, full_text: String },

    /// A complete sentence ready for TTS synthesis.
    SentenceReady { utterance_id: u64, sentence: String },

    /// TTS playback of the last sentence for an utterance has finished.
    PlaybackDone { utterance_id: u64 },

    /// System or background notification injected into the LLM as a system turn.
    SystemNotification { text: String },

    /// Result from a background tool/agent, continuing a prior LLM turn.
    AgentResult { task: String, result: String, tool_call_id: Option<String> },

    /// Text typed via TUI — treated as a user turn (no voice path).
    TextInput { text: String },
}
