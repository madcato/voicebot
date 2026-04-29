/// Global pipeline state, held in a `watch::Sender<PipelineState>`.
///
/// Each actor that owns a transition writes it directly — no central
/// coordinator sits on the hot path. Observers (TUI, logger) subscribe
/// to the `watch::Receiver` via `watch::Receiver::changed()`.
#[derive(Clone, Debug)]
pub enum PipelineState {
    /// No active utterance.
    Idle,

    /// VAD detected speech; STT is accumulating audio.
    Listening { utterance_id: u64 },

    /// Transcript ready; LLM is generating a response.
    Thinking { utterance_id: u64 },

    /// LLM done; TTS is playing the response.
    Speaking { utterance_id: u64 },

    /// Pipeline temporarily paused (e.g. consolidation running).
    Paused { reason: PauseReason },
}

#[derive(Clone, Debug, PartialEq)]
pub enum PauseReason {
    Consolidation,
}

impl PipelineState {
    pub fn utterance_id(&self) -> Option<u64> {
        match self {
            PipelineState::Listening { utterance_id }
            | PipelineState::Thinking { utterance_id }
            | PipelineState::Speaking { utterance_id } => Some(*utterance_id),
            _ => None,
        }
    }

    /// True when the pipeline is doing active work (not Idle).
    pub fn is_busy(&self) -> bool {
        !matches!(self, PipelineState::Idle)
    }
}
