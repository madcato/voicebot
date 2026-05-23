use std::sync::Arc;
use tokio::sync::{Notify, broadcast};

/// Signals and events for inter-task communication.
pub struct PipelineEvents {
    /// BARGE_IN: VAD SpeechStart — all tasks must cancel current work.
    /// Payload is the utterance_id of the new speech that interrupted.
    pub barge_in_tx: broadcast::Sender<u64>,
    /// LLM_POST_FINISHED: the LLM has streamed its complete response.
    /// Still used by consolidation_task to reset the idle timer.
    pub llm_post_finished: Arc<Notify>,
}

impl PipelineEvents {
    pub fn new() -> Self {
        let (barge_in_tx, _) = broadcast::channel(16);
        Self {
            barge_in_tx,
            llm_post_finished: Arc::new(Notify::new()),
        }
    }
}
