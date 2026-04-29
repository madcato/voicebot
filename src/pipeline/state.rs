use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::{broadcast, Notify};

/// Monotonically increasing counter for tagging each pipeline run with a unique ID.
pub static PIPELINE_RUN_ID: AtomicU64 = AtomicU64::new(0);

/// Shared state between the pipeline tasks.
pub struct SharedSession {
    /// Latest text transcribed by STT. Replaced on each new STT result.
    pub transliterated_text: Mutex<String>,
    /// LLM token stream buffer — text not yet split into sentences.
    pub assistant_text: Mutex<String>,
    /// Sentences ready for TTS playback.
    pub sentences: Mutex<VecDeque<String>>,
    /// True when the current LLM streaming POST has finished.
    pub llm_post_finished: AtomicBool,
    /// True while the LLM task is actively processing a turn.
    pub llm_busy: AtomicBool,
    /// True when the pending transliterated_text came from TUI text input (not voice).
    pub text_input_pending: AtomicBool,
    /// Timestamp of the most recent VAD SpeechEnd — used for end-to-end latency logging.
    pub t_vad_end: Mutex<Option<Instant>>,
    /// Timestamp of the POST sent to LLM — used to calculate TTFT.
    pub t_llm_post_send: Mutex<Option<Instant>>,
    /// Track if first sentence playback has been logged.
    pub first_speech_played: AtomicBool,
    /// True while the consolidation task is running.
    pub consolidation_active: AtomicBool,
    /// True when an STT result has been obtained but not yet fully processed by the LLM.
    pub stt_result_pending: AtomicBool,
    /// True when a background tool has delivered its result and the session already contains
    /// the tool_call + tool_result exchange.
    pub pending_tool_response: AtomicBool,
    /// True when the drained `transliterated_text` is a background-task notification that
    /// should be added to the session as a `system` turn (not a `user` turn).
    pub pending_system_injection: AtomicBool,
}

impl SharedSession {
    pub fn new() -> Self {
        Self {
            transliterated_text: Mutex::new(String::new()),
            assistant_text: Mutex::new(String::new()),
            sentences: Mutex::new(VecDeque::new()),
            llm_post_finished: AtomicBool::new(false),
            llm_busy: AtomicBool::new(false),
            text_input_pending: AtomicBool::new(false),
            t_vad_end: Mutex::new(None),
            t_llm_post_send: Mutex::new(None),
            first_speech_played: AtomicBool::new(false),
            consolidation_active: AtomicBool::new(false),
            stt_result_pending: AtomicBool::new(false),
            pending_tool_response: AtomicBool::new(false),
            pending_system_injection: AtomicBool::new(false),
        }
    }
}

/// Signals and events for inter-task communication.
pub struct PipelineEvents {
    /// VAD_DETECTED: broadcast cancellation. All tasks must stop immediately.
    pub cancel_tx: broadcast::Sender<()>,
    /// VAD_FINISH: silence detected; LLM task should start processing.
    pub vad_finish: Arc<Notify>,
    /// LLM_POST_RECEIVED: a token arrived from the LLM stream.
    pub llm_post_received: Arc<Notify>,
    /// SENTENCE_READY: a sentence has been pushed to shared.sentences.
    pub sentence_ready: Arc<Notify>,
    /// LLM_POST_FINISHED: the LLM has streamed its complete response.
    pub llm_post_finished: Arc<Notify>,
}

impl PipelineEvents {
    pub fn new() -> Self {
        let (cancel_tx, _) = broadcast::channel(16);
        Self {
            cancel_tx,
            vad_finish: Arc::new(Notify::new()),
            llm_post_received: Arc::new(Notify::new()),
            sentence_ready: Arc::new(Notify::new()),
            llm_post_finished: Arc::new(Notify::new()),
        }
    }
}
