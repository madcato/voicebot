use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use tokio::sync::{broadcast, mpsc, watch};

use super::broadcast::ControlBroadcast;
use crate::llm::LlmSession;
use crate::pipeline::frames::PipelineFrame;
use crate::pipeline::fsm::PipelineState;

pub struct ControlState {
    pub broadcast: ControlBroadcast,
    pub pipeline_state_rx: watch::Receiver<PipelineState>,
    pub tts_muted: Arc<AtomicBool>,
    pub play_cancel: Arc<AtomicBool>,
    pub barge_in_tx: broadcast::Sender<u64>,
    pub transcript_tx: mpsc::Sender<PipelineFrame>,
    pub llm_session: Arc<Mutex<LlmSession>>,
}
