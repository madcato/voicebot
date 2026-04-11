use tokio::sync::mpsc;

/// Pipeline state for the TUI status bar.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PipelineState {
    Idle,
    Listening,
    Transcribing,
    Thinking,
    Speaking,
}

/// Whether a user message originated from voice or keyboard.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InputSource {
    Voice,
    Text,
}

/// Events sent from the pipeline to the TUI for rendering.
#[derive(Clone, Debug)]
pub enum TuiEvent {
    /// Pipeline state changed.
    StateChange(PipelineState),
    /// User message finalized (from voice STT or typed input).
    UserMessage { text: String, source: InputSource },
    /// A new LLM token arrived (for streaming display).
    AssistantToken(String),
    /// LLM finished streaming this turn.
    AssistantDone,
    /// A tool was called by the LLM.
    ToolCall { name: String, result: String },
    /// A pipeline error occurred that the user should see.
    Error(String),
    /// Show the VOICEBOT splash screen on first render.
    Splash,
}

pub type TuiEventTx = mpsc::UnboundedSender<TuiEvent>;
pub type TuiEventRx = mpsc::UnboundedReceiver<TuiEvent>;
