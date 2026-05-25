pub mod config;
pub mod display_worker;
pub mod session_events;
pub mod session_manager;

pub use config::{AgentConfig, AgentRegistry};
pub use display_worker::{SessionDisplayWorker, session_log_path};
pub use session_events::{AcpSessionEvent, parse_session_update, create_event_channel};
pub use session_manager::{
    AcpSessionManager, SessionEntry, SessionEvent, SessionEventRx, SessionEventTx,
    SessionInfo, SessionStatus, create_session_event_channel,
};

/// Events that trigger proactive speech from Jarvis without a user utterance.
pub enum ProactiveEvent {
    /// A background agent task completed. Jarvis will vocalize the result.
    ///
    /// When `tool_call_id` is `Some`, the completion came from a background tool
    /// that was invoked by the LLM itself (e.g. `web_search`). The pipeline will
    /// inject the proper OpenAI tool result message into the session and let the
    /// LLM continue naturally instead of re-prompting via a user-role notification.
    AgentResult {
        task: String,
        result: String,
        tool_call_id: Option<String>,
    },
    /// The inference daemon decided there is something worth saying proactively.
    /// `message` is the raw observation text; `run_proactive_pipeline` will
    /// reformulate it in Jarvis's voice before speaking.
    InferenceDaemon { message: String },
    /// An ACP agent is requesting user permission for an action. Jarvis speaks
    /// the question, captures the next user utterance, and routes the answer
    /// back via `response_tx`.
    AgentQuestion {
        task_id: String,
        agent_name: String,
        question: String,
        options: Vec<String>,
        /// One-shot channel: send the ACP outcome string ("allow_once" / "reject_once")
        response_tx: tokio::sync::oneshot::Sender<String>,
    },
}

impl std::fmt::Debug for ProactiveEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AgentResult { task, .. } => write!(f, "AgentResult({task:?})"),
            Self::InferenceDaemon { message } => write!(f, "InferenceDaemon({message:?})"),
            Self::AgentQuestion {
                task_id,
                agent_name,
                question,
                options,
                ..
            } => {
                write!(
                    f,
                    "AgentQuestion(task={task_id}, agent={agent_name}, q={question:?}, opts={options:?})"
                )
            }
        }
    }
}
