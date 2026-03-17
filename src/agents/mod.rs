/// Events that trigger proactive speech from Jarvis without a user utterance.
pub enum ProactiveEvent {
    /// A background agent task completed. Jarvis will vocalize the result.
    AgentResult { task: String, result: String },
    /// The inference daemon decided there is something worth saying proactively.
    /// `message` is the raw observation text; `run_proactive_pipeline` will
    /// reformulate it in Jarvis's voice before speaking.
    InferenceDaemon { message: String },
    /// An ACP agent is requesting user permission for an action. Jarvis speaks
    /// the question, captures the next user utterance, and routes the answer
    /// back via `response_tx`.
    AgentQuestion {
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
            Self::AgentQuestion { question, options, .. } => {
                write!(f, "AgentQuestion({question:?}, {options:?})")
            }
        }
    }
}
