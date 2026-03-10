/// Events that trigger proactive speech from Jarvis without a user utterance.
#[derive(Debug)]
pub enum ProactiveEvent {
    /// A background agent task completed. Jarvis will vocalize the result.
    AgentResult { task: String, result: String },
    /// The inference daemon decided there is something worth saying proactively.
    /// `message` is the raw observation text; `run_proactive_pipeline` will
    /// reformulate it in Jarvis's voice before speaking.
    InferenceDaemon { message: String },
}
