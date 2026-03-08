/// Events that trigger proactive speech from Jarvis without a user utterance.
#[derive(Debug)]
pub enum ProactiveEvent {
    /// A background agent task completed. Jarvis will vocalize the result.
    AgentResult { task: String, result: String },
}
