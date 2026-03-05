pub mod agents;
pub mod audio;
pub mod config;
pub mod db;
pub mod mcp;
pub mod s2s;
pub mod session;
pub mod tools;

// Re-export commonly used types
pub use agents::AgentManager;
pub use audio::vad::{VoiceActivityDetector, VadResult};
pub use audio::buffer::AudioBuffer;
pub use audio::output::AudioOutput;
pub use db::Database;
pub use mcp::McpServer;
pub use s2s::{S2SAdapter, S2SModel, S2SRequest, S2SResponse, ModelType, ModelConfig};
pub use session::{SessionManager, ConversationContext, Message, MessageRole};
pub use tools::{ToolRouter, ToolRegistry};
