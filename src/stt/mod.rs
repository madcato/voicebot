// whisper-cpp-plus implementation with true streaming support + Metal GPU acceleration on macOS
pub mod whisper_plus;
pub mod stream;

pub use stream::SttStream;
// Re-export WhisperSttPlus for direct access (optional, kept for completeness)
#[allow(unused_imports)]
pub use whisper_plus::WhisperSttPlus;
// Primary export - alias for backward compatibility
pub use whisper_plus::WhisperSttPlus as WhisperStt;
