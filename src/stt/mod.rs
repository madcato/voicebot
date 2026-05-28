pub mod provider;
pub mod whisper;

#[cfg(feature = "parakeet")]
pub mod parakeet;

pub use provider::{SttProvider, create_provider};
#[cfg(feature = "parakeet")]
pub use parakeet::ParakeetSttProvider;
#[allow(unused_imports)]
pub use whisper::{WhisperSTTVAD, WhisperSTTVADConfig, WhisperSttProvider};

/// Events emitted while processing the audio stream.
#[derive(Debug, Clone)]
pub enum SpeechEvent {
    SpeechStart,
    #[allow(dead_code)]
    Speech(String),
    SpeechEnd(String),
}
