pub mod piper;
pub mod sentence;
#[cfg(feature = "kokoro")]
pub mod kokoro;
#[cfg(feature = "avspeech")]
pub mod avspeech;

pub use sentence::SentenceSplitter;
#[cfg(feature = "kokoro")]
pub use kokoro::KokoroTts;
#[cfg(feature = "avspeech")]
pub use avspeech::AvSpeechTts;

/// Unified TTS backend.
///
/// Select at startup via `TTS_PROVIDER` env var:
/// - `"avspeech"` (default) — Native macOS `AVSpeechSynthesizer`; requires `--features avspeech`.
///   Configured by `AVSPEECH_VOICE` / `AVSPEECH_RATE`.
/// - `"kokoro"` — Kokoro ONNX model; requires `--features kokoro` and espeak-ng.
///
/// All variants expose the same `synthesize` / `sample_rate` interface so the
/// rest of the pipeline is backend-agnostic.
pub enum TtsEngine {
    #[cfg(feature = "avspeech")]
    AvSpeech(AvSpeechTts),
    #[cfg(feature = "kokoro")]
    Kokoro(KokoroTts),
    /// Test-only variant: captures synthesized text instead of producing audio.
    /// Returns a single silent sample so AudioOutput.play_blocking() returns instantly.
    #[cfg(test)]
    Mock(mock_tts::MockTts),
}

impl TtsEngine {
    #[allow(unreachable_patterns)]
    pub fn synthesize(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        match self {
            #[cfg(feature = "avspeech")]
            Self::AvSpeech(t) => t.synthesize(text),
            #[cfg(feature = "kokoro")]
            Self::Kokoro(t) => t.synthesize(text),
            #[cfg(test)]
            Self::Mock(t) => t.synthesize(text),
            _ => unreachable!("no TTS backend enabled"),
        }
    }

    #[allow(unreachable_patterns)]
    pub fn sample_rate(&self) -> u32 {
        match self {
            #[cfg(feature = "avspeech")]
            Self::AvSpeech(t) => t.sample_rate(),
            #[cfg(feature = "kokoro")]
            Self::Kokoro(t) => t.sample_rate(),
            #[cfg(test)]
            Self::Mock(t) => t.sample_rate(),
            _ => unreachable!("no TTS backend enabled"),
        }
    }
}

/// Test-only TTS backend. Captures sentence text to a shared Vec instead of
/// synthesizing audio. Returns a single silent sample so AudioOutput returns
/// instantly without requiring real audio synthesis.
#[cfg(test)]
pub mod mock_tts {
    use std::sync::{Arc, Mutex};

    pub struct MockTts(pub Arc<Mutex<Vec<String>>>);

    impl MockTts {
        /// Returns `(MockTts, captured)` where `captured` is shared with the engine
        /// and accumulates every sentence text passed to `synthesize()`.
        pub fn new() -> (Self, Arc<Mutex<Vec<String>>>) {
            let cap = Arc::new(Mutex::new(Vec::new()));
            (Self(Arc::clone(&cap)), cap)
        }

        pub fn synthesize(&self, text: &str) -> anyhow::Result<Vec<f32>> {
            self.0.lock().unwrap().push(text.to_string());
            Ok(vec![0.0f32]) // 1 silent sample → play_blocking returns immediately
        }

        pub fn sample_rate(&self) -> u32 {
            22050
        }
    }
}
