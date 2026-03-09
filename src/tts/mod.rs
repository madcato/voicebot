pub mod piper;
pub mod say;
pub mod sentence;
#[cfg(feature = "kokoro")]
pub mod kokoro;

pub use say::SayTts;
pub use sentence::SentenceSplitter;
#[cfg(feature = "kokoro")]
pub use kokoro::KokoroTts;

/// Unified TTS backend.
///
/// Select at startup via `TTS_PROVIDER` env var:
/// - `"say"` (default) — macOS `say` subprocess, configured by `SAY_VOICE`
/// - `"kokoro"` — Kokoro ONNX model; requires `--features kokoro` and espeak-ng
///
/// Both variants expose the same `synthesize` / `sample_rate` interface so the
/// rest of the pipeline is backend-agnostic.
pub enum TtsEngine {
    Say(SayTts),
    #[cfg(feature = "kokoro")]
    Kokoro(KokoroTts),
}

impl TtsEngine {
    pub fn synthesize(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        match self {
            Self::Say(t) => t.synthesize(text),
            #[cfg(feature = "kokoro")]
            Self::Kokoro(t) => t.synthesize(text),
        }
    }

    pub fn sample_rate(&self) -> u32 {
        match self {
            Self::Say(t) => t.sample_rate(),
            #[cfg(feature = "kokoro")]
            Self::Kokoro(t) => t.sample_rate(),
        }
    }
}
