use anyhow::{Context, Result};

// Import path for kokorox (fork of kokoros/Kokoros).
// If the crate re-exports TTSKoko at the root, use: `kokorox::TTSKoko`.
// Check with: cargo doc --features kokoro --open
use kokorox::tts::koko::TTSKoko;

/// Kokoro TTS wrapper.
///
/// Wraps `TTSKoko` (ONNX-based, 24 kHz output) with the same `synthesize` /
/// `sample_rate` interface as `SayTts` so it can be swapped in via `TtsEngine`.
///
/// Requires:
/// - `brew install espeak-ng`  (macOS)
/// - `models/kokoro-v1.0.onnx` + `models/voices-v1.0.bin`
///   (download from <https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX>)
pub struct KokoroTts {
    inner: TTSKoko,
    /// Voice style name, e.g. `"af_bella"`, `"es_*"`.
    voice: String,
    /// BCP-47 language code passed to espeak-ng, e.g. `"en-us"`, `"es"`.
    language: String,
}

impl KokoroTts {
    /// Load the Kokoro ONNX model and voice embeddings asynchronously.
    ///
    /// `model_path`  — path to `kokoro-v1.0.onnx`
    /// `voices_path` — path to `voices-v1.0.bin`
    /// `voice`       — voice style name (see `tts.get_available_voices()`)
    /// `language`    — BCP-47 code for espeak-ng phonemisation
    pub async fn new(
        model_path: &str,
        voices_path: &str,
        voice: &str,
        language: &str,
    ) -> Result<Self> {
        let inner = TTSKoko::new(Some(model_path), Some(voices_path)).await;
        tracing::info!(
            "Kokoro TTS loaded: model={} voice={} lang={}",
            model_path, voice, language
        );
        Ok(Self {
            inner,
            voice: voice.to_string(),
            language: language.to_string(),
        })
    }

    /// Synthesise `text` into mono f32 PCM at 24 000 Hz.
    ///
    /// CPU-intensive — call from `tokio::task::spawn_blocking`.
    pub fn synthesize(&self, text: &str) -> Result<Vec<f32>> {
        self.inner
            .tts_raw_audio(
                text,
                &self.language,
                &self.voice,
                1.0,   // speed
                None,  // initial_silence
                false, // auto_detect_language (we set it explicitly)
                true, // force_style
                false, // phonemes mode
            )
            .map_err(|e| anyhow::anyhow!("Kokoro TTS: {}", e))
    }

    pub fn sample_rate(&self) -> u32 {
        24_000
    }
}
