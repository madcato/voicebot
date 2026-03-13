use anyhow::{Context, Result};
use std::sync::Mutex;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState};

/// Persistent Whisper STT.
///
/// `WhisperState` initialises the Metal GPU backend (`ggml_metal_init`) on
/// creation and tears it down (`ggml_metal_free`) on drop. Creating a fresh
/// state per utterance wastes ~200 ms re-loading Metal kernels each time.
///
/// We allocate the state once in `new()` and reuse it across calls via a
/// `Mutex`, keeping Metal resident in GPU memory for the process lifetime.
pub struct WhisperStt {
    _ctx: WhisperContext,
    /// Reusable inference state — owns Metal buffers and KV caches.
    state: Mutex<WhisperState>,
    language: String,
}

impl WhisperStt {
    pub fn new(model_path: &str, language: &str) -> Result<Self> {
        let ctx = WhisperContext::new_with_params(model_path, WhisperContextParameters::default())
            .context("Failed to load Whisper model")?;

        let state = ctx.create_state().context("Failed to create Whisper state")?;

        tracing::info!(
            target: "stt",
            "Whisper model loaded: {} (language: {}) — Metal state cached",
            model_path,
            language
        );

        Ok(Self {
            _ctx: ctx,
            state: Mutex::new(state),
            language: language.to_string(),
        })
    }

    /// Transcribe mono f32 audio sampled at 16 kHz.
    /// CPU-intensive — call from `tokio::task::spawn_blocking`.
    pub fn transcribe(&self, audio: &[f32]) -> Result<String> {
        let mut state = self.state.lock().unwrap();

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });
        params.set_language(Some(&self.language));
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_single_segment(false);
        // Don't predict timestamp tokens between words — saves decoder steps.
        params.set_no_timestamps(true);
        // Skip the word-level timestamp alignment post-processing pass.
        params.set_token_timestamps(false);

        state
            .full(params, audio)
            .context("Whisper transcription failed")?;

        let num_segments = state.full_n_segments();
        let mut text = String::new();
        for i in 0..num_segments {
            if let Some(seg) = state.get_segment(i) {
                let seg_text = seg.to_str().unwrap_or("").trim().to_string();
                if !seg_text.is_empty() {
                    if !text.is_empty() {
                        text.push(' ');
                    }
                    text.push_str(&seg_text);
                }
            }
        }

        Ok(text.trim().to_string())
    }
}
