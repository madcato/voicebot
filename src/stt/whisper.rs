use anyhow::{Context, Result};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

pub struct WhisperStt {
    ctx: WhisperContext,
    language: String,
}

impl WhisperStt {
    pub fn new(model_path: &str, language: &str) -> Result<Self> {
        let ctx = WhisperContext::new_with_params(model_path, WhisperContextParameters::default())
            .context("Failed to load Whisper model")?;

        tracing::info!("Whisper model loaded: {} (language: {})", model_path, language);

        Ok(Self {
            ctx,
            language: language.to_string(),
        })
    }

    /// Transcribe mono f32 audio sampled at 16 kHz.
    /// This is CPU-intensive — call from `tokio::task::spawn_blocking`.
    pub fn transcribe(&self, audio: &[f32]) -> Result<String> {
        let mut state = self.ctx.create_state().context("Failed to create Whisper state")?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });
        params.set_language(Some(&self.language));
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        // Single segment gives cleaner output for short utterances
        params.set_single_segment(false);

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
