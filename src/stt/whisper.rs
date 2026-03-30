use anyhow::{Context, Result};
use std::borrow::Cow;
use std::sync::Mutex;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState};

/// Minimum audio length fed to Whisper. Audio shorter than this is padded with
/// trailing silence so the model has enough context to decode short utterances
/// reliably ("yes", "ok", "sí", etc.).
const MIN_AUDIO_SAMPLES: usize = 16_000; // 1 second at 16 kHz

/// Pad `audio` with trailing zeros if shorter than `min_samples`.
/// Returns a `Cow` to avoid allocation when padding is not needed.
pub(crate) fn pad_to_min_duration(audio: &[f32], min_samples: usize) -> Cow<'_, [f32]> {
    if audio.len() >= min_samples {
        Cow::Borrowed(audio)
    } else {
        let mut padded = audio.to_vec();
        padded.resize(min_samples, 0.0);
        Cow::Owned(padded)
    }
}

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
    /// Number of CPU threads for whisper decoding. 0 = let whisper.cpp decide.
    threads: u32,
}

impl WhisperStt {
    pub fn new(model_path: &str, language: &str, threads: u32) -> Result<Self> {
        let ctx = WhisperContext::new_with_params(model_path, WhisperContextParameters::default())
            .context("Failed to load Whisper model")?;

        let state = ctx.create_state().context("Failed to create Whisper state")?;

        tracing::info!(
            target: "stt",
            "Whisper model loaded: {} (language: {}, threads: {}) — Metal state cached",
            model_path,
            language,
            if threads == 0 { "auto".to_string() } else { threads.to_string() }
        );

        Ok(Self {
            _ctx: ctx,
            state: Mutex::new(state),
            language: language.to_string(),
            threads,
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
        // Single segment mode: skip the "should I split?" heuristic — voice
        // utterances are short enough that a single decoder pass suffices.
        // Saves ~20-50ms by avoiding the segment-boundary search.
        params.set_single_segment(true);
        // Don't predict timestamp tokens between words — saves decoder steps.
        params.set_no_timestamps(true);
        // Skip the word-level timestamp alignment post-processing pass.
        params.set_token_timestamps(false);
        // Explicit thread count: match physical cores for optimal throughput.
        // Default 0 lets whisper.cpp pick, which may over-subscribe on HT CPUs.
        if self.threads > 0 {
            params.set_n_threads(self.threads as i32);
        }

        let audio = pad_to_min_duration(audio, MIN_AUDIO_SAMPLES);
        state
            .full(params, &audio)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_audio_is_padded_to_one_second() {
        let short = vec![0.1_f32; 4_000]; // 250 ms
        let padded = pad_to_min_duration(&short, 16_000);
        assert_eq!(padded.len(), 16_000);
        assert_eq!(&padded[..4_000], short.as_slice());
        assert!(padded[4_000..].iter().all(|&s| s == 0.0));
    }

    #[test]
    fn audio_at_exactly_one_second_is_not_reallocated() {
        let audio = vec![0.5_f32; 16_000];
        let result = pad_to_min_duration(&audio, 16_000);
        assert!(matches!(result, Cow::Borrowed(_)));
    }

    #[test]
    fn longer_audio_is_not_truncated() {
        let long = vec![0.3_f32; 32_000]; // 2 seconds
        let result = pad_to_min_duration(&long, 16_000);
        assert_eq!(result.len(), 32_000);
    }
}
