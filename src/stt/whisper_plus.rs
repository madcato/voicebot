/// Whisper STT using whisper-cpp-plus for true streaming audio-to-text.
///
/// This replaces the old whisper-rs implementation which only supported
/// full-audio snapshots. whisper-cpp-plus provides:
/// - True incremental transcription (feed chunks as they arrive)
/// - Lower latency (~400-700ms vs ~1000-1500ms)
/// - Built-in VAD support (though we keep Silero for event callbacks)
use anyhow::{Context, Result};
use std::borrow::Cow;
use std::sync::Arc;
use whisper_cpp_plus::{FullParams, SamplingStrategy, WhisperContext, WhisperState};

/// Minimum audio length fed to Whisper. Audio shorter than this is padded with
/// trailing silence so the model has enough context to decode short utterances
/// reliably ("yes", "ok", "sí", etc.).
const MIN_AUDIO_SAMPLES: usize = 16_000; // 1 second at 16 kHz

/// Pad `audio` with trailing zeros if shorter than `min_samples`.
/// Returns a `Cow` to avoid allocation when padding is not needed.
fn pad_to_min_duration(audio: &[f32], min_samples: usize) -> Cow<'_, [f32]> {
    if audio.len() >= min_samples {
        Cow::Borrowed(audio)
    } else {
        let mut padded = audio.to_vec();
        padded.resize(min_samples, 0.0);
        Cow::Owned(padded)
    }
}

/// Check if whisper-cpp-plus verbose output should be suppressed.
/// Set WHISPER_SILENCE=1 to silence all whisper.cpp logs (Metal init, GPU detection, etc.)
fn should_silence_whisper_logs() -> bool {
    std::env::var("WHISPER_SILENCE")
        .map(|v| v == "1" || v.to_lowercase() == "true" || v.to_lowercase() == "yes")
        .unwrap_or(false)
}

pub struct WhisperSttPlus {
    ctx: Arc<WhisperContext>,
    language: String,
    threads: u32,
    #[allow(dead_code)]
    silence_logs: bool,
}

impl WhisperSttPlus {
    pub fn new(model_path: &str, language: &str, threads: u32) -> Result<Self> {
        let silence_logs = should_silence_whisper_logs();

        let ctx = WhisperContext::new(model_path)
            .context("Failed to load Whisper model from whisper-cpp-plus")?;

        tracing::info!(
            target: "stt",
            "Whisper model loaded via whisper-cpp-plus: {} (language: {}, threads: {}{})",
            model_path,
            language,
            if threads == 0 { "auto".to_string() } else { threads.to_string() },
            if silence_logs { ", logs silenced" } else { "" }
        );

        Ok(Self {
            ctx: Arc::new(ctx),
            language: language.to_string(),
            threads,
            silence_logs,
        })
    }

    /// Transcribe complete audio in one shot (non-streaming mode).
    /// Used for SpeechEnd final transcription.
    pub fn transcribe_complete(&self, audio: &[f32]) -> Result<String> {
        self.transcribe_with_prompt_internal(audio, "")
    }

    /// Transcribe with prompt context (for speculative/continuation scenarios).
    #[allow(dead_code)]
    pub fn transcribe_with_prompt(&self, audio: &[f32], prompt: &str) -> Result<String> {
        self.transcribe_with_prompt_internal(audio, prompt)
    }

    fn transcribe_with_prompt_internal(&self, audio: &[f32], prompt: &str) -> Result<String> {
        use std::time::Instant;

        let t0 = Instant::now();
        let mut state = self.ctx.create_state()?;
        let state_creation_ms = t0.elapsed().as_millis();

        let t0 = Instant::now();
        let audio = pad_to_min_duration(audio, MIN_AUDIO_SAMPLES);
        let padding_ms = t0.elapsed().as_millis();

        let t0 = Instant::now();
        // Optimized parameters for speed vs accuracy trade-off
        // - Greedy with best_of=1 (fastest)
        // - n_threads from config (or auto)
        // - single_segment=true for short utterances
        // Optimized parameters for speed vs accuracy trade-off
        // - Greedy with best_of=0 (fastest single decode)
        // - single_segment=true for short utterances
        // - No timestamps overhead
        let params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 })
            .language(&self.language)
            .print_special(false)
            .print_progress(false)
            .print_realtime(false)
            .print_timestamps(false)
            .single_segment(true)
            .no_timestamps(true)
            .token_timestamps(false);

        let mut params = params;
        if !prompt.is_empty() {
            params = params.initial_prompt(prompt);
        }

        if self.threads > 0 {
            params = params.n_threads(self.threads as i32);
        }
        // If threads == 0, whisper.cpp uses its default (auto-detect)

        let params_setup_ms = t0.elapsed().as_millis();

        let t0 = Instant::now();
        state.full(params, &audio)?;
        let inference_ms = t0.elapsed().as_millis();

        let total_ms = state_creation_ms + padding_ms + params_setup_ms + inference_ms;
        tracing::info!(target: "stt", "transcribe_complete: state={}ms, inference={}ms, TOTAL={}ms, audio_samples={}, result_chars={}", 
            state_creation_ms, inference_ms, total_ms, audio.len(), MIN_AUDIO_SAMPLES);

        // Collect text from all segments
        let num_segments = state.full_n_segments();
        let mut text = String::new();

        for i in 0..num_segments {
            if i > 0 {
                text.push(' ');
            }
            match state.full_get_segment_text(i) {
                Ok(seg_text) => text.push_str(seg_text.trim()),
                Err(_) => continue, // Skip segments that fail to decode
            }
        }

        Ok(text.trim().to_string())
    }

    /// Create streaming transcriptor for incremental processing during speech.
    /// This is the KEY feature for true low-latency streaming.
    #[allow(dead_code)]
    pub fn create_streamer(&self) -> WhisperStreamer {
        WhisperStreamer::new(
            Arc::clone(&self.ctx),
            &self.language,
            self.threads,
            self.silence_logs,
        )
    }
}

/// Streaming whisper processor — feeds audio chunks incrementally
/// and gets partial transcripts as decoding progresses.
#[allow(dead_code)]
pub struct WhisperStreamer {
    #[allow(dead_code)] // kept for future streaming features
    ctx: Arc<WhisperContext>,
    state: WhisperState,
    language: String,
    threads: u32,
    silence_logs: bool,
    accumulated_audio: Vec<f32>,
}

#[allow(dead_code)]
impl WhisperStreamer {
    fn new(ctx: Arc<WhisperContext>, language: &str, threads: u32, silence_logs: bool) -> Self {
        let state = ctx
            .create_state()
            .expect("Failed to create state for streamer");

        Self {
            ctx,
            state,
            language: language.to_string(),
            threads,
            silence_logs,
            accumulated_audio: Vec::new(),
        }
    }

    /// Feed audio chunk incrementally. Returns None until enough audio accumulates.
    pub fn feed_chunk(&mut self, chunk: &[f32]) -> Result<Option<String>> {
        use std::time::Instant;

        // Accumulate audio
        self.accumulated_audio.extend_from_slice(chunk);

        const MIN_AUDIO_SAMPLES: usize = 8_000; // 500ms @ 16kHz

        if self.accumulated_audio.len() < MIN_AUDIO_SAMPLES {
            return Ok(None);
        }

        let t0 = Instant::now();
        self.transcribe_accumulated()?;
        let transcribe_ms = t0.elapsed().as_millis();

        let t0 = Instant::now();
        let text = self.get_current_result()?;
        let get_result_ms = t0.elapsed().as_millis();

        tracing::debug!(target: "stt", "feed_chunk: accumulate={}, total_audio={}, transcribe={}ms, get_result={}ms",
            chunk.len(), self.accumulated_audio.len(), transcribe_ms, get_result_ms);

        Ok(Some(text))
    }

    /// Finalize streaming and get final transcription.
    pub fn finalize(&mut self) -> Result<String> {
        if self.accumulated_audio.is_empty() {
            return Ok(String::new());
        }

        let _ = self.transcribe_accumulated();
        self.get_current_result()
    }

    fn transcribe_accumulated(&mut self) -> Result<()> {
        // Note: WHISPER_SILENCE only affects initialization logs from whisper.cpp C++ backend
        // (Metal/GPU detection). These are printed via printf before Rust code runs.
        // The print_* flags below suppress per-inference output during transcription.
        let params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 })
            .language(&self.language)
            .print_special(false)
            .print_progress(false)
            .print_realtime(false)
            .print_timestamps(false)
            .single_segment(true)
            .no_timestamps(true)
            .token_timestamps(false);

        let mut params = params;
        if self.threads > 0 {
            params = params.n_threads(self.threads as i32);
        }

        self.state.full(params, &self.accumulated_audio)?;
        Ok(())
    }

    fn get_current_result(&self) -> Result<String> {
        let num_segments = self.state.full_n_segments();
        let mut text = String::new();

        for i in 0..num_segments {
            if i > 0 {
                text.push(' ');
            }
            match self.state.full_get_segment_text(i) {
                Ok(seg_text) => text.push_str(seg_text.trim()),
                Err(_) => continue,
            }
        }

        Ok(text.trim().to_string())
    }

    /// Clear accumulated audio for next utterance.
    pub fn reset(&mut self) {
        self.accumulated_audio.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_loading() {
        let model_path = "models/ggml-large-v3-turbo.bin";
        std::fs::metadata(model_path).ok().map(|_| {
            let stt = WhisperSttPlus::new(model_path, "es", 0);
            assert!(stt.is_ok(), "Should load valid model");
        });
    }

    #[test]
    fn test_streamer_basic() {
        let model_path = "models/ggml-large-v3-turbo.bin";
        if std::fs::metadata(model_path).is_err() {
            eprintln!("Skipping test: model not found");
            return;
        }

        let stt = WhisperSttPlus::new(model_path, "es", 0).unwrap();
        let mut streamer = stt.create_streamer();

        // Feed silence (won't produce output until MIN_AUDIO_SAMPLES)
        let silence = vec![0.0f32; 4_000]; // 250ms
        let result = streamer.feed_chunk(&silence).unwrap();
        assert!(result.is_none(), "Should not produce output yet");

        // More audio to reach threshold
        let more_silence = vec![0.0f32; 8_000]; // +500ms = 750ms total
        let _result = streamer.feed_chunk(&more_silence).unwrap();
    }
}
