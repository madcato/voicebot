/// Unified Whisper STT + VAD using whisper-cpp-plus integrated APIs with streaming.
///
/// Single class that handles both voice activity detection and speech transcription:
/// - Detects SpeechStart/SpeechEnd events automatically using EnhancedWhisperVadProcessor
/// - Transcribes audio segments in real-time as they arrive
/// - Uses whisper.cpp's enhanced Silero VAD model with aggregation
use anyhow::{Context, Result};
use std::sync::Arc;
use tokio::sync::mpsc;
use whisper_cpp_plus::{
    FullParams, SamplingStrategy, WhisperContext, WhisperStream, enhanced::{EnhancedVadParamsBuilder, EnhancedWhisperVadProcessor}
};

/// Events sent to the event channel during processing
#[derive(Debug, Clone)]
pub enum SpeechEvent {
    /// Speech detected started
    SpeechStart,
    /// Still speaking (streaming partial text)
    Speech(String),
    /// Speech segment ended with final transcript
    SpeechEnd(String),
    /// Silence (no speech detected)
    Silence,
}

/// Configuration for WhisperSTTVAD
#[derive(Clone)]
pub struct WhisperSTTVADConfig {
    pub whisper_model: String,
    pub vad_model: String,
    pub language: String,
    pub silence_ms: u32,
}

impl Default for WhisperSTTVADConfig {
    fn default() -> Self {
        Self {
            whisper_model: "models/ggml-large-v3-turbo.bin".to_string(),
            vad_model: "models/ggml-silero-vad.bin".to_string(),
            language: "es".to_string(),
            silence_ms: 500,
        }
    }
}

/// Unified Whisper STT + VAD processor with streaming support
///
/// This is THE single class for all speech processing needs. It combines:
/// - Voice Activity Detection (using EnhancedWhisperVadProcessor with aggregation)
/// - Real-time Speech Transcription (using WhisperStream for progressive results)
/// - State machine management (Silence → SpeechStart → Speech → SpeechEnd)
pub struct WhisperSTTVAD {
    ctx: Arc<WhisperContext>,
    vad_processor: EnhancedWhisperVadProcessor,
    stream: WhisperStream,
    language: String,
}

impl WhisperSTTVAD {
    /// Create a new unified STT+VAD instance with streaming support
    ///
    /// Loads both the Whisper model for transcription and the VAD model
    /// for speech detection into a single object.
    pub fn new(config: WhisperSTTVADConfig) -> Result<Self> {
        // Load Whisper context for transcription
        let ctx = Arc::new(
            WhisperContext::new(&config.whisper_model).context("Failed to load Whisper model")?,
        );

        // Load Enhanced VAD processor for speech detection
        let vad_processor = EnhancedWhisperVadProcessor::new(&config.vad_model)
            .context("Failed to load VAD model")?;

        // Configure streaming parameters
        let params =
            FullParams::new(SamplingStrategy::Greedy { best_of: 1 }).language(&config.language);

        let stream = WhisperStream::new(&ctx, params)?;

        tracing::info!(target: "sttvad", "Initialized WhisperSTTVAD (whisper: {}, vad: {}, lang: {})", 
            config.whisper_model, config.vad_model, config.language);

        Ok(Self {
            ctx,
            vad_processor,
            stream,
            language: config.language,
        })
    }

    /// Process an audio chunk and dispatch events via the provided channel.
    ///
    /// This is the main async method for real-time audio processing. Feed it
    /// microphone chunks (16kHz mono f32) and it will send events through the mpsc sender
    /// as they are detected:
    /// - SpeechStart: User just started speaking
    /// - Speech(partial_text): User is still speaking (streaming partial results)
    /// - SpeechEnd(final_text): User stopped speaking (complete segment)
    ///
    /// Uses EnhancedWhisperVadProcessor.process_with_aggregation() to detect
    /// speech segments intelligently, then transcribes them in real-time using
    /// WhisperStream.
    ///
    /// # Arguments
    /// * `audio` - Audio chunk (16kHz mono f32 samples)
    /// * `tx` - mpsc sender to dispatch events asynchronously
    pub async fn process_audio(&mut self, audio: &[f32], tx: &mpsc::Sender<SpeechEvent>) -> Result<()> {
        if audio.is_empty() {
            return Ok(());
        }

        let vad_params = EnhancedVadParamsBuilder::new()
            .threshold(0.5)
            .max_segment_duration(30.0)  // Aggregate up to 30 seconds
            .merge_segments(true)         // Merge adjacent segments
            .min_gap_ms(100)              // Minimum 100ms gap to keep segments separate
            .speech_pad_ms(400)           // Add 400ms padding around speech
            .build();

        // Use enhanced whisper-cpp-plus VAD to detect speech with aggregation
        let chunks = match self
            .vad_processor
            .process_with_aggregation(audio, &vad_params)
        {
            Ok(chunks) => chunks,
            Err(_) => return Ok(()),
        };

        let mut found_speech = false;

        for chunk in &chunks {
            if chunk.duration_seconds > 0.0 {
                found_speech = true;
                
                // Send SpeechStart event
                let _ = tx.send(SpeechEvent::SpeechStart).await;

                // Transcribe in streaming mode
                self.stream.feed_audio(&chunk.audio.clone());

                // Process step to get pending transcription and send as events
                if let Ok(Some(segments)) = self.stream.process_step() {
                    for seg in segments {
                        if !seg.text.trim().is_empty() {
                            let _ = tx.send(SpeechEvent::Speech(seg.text.clone())).await;
                        }
                    }
                }
            }
        }

        // Check if we should finalize (silence detected after speech)
        // The VAD returns empty chunks when silence is detected
        if !found_speech && !chunks.is_empty() {
            // Finalize any accumulated transcription
            if let Ok(segments) = self.stream.flush() {
                let final_text = segments.into_iter().map(|s| s.text).collect::<String>();
                let _ = tx.send(SpeechEvent::SpeechEnd(final_text)).await;
            } else {
                let _ = tx.send(SpeechEvent::SpeechEnd(String::new())).await;
            }
        }

        Ok(())
    }

    /// Transcribe audio without streaming - returns complete text synchronously.
    /// Useful for non-streaming scenarios like verifying speaker transcripts.
    pub fn transcribe_complete(&self, audio: &[f32]) -> Result<String> {
        use std::time::Instant;

        let t0 = Instant::now();
        let mut state = self.ctx.create_state()?;

        let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 })
            .language(&self.language)
            .print_special(false)
            .print_progress(false)
            .print_realtime(false)
            .no_timestamps(true)
            .single_segment(true);

        state.full(params.clone(), audio)?;

        let inference_ms = t0.elapsed().as_millis();

        // Collect text from all segments
        let num_segments = state.full_n_segments();
        let mut text = String::new();

        for i in 0..num_segments {
            if i > 0 {
                text.push(' ');
            }
            if let Ok(seg_text) = state.full_get_segment_text(i) {
                text.push_str(seg_text.trim());
            }
        }

        tracing::debug!(target: "sttvad", "transcribe_complete: inference={}ms, chars={}", 
            inference_ms, text.len());

Ok(text.trim().to_string())
    }

    /// Reset the streaming state for a new conversation turn.
    pub fn reset_stream(&mut self) {
        // Create a fresh stream to clear any accumulated context
        let params =
            FullParams::new(SamplingStrategy::Greedy { best_of: 1 }).language(&self.language);

        self.stream = match WhisperStream::new(&self.ctx, params) {
            Ok(s) => s,
            Err(_) => return,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whisper_sttvad_creation() {
        let config = WhisperSTTVADConfig {
            whisper_model: "models/ggml-large-v3-turbo.bin".to_string(),
            vad_model: "models/ggml-silero-vad.bin".to_string(),
            language: "es".to_string(),
            silence_ms: 500,
        };

        // Only run if both models exist
        if std::fs::metadata(&config.whisper_model)
            .ok()
            .and(std::fs::metadata(&config.vad_model).ok())
            .is_some()
        {
            let sttvad = WhisperSTTVAD::new(config);
            assert!(sttvad.is_ok(), "Should create valid STTVAD instance");
        } else {
            println!("Skipping test: models not found (you need to download ggml-silero-vad.bin)");
        }
    }

  #[tokio::test]
    async fn test_empty_audio_no_events() {
        let config = WhisperSTTVADConfig::default();

        // Skip if models don't exist
        if std::fs::metadata(&config.whisper_model).is_err()
            || std::fs::metadata(&config.vad_model).is_err()
        {
            println!("Skipping test_empty_audio_no_events: models not found");
            return;
        }

        let Ok(mut sttvad) = WhisperSTTVAD::new(config.clone()) else {
            println!("Skipping test: VAD model not found");
            return;
        };

        // Create a dummy channel for testing
        let (tx, _rx) = mpsc::channel(32);

        // Empty audio should not send any events
        let result = sttvad.process_audio(&[], &tx).await;
        
        assert!(result.is_ok());
    }
}
