use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use crate::stt::whisper_plus::WhisperSttPlus;

/// Streaming STT using whisper-cpp-plus for true live audio-to-text.
/// 
/// With whisper-cpp-plus native streaming support, this is now minimal:
/// - submit() fires-and-forget background transcription
/// - await_result(audio) does immediate synchronous transcription
pub struct SttStream {
    stt: Arc<WhisperSttPlus>,
    epoch: AtomicU64,
}

impl SttStream {
    pub fn new(stt: Arc<WhisperSttPlus>) -> Arc<Self> {
        Arc::new(Self {
            stt,
            epoch: AtomicU64::new(0),
        })
    }

    /// Get current utterance epoch
    pub fn get_epoch(&self) -> u64 {
        self.epoch.load(Ordering::SeqCst)
    }

    /// Increment epoch (called at SpeechStart)
    pub fn next_epoch(&self) -> u64 {
        self.epoch.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Warm up the model (JIT compilation on first inference)
    pub async fn warmup(&self) {
        let stt = Arc::clone(&self.stt);
        tokio::task::spawn_blocking(move || {
            // Feed silence to trigger JIT/ANE initialization
            let silence = vec![0.0f32; 16_000]; // 1 second @ 16kHz
            let _ = stt.transcribe_complete(&silence);
        }).await.ok();
    }

    /// Submit audio for background transcription (fire-and-forget).
    /// The id parameter is kept for API compatibility but ignored.
    pub fn submit(&self, audio: Vec<f32>, _id: u64) -> u64 {
        let stt = Arc::clone(&self.stt);
        
        tokio::spawn(async move {
            let _result = tokio::task::spawn_blocking(move || {
                stt.transcribe_complete(&audio)
            })
            .await;
            // Results are not stored/retrieved - just fire and forget
        });

        // Return ID for API compatibility (ignored by caller)
        _id
    }

    /// Transcribe audio synchronously. Returns immediately with result.
    /// This is the fast path for final transcription when VAD ends.
    pub async fn await_result(&self, audio: Vec<f32>) -> anyhow::Result<String> {
        let stt = Arc::clone(&self.stt);
        
        tokio::task::spawn_blocking(move || {
            stt.transcribe_complete(&audio)
        })
        .await?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_creation() {
        let model_path = std::env::var("WHISPER_MODEL")
            .unwrap_or_else(|_| "models/ggml-large-v3-turbo.bin".to_string());

        if std::fs::metadata(&model_path).is_err() {
            eprintln!("Skipping test: model not found");
            return;
        }

        let stt = Arc::new(WhisperSttPlus::new(&model_path, "es", 0).unwrap());
        let stream = SttStream::new(stt);

        // Test basic operations
        assert!(stream.get_epoch() == 0);
        assert!(stream.next_epoch() == 1);
    }
}
