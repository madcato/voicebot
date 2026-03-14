use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::watch;
use super::WhisperStt;

/// Runs Whisper continuously in the background so the transcript is ready
/// (or nearly ready) when VAD fires SpeechEnd.
///
/// # How it works
///
/// The audio loop calls `submit()` whenever the speech buffer grows —
/// starting immediately at SpeechStart and then every ~500ms of new audio.
/// A background task picks up each new audio snapshot and runs Whisper.
/// Because Whisper is always working, by the time SpeechEnd fires the result
/// for the final audio is usually complete or only milliseconds away.
///
/// `submit()` returns a sequence number.  `await_result(seq)` blocks until
/// a result for that sequence (or a later one) is available.
pub struct SttStream {
    seq: AtomicU64,
    audio_tx: watch::Sender<(u64, Vec<f32>)>,
    /// Kept so callers can clone receivers for `await_result`.
    result_rx: watch::Receiver<(u64, String)>,
}

impl SttStream {
    pub fn new(stt: Arc<WhisperStt>) -> Arc<Self> {
        let (audio_tx, mut audio_rx) = watch::channel::<(u64, Vec<f32>)>((0, vec![]));
        let (result_tx, result_rx) = watch::channel::<(u64, String)>((0, String::new()));

        tokio::spawn(async move {
            loop {
                // Block until a new audio snapshot is submitted.
                if audio_rx.changed().await.is_err() {
                    break;
                }
                let (seq, audio) = audio_rx.borrow_and_update().clone();
                if audio.is_empty() {
                    continue;
                }

                let stt_c = Arc::clone(&stt);
                let secs = audio.len() as f32 / 16_000.0;
                tracing::debug!(target: "stt", "Streaming STT seq={seq} ({secs:.1}s audio)");
                let t = std::time::Instant::now();

                match tokio::task::spawn_blocking(move || stt_c.transcribe(&audio)).await {
                    Ok(Ok(text)) => {
                        tracing::debug!(
                            target: "stt",
                            "Streaming STT seq={seq} done in {}ms: {:?}",
                            t.elapsed().as_millis(),
                            &text[..text.len().min(60)]
                        );
                        let _ = result_tx.send((seq, text));
                    }
                    Ok(Err(e)) => tracing::error!(target: "stt", "STT error: {e}"),
                    Err(e) => tracing::error!(target: "stt", "STT task panicked: {e}"),
                }
            }
        });

        Arc::new(Self { seq: AtomicU64::new(0), audio_tx, result_rx })
    }

    /// Submit a new audio snapshot for processing.
    /// Returns the sequence number — pass it to `await_result`.
    pub fn submit(&self, audio: Vec<f32>) -> u64 {
        let s = self.seq.fetch_add(1, Ordering::SeqCst) + 1;
        let _ = self.audio_tx.send((s, audio));
        s
    }

    /// Wait until a Whisper result for sequence >= `min_seq` is ready.
    pub async fn await_result(&self, min_seq: u64) -> String {
        let mut rx = self.result_rx.clone();
        loop {
            {
                let guard = rx.borrow();
                if guard.0 >= min_seq {
                    return guard.1.clone();
                }
            }
            if rx.changed().await.is_err() {
                return String::new();
            }
        }
    }
}

#[cfg(test)]
impl SttStream {
    /// Creates a mock SttStream that immediately returns `transcript` for any
    /// `await_result(seq)` where seq <= 1. No background Whisper task is started.
    /// Use in integration tests to bypass STT and inject a known transcript.
    pub(crate) fn mock(transcript: String) -> Arc<Self> {
        let (audio_tx, _audio_rx) = watch::channel::<(u64, Vec<f32>)>((0, vec![]));
        let (result_tx, result_rx) = watch::channel::<(u64, String)>((0, String::new()));
        // Pre-load seq=1 so await_result(1) returns immediately without blocking.
        let _ = result_tx.send((1, transcript));
        // result_tx drops here — that's fine because await_result reads the pre-loaded
        // value (1 >= 1) without calling changed().
        Arc::new(Self { seq: AtomicU64::new(1), audio_tx, result_rx })
    }
}
