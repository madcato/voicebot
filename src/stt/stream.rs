use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::watch;
use super::WhisperStt;

/// Runs Whisper continuously in the background so the transcript is ready
/// (or nearly ready) when VAD fires SpeechEnd.
///
/// # How it works
///
/// The audio loop calls `submit()` with a growing full snapshot of the current
/// utterance — starting at `stt_min_submit_ms` and then every
/// `stt_submit_interval_ms` of new audio during speech.
///
/// A background worker picks up each new snapshot and runs the full Whisper
/// decoder on it. Because Whisper is always working on the latest audio, by
/// the time `SpeechEnd` fires the result for the final audio is usually
/// complete or only a short delta away.
///
/// # Why full snapshots (not deltas)
///
/// Delta decoding (processing only the new tail with a text prompt for context)
/// causes word-boundary errors: the encoder has no acoustic context for the
/// first word of each delta, and silence-only deltas at SpeechEnd cause
/// hallucinations. Full-snapshot decoding is slower per call but correct.
///
/// # Utterance isolation
///
/// Each `submit()` carries an `utterance_id`. A new utterance_id signals that
/// the user started speaking again; any result published after that point
/// belongs to the new utterance. `await_result` relies only on seq numbers,
/// so stale-epoch checks in main.rs handle discard of old results.
///
/// # Sequence numbers
///
/// `submit()` returns a monotonically increasing `seq`. `await_result(seq)`
/// blocks until a result for that seq (or later) is available. A result is
/// **always** published for every submit — even if skipped — so `await_result`
/// never blocks forever.
pub struct SttStream {
    seq: AtomicU64,
    /// Carries (seq, full_audio_snapshot, utterance_id).
    audio_tx: watch::Sender<(u64, Vec<f32>, u64)>,
    result_rx: watch::Receiver<(u64, Result<String, String>)>,
}

impl SttStream {
    pub fn new(stt: Arc<WhisperStt>) -> Arc<Self> {
        let (audio_tx, mut audio_rx) = watch::channel::<(u64, Vec<f32>, u64)>((0, vec![], 0));
        let (result_tx, result_rx) =
            watch::channel::<(u64, Result<String, String>)>((0, Ok(String::new())));

        tokio::spawn(async move {
            loop {
                if audio_rx.changed().await.is_err() {
                    break;
                }
                let (seq, audio, _utterance_id) = audio_rx.borrow_and_update().clone();
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
                        let _ = result_tx.send((seq, Ok(text)));
                    }
                    Ok(Err(e)) => {
                        tracing::error!(target: "stt", "STT error: {e}");
                        let _ = result_tx.send((seq, Err(e.to_string())));
                    }
                    Err(e) => {
                        tracing::error!(target: "stt", "STT task panicked: {e}");
                        let _ = result_tx.send((seq, Err(format!("STT task panicked: {e}"))));
                    }
                }
            }
        });

        Arc::new(Self {
            seq: AtomicU64::new(0),
            audio_tx,
            result_rx,
        })
    }

    /// Submit a new audio snapshot for processing.
    ///
    /// `utterance_id` is the current utterance epoch — pass the same value for
    /// all submits within one utterance (speculative and final).
    ///
    /// Returns the sequence number; pass it to `await_result`.
    pub fn submit(&self, audio: Vec<f32>, utterance_id: u64) -> u64 {
        let s = self.seq.fetch_add(1, Ordering::SeqCst) + 1;
        let _ = self.audio_tx.send((s, audio, utterance_id));
        s
    }

    /// Returns a cloned watch receiver for monitoring STT results.
    pub fn result_receiver(&self) -> watch::Receiver<(u64, Result<String, String>)> {
        self.result_rx.clone()
    }

    /// Wait until a Whisper result for sequence >= `min_seq` is ready.
    /// Returns `Ok(text)` on success or `Err(msg)` if Whisper failed.
    pub async fn await_result(&self, min_seq: u64) -> Result<String, String> {
        let mut rx = self.result_rx.clone();
        loop {
            {
                let guard = rx.borrow();
                if guard.0 >= min_seq {
                    return guard.1.clone();
                }
            }
            if rx.changed().await.is_err() {
                return Ok(String::new());
            }
        }
    }
}

#[cfg(test)]
impl SttStream {
    /// Creates a mock SttStream that immediately returns `transcript` for any
    /// `await_result(seq)` where seq <= 1.
    pub(crate) fn mock(transcript: String) -> Arc<Self> {
        let (audio_tx, _audio_rx) =
            watch::channel::<(u64, Vec<f32>, u64)>((0, vec![], 0));
        let (result_tx, result_rx) =
            watch::channel::<(u64, Result<String, String>)>((0, Ok(String::new())));
        let _ = result_tx.send((1, Ok(transcript)));
        Arc::new(Self {
            seq: AtomicU64::new(1),
            audio_tx,
            result_rx,
        })
    }
}
