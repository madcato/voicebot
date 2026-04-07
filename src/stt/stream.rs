use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc as std_mpsc;
use tokio::sync::{oneshot, watch};
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
///
/// # Thread affinity
///
/// CoreML and Metal backends in whisper.cpp have thread-local state. Running
/// two consecutive `whisper_full_with_state()` calls on different OS threads
/// (as `tokio::task::spawn_blocking` may do) can corrupt the decoder state and
/// cause every-other-transcription failures.
///
/// This implementation uses a single dedicated `std::thread` (named
/// "whisper-worker") for ALL transcription calls. The tokio worker task sends
/// audio via `std::sync::mpsc` and receives results via `tokio::sync::oneshot`.
pub struct SttStream {
    seq: AtomicU64,
    /// Carries (seq, full_audio_snapshot, utterance_id).
    audio_tx: watch::Sender<(u64, Vec<f32>, u64)>,
    result_rx: watch::Receiver<(u64, Result<String, String>)>,
}

/// Message sent to the dedicated Whisper OS thread.
struct WhisperWork {
    audio: Vec<f32>,
    reply: oneshot::Sender<Result<String, anyhow::Error>>,
}

impl SttStream {
    pub fn new(stt: Arc<WhisperStt>) -> Arc<Self> {
        // ── Dedicated Whisper thread ──────────────────────────────────────────
        // All `stt.transcribe()` calls happen on this single OS thread so that
        // CoreML / Metal thread-local state is never split across pool threads.
        let (work_tx, work_rx) = std_mpsc::channel::<WhisperWork>();

        std::thread::Builder::new()
            .name("whisper-worker".to_string())
            .spawn(move || {
                while let Ok(WhisperWork { audio, reply }) = work_rx.recv() {
                    let result = stt.transcribe(&audio);
                    // reply_rx may have been dropped if the 20-s timeout fired.
                    let _ = reply.send(result);
                }
                tracing::warn!(target: "stt", "Whisper worker thread exiting (channel closed)");
            })
            .expect("Failed to spawn whisper-worker thread");

        // ── Async coordinator task ────────────────────────────────────────────
        // Picks up the latest audio snapshot from the watch channel, dispatches
        // it to the dedicated thread, and publishes the result.
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

                let secs = audio.len() as f32 / 16_000.0;
                tracing::debug!(target: "stt", "Streaming STT seq={seq} ({secs:.1}s audio)");
                let t = std::time::Instant::now();

                let (reply_tx, reply_rx) = oneshot::channel();
                if work_tx.send(WhisperWork { audio, reply: reply_tx }).is_err() {
                    tracing::error!(target: "stt", "Whisper worker thread is gone — cannot transcribe");
                    let _ = result_tx.send((seq, Err("Whisper worker died".to_string())));
                    continue;
                }

                match tokio::time::timeout(std::time::Duration::from_secs(20), reply_rx).await {
                    Ok(Ok(Ok(text))) => {
                        tracing::debug!(
                            target: "stt",
                            "Streaming STT seq={seq} done in {}ms: {:?}",
                            t.elapsed().as_millis(),
                            &text[..text.len().min(60)]
                        );
                        let _ = result_tx.send((seq, Ok(text)));
                    }
                    Ok(Ok(Err(e))) => {
                        tracing::error!(target: "stt", "STT error: {e}");
                        let _ = result_tx.send((seq, Err(e.to_string())));
                    }
                    Ok(Err(_cancelled)) => {
                        // oneshot dropped — shouldn't happen but publish an error
                        tracing::error!(target: "stt", "STT reply channel dropped for seq={seq}");
                        let _ = result_tx.send((seq, Err("STT reply channel dropped".to_string())));
                    }
                    Err(_timeout) => {
                        // Whisper hung (CoreML/ANE deadlock or runaway inference).
                        // The work_tx message is in the thread's queue but we unblock
                        // the pipeline. The dedicated thread will remain stuck — there
                        // is no clean way to kill it, so we restart the process.
                        tracing::error!(
                            target: "stt",
                            "STT timeout after 20s (seq={seq}, {secs:.1}s audio) — Whisper hung. Restarting process."
                        );
                        let _ = result_tx.send((seq, Err("STT timeout — restarting".to_string())));
                        // Give the error a moment to propagate before exit.
                        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                        std::process::exit(1);
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

    /// Submit a 1-second silence to warm up the CoreML / ANE JIT on the
    /// dedicated Whisper thread. Call once after `SttStream::new()`.
    pub async fn warmup(&self) {
        let silence = vec![0.0f32; 16_000];
        let seq = self.submit(silence, 0);
        // Ignore the result — we just need the thread to run once.
        let _ = self.await_result(seq).await;
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
