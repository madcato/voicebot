use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::error;

use crate::audio::output::AudioOutput;
use crate::tts::TtsEngine;
use super::state::{SharedSession, PipelineEvents};

/// Await a pending playback handle, logging any error. No-op if `None`.
#[allow(dead_code)]
pub async fn drain_play(handle: &mut Option<tokio::task::JoinHandle<anyhow::Result<()>>>) {
    if let Some(h) = handle.take() {
        match h.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => error!(target: "audio", "Playback error: {}", e),
            Err(e) => error!(target: "audio", "Playback task panicked: {}", e),
        }
    }
}

/// TTS task: blocks on SENTENCE_READY, synthesizes and plays each sentence.
pub async fn tts_task(
    shared: Arc<SharedSession>,
    events: Arc<PipelineEvents>,
    tts: Arc<TtsEngine>,
    audio_output: Arc<AudioOutput>,
    tts_sample_rate: u32,
    play_cancel: Arc<AtomicBool>,
    tts_muted: Arc<AtomicBool>,
    #[cfg(feature = "tui")]
    tui_tx: crate::tui::events::TuiEventTx,
    #[cfg(feature = "remote")]
    remote_tts_tx: Arc<
        tokio::sync::Mutex<
            Option<tokio::sync::mpsc::Sender<crate::remote::protocol::TtsAudioPacket>>,
        >,
    >,
) {
    let mut cancel_rx = events.cancel_tx.subscribe();
    let mut play_handle: Option<tokio::task::JoinHandle<anyhow::Result<()>>> = None;
    let mut first_sentence = true;

    loop {
        // Drain the queue immediately; only block when it's empty.
        let sentence = shared.sentences.lock().unwrap().pop_front();
        let sentence = if let Some(s) = sentence {
            s
        } else {
            let cancelled = tokio::select! {
                _ = events.sentence_ready.notified() => false,
                _ = cancel_rx.recv() => true,
            };
            if cancelled {
                play_cancel.store(true, Ordering::SeqCst);
                if let Some(h) = play_handle.take() {
                    h.abort();
                }
                shared.sentences.lock().unwrap().clear();
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                play_cancel.store(false, Ordering::SeqCst);
                first_sentence = true;
                while cancel_rx.try_recv().is_ok() {}
                continue;
            }
            match shared.sentences.lock().unwrap().pop_front() {
                Some(s) => s,
                None => continue,
            }
        };

        if tts_muted.load(Ordering::SeqCst) {
            continue;
        }

        #[cfg(feature = "tui")]
        tui_tx
            .send(crate::tui::events::TuiEvent::StateChange(
                crate::tui::events::PipelineState::Speaking,
            ))
            .ok();

        let tts_c = Arc::clone(&tts);
        let sentence_c = sentence.clone();
        let synth_handle = tokio::task::spawn_blocking(move || tts_c.synthesize(&sentence_c));

        if let Some(h) = play_handle.take() {
            let cancelled = tokio::select! {
                result = h => {
                    if let Ok(Err(e)) = result {
                        error!(target: "audio", "Playback error: {}", e);
                    }
                    false
                },
                _ = cancel_rx.recv() => true,
            };
            if cancelled {
                synth_handle.abort();
                play_cancel.store(true, Ordering::SeqCst);
                shared.sentences.lock().unwrap().clear();
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                play_cancel.store(false, Ordering::SeqCst);
                first_sentence = true;
                while cancel_rx.try_recv().is_ok() {}
                continue;
            }
        }

        if cancel_rx.try_recv().is_ok() {
            synth_handle.abort();
            shared.sentences.lock().unwrap().clear();
            first_sentence = true;
            while cancel_rx.try_recv().is_ok() {}
            continue;
        }

        let samples = match synth_handle.await {
            Ok(Ok(s)) => s,
            Ok(Err(e)) => {
                error!(target: "tts", "TTS synthesis error: {}", e);
                #[cfg(feature = "tui")]
                tui_tx
                    .send(crate::tui::events::TuiEvent::Error(format!(
                        "TTS synthesis error: {e}"
                    )))
                    .ok();
                continue;
            }
            Err(e) => {
                error!(target: "tts", "TTS task panicked: {}", e);
                #[cfg(feature = "tui")]
                tui_tx
                    .send(crate::tui::events::TuiEvent::Error(format!(
                        "TTS task panicked: {e}"
                    )))
                    .ok();
                continue;
            }
        };

        if first_sentence {
            first_sentence = false;
            shared.first_speech_played.store(true, Ordering::SeqCst);
            if let Some(t0) = shared.t_vad_end.lock().unwrap().as_ref() {
                let latency_ms = t0.elapsed().as_millis();
                tracing::info!(target: "performance", "[+{}ms] SpeechStart → FirstAudioPlayback", latency_ms);
            }
        }

        #[cfg(feature = "remote")]
        {
            let maybe_tx = remote_tts_tx.lock().await.clone();
            if let Some(tx) = maybe_tx {
                let packet = crate::remote::protocol::TtsAudioPacket {
                    samples,
                    sample_rate: tts_sample_rate,
                };
                if tx.send(packet).await.is_err() {
                    tracing::warn!(target: "remote", "Remote TTS channel closed");
                }
                continue;
            }
        }

        let out_c = Arc::clone(&audio_output);
        let cancel_c = Arc::clone(&play_cancel);
        play_handle = Some(tokio::task::spawn_blocking(move || {
            out_c.play_blocking(&samples, tts_sample_rate, &cancel_c)
        }));
    }
}
