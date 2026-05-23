use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::broadcast;
use tokio::sync::mpsc;
use tracing::error;

use super::state::PipelineEvents;
use crate::audio::output::AudioOutput;
use crate::pipeline::frames::PipelineFrame;
use crate::tts::TtsEngine;

/// TTS task: receives sentences from sen_task (and llm_task error paths) via typed channel,
/// synthesizes and plays each one.
#[allow(clippy::too_many_arguments)]
pub async fn tts_task(
    events: Arc<PipelineEvents>,
    t_vad_end: Arc<Mutex<Option<Instant>>>,
    mut sentences_rx: mpsc::Receiver<PipelineFrame>,
    tts: Arc<TtsEngine>,
    audio_output: Arc<AudioOutput>,
    tts_sample_rate: u32,
    play_cancel: Arc<AtomicBool>,
    tts_muted: Arc<AtomicBool>,
    #[cfg(feature = "tui")] tui_tx: crate::tui::events::TuiEventTx,
    #[cfg(feature = "remote")] remote_tts_tx: Arc<
        tokio::sync::Mutex<
            Option<tokio::sync::mpsc::Sender<crate::remote::protocol::TtsAudioPacket>>,
        >,
    >,
    #[cfg(feature = "control")] control_broadcast: crate::control::broadcast::ControlBroadcast,
) {
    let mut cancel_rx = events.barge_in_tx.subscribe();
    let mut play_handle: Option<tokio::task::JoinHandle<anyhow::Result<()>>> = None;
    let mut first_sentence = true;

    loop {
        // Drain queue first for low latency; block only when empty.
        let next = match sentences_rx.try_recv() {
            Ok(frame) => Some(frame),
            Err(mpsc::error::TryRecvError::Empty) => {
                tokio::select! {
                    frame = sentences_rx.recv() => frame,
                    _ = cancel_rx.recv() => {
                        handle_barge_in(
                            &play_cancel,
                            &mut play_handle,
                            &mut sentences_rx,
                            &mut cancel_rx,
                            &mut first_sentence,
                        ).await;
                        continue;
                    }
                }
            }
            Err(mpsc::error::TryRecvError::Disconnected) => break,
        };

        let (sentence, utterance_id) = match next {
            Some(PipelineFrame::SentenceReady {
                sentence,
                utterance_id,
            }) => (sentence, utterance_id),
            Some(_) => continue,
            None => break,
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
        #[cfg(feature = "control")]
        control_broadcast.send(crate::control::broadcast::ControlEvent::TtsStart { utterance_id });

        // Ensure previous playback fully stops before starting next sentence.
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
                handle_barge_in(
                    &play_cancel,
                    &mut play_handle,
                    &mut sentences_rx,
                    &mut cancel_rx,
                    &mut first_sentence,
                )
                .await;
                continue;
            }
        }

        let tts_c = Arc::clone(&tts);
        let sentence_c = sentence.clone();
        let mut synth_handle = tokio::task::spawn_blocking(move || tts_c.synthesize(&sentence_c));

        let samples = tokio::select! {
            _ = cancel_rx.recv() => {
                synth_handle.abort();
                handle_barge_in(
                    &play_cancel,
                    &mut play_handle,
                    &mut sentences_rx,
                    &mut cancel_rx,
                    &mut first_sentence,
                ).await;
                continue;
            }
            result = &mut synth_handle => {
                match result {
                    Ok(Ok(s)) => s,
                    Ok(Err(e)) => {
                        error!(target: "tts", "TTS synthesis error: {}", e);
                        #[cfg(feature = "tui")]
                        tui_tx.send(crate::tui::events::TuiEvent::Error(format!("TTS synthesis error: {e}"))).ok();
                        continue;
                    }
                    Err(e) => {
                        error!(target: "tts", "TTS task panicked: {}", e);
                        #[cfg(feature = "tui")]
                        tui_tx.send(crate::tui::events::TuiEvent::Error(format!("TTS task panicked: {e}"))).ok();
                        continue;
                    }
                }
            }
        };

        if first_sentence {
            first_sentence = false;
            if let Some(t0) = t_vad_end.lock().unwrap().as_ref() {
                tracing::info!(target: "performance", "[+{}ms] SpeechStart → FirstAudioPlayback", t0.elapsed().as_millis());
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

async fn handle_barge_in(
    play_cancel: &Arc<AtomicBool>,
    play_handle: &mut Option<tokio::task::JoinHandle<anyhow::Result<()>>>,
    sentences_rx: &mut mpsc::Receiver<PipelineFrame>,
    cancel_rx: &mut broadcast::Receiver<u64>,
    first_sentence: &mut bool,
) {
    // Single ownership of play_cancel in this task avoids cross-writer races.
    play_cancel.store(true, Ordering::SeqCst);

    if let Some(handle) = play_handle.take() {
        match handle.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => error!(target: "audio", "Playback error during barge-in: {}", e),
            Err(e) => error!(target: "audio", "Playback task join failed during barge-in: {}", e),
        }
    }

    while sentences_rx.try_recv().is_ok() {}
    while cancel_rx.try_recv().is_ok() {}

    *first_sentence = true;
    play_cancel.store(false, Ordering::SeqCst);
}
