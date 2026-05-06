use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::mpsc;
use tracing::error;

use crate::audio::output::AudioOutput;
use crate::pipeline::frames::PipelineFrame;
use crate::tts::TtsEngine;
use super::state::PipelineEvents;

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
    #[cfg(feature = "tui")]
    tui_tx: crate::tui::events::TuiEventTx,
    #[cfg(feature = "remote")]
    remote_tts_tx: Arc<
        tokio::sync::Mutex<
            Option<tokio::sync::mpsc::Sender<crate::remote::protocol::TtsAudioPacket>>,
        >,
    >,
    #[cfg(feature = "control")]
    control_broadcast: crate::control::broadcast::ControlBroadcast,
) {
    let mut cancel_rx = events.barge_in_tx.subscribe();
    let mut play_handle: Option<tokio::task::JoinHandle<anyhow::Result<()>>> = None;
    let mut first_sentence = true;

    loop {
        // Drain the queue immediately; only block when it is empty.
        let (sentence, utterance_id) = match sentences_rx.try_recv() {
            Ok(PipelineFrame::SentenceReady { sentence, utterance_id }) => (sentence, utterance_id),
            Ok(_) => continue,
            Err(_) => {
                // Channel empty — block until next sentence or cancellation.
                let cancelled = tokio::select! {
                    frame = sentences_rx.recv() => {
                        match frame {
                            Some(PipelineFrame::SentenceReady { sentence, utterance_id: uid }) => {
                                // Process this sentence in the main path below.
                                // Re-enter the loop with the sentence queued so we fall through.
                                // The simplest approach: just loop back; try_recv picks it up.
                                // Instead, directly yield it via a synthetic Ok path by
                                // using a one-element buffer trick:
                                // We handle it here directly to avoid extra machinery.
                                if tts_muted.load(Ordering::SeqCst) { continue; }

                                #[cfg(feature = "tui")]
                                tui_tx.send(crate::tui::events::TuiEvent::StateChange(
                                    crate::tui::events::PipelineState::Speaking,
                                )).ok();
                                #[cfg(feature = "control")]
                                control_broadcast.send(crate::control::broadcast::ControlEvent::TtsStart {
                                    utterance_id: uid,
                                });

                                // Await previous play handle before synthesizing.
                                if let Some(h) = play_handle.take() {
                                    if let Ok(Err(e)) = h.await {
                                        error!(target: "audio", "Playback error: {}", e);
                                    }
                                }

                                let tts_c = Arc::clone(&tts);
                                let sentence_c = sentence.clone();
                                let samples = match tokio::task::spawn_blocking(move || tts_c.synthesize(&sentence_c)).await {
                                    Ok(Ok(s)) => s,
                                    Ok(Err(e)) => { error!(target: "tts", "TTS synthesis error: {}", e); continue; }
                                    Err(e) => { error!(target: "tts", "TTS task panicked: {}", e); continue; }
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
                                        let packet = crate::remote::protocol::TtsAudioPacket { samples, sample_rate: tts_sample_rate };
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
                                false
                            }
                            Some(_) => false,
                            None => false,
                        }
                    },
                    _ = cancel_rx.recv() => true,
                };
                if cancelled {
                    play_cancel.store(true, Ordering::SeqCst);
                    if let Some(h) = play_handle.take() { h.abort(); }
                    while sentences_rx.try_recv().is_ok() {}
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                    play_cancel.store(false, Ordering::SeqCst);
                    first_sentence = true;
                    while cancel_rx.try_recv().is_ok() {}
                }
                continue;
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
        #[cfg(feature = "control")]
        control_broadcast.send(crate::control::broadcast::ControlEvent::TtsStart {
            utterance_id,
        });

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
                while sentences_rx.try_recv().is_ok() {}
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                play_cancel.store(false, Ordering::SeqCst);
                first_sentence = true;
                while cancel_rx.try_recv().is_ok() {}
                continue;
            }
        }

        if cancel_rx.try_recv().is_ok() {
            synth_handle.abort();
            while sentences_rx.try_recv().is_ok() {}
            first_sentence = true;
            while cancel_rx.try_recv().is_ok() {}
            continue;
        }

        let samples = match synth_handle.await {
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
