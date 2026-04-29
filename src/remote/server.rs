use axum::extract::ws::{Message, WebSocket};
use axum::extract::{State, WebSocketUpgrade};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use futures_util::stream::StreamExt;
use futures_util::SinkExt;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, Mutex};
use tracing::{error, info, warn};

use crate::audio::audio_capture::AudioChunk;

use super::protocol::{ClientMessage, ServerMessage, TtsAudioPacket};

/// Shared state for the WebSocket server.
pub struct RemoteState {
    /// Pipeline audio input channel (same one AudioCapture writes to).
    pub audio_tx: async_channel::Sender<AudioChunk>,
    /// Samples per audio chunk (matches pipeline expectation).
    pub samples_per_chunk: usize,
    /// Barge-in: broadcast cancel signal (payload = utterance_id).
    pub barge_in_tx: broadcast::Sender<u64>,
    /// Barge-in: atomic flag for play_blocking.
    pub play_cancel: Arc<AtomicBool>,
    /// TTS audio routing: when Some, TTS sends audio here instead of CPAL.
    pub tts_audio_tx: Arc<Mutex<Option<mpsc::Sender<TtsAudioPacket>>>>,
    /// True when a remote client is connected.
    pub connected: AtomicBool,
}

/// Start the WebSocket server. Returns a JoinHandle for the server task.
pub async fn start_server(
    port: u16,
    state: Arc<RemoteState>,
) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/ws", get(ws_upgrade))
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!(target: "remote", "WebSocket server listening on {addr}");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn ws_upgrade(
    ws: WebSocketUpgrade,
    State(state): State<Arc<RemoteState>>,
) -> impl IntoResponse {
    // Only allow one connection at a time.
    if state.connected.load(Ordering::SeqCst) {
        return (StatusCode::CONFLICT, "Another remote client is already connected")
            .into_response();
    }

    ws.on_upgrade(move |socket| handle_connection(socket, state))
        .into_response()
}

async fn handle_connection(socket: WebSocket, state: Arc<RemoteState>) {
    // Mark as connected.
    if state
        .connected
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        // Race: another connection won.
        let (mut tx, _) = socket.split();
        let msg = serde_json::to_string(&ServerMessage::Error {
            message: "Another client connected simultaneously".into(),
        })
        .unwrap();
        let _ = tx.send(Message::Text(msg.into())).await;
        return;
    }

    info!(target: "remote", "Remote client connected");

    let (ws_write, ws_read) = socket.split();

    // Channel for TTS audio → WS sink.
    let (tts_tx, tts_rx) = mpsc::channel::<TtsAudioPacket>(32);

    // Install TTS routing: pipeline will send audio here instead of CPAL.
    {
        let mut guard = state.tts_audio_tx.lock().await;
        *guard = Some(tts_tx);
    }

    // We need to split ws_read into two paths:
    // 1. Binary frames → ws_audio_source (audio pipeline)
    // 2. Text frames → control message handler
    //
    // Since axum gives us a single stream, we use a fan-out channel.
    let (binary_tx, binary_rx) = async_channel::bounded::<Vec<u8>>(200);

    // Wrap binary_rx as a fake SplitStream by converting binary data back into
    // AudioChunks inside a dedicated task. This avoids the SplitStream type
    // requirement and gives us a cleaner separation.
    let audio_tx = state.audio_tx.clone();
    let samples_per_chunk = state.samples_per_chunk;

    // Task: read WS messages, fan out binary vs text.
    let barge_in_tx = state.barge_in_tx.clone();
    let play_cancel = Arc::clone(&state.play_cancel);
    let ws_write = Arc::new(tokio::sync::Mutex::new(ws_write));
    let ws_write_ctrl = Arc::clone(&ws_write);

    let reader_handle = tokio::spawn(async move {
        let mut ws_read = ws_read;
        while let Some(msg) = ws_read.next().await {
            let msg = match msg {
                Ok(m) => m,
                Err(e) => {
                    warn!(target: "remote", "WS read error: {e}");
                    break;
                }
            };

            match msg {
                Message::Binary(data) => {
                    if binary_tx.try_send(data.to_vec()).is_err() {
                        warn!(target: "remote", "Binary channel full, dropping frame");
                    }
                }
                Message::Text(text) => {
                    match serde_json::from_str::<ClientMessage>(&text) {
                        Ok(ClientMessage::SessionStart { sample_rate: _ }) => {
                            info!(target: "remote", "Session started");
                            let ready =
                                serde_json::to_string(&ServerMessage::SessionReady).unwrap();
                            let mut tx = ws_write_ctrl.lock().await;
                            let _ = tx.send(Message::Text(ready.into())).await;
                        }
                        Ok(ClientMessage::BargeIn) => {
                            info!(target: "remote", "Barge-in from remote client");
                            play_cancel.store(true, Ordering::SeqCst);
                            let _ = barge_in_tx.send(0);
                        }
                        Err(e) => {
                            warn!(target: "remote", "Unknown control message: {e}");
                        }
                    }
                }
                Message::Close(_) => {
                    info!(target: "remote", "WS close frame received");
                    break;
                }
                _ => {}
            }
        }
        // Signal binary channel closed.
        binary_tx.close();
    });

    // Task: read binary audio data → AudioChunk → pipeline.
    let audio_handle = tokio::spawn(async move {
        let mut buffer: Vec<f32> = Vec::with_capacity(samples_per_chunk * 2);

        while let Ok(data) = binary_rx.recv().await {
            for pair in data.chunks_exact(2) {
                let sample = i16::from_le_bytes([pair[0], pair[1]]);
                buffer.push(sample as f32 / i16::MAX as f32);
            }

            while buffer.len() >= samples_per_chunk {
                let chunk_samples: Vec<f32> = buffer.drain(..samples_per_chunk).collect();
                let _ = audio_tx.try_send(AudioChunk {
                    samples: chunk_samples,
                    sample_rate: 16_000,
                    channels: 1,
                });
            }
        }
    });

    // Task: TTS audio → WS binary frames.
    let play_cancel_sink = Arc::clone(&state.play_cancel);
    let ws_write_sink = ws_write;
    let sink_handle = tokio::spawn(async move {
        // We need to send binary frames from tts_rx through ws_write_sink.
        // Since ws_audio_sink expects a SplitSink, we'll implement the send
        // loop directly here for simplicity.
        let mut tts_rx = tts_rx;

        while let Some(packet) = tts_rx.recv().await {
            if play_cancel_sink.load(Ordering::Relaxed) {
                continue;
            }

            let mono = if packet.sample_rate != 16_000 {
                match resample_mono_simple(&packet.samples, packet.sample_rate, 16_000) {
                    Ok(r) => r,
                    Err(e) => {
                        error!(target: "remote", "Resample error: {e}");
                        continue;
                    }
                }
            } else {
                packet.samples
            };

            let mut tx = ws_write_sink.lock().await;

            // audio.start
            let start_json = serde_json::to_string(&ServerMessage::AudioStart).unwrap();
            if tx.send(Message::Text(start_json.into())).await.is_err() {
                break;
            }

            // Send audio in 20ms frames (320 samples @ 16kHz = 640 bytes).
            for chunk in mono.chunks(320) {
                if play_cancel_sink.load(Ordering::Relaxed) {
                    break;
                }
                let bytes = f32_to_i16le(chunk);
                if tx.send(Message::Binary(bytes.into_iter().collect())).await.is_err() {
                    return;
                }
            }

            // audio.end
            let end_json = serde_json::to_string(&ServerMessage::AudioEnd).unwrap();
            if tx.send(Message::Text(end_json.into())).await.is_err() {
                break;
            }
        }
    });

    // Wait for reader to finish (client disconnected).
    let _ = reader_handle.await;

    // Clean up.
    audio_handle.abort();
    sink_handle.abort();

    // Remove TTS routing — audio goes back to local speakers.
    {
        let mut guard = state.tts_audio_tx.lock().await;
        *guard = None;
    }

    state.connected.store(false, Ordering::SeqCst);
    info!(target: "remote", "Remote client disconnected, restored local audio");
}

fn f32_to_i16le(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let i = (clamped * i16::MAX as f32) as i16;
        bytes.extend_from_slice(&i.to_le_bytes());
    }
    bytes
}

fn resample_mono_simple(
    samples: &[f32],
    from_rate: u32,
    to_rate: u32,
) -> anyhow::Result<Vec<f32>> {
    use rubato::{FftFixedIn, Resampler};

    let chunk_size = 1024usize;
    let mut resampler = FftFixedIn::<f32>::new(
        from_rate as usize,
        to_rate as usize,
        chunk_size,
        2,
        1,
    )?;

    let mut output = Vec::new();
    let mut pos = 0;

    while pos + chunk_size <= samples.len() {
        let input = vec![samples[pos..pos + chunk_size].to_vec()];
        let result = resampler.process(&input, None)?;
        output.extend_from_slice(&result[0]);
        pos += chunk_size;
    }

    if pos < samples.len() {
        let mut tail = samples[pos..].to_vec();
        tail.resize(chunk_size, 0.0);
        let input = vec![tail];
        let result = resampler.process(&input, None)?;
        let remaining = samples.len() - pos;
        let expected =
            (remaining as f64 * to_rate as f64 / from_rate as f64).ceil() as usize;
        let take = expected.min(result[0].len());
        output.extend_from_slice(&result[0][..take]);
    }

    Ok(output)
}
